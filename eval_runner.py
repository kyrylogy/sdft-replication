"""Unified, config-driven, adapter-aware evaluation.

(Named eval_runner, NOT evaluate — a module called evaluate.py on sys.path would
shadow HuggingFace's `evaluate` library that lm-eval imports in the forgetting run.)

One entrypoint for both tasks, both engines, and both tunings:
  - accuracy eval on tooluse (strict|current scorer) and science (<answer> match)
  - LoRA adapters via the HF engine; full-FT runs via their saved final_model/checkpoints
  - the base anchor (no adapter) through the SAME code path
  - a checkpoint curve (evaluate every saved checkpoint -> forgetting trajectory)
  - Wilson 95% CIs on every accuracy
  - the lm-eval forgetting battery (per-task; base anchor uses identical settings)

Generation + scoring are importable so train.py's stage-2 retention gate scores with
the exact same logic used for reported numbers.

  python eval_runner.py --config <cfg> [--adapter <dir>|--base] [--mode accuracy|forgetting]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

from exp_config import load_config, stamp_run
import datasets_sdft as D


# ---------------------------------------------------------------------------
# Scorers (tooluse strict = the audit's ordered element-wise pairing; default)
# ---------------------------------------------------------------------------
def _extract_action_pairs(text):
    pairs = []
    for m in re.compile(r'Action:\s*(\w+)\s*\n\s*Action Input:\s*(\{.*?\})', re.DOTALL).finditer(text):
        try:
            pairs.append((m.group(1), json.loads(m.group(2))))
        except json.JSONDecodeError:
            continue
    return pairs


def score_tooluse_strict(response, golden_answer):
    """Ordered element-wise (action, input) match. Returns 1/0, or None if gold ungradeable."""
    gold = []
    for item in golden_answer:
        try:
            gold.append((item["Action"], json.loads(item["Action_Input"])))
        except Exception:
            return None
    pred = _extract_action_pairs(response)
    if len(pred) != len(gold):
        return 0
    return int(all(pn == gn and pi == gi for (pn, pi), (gn, gi) in zip(pred, gold)))


def score_tooluse_current(response, golden_answer):
    """Legacy order-insensitive / merged-input scorer (kept for comparison)."""
    pred_actions = re.findall(r'Action:\s*(\w+)', response)
    pred_inputs = {}
    for block in re.findall(r'Action Input:\s*({.*?})', response, re.DOTALL):
        try:
            pred_inputs.update(json.loads(block))
        except json.JSONDecodeError:
            continue
    gt_actions = [i["Action"] for i in golden_answer]
    gt_inputs = {}
    for i in golden_answer:
        try:
            gt_inputs.update(json.loads(i["Action_Input"]))
        except Exception:
            pass
    return int(Counter(pred_actions) == Counter(gt_actions) and pred_inputs == gt_inputs)


def score_science(response, answer):
    ext = response.split("<answer>")[-1].split("</answer>")[0].strip()
    return int(ext == answer)


def score_rows(dataset, rows, responses, scorer="strict"):
    """Return (accuracy, n_effective, per_sample) skipping ungradeable rows (None)."""
    per = []
    for row, resp in zip(rows, responses):
        if dataset == "tooluse":
            s = score_tooluse_strict(resp, row["golden_answer"]) if scorer == "strict" \
                else score_tooluse_current(resp, row["golden_answer"])
        elif dataset == "science":
            s = score_science(resp, row["answer"])
        else:
            raise ValueError(dataset)
        per.append(s)
    graded = [s for s in per if s is not None]
    n_eff = len(graded)
    acc = (sum(graded) / n_eff) if n_eff else 0.0
    return acc, n_eff, per


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Model loading + generation
# ---------------------------------------------------------------------------
def _device(cfg):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(name, device):
    if device == "cpu":
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_hf(model_id, dtype, device, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_src = adapter if (adapter and os.path.isfile(os.path.join(adapter, "tokenizer_config.json"))) else model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate_hf(model, tokenizer, prompts, device, max_new_tokens=2048, temperature=0.0):
    from tqdm import tqdm
    do_sample = temperature > 0
    out = []
    for prompt in tqdm(prompts, desc="generate(hf)"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        out.append(tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


# ---------------------------------------------------------------------------
# In-process eval (reused by the stage-2 retention gate)
# ---------------------------------------------------------------------------
def eval_in_process(model, tokenizer, dataset, set_name, device,
                    scorer="strict", max_new_tokens=2048, temperature=0.0, sample_size=None):
    """Score an already-loaded model on one split. Returns dict with accuracy."""
    rows, _ = D.load_eval(dataset, set_name, tokenizer, teacher_ceiling=False, max_samples=sample_size)
    prompts = [r["text"] for r in rows]
    responses = generate_hf(model, tokenizer, prompts, device, max_new_tokens, temperature)
    acc, n_eff, per = score_rows(dataset, rows, responses, scorer)
    return {"accuracy": acc, "n_effective": n_eff, "n_total": len(rows)}


# ---------------------------------------------------------------------------
# Accuracy mode (CLI)
# ---------------------------------------------------------------------------
def _eval_one(model_id, adapter, cfg, device, dtype, out_dir, label):
    """Evaluate one target. `adapter` is a LoRA dir (lora) or None (full/base):
    for full-FT, model_id is the full-weight dir itself."""
    dataset = cfg["data"]["dataset"]
    sets = cfg["eval"]["sets"]
    scorer = cfg["eval"]["scorer"]
    engine = cfg["eval"]["engine"]
    mnt = cfg["eval"]["max_new_tokens"]
    temp = cfg["eval"]["temperature"]
    ceiling = cfg["eval"].get("teacher_ceiling", False)

    if engine == "vllm" and adapter:
        raise SystemExit("eval.engine=vllm cannot load a LoRA adapter; set eval.engine=hf")

    # Load the model/engine ONCE, then reuse across all eval sets.
    if engine == "hf":
        model, tokenizer = load_hf(model_id, dtype, device, adapter=adapter)
        def gen(prompts):
            return generate_hf(model, tokenizer, prompts, device, mnt, temp)
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
        llm = LLM(model=model_id, gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
                  enforce_eager=True, dtype=torch.bfloat16, trust_remote_code=True)
        sp = SamplingParams(temperature=temp, max_tokens=mnt,
                            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None)
        def gen(prompts):
            return [o.outputs[0].text for o in llm.generate(prompts, sp)]

    results = {}
    for set_name in sets:
        rows, demo_source = D.load_eval(dataset, set_name, tokenizer, teacher_ceiling=ceiling,
                                        max_samples=cfg["eval"].get("max_samples"))
        responses = gen([r["text"] for r in rows])
        acc, n_eff, per = score_rows(dataset, rows, responses, scorer)
        k = int(round(acc * n_eff))
        lo, hi = wilson_ci(k, n_eff)
        rec = {"accuracy": acc, "n_correct": k, "n_effective": n_eff, "n_total": len(rows),
               "wilson95": [lo, hi], "scorer": scorer, "demo_source": demo_source,
               "teacher_ceiling": ceiling, "per_sample_scores": per}
        results[set_name] = rec
        sub = out_dir / f"{dataset}_{set_name}"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "eval_results.json").write_text(json.dumps(
            {**rec, "label": label, "model_id": model_id, "adapter": adapter}, indent=2))
        (sub / "eval_responses.json").write_text(json.dumps([
            {"prompt": rows[i]["text"], "response": responses[i], "correct": per[i]}
            for i in range(len(rows))], indent=2))
        print(f"[eval:{label}] {dataset}/{set_name}: acc={acc:.4f} ({k}/{n_eff})  95%CI=[{lo:.3f},{hi:.3f}]")
    return results


def _checkpoint_targets(cfg, run_dir, base_id, adapter_override):
    """Return [(model_id, adapter, label)] over the checkpoint curve + final, tuning-aware."""
    tuning = cfg["method"]["tuning"]
    curve = cfg["eval"].get("checkpoint_curve")
    targets = []
    ckpts = sorted(run_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    if tuning == "lora":
        if curve:
            for ck in ckpts:
                if (ck / "adapter_config.json").exists():
                    targets.append((base_id, str(ck), f"step{ck.name.split('-')[-1]}"))
        targets.append((base_id, adapter_override or str(run_dir / "lora_adapter"), "final"))
    else:  # full: the checkpoint dir / final_model IS the model
        if curve:
            for ck in ckpts:
                if (ck / "config.json").exists():
                    targets.append((str(ck), None, f"step{ck.name.split('-')[-1]}"))
        targets.append((adapter_override or str(run_dir / "final_model"), None, "final"))
    return targets


def _wandb_log_eval(cfg, all_res):
    """Log eval accuracies + Wilson CIs to a wandb summary run, grouped by scale×dataset,
    so the whole matrix shows up in the dashboard. Training already logs its own run; this
    is a separate eval-only summary. Guarded — never fails the eval."""
    if cfg.get("runtime", {}).get("report_to") != "wandb":
        return
    try:
        import wandb
        os.environ.setdefault("WANDB_PROJECT", cfg["runtime"]["wandb_project"])
        wandb.init(project=cfg["runtime"]["wandb_project"],
                   name=f"{cfg['_derived']['run_stem']}__eval",
                   group=f"{cfg['model']['scale']}_{cfg['data']['dataset']}",
                   job_type="eval", reinit=True,
                   config={"scorer": cfg["eval"]["scorer"], "sets": cfg["eval"]["sets"],
                           "git_sha": (cfg.get("_provenance", {}) or {}).get("git_sha")})
        for label, sets in all_res.items():
            for set_name, rec in sets.items():
                key = f"eval/{label}/{cfg['data']['dataset']}_{set_name}"
                wandb.summary[f"{key}/accuracy"] = rec["accuracy"]
                wandb.summary[f"{key}/n"] = rec["n_effective"]
                wandb.summary[f"{key}/wilson_lo"], wandb.summary[f"{key}/wilson_hi"] = rec["wilson95"]
        wandb.finish()
        print("[wandb] logged eval summary")
    except Exception as e:
        print(f"[wandb] eval logging skipped: {e.__class__.__name__}: {e}")


def run_accuracy(cfg, adapter_override=None, use_base=False):
    device = _device(cfg)
    dtype = _dtype(cfg["model"]["dtype"], device)
    base_id = cfg["model"]["id"]
    out_root = Path(cfg["_derived"]["output_dir"]) / "eval"

    if use_base:
        # Base model with the demo in-context is the CEILING, not the floor — write it to its
        # own dir so `--base` (floor) and `--base --set eval.teacher_ceiling=true` (ceiling)
        # never overwrite each other.
        label = "ceiling" if cfg["eval"].get("teacher_ceiling") else "base_anchor"
        stamp_run(cfg, out_root / label, extra={"mode": "accuracy", "target": label,
                                                "data_fingerprint": D.fingerprint(cfg["data"]["dataset"])})
        # --base runs never go through train.py, so nothing else stamps the run's top level;
        # collect_results.load_runs() discovers runs via runs/*/resolved_config.yaml (one level
        # deep) and would never see this run dir without it.
        stamp_run(cfg, out_root.parent, extra={"mode": "accuracy", "target": label})
        res = {label: _eval_one(base_id, None, cfg, device, dtype, out_root / label, label)}
        _wandb_log_eval(cfg, res)
        return res

    run_dir = Path(cfg["_derived"]["output_dir"])
    targets = _checkpoint_targets(cfg, run_dir, base_id, adapter_override)

    all_res = {}
    for model_id, adapter, label in targets:
        probe = adapter if adapter else model_id
        if not Path(probe).exists():
            print(f"[eval] skip {label}: {probe} missing")
            continue
        all_res[label] = _eval_one(model_id, adapter, cfg, device, dtype, out_root / label, label)
    stamp_run(cfg, out_root, extra={"mode": "accuracy", "targets": [t[2] for t in targets],
                                    "data_fingerprint": D.fingerprint(cfg["data"]["dataset"])})
    _wandb_log_eval(cfg, all_res)
    return all_res


# ---------------------------------------------------------------------------
# Forgetting mode (lm-eval; base anchor uses identical settings)
# ---------------------------------------------------------------------------
def run_forgetting(cfg, adapter_override=None, use_base=False):
    fg = cfg["eval"]["forgetting"]
    if not fg.get("enabled"):
        raise SystemExit("eval.forgetting.enabled is false; set it true (and num_fewshot) to run the battery")
    base_id = cfg["model"]["id"]
    tuning = cfg["method"]["tuning"]
    run_dir = Path(cfg["_derived"]["output_dir"])
    out_dir = run_dir / ("eval/forgetting_base" if use_base else "eval/forgetting")
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_base:
        model_args = [f"pretrained={base_id}", "dtype=bfloat16"]
    elif tuning == "lora":
        adapter = adapter_override or str(run_dir / "lora_adapter")
        model_args = [f"pretrained={base_id}", f"peft={adapter}", "dtype=bfloat16"]
    else:  # full: point pretrained at the saved full model
        model_args = [f"pretrained={adapter_override or str(run_dir / 'final_model')}", "dtype=bfloat16"]

    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf",
           "--model_args", ",".join(model_args),
           "--tasks", ",".join(fg["tasks"]),
           "--batch_size", str(fg["batch_size"]),
           "--output_path", str(out_dir),
           "--confirm_run_unsafe_code"]
    if fg.get("num_fewshot") is not None:
        cmd += ["--num_fewshot", str(fg["num_fewshot"])]
    env = {**os.environ, "HF_ALLOW_CODE_EVAL": "1"}
    stamp_run(cfg, out_dir, extra={"mode": "forgetting", "target": "base" if use_base else tuning,
                                   "lm_eval_cmd": " ".join(cmd), "num_fewshot": fg.get("num_fewshot")})
    if use_base:
        # same reason as run_accuracy's use_base branch: no train.py stamp exists for this run dir.
        stamp_run(cfg, run_dir, extra={"mode": "forgetting", "target": "base"})
    print(f"[forgetting] {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def dry_report(cfg, use_base, adapter_override):
    """Validate the eval plan (targets, splits, output paths) without loading weights."""
    d, e = cfg["data"], cfg["eval"]
    base_id = cfg["model"]["id"]
    run_dir = Path(cfg["_derived"]["output_dir"])
    print("=" * 72)
    print(f"DRY RUN (eval) — {cfg['_derived']['run_stem']}")
    print("=" * 72)
    print(f"  dataset={d['dataset']} engine={e['engine']} scorer={e['scorer']} sets={e['sets']} "
          f"max_samples={e.get('max_samples')} teacher_ceiling={e.get('teacher_ceiling')}")
    if use_base:
        print(f"  target: BASE anchor {base_id} (no adapter)")
    else:
        for mid, ad, label in _checkpoint_targets(cfg, run_dir, base_id, adapter_override):
            probe = ad if ad else mid
            print(f"  target[{label}]: {'OK' if Path(probe).exists() else 'MISSING'}  {probe}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_id)
        for s in e["sets"]:
            rows, _ = D.load_eval(d["dataset"], s, tok, teacher_ceiling=e.get("teacher_ceiling", False), max_samples=2)
            print(f"  [data] {d['dataset']}/{s}: {len(rows)} rows format OK")
    except Exception as ex:
        print(f"  [data] tokenizer/format check skipped: {ex.__class__.__name__}")
    print(f"  output -> {run_dir / 'eval'}")
    print("DRY RUN OK — nothing evaluated.")


def main():
    ap = argparse.ArgumentParser(description="Unified adapter-aware evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--mode", choices=["accuracy", "forgetting"], default="accuracy")
    ap.add_argument("--adapter", default=None, help="override adapter/model dir (else <output_dir>/{lora_adapter,final_model})")
    ap.add_argument("--base", action="store_true", help="evaluate the bare base model (the anchor)")
    ap.add_argument("--dry_run", action="store_true", help="Validate the eval plan without loading weights.")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.dry_run:
        dry_report(cfg, args.base, args.adapter)
        return
    if args.mode == "accuracy":
        run_accuracy(cfg, adapter_override=args.adapter, use_base=args.base)
    else:
        run_forgetting(cfg, adapter_override=args.adapter, use_base=args.base)


if __name__ == "__main__":
    main()
