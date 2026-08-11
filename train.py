"""Unified, config-driven trainer for every arm.

One entrypoint spans the whole experiment cube:
    objective ∈ {sft, sdft, online_sft}
    tuning    ∈ {lora, full}
    teacher   ∈ {ema, frozen, none}
    dataset   ∈ {tooluse, science}
    stage     ∈ {1 (fresh), 2 (continue from a stage-1 adapter/checkpoint)}
selected purely by config — no per-arm scripts, so SFT and SDFT share identical
data loading, masking, and defaults and differ ONLY by the objective/teacher flags.

Hardening requirements baked in (see WORK_LOG / memory):
  A. stage-2 EMA teacher carries the stage-1 init (teacher continues, like the student)
  B. stage-2 step-0 retention gate proves the stage-1 skill loaded before any gradient
  C. stage-2 train config is whatever the config says — set it once in a shared parent
     and every arm `extends` it, so it is identical by construction, not by luck
  D. this file IS the unified lora/full entrypoint

Run via run.sh (which sets the GPU), or directly:
    python train.py --config configs/experiments/<arm>.yaml [--set k=v ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainerCallback,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from exp_config import load_config, stamp_run
import datasets_sdft as D


# ---------------------------------------------------------------------------
# Callbacks / collators (defined once, here — no cross-file copies)
# ---------------------------------------------------------------------------
class LoRAEMACallback(TrainerCallback):
    """EMA-mix the student's LoRA into the teacher's LoRA every step:
    teacher_lora <- (1-alpha)*teacher_lora + alpha*student_lora. Recovers the
    paper's Appendix-A.3 EMA-of-student teacher in adapter space."""

    def __init__(self, student_module, teacher_module, alpha: float):
        self.alpha = float(alpha)
        s = {n: p for n, p in student_module.named_parameters() if "lora_" in n}
        t = {n: p for n, p in teacher_module.named_parameters() if "lora_" in n}
        self.pairs = [(sp, t[n]) for n, sp in s.items() if n in t and t[n].shape == sp.shape]
        print(f"[ema] matched {len(self.pairs)} LoRA pairs (student={len(s)}, teacher={len(t)}); alpha={self.alpha}")
        assert self.pairs, "LoRAEMACallback found no matching LoRA pairs — teacher wrap failed"

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        for sp, tp in self.pairs:
            tp.data.mul_(1.0 - self.alpha).add_(sp.data, alpha=self.alpha)


class ResponseOnlyCollator(DataCollatorForLanguageModeling):
    """Mask everything up to and including the last assistant marker -> loss on
    response tokens only (Qwen2.5 chat template lacks {% generation %} tags)."""

    def __init__(self, tokenizer, response_template: str):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.resp_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        L = len(self.resp_ids)
        labels = batch["labels"]
        for i in range(labels.size(0)):
            ids = batch["input_ids"][i].tolist()
            last = -1
            for j in range(len(ids) - L + 1):
                if ids[j:j + L] == self.resp_ids:
                    last = j
            if last < 0:
                labels[i, :] = -100
            else:
                labels[i, :last + L] = -100
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _lora_config(cfg):
    lc = cfg["lora"]
    return LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                      bias=lc["bias"], target_modules=list(lc["target_modules"]),
                      task_type=TaskType.CAUSAL_LM)


def _load_base(model_id, dtype):
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)


# ---------------------------------------------------------------------------
# Student / teacher construction (all four arms, both stages)
# ---------------------------------------------------------------------------
def build_student(cfg, dtype):
    model_id = cfg["model"]["id"]
    tuning = cfg["method"]["tuning"]
    stage = cfg["data"]["stage"]
    base = _load_base(model_id, dtype)

    if tuning == "full":
        if stage == 2:
            # Continue full weights from stage-1 (a saved full model dir).
            base = _load_base(cfg["data"]["init_checkpoint"], dtype)
        student = base  # all params trainable
    else:  # lora
        if stage == 1:
            student = get_peft_model(base, _lora_config(cfg))
        else:
            # Requirement: the student continues the SAME adapter it learned skill 1 in.
            student = PeftModel.from_pretrained(base, cfg["data"]["init_adapter"], is_trainable=True)
        if cfg["runtime"].get("enable_input_require_grads"):
            student.enable_input_require_grads()
    return student


def build_teacher(cfg, dtype):
    """Return (teacher_module, is_peft_ema). None when objective=sft."""
    objective = cfg["method"]["objective"]
    if objective == "sft":
        return None, False
    teacher_mode = cfg["method"]["teacher"]
    tuning = cfg["method"]["tuning"]
    stage = cfg["data"]["stage"]
    base = _load_base(cfg["model"]["id"], dtype)
    for p in base.parameters():
        p.requires_grad_(False)

    if teacher_mode == "frozen":
        return base, False   # bare base; demonstration enters via teacher_prompt

    if teacher_mode == "ema":
        if tuning == "full":
            # Full-FT EMA is handled by DistilTrainer's sync_ref_model machinery.
            # Requirement A (full-FT path): at stage 2 the teacher must CONTINUE from the
            # stage-1 weights like the student, not restart from the bare base.
            if stage == 2:
                base = _load_base(cfg["data"]["init_checkpoint"], dtype)
                for p in base.parameters():
                    p.requires_grad_(False)
                print(f"[ema] full-FT teacher initialized from stage-1: {cfg['data']['init_checkpoint']}")
            return base, False
        # LoRA EMA: teacher carries its own adapter.
        if stage == 1:
            teacher = get_peft_model(base, _lora_config(cfg))  # zero-init B -> teacher == bare base at t0
        else:
            # Requirement A: at stage-2 start the teacher already equals the stage-1
            # model, so its distillation target is conditioned on a Tool-Use-trained
            # model — not a bare base that never learned skill 1.
            teacher = PeftModel.from_pretrained(base, cfg["data"]["init_adapter"], is_trainable=False)
            print(f"[ema] teacher adapter initialized from stage-1: {cfg['data']['init_adapter']}")
        for p in teacher.parameters():
            p.requires_grad_(False)   # EMA touches .data only
        return teacher, True

    raise ValueError(f"unhandled teacher mode {teacher_mode!r}")


# ---------------------------------------------------------------------------
# Trainer configs
# ---------------------------------------------------------------------------
def _epoch_step_args(cfg):
    return dict(num_train_epochs=cfg["train"]["num_train_epochs"], max_steps=-1)


def _save_total_limit(cfg):
    """Keep ALL checkpoints (no GC) when a LoRA forgetting curve is requested — otherwise
    Trainer would prune to save_total_limit and the curve loses its early points. Full-FT
    keeps the limit (full checkpoints are large)."""
    if cfg["eval"].get("checkpoint_curve") and cfg["method"]["tuning"] == "lora":
        return None
    return cfg["runtime"]["save_total_limit"]


def build_distil_config(cfg):
    from distil_config import DistilConfig
    t, s, v, rt = cfg["train"], cfg["sdft"], cfg["vllm"], cfg["runtime"]
    objective = cfg["method"]["objective"]
    tuning, teacher = cfg["method"]["tuning"], cfg["method"]["teacher"]
    full_ema = (tuning == "full" and teacher == "ema")
    return DistilConfig(
        output_dir=cfg["_derived"]["output_dir"],
        # generation backend
        use_vllm=v["enabled"], vllm_mode=v["mode"],
        vllm_gpu_memory_utilization=v["gpu_memory_utilization"],
        vllm_enable_sleep_mode=v["enable_sleep_mode"],
        vllm_importance_sampling_correction=v["importance_sampling_correction"],
        generate_from_teacher=(objective == "online_sft"),
        # precision
        bf16=(cfg["model"]["dtype"] == "bfloat16"), fp16=(cfg["model"]["dtype"] == "float16"),
        # batch / steps
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_prompt_length=t["max_prompt_length"], max_completion_length=t["max_completion_length"],
        num_generations=1, num_iterations=1, **_epoch_step_args(cfg),
        # optimizer
        learning_rate=t["learning_rate"], warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"], max_grad_norm=t["max_grad_norm"],
        # logging / checkpoints
        logging_steps=1, save_steps=rt["save_steps"], save_total_limit=_save_total_limit(cfg),
        save_strategy="steps", report_to=rt["report_to"], run_name=cfg["_derived"]["run_name"],
        log_completions=False,
        # teacher EMA: full-FT via sync_ref_model; LoRA via LoRAEMACallback (added separately)
        sync_ref_model=full_ema, ref_model_sync_steps=1, ref_model_mixup_alpha=s["ema_alpha"],
        # KL / loss
        alpha=s["kl_alpha"], beta=s["beta"], num_loss_tokens_to_skip=s["num_loss_tokens_to_skip"],
        temperature=s["temperature"],
        # misc / memory
        use_transformers_paged=False, cache_implementation=None,
        gradient_checkpointing=t["gradient_checkpointing"], dataloader_pin_memory=False,
        remove_unused_columns=False, seed=t["seed"],
    )


def build_sft_config(cfg):
    from trl import SFTConfig
    t, rt = cfg["train"], cfg["runtime"]
    return SFTConfig(
        output_dir=cfg["_derived"]["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        **_epoch_step_args(cfg),
        learning_rate=t["learning_rate"], warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"], max_grad_norm=t["max_grad_norm"],
        logging_steps=1, save_steps=rt["save_steps"], save_total_limit=_save_total_limit(cfg),
        save_strategy="steps", report_to=rt["report_to"], run_name=cfg["_derived"]["run_name"],
        bf16=(cfg["model"]["dtype"] == "bfloat16"), fp16=(cfg["model"]["dtype"] == "float16"),
        gradient_checkpointing=t["gradient_checkpointing"], dataloader_pin_memory=False,
        remove_unused_columns=True, seed=t["seed"],
        max_length=t["max_prompt_length"] + t["max_completion_length"],
        packing=False, dataset_text_field="text",
    )


# ---------------------------------------------------------------------------
# Requirement B — stage-2 retention gate
# ---------------------------------------------------------------------------
def _read_stage1_accuracy(cfg):
    """Auto-read the stage-1 final accuracy for retention_gate.task/set from the
    stage-1 run's eval outputs, if present. Returns float or None."""
    gate = cfg["retention_gate"]
    if gate.get("expected_accuracy") is not None:
        return float(gate["expected_accuracy"])
    init = cfg["data"].get("init_adapter")
    if not init:
        return None
    stage1_root = Path(init).parent  # <stage1_out>/lora_adapter -> <stage1_out>
    cand = stage1_root / "eval" / "final" / f"{gate['task']}_{gate['set']}" / "eval_results.json"
    if cand.exists():
        return float(json.loads(cand.read_text())["accuracy"])
    return None


def retention_gate(cfg, student, tokenizer):
    gate = cfg["retention_gate"]
    if cfg["data"]["stage"] != 2 or not gate.get("enabled"):
        return

    # Structural: adapter must have loaded into the TRAINABLE path with learned (non-zero)
    # weights. Explicit raises (not assert) so they survive `python -O`.
    if gate.get("structural", True) and cfg["method"]["tuning"] == "lora":
        trainable = [n for n, p in student.named_parameters() if "lora_" in n and p.requires_grad]
        b_nonzero = any(p.abs().sum().item() > 0 for n, p in student.named_parameters() if "lora_B" in n)
        if not trainable:
            raise SystemExit("[gate] no trainable LoRA params — stage-1 adapter did not load into the trainable path")
        if not b_nonzero:
            raise SystemExit("[gate] all LoRA-B are zero — stage-1 adapter is untrained/empty (not the continuation you want)")
        print(f"[gate] structural OK: {len(trainable)} trainable LoRA tensors, B non-zero")

    # Behavioral: the loaded model must reproduce the stage-1 skill number. The model is
    # still on CPU here (the Trainer places it only later), so move it to the eval device
    # first — otherwise this decodes a 7B model on CPU AND compares against the GPU-produced
    # stage-1 number (device/precision drift can trip the tolerance).
    import eval_runner as _ev
    device = _ev._device(cfg)
    student.to(device)
    was_training = student.training
    student.eval()
    res = _ev.eval_in_process(student, tokenizer, gate["task"], gate["set"], device,
                              scorer=cfg["eval"]["scorer"], max_new_tokens=cfg["eval"]["max_new_tokens"],
                              temperature=0.0, sample_size=cfg["eval"].get("max_samples"))
    if was_training:
        student.train()
    loaded = res["accuracy"]
    expected = _read_stage1_accuracy(cfg)
    print(f"[gate] behavioral: loaded {gate['task']}/{gate['set']} acc={loaded:.4f} "
          f"(n={res['n_effective']}); expected={expected}")
    if expected is None:
        # Fail CLOSED: a safety gate that can't find its baseline must stop, not shrug.
        if not gate.get("allow_missing_baseline", False):
            raise SystemExit(
                "[gate] FAIL-CLOSED: no stage-1 expected accuracy. Run the stage-1 eval first "
                "(./run.sh eval <stage1_cfg>) so <init_adapter>/../eval/final/<task>_<set>/eval_results.json "
                "exists, OR set retention_gate.expected_accuracy explicitly. "
                "(retention_gate.allow_missing_baseline=true proceeds on the structural check alone — not recommended.)")
        print("[gate][warn] allow_missing_baseline=true — proceeding on the structural check only; "
              "behavioral retention is NOT enforced.")
    else:
        tol = gate["tolerance"]
        if abs(loaded - expected) > tol:
            raise SystemExit(f"[gate] FAILED: loaded acc {loaded:.4f} deviates from stage-1 "
                             f"{expected:.4f} by > {tol}. Adapter likely didn't continue correctly.")
        print(f"[gate] PASS: within tolerance {tol}")
    return {"loaded_accuracy": loaded, "expected_accuracy": expected, "n": res["n_effective"]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def dry_report(cfg):
    """Validate the whole plan without loading model weights or training: resolved plan,
    trainer knobs (computed, not constructed — avoids bf16-on-CPU errors), data paths +
    a 2-row format smoke, and the stage-2 continuation / retention-gate readiness."""
    m, meth, t, s, d = cfg["model"], cfg["method"], cfg["train"], cfg["sdft"], cfg["data"]
    print("=" * 72)
    print(f"DRY RUN (train) — {cfg['_derived']['run_stem']}")
    print("=" * 72)
    print(f"  plan    : {meth['objective']} / {meth['tuning']} / teacher={meth['teacher']}  stage={d['stage']}")
    print(f"  model   : {m['id']}  ({m['scale']}, {m['dtype']})")
    print(f"  dataset : {d['dataset']}  holdout_filter={d['holdout_filter']}  max_samples={d['max_samples']}")
    print(f"  output  : {cfg['_derived']['output_dir']}")
    eff = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    print(f"  optim   : lr={t['learning_rate']} epochs={t['num_train_epochs']} eff_batch={eff} "
          f"({t['per_device_train_batch_size']}x{t['gradient_accumulation_steps']}) sched={t['lr_scheduler_type']}")
    if meth["objective"] != "sft":
        full_ema = meth["tuning"] == "full" and meth["teacher"] == "ema"
        lora_ema = meth["tuning"] == "lora" and meth["teacher"] == "ema"
        print(f"  sdft    : kl_alpha={s['kl_alpha']} beta={s['beta']} temp={s['temperature']} "
              f"skip={s['num_loss_tokens_to_skip']} ema_alpha={s['ema_alpha']}")
        print(f"  teacher : sync_ref_model={full_ema} | LoRAEMACallback={lora_ema} | "
              f"generate_from_teacher={meth['objective'] == 'online_sft'} | vllm={cfg['vllm']['enabled']}")
    print(f"  save    : every {cfg['runtime']['save_steps']} steps, keep={_save_total_limit(cfg)} (None=all)")

    spec = D.DATASETS[d["dataset"]]
    print(f"  [data] train path {'OK' if spec['train'].exists() else 'MISSING'}: {spec['train']}")
    tok = None
    if meth["objective"] == "sft":
        try:
            tok = AutoTokenizer.from_pretrained(m["id"])
        except Exception as e:
            print(f"  [data] tokenizer unavailable ({e.__class__.__name__}); skipping sft format smoke")
    if spec["train"].exists() and (meth["objective"] != "sft" or tok is not None):
        try:
            ds = D.load_train(d["dataset"], meth["objective"], t["seed"], tokenizer=tok,
                              holdout_filter=d["holdout_filter"], max_samples=2)
            print(f"  [data] format OK: cols={ds.column_names}")
        except Exception as e:
            print(f"  [data] FORMAT ERROR: {e.__class__.__name__}: {e}")

    if d["stage"] == 2:
        init = d.get("init_adapter") if meth["tuning"] == "lora" else d.get("init_checkpoint")
        print(f"  [stage2] init {'OK' if (init and Path(init).exists()) else 'MISSING'}: {init}")
        gate = cfg["retention_gate"]
        if gate.get("enabled"):
            base = _read_stage1_accuracy(cfg)
            if base is None and not gate.get("allow_missing_baseline"):
                print("  [gate] WOULD FAIL CLOSED: no stage-1 baseline. Run `./run.sh eval <stage1_cfg>` first, "
                      "or set retention_gate.expected_accuracy.")
            else:
                rows = cfg["eval"].get("max_samples") or "all"
                print(f"  [gate] baseline={base} tol={gate['tolerance']} task={gate['task']}/{gate['set']} "
                      f"(enforced on {rows} rows)")
    print("DRY RUN OK — nothing trained.")


def main():
    ap = argparse.ArgumentParser(description="Unified SDFT/SFT trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--dry_run", action="store_true",
                    help="Validate config/data/stage-2 plan without loading weights or training.")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.dry_run:
        dry_report(cfg)
        return
    objective = cfg["method"]["objective"]
    tuning = cfg["method"]["tuning"]
    output_dir = cfg["_derived"]["output_dir"]
    dtype = _dtype(cfg["model"]["dtype"])

    if cfg["runtime"].get("report_to") == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg["runtime"]["wandb_project"])
        os.environ.setdefault("WANDB_WATCH", "false")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[run] {cfg['_derived']['run_stem']} :: {objective}/{tuning}/{cfg['method']['teacher']} "
          f"stage={cfg['data']['stage']} dataset={cfg['data']['dataset']}")

    # ---- SFT path (SFTTrainer) ----
    if objective == "sft":
        from trl import SFTTrainer
        student = build_student(cfg, dtype)
        if tuning == "lora" and cfg["data"]["stage"] == 1:
            student.enable_input_require_grads()
        train_ds = D.load_train(cfg["data"]["dataset"], "sft", cfg["train"]["seed"], tokenizer=tokenizer,
                                holdout_filter=cfg["data"]["holdout_filter"], max_samples=cfg["data"]["max_samples"])
        gate_info = retention_gate(cfg, student, tokenizer)
        stamp_run(cfg, output_dir, extra={"stage": cfg["data"]["stage"], "retention_gate": gate_info,
                                          "data_fingerprint": D.fingerprint(cfg["data"]["dataset"])})
        trainer = SFTTrainer(
            model=student, args=build_sft_config(cfg), train_dataset=train_ds,
            processing_class=tokenizer,
            data_collator=ResponseOnlyCollator(tokenizer, D.RESPONSE_TEMPLATE))
        trainer.train()

    # ---- SDFT / online-SFT path (DistilTrainer) ----
    else:
        from distil_trainer import DistilTrainer
        if objective == "online_sft" and not cfg["vllm"]["enabled"]:
            raise SystemExit("online_sft requires vllm.enabled=true")
        student = build_student(cfg, dtype)
        teacher, is_peft_ema = build_teacher(cfg, dtype)
        train_ds = D.load_train(cfg["data"]["dataset"], objective, cfg["train"]["seed"],
                                holdout_filter=cfg["data"]["holdout_filter"], max_samples=cfg["data"]["max_samples"])

        gate_info = retention_gate(cfg, student, tokenizer)
        stamp_run(cfg, output_dir, extra={"stage": cfg["data"]["stage"], "retention_gate": gate_info,
                                          "data_fingerprint": D.fingerprint(cfg["data"]["dataset"])})

        trainer = DistilTrainer(
            model=student, ref_model=teacher, args=build_distil_config(cfg),
            train_dataset=train_ds, processing_class=tokenizer)

        # LoRA EMA teacher tracking (full-FT EMA is wired via sync_ref_model instead).
        if is_peft_ema:
            trainer.add_callback(LoRAEMACallback(student, teacher, alpha=cfg["sdft"]["ema_alpha"]))

        assert trainer.ref_model is not None, "ref_model nulled — teacher routing collapsed"
        trainer.train()

    # ---- Save artifact ----
    if trainer.accelerator.is_main_process:
        if tuning == "lora":
            adapter_dir = os.path.join(output_dir, "lora_adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            student.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            print(f"[save] adapter -> {adapter_dir}")
        else:
            final_dir = os.path.join(output_dir, "final_model")
            student.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            print(f"[save] full model -> {final_dir}")


if __name__ == "__main__":
    main()
