"""Aggregate every run under runs/ into paper-ready tables (see METRICS.md for definitions).

Statistics are applied at the level the claim lives at:
  - retention (skill-1 after skill-2) is a PAIRED comparison on the same holdout items,
    so it carries a McNemar exact test (stage-1 vs stage-2), not two separate Wilson bands.
  - the METHOD-level claim (SDFT beats SFT) is the per-seed paired difference with sign
    consistency (method_effect.csv) — McNemar (within a fixed seed) is a narrower claim.
  - continual metrics use the standard names: BWT (= retention_abs) and ACC (mean task acc
    after the last stage), so results are comparable to the CL literature.

Emits: results_long, runs_index, retention, continual_metrics, method_effect, forgetting,
gap_closed, significance, aggregate, and pairing_issues (invariant violations — a retention
or forgetting row is EXCLUDED, not silently subtracted, if its two sides disagree on
scale / seed / scorer / eval-set / num_fewshot).

Usage: python collect_results.py [--root runs] [--out analysis] [--strict]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent

LMEVAL_PRIMARY = ["acc_norm,none", "acc,none", "mc2,none", "exact_match,none",
                  "exact_match,strict-match", "prompt_level_strict_acc,none",
                  "inst_level_strict_acc,none", "pass@1,none"]

# Arm identity (fixed), so SDFT-EMA vs SFT pairing is unambiguous.
def arm_key(objective, teacher):
    if objective == "online_sft":
        return "online_sft"
    if objective == "sdft":
        return "sdft_ema" if teacher == "ema" else "sdft_frozen"
    return "sft"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def _meta(cfg, run_dir):
    m, mo, d, t = cfg.get("method", {}), cfg.get("model", {}), cfg.get("data", {}), cfg.get("train", {})
    prov, run_extra = cfg.get("_provenance", {}), cfg.get("_run", {}) or {}
    init = d.get("init_adapter") or d.get("init_checkpoint")
    return {
        "run": run_dir.name, "objective": m.get("objective"), "tuning": m.get("tuning"),
        "teacher": m.get("teacher"), "arm": arm_key(m.get("objective"), m.get("teacher")),
        "scale": mo.get("scale"), "dataset": d.get("dataset"), "stage": d.get("stage"),
        "seed": t.get("seed"), "scorer": cfg.get("eval", {}).get("scorer"),
        "git_sha": (prov.get("git_sha") or "")[:8],
        "stage1_run": Path(init).parent.name if init else None,
        "num_fewshot": (cfg.get("eval", {}).get("forgetting", {}) or {}).get("num_fewshot"),
    }


def _pick_lmeval(tm):
    for key in LMEVAL_PRIMARY:
        if key in tm and isinstance(tm[key], (int, float)):
            return key, float(tm[key])
    for key, val in tm.items():
        if isinstance(val, (int, float)) and "stderr" not in key:
            return key, float(val)
    return None, None


def _stderr_for(tm, metric_key):
    if not metric_key or "," not in metric_key:
        return None
    name, filt = metric_key.split(",", 1)
    v = tm.get(f"{name}_stderr,{filt}")
    return float(v) if isinstance(v, (int, float)) else None


def load_runs(root):
    """runs=[(dir, meta)]; acc={(run,ckpt,edset,eset):rec}; fg=[rows]."""
    runs, acc, fg = [], {}, []
    for cfg_path in sorted(Path(root).glob("*/resolved_config.yaml")):
        run_dir = cfg_path.parent
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except Exception as e:
            print(f"[skip] {cfg_path}: {e}")
            continue
        meta = _meta(cfg, run_dir)
        runs.append((run_dir, meta))

        for rj in run_dir.glob("eval/*/*/eval_results.json"):
            ckpt = rj.parent.parent.name
            edset, _, eset = rj.parent.name.partition("_")
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            w = r.get("wilson95", [None, None])
            acc[(meta["run"], ckpt, edset, eset)] = {
                **meta, "checkpoint": ckpt, "eval_dataset": edset, "eval_set": eset,
                "accuracy": r.get("accuracy"), "n": r.get("n_effective"),
                "ci_lo": w[0], "ci_hi": w[1], "teacher_ceiling": r.get("teacher_ceiling", False),
                "per_sample": r.get("per_sample_scores"), "path": str(rj)}

        for rj in run_dir.glob("eval/forgetting*/**/results*.json"):
            label = "forgetting_base" if "forgetting_base" in str(rj) else "forgetting"
            try:
                results = json.loads(rj.read_text()).get("results", {})
            except Exception:
                continue
            for task, tm in results.items():
                mk, val = _pick_lmeval(tm)
                if val is not None:
                    fg.append({"run": meta["run"], "scale": meta["scale"], "label": label,
                               "task": task, "value": val, "stderr": _stderr_for(tm, mk),
                               "num_fewshot": meta["num_fewshot"]})
    return runs, acc, fg


# ---------------------------------------------------------------------------
# Paired McNemar exact (two-sided binomial on discordant pairs)
# ---------------------------------------------------------------------------
def mcnemar_exact(a, b):
    """Returns (b01, c10, n_disc, p). b01 = a-wrong/b-right, c10 = a-right/b-wrong."""
    b01 = c10 = 0
    for x, y in zip(a or [], b or []):
        if x is None or y is None:
            continue
        if x == 0 and y == 1:
            b01 += 1
        elif x == 1 and y == 0:
            c10 += 1
    n = b01 + c10
    if n == 0:
        return b01, c10, 0, 1.0
    k = min(b01, c10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))
    return b01, c10, n, p


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path.name}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate runs/ into paper-ready tables")
    ap.add_argument("--root", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "analysis"))
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any pairing invariant fails")
    args = ap.parse_args()
    out = Path(args.out)

    runs, acc, fg = load_runs(args.root)
    if not runs:
        print(f"[collect] no runs with resolved_config.yaml under {args.root}. Nothing to do yet.")
        return
    meta_by_run = {m["run"]: m for _, m in runs}
    issues = []  # pairing-invariant violations -> excluded, logged

    def acc_get(run, ckpt, edset, eset):
        return acc.get((run, ckpt, edset, eset))

    # --- results_long ---
    long_rows = [{**rec, "metric": "accuracy", "value": rec["accuracy"]} for rec in acc.values()]
    for r in fg:
        m = meta_by_run.get(r["run"], {})
        long_rows.append({**m, "checkpoint": r["label"], "metric": f"lmeval:{r['task']}",
                          "value": r["value"], "n": None})
    write_csv(out / "results_long.csv", long_rows,
              ["run", "objective", "tuning", "teacher", "scale", "dataset", "stage", "seed",
               "checkpoint", "eval_dataset", "eval_set", "metric", "value", "ci_lo", "ci_hi", "n", "git_sha"])

    # --- runs_index ---
    idx = []
    for run_dir, m in runs:
        idx.append({**m,
                    "has_adapter": (run_dir / "lora_adapter").exists() or (run_dir / "final_model").exists(),
                    "has_final_eval": any(k[0] == m["run"] and k[1] == "final" for k in acc),
                    "n_checkpoint_evals": len({k[1] for k in acc if k[0] == m["run"] and k[1].startswith("step")}),
                    "has_forgetting": any(r["run"] == m["run"] and r["label"] == "forgetting" for r in fg),
                    "has_base_forgetting": any(r["run"] == m["run"] and r["label"] == "forgetting_base" for r in fg)})
    write_csv(out / "runs_index.csv", idx,
              ["run", "objective", "tuning", "teacher", "scale", "dataset", "stage", "seed",
               "has_adapter", "has_final_eval", "n_checkpoint_evals", "has_forgetting",
               "has_base_forgetting", "git_sha"])

    # --- retention (CONTINUAL / BWT) with PAIRED McNemar + invariant checks ---
    ret, cont = [], []
    excluded_pairs = set()   # (stage2_run, stage1_run) that failed an invariant -> keep them out of BWT too
    for _, m in runs:
        if m["stage"] != 2 or not m["stage1_run"]:
            continue
        s1m = meta_by_run.get(m["stage1_run"], {})
        for edset, eset in [("tooluse", "holdout"), ("tooluse", "eval_data")]:
            s2 = acc_get(m["run"], "final", edset, eset)
            s1 = acc_get(m["stage1_run"], "final", edset, eset)
            if not (s2 and s1 and s2["accuracy"] is not None and s1["accuracy"]):
                continue
            # invariant: same scale/scorer/eval-set; warn on seed mismatch
            bad = []
            if s1["scale"] != s2["scale"]:
                bad.append(f"scale {s1['scale']}!={s2['scale']}")
            if s1m.get("scorer") != m.get("scorer"):
                bad.append(f"scorer {s1m.get('scorer')}!={m.get('scorer')}")
            if s1m.get("seed") != m.get("seed"):
                bad.append(f"seed {s1m.get('seed')}!={m.get('seed')}")
            if bad:
                issues.append({"kind": "retention", "arm": m["run"], "vs": m["stage1_run"],
                               "eval_set": f"{edset}/{eset}", "problem": "; ".join(bad)})
                excluded_pairs.add((m["run"], m["stage1_run"]))
                continue
            b01, c10, ndisc, p = mcnemar_exact(s1["per_sample"], s2["per_sample"])
            ret.append({"arm": m["run"], "arm_id": m["arm"], "stage1_run": m["stage1_run"],
                        "scale": m["scale"], "seed": m["seed"], "eval_set": f"{edset}/{eset}",
                        "n": s2["n"], "acquisition": round(s1["accuracy"], 4),
                        "stage1_acc": round(s1["accuracy"], 4), "stage2_acc": round(s2["accuracy"], 4),
                        "bwt": round(s2["accuracy"] - s1["accuracy"], 4),
                        "retention_abs": round(s2["accuracy"] - s1["accuracy"], 4),
                        "retention_pct": round(100 * s2["accuracy"] / s1["accuracy"], 1),
                        "forgot": c10, "gained": b01, "n_discordant": ndisc, "mcnemar_p": round(p, 4),
                        "sig_05": p < 0.05})
        # continual metrics: ACC = mean(science acq, tool-use retention) — both stage-2, always safe.
        # bwt_tooluse needs the stage-1 pair, so it INHERITS the retention exclusion (no leak): a
        # pair dropped from retention.csv yields bwt_tooluse=None here, not a contaminated number.
        sci = acc_get(m["run"], "final", "science", "eval_data")
        tu = acc_get(m["run"], "final", "tooluse", "holdout")
        s1tu = acc_get(m["stage1_run"], "final", "tooluse", "holdout")
        if sci and tu and None not in (sci["accuracy"], tu["accuracy"]):
            pair_ok = (m["run"], m["stage1_run"]) not in excluded_pairs
            bwt = (round(tu["accuracy"] - s1tu["accuracy"], 4)
                   if s1tu and s1tu["accuracy"] is not None and pair_ok else None)
            cont.append({"arm": m["run"], "arm_id": m["arm"], "scale": m["scale"], "seed": m["seed"],
                         "science_acc": round(sci["accuracy"], 4),
                         "tooluse_retained": round(tu["accuracy"], 4),
                         "bwt_tooluse": bwt,
                         "ACC": round((sci["accuracy"] + tu["accuracy"]) / 2, 4)})
    write_csv(out / "retention.csv", ret,
              ["arm", "arm_id", "stage1_run", "scale", "seed", "eval_set", "n", "acquisition",
               "stage1_acc", "stage2_acc", "bwt", "retention_abs", "retention_pct",
               "forgot", "gained", "n_discordant", "mcnemar_p", "sig_05"])
    write_csv(out / "continual_metrics.csv", cont,
              ["arm", "arm_id", "scale", "seed", "science_acc", "tooluse_retained", "bwt_tooluse", "ACC"])

    # --- method_effect: per-seed SDFT-EMA minus SFT, with sign consistency ---
    def diffs(getter, protocol):
        by_scale = defaultdict(dict)   # scale -> seed -> {arm_id: value}
        for row in getter:
            by_scale[row["scale"]].setdefault(row["seed"], {})[row["arm_id"]] = row["value"]
        rows = []
        for scale, seeds in by_scale.items():
            ds = []
            for seed, arms in sorted(seeds.items(), key=lambda x: str(x[0])):
                if "sdft_ema" in arms and "sft" in arms:
                    d = arms["sdft_ema"] - arms["sft"]
                    ds.append(d)
                    rows.append({"scale": scale, "protocol": protocol, "seed": seed,
                                 "sdft_ema": round(arms["sdft_ema"], 4), "sft": round(arms["sft"], 4),
                                 "diff": round(d, 4)})
            if ds:
                rows.append({"scale": scale, "protocol": protocol, "seed": "MEAN",
                             "diff": round(statistics.mean(ds), 4), "n_seeds": len(ds),
                             "sign_consistent": all(x > 0 for x in ds) or all(x < 0 for x in ds)})
        return rows

    acq_pts = [{"scale": r["scale"], "seed": r["seed"], "arm_id": r["arm"], "value": r["accuracy"]}
               for r in acc.values() if r["stage"] == 1 and r["checkpoint"] == "final"
               and r["eval_dataset"] == "tooluse" and r["eval_set"] == "holdout" and r["accuracy"] is not None]
    ret_pts = [{"scale": r["scale"], "seed": r["seed"], "arm_id": r["arm_id"], "value": r["retention_abs"]}
               for r in ret if r["eval_set"] == "tooluse/holdout"]
    method = diffs(acq_pts, "acquisition") + diffs(ret_pts, "retention_bwt")
    write_csv(out / "method_effect.csv", method,
              ["scale", "protocol", "seed", "sdft_ema", "sft", "diff", "n_seeds", "sign_consistent"])

    # --- forgetting (GENERAL): base paired BY SCALE, per task, with stderr ---
    base_fg = {(r["scale"], r["task"]): r for r in fg if r["label"] == "forgetting_base"}
    forg = []
    for r in fg:
        if r["label"] != "forgetting":
            continue
        b = base_fg.get((r["scale"], r["task"]))
        m = meta_by_run.get(r["run"], {})
        delta = fewshot_ok = None
        if b is not None:
            if b["num_fewshot"] != r["num_fewshot"]:
                issues.append({"kind": "forgetting", "arm": r["run"], "vs": "base",
                               "eval_set": r["task"],
                               "problem": f"num_fewshot {b['num_fewshot']}!={r['num_fewshot']}"})
                continue
            delta = round(b["value"] - r["value"], 4)
            fewshot_ok = True
        forg.append({"arm": r["run"], "arm_id": m.get("arm"), "scale": r["scale"], "task": r["task"],
                     "base": round(b["value"], 4) if b else None, "base_stderr": b["stderr"] if b else None,
                     "adapted": round(r["value"], 4), "adapted_stderr": r["stderr"],
                     "forgetting": delta, "num_fewshot": r["num_fewshot"]})
    write_csv(out / "forgetting.csv", forg,
              ["arm", "arm_id", "scale", "task", "base", "base_stderr", "adapted", "adapted_stderr",
               "forgetting", "num_fewshot"])

    # --- gap_closed: HOLDOUT ceiling only (descriptive; see METRICS.md) ---
    base_acc = {(r["scale"], r["eval_dataset"], r["eval_set"]): r["accuracy"]
                for r in acc.values() if r["checkpoint"] == "base_anchor" and r["eval_set"] == "holdout"}
    ceil_acc = {(r["scale"], r["eval_dataset"], r["eval_set"]): r["accuracy"]
                for r in acc.values() if r.get("teacher_ceiling") and r["eval_set"] == "holdout"}
    gap = []
    for r in acc.values():
        if r["checkpoint"] != "final" or r["eval_set"] != "holdout":
            continue
        key = (r["scale"], r["eval_dataset"], "holdout")
        b, c = base_acc.get(key), ceil_acc.get(key)
        if b is not None and c is not None and c != b and r["accuracy"] is not None:
            gap.append({"arm": r["run"], "scale": r["scale"], "eval_set": f"{r['eval_dataset']}/holdout",
                        "base": round(b, 4), "arm_acc": round(r["accuracy"], 4), "ceiling": round(c, 4),
                        "gap_closed_pct": round(100 * (r["accuracy"] - b) / (c - b), 1), "note": "descriptive"})
    write_csv(out / "gap_closed.csv", gap,
              ["arm", "scale", "eval_set", "base", "arm_acc", "ceiling", "gap_closed_pct", "note"])

    # --- significance (arm vs arm, EXPLORATORY, uncorrected) ---
    # Grouped by STAGE too, so stage-1 (acquisition) arms are never McNemar'd against stage-2
    # (retention) arms — that cross-stage comparison would confound a stage effect with a method
    # effect. p-values are per-pair and UNCORRECTED (see METRICS.md): the confirmatory endpoint is
    # retention.csv + method_effect.csv, not this table.
    by_eval = defaultdict(list)
    for r in acc.values():
        if r["checkpoint"] == "final" and r.get("per_sample"):
            by_eval[(r["scale"], r["seed"], r["stage"], r["eval_dataset"], r["eval_set"])].append(r)
    sig = []
    for (scale, seed, stage, edset, eset), recs in by_eval.items():
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if a["arm"] == b["arm"]:
                    continue   # same arm (shouldn't collide within a cell); skip degenerate pairs
                b01, c10, ndisc, p = mcnemar_exact(a["per_sample"], b["per_sample"])
                sig.append({"scale": scale, "seed": seed, "stage": stage, "eval_set": f"{edset}/{eset}",
                            "arm_id_a": a["arm"], "arm_a": a["run"], "arm_id_b": b["arm"], "arm_b": b["run"],
                            "acc_a": round(a["accuracy"], 4), "acc_b": round(b["accuracy"], 4),
                            "n_discordant": ndisc, "mcnemar_p": round(p, 4), "sig_05_uncorrected": p < 0.05})
    write_csv(out / "significance.csv", sig,
              ["scale", "seed", "stage", "eval_set", "arm_id_a", "arm_a", "arm_id_b", "arm_b",
               "acc_a", "acc_b", "n_discordant", "mcnemar_p", "sig_05_uncorrected"])

    # --- aggregate (seed mean +/- std) ---
    groups = defaultdict(list)
    for r in acc.values():
        if r["checkpoint"] == "final" and r["accuracy"] is not None:
            groups[(r["arm"], r["scale"], r["dataset"], r["stage"], r["eval_dataset"], r["eval_set"])].append(r["accuracy"])
    agg = [{"arm": k[0], "scale": k[1], "dataset": k[2], "stage": k[3], "eval_set": f"{k[4]}/{k[5]}",
            "n_seeds": len(v), "mean_acc": round(statistics.mean(v), 4),
            "std_acc": round(statistics.pstdev(v), 4) if len(v) > 1 else 0.0} for k, v in groups.items()]
    write_csv(out / "aggregate.csv", agg,
              ["arm", "scale", "dataset", "stage", "eval_set", "n_seeds", "mean_acc", "std_acc"])

    # --- pairing issues ---
    if issues:
        write_csv(out / "pairing_issues.csv", issues, ["kind", "arm", "vs", "eval_set", "problem"])
        print(f"\n[collect] !! {len(issues)} pairing invariant(s) FAILED — those rows were EXCLUDED. "
              f"See pairing_issues.csv")

    print(f"\n[collect] {len(runs)} runs | retention:{len(ret)} continual:{len(cont)} "
          f"method_effect:{len([r for r in method if r['seed']=='MEAN'])} "
          f"forgetting:{len(forg)} gap:{len(gap)} sig:{len(sig)} | issues:{len(issues)}")
    print(f"[collect] tables -> {out}/")
    if args.strict and issues:
        raise SystemExit(f"[collect] --strict: {len(issues)} pairing invariant(s) failed")


if __name__ == "__main__":
    main()
