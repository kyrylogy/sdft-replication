"""Aggregate every run under runs/ into paper-ready tables.

Walks runs/<name>/, joins each resolved_config.yaml with its eval outputs, and emits
tidy CSVs (see METRICS.md for exact metric definitions):

  results_long.csv   one row per (run, checkpoint, eval_set, metric): accuracy + Wilson CI,
                     and lm-eval forgetting tasks. The raw table everything else derives from.
  runs_index.csv     one row per run: coverage flags (adapter? final eval? forgetting? #ckpts).
  retention.csv      CONTINUAL forgetting: stage-2 vs stage-1 skill level (abs + % retained).
  forgetting.csv     GENERAL forgetting: base-anchor minus adapted, per lm-eval task.
  gap_closed.csv     (arm - base) / (ceiling - base) per eval set.
  significance.csv   McNemar exact test between arms on the same eval set (from per-sample scores).
  aggregate.csv      seed mean +/- std per (arm, scale, dataset, eval_set).

Usage: python collect_results.py [--root runs] [--out analysis]
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

# lm-eval primary-metric preference per task (first match wins).
LMEVAL_PRIMARY = ["acc_norm,none", "acc,none", "mc2,none", "exact_match,none",
                  "exact_match,strict-match", "prompt_level_strict_acc,none",
                  "inst_level_strict_acc,none", "pass@1,none"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def _meta(cfg, run_dir):
    m, mo, d, t = cfg.get("method", {}), cfg.get("model", {}), cfg.get("data", {}), cfg.get("train", {})
    prov = cfg.get("_provenance", {})
    init = d.get("init_adapter") or d.get("init_checkpoint")
    stage1_run = Path(init).parent.name if init else None   # runs/<s1>/lora_adapter -> <s1>
    return {
        "run": run_dir.name, "objective": m.get("objective"), "tuning": m.get("tuning"),
        "teacher": m.get("teacher"), "scale": mo.get("scale"), "dataset": d.get("dataset"),
        "stage": d.get("stage"), "seed": t.get("seed"), "git_sha": (prov.get("git_sha") or "")[:8],
        "stage1_run": stage1_run,
    }


def _pick_lmeval(task_metrics):
    for key in LMEVAL_PRIMARY:
        if key in task_metrics and isinstance(task_metrics[key], (int, float)):
            return key, float(task_metrics[key])
    for key, val in task_metrics.items():
        if isinstance(val, (int, float)) and "stderr" not in key:
            return key, float(val)
    return None, None


def load_runs(root):
    """Return runs=[(run_dir, meta)], acc={(run,ckpt,edset,eset):record}, fg={(run,label,task):value}."""
    runs, acc, fg = [], {}, {}
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
            ckpt = rj.parent.parent.name          # final | step50 | base_anchor
            leaf = rj.parent.name                  # <edataset>_<eset>
            edset, _, eset = leaf.partition("_")
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            w = r.get("wilson95", [None, None])
            acc[(meta["run"], ckpt, edset, eset)] = {
                **meta, "checkpoint": ckpt, "eval_dataset": edset, "eval_set": eset,
                "accuracy": r.get("accuracy"), "n": r.get("n_effective"),
                "ci_lo": w[0], "ci_hi": w[1], "teacher_ceiling": r.get("teacher_ceiling", False),
                "per_sample": r.get("per_sample_scores"), "path": str(rj),
            }

        for rj in run_dir.glob("eval/forgetting*/**/results*.json"):
            label = "forgetting_base" if "forgetting_base" in str(rj) else "forgetting"
            try:
                results = json.loads(rj.read_text()).get("results", {})
            except Exception:
                continue
            for task, tm in results.items():
                _, val = _pick_lmeval(tm)
                if val is not None:
                    fg[(meta["run"], label, task)] = val
    return runs, acc, fg


# ---------------------------------------------------------------------------
# McNemar exact (two-sided binomial on discordant pairs)
# ---------------------------------------------------------------------------
def mcnemar_exact(a, b):
    bb = cc = 0
    for x, y in zip(a or [], b or []):
        if x is None or y is None:
            continue
        if x == 0 and y == 1:
            bb += 1
        elif x == 1 and y == 0:
            cc += 1
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))
    return bb, cc, p


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------
def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate runs/ into paper-ready tables")
    ap.add_argument("--root", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "analysis"))
    args = ap.parse_args()
    out = Path(args.out)

    runs, acc, fg = load_runs(args.root)
    if not runs:
        print(f"[collect] no runs with resolved_config.yaml under {args.root}. Nothing to do yet.")
        return

    # --- results_long.csv (accuracy + forgetting) ---
    long_rows = []
    for rec in acc.values():
        long_rows.append({**rec, "metric": "accuracy", "value": rec["accuracy"]})
    for (run, label, task), val in fg.items():
        m = next((mt for rd, mt in runs if mt["run"] == run), {})
        long_rows.append({**m, "checkpoint": label, "metric": f"lmeval:{task}", "value": val, "n": None})
    long_cols = ["run", "objective", "tuning", "teacher", "scale", "dataset", "stage", "seed",
                 "checkpoint", "eval_dataset", "eval_set", "metric", "value", "ci_lo", "ci_hi", "n", "git_sha"]
    write_csv(out / "results_long.csv", long_rows, long_cols)

    # --- runs_index.csv (coverage) ---
    idx = []
    for run_dir, m in runs:
        has_final = any(k[0] == m["run"] and k[1] == "final" for k in acc)
        n_ckpt = len({k[1] for k in acc if k[0] == m["run"] and k[1].startswith("step")})
        idx.append({**m,
                    "has_adapter": (run_dir / "lora_adapter").exists() or (run_dir / "final_model").exists(),
                    "has_final_eval": has_final, "n_checkpoint_evals": n_ckpt,
                    "has_forgetting": any(k[0] == m["run"] and k[1] == "forgetting" for k in fg),
                    "has_base_forgetting": any(k[0] == m["run"] and k[1] == "forgetting_base" for k in fg)})
    write_csv(out / "runs_index.csv", idx,
              ["run", "objective", "tuning", "teacher", "scale", "dataset", "stage", "seed",
               "has_adapter", "has_final_eval", "n_checkpoint_evals", "has_forgetting",
               "has_base_forgetting", "git_sha"])

    # --- retention.csv (CONTINUAL forgetting: stage-2 vs stage-1 skill level) ---
    ret = []
    for _, m in runs:
        if m["stage"] != 2 or not m["stage1_run"]:
            continue
        for (edset, eset) in [("tooluse", "holdout"), ("tooluse", "eval_data")]:
            s2 = acc.get((m["run"], "final", edset, eset))
            s1 = acc.get((m["stage1_run"], "final", edset, eset))
            if s2 and s1 and s2["accuracy"] is not None and s1["accuracy"]:
                ret.append({"arm": m["run"], "stage1_run": m["stage1_run"], "scale": m["scale"],
                            "objective": m["objective"], "eval_set": f"{edset}/{eset}",
                            "stage1_acc": round(s1["accuracy"], 4), "stage2_acc": round(s2["accuracy"], 4),
                            "retention_abs": round(s2["accuracy"] - s1["accuracy"], 4),
                            "retention_pct": round(100 * s2["accuracy"] / s1["accuracy"], 1)})
    write_csv(out / "retention.csv", ret,
              ["arm", "stage1_run", "scale", "objective", "eval_set",
               "stage1_acc", "stage2_acc", "retention_abs", "retention_pct"])

    # --- forgetting.csv (GENERAL: base anchor minus adapted, per task) ---
    forg = []
    for (run, label, task), val in fg.items():
        if label != "forgetting":
            continue
        base = fg.get((run, "forgetting_base", task))
        m = next((mt for rd, mt in runs if mt["run"] == run), {})
        forg.append({"arm": run, "scale": m.get("scale"), "objective": m.get("objective"),
                     "task": task, "adapted": round(val, 4),
                     "base": round(base, 4) if base is not None else None,
                     "forgetting": round(base - val, 4) if base is not None else None})
    write_csv(out / "forgetting.csv", forg,
              ["arm", "scale", "objective", "task", "base", "adapted", "forgetting"])

    # --- gap_closed.csv ((arm - base) / (ceiling - base)) ---
    base_acc, ceil_acc = {}, {}
    for rec in acc.values():
        key = (rec["scale"], rec["eval_dataset"], rec["eval_set"])
        if rec["checkpoint"] == "base_anchor":
            base_acc[key] = rec["accuracy"]
        if rec.get("teacher_ceiling"):
            ceil_acc[key] = rec["accuracy"]
    gap = []
    for rec in acc.values():
        if rec["checkpoint"] != "final":
            continue
        key = (rec["scale"], rec["eval_dataset"], rec["eval_set"])
        b, c = base_acc.get(key), ceil_acc.get(key)
        if b is not None and c is not None and c != b and rec["accuracy"] is not None:
            gap.append({"arm": rec["run"], "scale": rec["scale"], "eval_set": f"{rec['eval_dataset']}/{rec['eval_set']}",
                        "base": round(b, 4), "arm_acc": round(rec["accuracy"], 4), "ceiling": round(c, 4),
                        "gap_closed_pct": round(100 * (rec["accuracy"] - b) / (c - b), 1)})
    write_csv(out / "gap_closed.csv", gap,
              ["arm", "scale", "eval_set", "base", "arm_acc", "ceiling", "gap_closed_pct"])

    # --- significance.csv (McNemar between arms, same eval set, final checkpoint) ---
    by_eval = defaultdict(list)
    for rec in acc.values():
        if rec["checkpoint"] == "final" and rec.get("per_sample"):
            by_eval[(rec["scale"], rec["eval_dataset"], rec["eval_set"])].append(rec)
    sig = []
    for (scale, edset, eset), recs in by_eval.items():
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                bb, cc, p = mcnemar_exact(a["per_sample"], b["per_sample"])
                sig.append({"scale": scale, "eval_set": f"{edset}/{eset}",
                            "arm_a": a["run"], "arm_b": b["run"],
                            "acc_a": round(a["accuracy"], 4), "acc_b": round(b["accuracy"], 4),
                            "discordant_a_wrong_b_right": bb, "discordant_a_right_b_wrong": cc,
                            "mcnemar_p": round(p, 4), "significant_05": p < 0.05})
    write_csv(out / "significance.csv", sig,
              ["scale", "eval_set", "arm_a", "arm_b", "acc_a", "acc_b",
               "discordant_a_wrong_b_right", "discordant_a_right_b_wrong", "mcnemar_p", "significant_05"])

    # --- aggregate.csv (seed mean +/- std) ---
    groups = defaultdict(list)
    for rec in acc.values():
        if rec["checkpoint"] == "final" and rec["accuracy"] is not None:
            groups[(rec["objective"], rec["tuning"], rec["teacher"], rec["scale"],
                    rec["dataset"], rec["stage"], rec["eval_dataset"], rec["eval_set"])].append(rec["accuracy"])
    agg = []
    for k, vals in groups.items():
        agg.append({"objective": k[0], "tuning": k[1], "teacher": k[2], "scale": k[3],
                    "dataset": k[4], "stage": k[5], "eval_set": f"{k[6]}/{k[7]}",
                    "n_seeds": len(vals), "mean_acc": round(statistics.mean(vals), 4),
                    "std_acc": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0})
    write_csv(out / "aggregate.csv", agg,
              ["objective", "tuning", "teacher", "scale", "dataset", "stage", "eval_set",
               "n_seeds", "mean_acc", "std_acc"])

    # --- console summary ---
    print(f"\n[collect] {len(runs)} runs | {len(long_rows)} result rows | "
          f"{len(ret)} retention, {len(forg)} forgetting, {len(gap)} gap-closed, {len(sig)} McNemar pairs")
    print(f"[collect] tables -> {out}/")


if __name__ == "__main__":
    main()
