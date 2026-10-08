"""Write analysis/SUMMARY.md: every result in one readable page (GitHub renders it as tables).

Per arm, stage and seed: task accuracy, retention (BWT, forgotten/gained items), the continual
average (ACC), the general-capability battery after each stage, LoRA update norms, training cost,
and the per-item tests from final_stats.txt. Cells read "mean (seed 42 / 1234 / 2024)".

Reads analysis/*.csv (collect_results.py), analysis/adapter_distance.csv, analysis/final_stats.txt
and wandb_export/runs_index.json. Run it last:

    python collect_results.py && python stats_final.py && python make_figures.py && python make_summary.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import pandas as pd

from make_figures import ARM_STYLE, HEADLINE_TASKS, TASK_LABEL, _arm_of, load_tables

REPO = Path(__file__).resolve().parent
SEEDS = [42, 1234, 2024]
LABEL = {**{k: v["label"] for k, v in ARM_STYLE.items()}, "sft_joint": "SFT joint (ceiling)"}
ARMS = ["sft", "sft_acq50", "sdft_ema", "sdft_frozen", "online_sft", "sft_joint"]
SETS = [("tooluse", "holdout", "Tool-Use holdout"), ("tooluse", "eval_data", "Tool-Use eval_data"),
        ("science", "eval_data", "Science")]


def cell(by_seed, fmt="{:.3f}"):
    """'mean (s42 / s1234 / s2024)' from {seed: value}; a single seed shows just its value."""
    vals = [by_seed[s] for s in SEEDS if s in by_seed] or list(by_seed.values())
    if not vals:
        return ""
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(st.mean(vals))} ({' / '.join(fmt.format(v) for v in vals)})"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def arm_col(df):
    return [_arm_of(None, o, t, r) for o, t, r in zip(df["objective"], df["teacher"], df["run"])]


def accuracy_section(rl):
    acc = rl[(rl["metric"] == "accuracy") & (rl["checkpoint"] == "final")].copy()
    acc["arm"] = arm_col(acc)
    rows = []
    for stage in (1, 2):
        for a in ARMS:
            sub = acc[(acc["arm"] == a) & (acc["stage"] == stage)]
            if sub.empty:
                continue
            rows.append([LABEL[a], stage] + [cell(dict(zip(g["seed"], g["value"]))) for g in
                         (sub[(sub["eval_dataset"] == d) & (sub["eval_set"] == e)] for d, e, _ in SETS)])
    anchors = rl[(rl["metric"] == "accuracy") & rl["checkpoint"].isin(["base_anchor", "ceiling"])]
    for ck, name in (("base_anchor", "Base model"), ("ceiling", "Teacher ceiling (demo in context)")):
        sub = anchors[anchors["checkpoint"] == ck]
        if not sub.empty:
            rows.append([name, "–"] + [cell(dict(zip(g["seed"], g["value"]))) for g in
                         (sub[(sub["eval_dataset"] == d) & (sub["eval_set"] == e)] for d, e, _ in SETS)])
    return table(["Arm", "After stage", *(lab for *_, lab in SETS)], rows)


def exact(acc, n):
    """The CSVs round accuracies to 4 dp; k/n recovers the exact value (k items right)."""
    return [round(a * m) / m for a, m in zip(acc, n)]


def retention_section(ret):
    rows = []
    for es in ("tooluse/holdout", "tooluse/eval_data"):
        for a in ARMS:
            g = ret[(ret["arm_id"] == a) & (ret["eval_set"] == es)]
            if g.empty:
                continue
            n = g["n"].iloc[0]
            right1 = round((g["stage1_acc"] * g["n"]).sum())
            s1, s2 = exact(g["stage1_acc"], g["n"]), exact(g["stage2_acc"], g["n"])
            rows.append([es, LABEL[a], cell(dict(zip(g["seed"], s1))), cell(dict(zip(g["seed"], s2))),
                         cell(dict(zip(g["seed"], [100 * (y - x) for x, y in zip(s1, s2)])), "{:+.1f}"),
                         f"{g['forgot'].sum():.0f}/{right1} ({g['forgot'].sum() / right1:.1%})",
                         f"{g['gained'].sum():.0f}/{n * len(g) - right1} ({g['gained'].sum() / (n * len(g) - right1):.1%})",
                         " / ".join(f"{p:.3f}" for p in g.set_index("seed").reindex(SEEDS)["mcnemar_p"].dropna())])
    return table(["Set", "Arm", "After stage 1", "After stage 2", "BWT (pp)", "Forgot (pooled)",
                  "Gained (pooled)", "McNemar p per seed"], rows)


def continual_section(cm, rl):
    final = rl[(rl["metric"] == "accuracy") & (rl["checkpoint"] == "final")]
    value = {(r, d, e): v for r, d, e, v in zip(final["run"], final["eval_dataset"], final["eval_set"], final["value"])}
    rows = []
    for a in ARMS:
        g = cm[cm["arm_id"] == a]
        if g.empty:
            continue
        sci = [value.get((r, "science", "eval_data"), s) for r, s in zip(g["arm"], g["science_acc"])]
        kept = [value.get((r, "tooluse", "holdout"), k) for r, k in zip(g["arm"], g["tooluse_retained"])]
        rows.append([LABEL[a], cell(dict(zip(g["seed"], sci))), cell(dict(zip(g["seed"], kept))),
                     cell(dict(zip(g["seed"], [(x + y) / 2 for x, y in zip(sci, kept)])))])
    joint = final[final["dataset"] == "joint"]
    sci = joint[(joint["eval_dataset"] == "science")]["value"]
    hold = joint[(joint["eval_dataset"] == "tooluse") & (joint["eval_set"] == "holdout")]["value"]
    if len(sci) and len(hold):
        rows.append([LABEL["sft_joint"] + ", both tasks at once", f"{sci.mean():.3f}", f"{hold.mean():.3f}",
                     f"{(sci.mean() + hold.mean()) / 2:.3f}"])
    return table(["Arm", "Science", "Tool-Use kept (holdout)", "ACC"], rows)


def battery_section(rl, ri):
    lm = rl[rl["metric"].str.startswith("lmeval:")].copy()
    lm["task"] = lm["metric"].str.slice(7)
    base = lm[lm["checkpoint"] == "forgetting_base"].drop_duplicates("task").set_index("task")["value"]
    lm = lm[lm["checkpoint"] == "forgetting"].copy()
    lm["arm"] = arm_col(lm)
    lm["stage"] = lm["run"].map(dict(zip(ri["run"], ri["stage"])))
    lm["delta"] = 100 * (lm["value"] - lm["task"].map(base))
    parts = []
    for stage in sorted(lm["stage"].dropna().unique()):
        arms = [a for a in ARMS if not lm[(lm["arm"] == a) & (lm["stage"] == stage)].empty]
        rows = []
        for tk in HEADLINE_TASKS:
            rows.append([TASK_LABEL[tk]] + [cell(dict(zip(g["seed"], g["delta"])), "{:+.2f}") for g in
                        (lm[(lm["arm"] == a) & (lm["stage"] == stage) & (lm["task"] == tk)] for a in arms)]
                        + [f"{100 * base[tk]:.1f}"])
        what = "Tool-Use adapters" if stage == 1 else "Tool-Use → Science adapters"
        parts.append(f"**After stage {int(stage)}** ({what}), change vs the base model in pp:\n\n"
                     + table(["Task", *(LABEL[a] for a in arms), "Base model (%)"], rows))
    return "\n\n".join(parts)


def norms_section(ad):
    ad = ad[ad["adapter"].str.contains(r"checkpoint-\d+$")].copy()
    ad["run"] = ad["adapter"].str.extract(r"runs/([^/]+)/")[0]
    ad["step"] = ad["adapter"].str.extract(r"checkpoint-(\d+)$")[0].astype(int)
    ad["seed"] = ad["run"].str.extract(r"seed(\d+)")[0].astype(float).astype("Int64")
    ad["arm"] = [_arm_of(None, "sdft" if r.startswith("sdft") else "online_sft" if r.startswith("online") else "sft",
                         "ema" if "ema" in r else "frozen", r) for r in ad["run"]]
    ad["stage"] = [2 if "science_s2" in r else 1 for r in ad["run"]]
    rows = []
    for (a, stage, step), g in ad.groupby(["arm", "stage", "step"]):
        rows.append([LABEL.get(a, a), stage, step, cell(dict(zip(g["seed"].astype(int), g["frobenius"])), "{:.2f}")])
    rows.sort(key=lambda r: (ARMS.index(next(k for k, v in LABEL.items() if v == r[0])), r[1], r[2]))
    return table(["Arm", "Stage", "Checkpoint step", "‖ΔW‖F"], rows)


def cost_section(index_path):
    if not index_path.exists():
        return "(no wandb_export/runs_index.json)"
    per = {}
    for r in json.loads(index_path.read_text(encoding="utf-8")):
        s = r.get("summary", {})
        if "_7b_" in r["name"] and "train_runtime" in s and s.get("train/global_step"):
            per.setdefault(re.sub(r"_seed\d+$", "", r["name"]), []).append((s["train_runtime"], s["train/global_step"]))
    rows = [[k, len(v), f"{st.mean(rt / gs for rt, gs in v):.2f}", f"{st.mean(rt for rt, _ in v) / 3600:.2f}"]
            for k, v in sorted(per.items())]
    return table(["Config", "W&B runs", "Seconds per step", "GPU-hours per run"], rows)


def main():
    ap = argparse.ArgumentParser(description="analysis/*.csv -> analysis/SUMMARY.md")
    ap.add_argument("--analysis", default=str(REPO / "analysis"))
    ap.add_argument("--wandb", default=str(REPO / "wandb_export" / "runs_index.json"))
    ap.add_argument("--out", default=None, help="default: <analysis>/SUMMARY.md")
    args = ap.parse_args()
    adir = Path(args.analysis)
    t = load_tables(adir)
    rl, ri = t["results_long"], t["runs_index"]
    ad_path, fs_path = adir / "adapter_distance.csv", adir / "final_stats.txt"
    doc = [
        "# Results summary",
        "",
        "Generated by `make_summary.py` from the tables in this folder; do not edit by hand. Cells read "
        "**mean (seed 42 / 1234 / 2024)**. Definitions are in [METRICS.md](../METRICS.md), and "
        "[RESULTS.md](../RESULTS.md) says which file each number comes from.",
        "",
        "## Task accuracy (final checkpoint)", "", accuracy_section(rl), "",
        "## Retention of Tool-Use while learning Science", "",
        "BWT = accuracy after stage 2 − after stage 1. Forgot / gained: items right after stage 1 and wrong "
        "after stage 2, and the reverse, pooled over seeds.", "", retention_section(t["retention"]), "",
        "## Continual-learning average", "", continual_section(t["continual_metrics"], rl), "",
        "## General capability (lm-eval, 0-shot, no chat template)", "", battery_section(rl, ri), "",
        "## LoRA update norm", "",
        norms_section(pd.read_csv(ad_path)) if ad_path.exists() else "(no adapter_distance.csv)", "",
        "## Training cost (W&B)", "", cost_section(Path(args.wandb)), "",
        "## Per-item tests (`final_stats.txt`)", "",
        "```", fs_path.read_text(encoding="utf-8").rstrip() if fs_path.exists() else "(missing)", "```", "",
    ]
    out = Path(args.out) if args.out else adir / "SUMMARY.md"
    out.write_text("\n".join(doc), encoding="utf-8", newline="\n")
    print(f"[summary] {out}")


if __name__ == "__main__":
    main()
