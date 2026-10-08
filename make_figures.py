"""Paper figures from the analysis/*.csv tables (produced by collect_results.py).

Colour + marker + linestyle are fixed PER ARM (identity, not rank; never colour-alone,
so the figures survive greyscale printing). Palette is Okabe-Ito. Data-prep (pandas) is
separate from plotting (matplotlib) so prep is testable without a plot backend.

Figures (each skips cleanly if its data isn't there):
  fig_forgetting_curve  Tool-Use retention over stage-2 checkpoints (Fig-4 analogue)
  fig_tradeoff          new-task acquisition vs. skill-1 retention (BWT), one point per arm
  fig_scale_trend       accuracy vs scale, line = seed mean, individual SEED POINTS overlaid
                        (skipped when fewer than 2 scales have data: no trend to show)
  fig_forgetting_bars   general capability delta vs base (adapted-base, pp) on the 6 headline
                        lm-eval tasks, one panel per stage: bar = seed mean per arm, dots = seeds
  fig_retention_endpoints  Tool-Use accuracy after stage 1 -> after stage 2, per arm, one panel
                        per retention set (holdout = seen APIs, eval_data = unseen APIs)

Usage: python make_figures.py [--analysis analysis] [--out analysis/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent

ARM_STYLE = {
    "sft":         dict(color="#0072B2", marker="o", ls="-",  label="SFT"),
    "sft_acq50":   dict(color="#56B4E9", marker="v", ls="--", label="SFT-acq50"),
    "sdft_ema":    dict(color="#D55E00", marker="s", ls="-",  label="SDFT-EMA"),
    "sdft_frozen": dict(color="#009E73", marker="^", ls="--", label="SDFT-frozen"),
    "online_sft":  dict(color="#CC79A7", marker="D", ls=":",  label="online-SFT"),
}
SCALE_ORDER = {"3b": 0, "7b": 1, "14b": 2}
# The 6-task battery the thesis reports; forgetting.csv also carries ~60 mmlu_* subtask rows.
HEADLINE_TASKS = ["ifeval", "truthfulqa_mc2", "humaneval", "hellaswag", "mmlu", "winogrande"]
TASK_LABEL = {"ifeval": "IFEval", "truthfulqa_mc2": "TruthfulQA-mc2", "humaneval": "HumanEval",
              "hellaswag": "HellaSwag", "mmlu": "MMLU", "winogrande": "WinoGrande"}
RETENTION_SETS = {"tooluse/holdout": "Tool-Use holdout (seen APIs, n=100)",
                  "tooluse/eval_data": "Tool-Use eval_data (unseen APIs, n=97)"}
TABLES = ["results_long", "retention", "continual_metrics", "method_effect",
          "forgetting", "gap_closed", "aggregate", "significance", "runs_index"]


def load_tables(adir):
    adir = Path(adir)
    return {n: (pd.read_csv(adir / f"{n}.csv") if (adir / f"{n}.csv").exists() else pd.DataFrame())
            for n in TABLES}


# ---------------------------------------------------------------------------
# Data prep (pure pandas — testable without matplotlib)
# ---------------------------------------------------------------------------
def prep_forgetting_curve(t, eval_dataset="tooluse", eval_set="holdout"):
    """Stage-2 ONLY: skill-1 retention while skill-2 is being learned.

    The stage filter is load-bearing. Without it this mask also matches every
    stage-1 run's own acquisition curve on the same task/set (8 extra curves at
    7B), which then get drawn on an axis labelled "Stage-2 training step" —
    stage-1 runs to step 248 where stage-2 ends at 168, so the contamination
    reads as a late recovery that never happened.

    step 0 is the stage-1 final accuracy (the value BWT subtracts), so the
    curve starts at the level the adapter actually carried into stage 2 rather
    than at its first *saved* checkpoint 50 steps in. retention.csv has one row
    per (run, eval_set), so the anchor MUST be taken from this curve's eval_set:
    unfiltered, the later tooluse/eval_data row overwrote the holdout one and
    every holdout curve started at its eval_data accuracy (~0.6-0.7, not ~0.3).
    """
    df = t["results_long"]
    if df.empty:
        return {}
    m = df[(df["metric"] == "accuracy") & (df["eval_dataset"] == eval_dataset)
           & (df["eval_set"] == eval_set) & (df["stage"] == 2)
           & (df["checkpoint"].astype(str).str.match(r"step\d+"))]
    ret = t.get("retention")
    anchor = {}
    if ret is not None and not ret.empty and "stage1_acc" in ret:
        r1 = ret[ret["eval_set"] == f"{eval_dataset}/{eval_set}"]
        anchor = dict(zip(r1["arm"], r1["stage1_acc"]))
    out = {}
    for run, g in m.groupby("run"):
        g = g.copy()
        g["step"] = g["checkpoint"].str.slice(4).astype(int)
        g = g.sort_values("step")
        steps, accs = g["step"].tolist(), g["value"].tolist()
        if run in anchor:
            steps, accs = [0] + steps, [anchor[run]] + accs
        out[run] = (steps, accs,
                    _arm_of(t, g["objective"].iloc[0], g.get("teacher").iloc[0] if "teacher" in g else None, run))
    return out


def _arm_of(t, objective, teacher, run=""):
    """Mirror of collect_results.arm_key (acq50 control and joint ceiling, marked by run name)."""
    if objective == "online_sft":
        arm = "online_sft"
    elif objective == "sdft":
        arm = "sdft_ema" if teacher == "ema" else "sdft_frozen"
    else:
        arm = "sft"
    if "_joint_" in str(run):
        return arm + "_joint"
    return arm + ("_acq50" if "acq50" in str(run) else "")


def _arm_id(arm_id, run):
    """arm_id from a table; tables written before the acq50/joint fixes label those runs "sft"."""
    for marker, suffix in (("_joint_", "_joint"), ("acq50", "_acq50")):
        if marker in str(run):
            return arm_id if str(arm_id).endswith(suffix) else arm_id + suffix
    return arm_id


def prep_tradeoff(t):
    """One point per stage-2 arm: (science acquisition, tool-use BWT, arm_id)."""
    c = t["continual_metrics"]
    if c.empty:
        return []
    return [{"x": r["science_acc"], "y": r["bwt_tooluse"], "arm": _arm_id(r["arm_id"], r["arm"]), "label": r["arm"]}
            for _, r in c.iterrows()]


def prep_scale_trend(t, dataset="tooluse", eval_set="tooluse/holdout", stage=1):
    """Per arm: mean line + individual seed points across scales."""
    agg, rl = t["aggregate"], t["results_long"]
    if agg.empty:
        return {}
    m = agg[(agg["dataset"] == dataset) & (agg["eval_set"] == eval_set) & (agg["stage"] == stage)]
    out = {}
    for _, r in m.iterrows():
        out.setdefault(r["arm"], {"line": [], "points": []})["line"].append(
            (r["scale"], r["mean_acc"], r.get("std_acc", 0.0)))
    # seed points from results_long
    if not rl.empty:
        edset, eset = eval_set.split("/", 1)
        pts = rl[(rl["metric"] == "accuracy") & (rl["checkpoint"] == "final")
                 & (rl["dataset"] == dataset) & (rl["stage"] == stage)
                 & (rl["eval_dataset"] == edset) & (rl["eval_set"] == eset)]
        for _, r in pts.iterrows():
            ak = _arm_of(t, r["objective"], r.get("teacher"), r["run"])
            if ak in out:
                out[ak]["points"].append((r["scale"], r["value"]))
    for k in out:
        out[k]["line"] = sorted(out[k]["line"], key=lambda s: SCALE_ORDER.get(s[0], 9))
    return out


def prep_retention_endpoints(t, sets=tuple(RETENTION_SETS), scale="7b"):
    """{eval_set: {arm_id: [(stage1_acc, stage2_acc) per stage-2 run]}} from retention.csv, one
    scale. Both endpoints of every chain, so the plot shows what BWT (= stage2 - stage1) is made
    of: where an arm started as much as how far it fell."""
    r = t.get("retention")
    if r is None or r.empty:
        return {}
    if "scale" in r:
        r = r[r["scale"] == scale]
    out = {}
    for es in sets:
        sub = r[r["eval_set"] == es]
        for run, aid, s1, s2 in zip(sub["arm"], sub["arm_id"], sub["stage1_acc"], sub["stage2_acc"]):
            out.setdefault(es, {}).setdefault(_arm_id(aid, run), []).append((s1, s2))
    return out


def prep_forgetting_bars(t, tasks=HEADLINE_TASKS, stage=1, scale="7b"):
    """Per (arm_id, task): delta vs base in pp, adapted - base (= -forgetting; negative =
    damage) -- seed mean `mean_pp`, every seed `seeds_pp`, and `se_pp`, the mean per-run
    stderr of the delta, sqrt(base_se^2 + adapted_se^2) (treats the two evals as independent,
    so conservative for same-item evals). Headline tasks only: forgetting.csv also carries the
    mmlu_* subtasks. Stage-`stage` adapters only (stage from runs_index): forgetting.csv has
    no stage column, so a stage-2 battery would otherwise be pooled into its arm's bar."""
    f = t["forgetting"]
    if f.empty or "forgetting" not in f.columns:
        return pd.DataFrame()
    f = f[f["task"].isin(tasks)].dropna(subset=["forgetting"]).copy()
    if "scale" in f:   # one scale per figure: never average a 3B and a 7B run into one bar
        f = f[f["scale"] == scale]
    ri = t.get("runs_index")
    if stage is not None and ri is not None and not ri.empty:
        f = f[f["arm"].map(dict(zip(ri["run"], ri["stage"]))) == stage]
    if f.empty:
        return pd.DataFrame()
    f["arm_id"] = [_arm_id(a, r) for a, r in zip(f["arm_id"], f["arm"])]
    f["delta_pp"] = -100 * f["forgetting"]
    def _se(row):
        b, a = row.get("base_stderr"), row.get("adapted_stderr")
        return (float(b) ** 2 + float(a) ** 2) ** 0.5 if pd.notna(b) and pd.notna(a) else 0.0
    f["se_pp"] = 100 * f.apply(_se, axis=1)
    g = f.groupby(["arm_id", "task"])
    return pd.DataFrame({"mean_pp": g["delta_pp"].mean(), "seeds_pp": g["delta_pp"].agg(list),
                         "n_seeds": g["delta_pp"].size(), "se_pp": g["se_pp"].mean()}).reset_index()


# ---------------------------------------------------------------------------
# Plotting (matplotlib)
# ---------------------------------------------------------------------------
def _new_ax():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.grid(True, lw=0.4, color="0.85", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[fig] {out}/{name}.png/.pdf")


def fig_forgetting_curve(t, out):
    data = prep_forgetting_curve(t)
    if not data:
        return False
    fig, ax = _new_ax()
    seen = set()   # one legend entry per arm, not one per run
    for run, (steps, accs, ak) in data.items():
        s = ARM_STYLE[ak]
        ax.plot(steps, accs, color=s["color"], marker=s["marker"], ls=s["ls"], lw=2, ms=6,
                label=s["label"] if ak not in seen else None, zorder=3)
        seen.add(ak)
    ax.set_xlabel("Stage-2 (Science) training step   (0 = stage-1 final)")
    ax.set_ylabel("Tool-Use retention (holdout acc)")
    ax.set_title("Skill-1 retention during skill-2 learning")
    ax.legend(frameon=False)
    _save(fig, out, "fig_forgetting_curve")
    return True


def fig_tradeoff(t, out):
    pts = prep_tradeoff(t)
    if not pts:
        return False
    fig, ax = _new_ax()
    seen = set()
    for p in pts:
        s = ARM_STYLE[p["arm"]]
        ax.scatter(p["x"], p["y"], color=s["color"], marker=s["marker"], s=90, zorder=3,
                   edgecolor="white", linewidth=0.8, label=s["label"] if p["arm"] not in seen else None)
        seen.add(p["arm"])
    ax.axhline(0, color="0.6", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("New-task (Science) acquisition")
    ax.set_ylabel("Skill-1 retention  (BWT, Δ vs stage-1)")
    ax.set_title("Acquisition vs. forgetting")
    ax.legend(frameon=False)
    _save(fig, out, "fig_tradeoff")
    return True


def fig_scale_trend(t, out):
    data = prep_scale_trend(t)
    if not data:
        return False
    scales = {sc for d in data.values() for sc, _, _ in d["line"]}
    if len(scales) < 2:   # one scale = no trend; it would only restate aggregate.csv on 3B..14B ticks
        print(f"[fig] fig_scale_trend skipped: data at {len(scales)} scale(s) {sorted(scales)}, need >= 2")
        return False
    import numpy as np
    fig, ax = _new_ax()
    for i, (ak, d) in enumerate(data.items()):
        s = ARM_STYLE[ak]
        xs = [SCALE_ORDER.get(sc, 9) for sc, _, _ in d["line"]]
        ys = [a for _, a, _ in d["line"]]
        ax.plot(xs, ys, color=s["color"], marker=s["marker"], ls=s["ls"], lw=2, ms=7, label=s["label"], zorder=3)
        # individual seed points, jittered so overlapping seeds are visible
        jitter = (i - len(data) / 2) * 0.05
        for sc, acc in d["points"]:
            ax.scatter(SCALE_ORDER.get(sc, 9) + jitter, acc, color=s["color"], marker=s["marker"],
                       s=28, alpha=0.5, zorder=2, edgecolor="none")
    ax.set_xticks(list(SCALE_ORDER.values()))
    ax.set_xticklabels([s.upper() for s in SCALE_ORDER])
    ax.set_xlabel("Model scale")
    ax.set_ylabel("Tool-Use accuracy (holdout)")
    ax.set_title("Scaling trend (line = seed mean; dots = seeds)")
    ax.legend(frameon=False)
    _save(fig, out, "fig_scale_trend")
    return True


def fig_forgetting_bars(t, out):
    """One panel per stage that has batteries (after stage 1, after stage 2), shared y-axis."""
    import matplotlib.pyplot as plt
    import numpy as np
    panels = [(st, f) for st in (1, 2) for f in [prep_forgetting_bars(t, stage=st)] if not f.empty]
    if not panels:
        return False
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 3.4 * len(panels) + 0.4), sharey=True, squeeze=False)
    for ax, (stage, f) in zip(axes[:, 0], panels):
        ax.grid(True, lw=0.4, color="0.85", zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        tasks = [tk for tk in HEADLINE_TASKS if tk in set(f["task"])]
        arms = [a for a in ARM_STYLE if a in set(f["arm_id"])]
        w = 0.8 / max(len(arms), 1)
        for i, ak in enumerate(arms):
            s = ARM_STYLE[ak]
            sub = f[f["arm_id"] == ak].set_index("task").reindex(tasks)   # missing task -> no bar, not 0
            n = int(sub["n_seeds"].max())
            x = np.arange(len(tasks)) + i * w
            ax.bar(x, sub["mean_pp"], width=w * 0.92, color=s["color"], zorder=3,
                   label=f"{s['label']} (n={n})", yerr=sub["se_pp"].fillna(0.0), capsize=2,
                   error_kw=dict(lw=0.8, ecolor="0.35"))
            for xi, pts in zip(x, sub["seeds_pp"]):   # every seed, not just the mean (METRICS.md)
                if isinstance(pts, list):
                    ax.scatter([xi] * len(pts), pts, s=9, color="black", zorder=4, linewidth=0)
        ax.axhline(0, color="0.4", lw=0.7)
        ax.set_xticks(np.arange(len(tasks)) + w * (len(arms) - 1) / 2)
        ax.set_xticklabels([TASK_LABEL.get(tk, tk) for tk in tasks], rotation=20, ha="right")
        ax.set_ylabel("Δ vs base (pp)")
        ax.set_title(f"After stage {stage} ({'Tool-Use' if stage == 1 else 'Tool-Use, then Science'})", fontsize=10)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("General capability vs the base model (adapted − base)\n"
                 "bar = seed mean · dots = seeds · whisker = ±1 s.e. of one run's Δ", fontsize=10)
    fig.tight_layout()
    _save(fig, out, "fig_forgetting_bars")
    return True


def fig_retention_endpoints(t, out):
    import matplotlib.pyplot as plt
    data = prep_retention_endpoints(t)
    if not data:
        return False
    fig, axes = plt.subplots(1, len(data), figsize=(3.4 * len(data), 3.5), squeeze=False)
    for ax, (es, arms) in zip(axes[0], data.items()):
        ax.grid(True, axis="y", lw=0.4, color="0.85", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for ak in [a for a in ARM_STYLE if a in arms]:
            s, pts = ARM_STYLE[ak], arms[ak]
            for s1, s2 in pts:   # every seed, faint
                ax.plot([0, 1], [s1, s2], color=s["color"], ls=s["ls"], lw=1, alpha=0.35, zorder=2)
            m1, m2 = (sum(p[i] for p in pts) / len(pts) for i in (0, 1))
            ax.plot([0, 1], [m1, m2], color=s["color"], marker=s["marker"], ls=s["ls"], lw=2.2, ms=7,
                    label=f"{s['label']} (n={len(pts)})", zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["after stage 1\n(Tool-Use)", "after stage 2\n(+ Science)"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(RETENTION_SETS.get(es, es), fontsize=10)
    axes[0][0].set_ylabel("Tool-Use accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle("Tool-Use accuracy before and after learning Science (bold = seed mean, faint = seeds)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, out, "fig_retention_endpoints")
    return True


def main():
    ap = argparse.ArgumentParser(description="Render paper figures from analysis/*.csv")
    ap.add_argument("--analysis", default=str(REPO / "analysis"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(args.analysis) / "figures"

    t = load_tables(args.analysis)
    if all(df.empty for df in t.values()):
        print(f"[figures] no analysis tables under {args.analysis}. Run collect_results.py first.")
        return
    made = {"forgetting_curve": fig_forgetting_curve(t, out), "tradeoff": fig_tradeoff(t, out),
            "scale_trend": fig_scale_trend(t, out), "forgetting_bars": fig_forgetting_bars(t, out),
            "retention_endpoints": fig_retention_endpoints(t, out)}
    done = [k for k, v in made.items() if v]
    skipped = [k for k, v in made.items() if not v]
    print(f"[figures] rendered: {done or 'none'}" + (f" | skipped (no data / <2 scales): {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
