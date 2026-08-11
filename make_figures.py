"""Paper figures from the analysis/*.csv tables (produced by collect_results.py).

Design: colour + marker + linestyle are fixed PER ARM (identity, not rank; never
colour-alone, so the figures survive greyscale printing). Palette is Okabe-Ito,
the colourblind-safe standard for scientific figures. Data-prep (pandas) is kept
separate from plotting (matplotlib) so the prep is testable without a plot backend.

Figures (each skips cleanly if its data isn't there yet):
  fig_forgetting_curve.{png,pdf}  Tool-Use retention over stage-2 checkpoints (Fig-4 analogue)
  fig_tradeoff.{png,pdf}          new-task acquisition vs. skill-1 retention, one point per arm
  fig_scale_trend.{png,pdf}       accuracy vs scale (3B/7B/14B), line per arm
  fig_forgetting_bars.{png,pdf}   general forgetting (base-adapted) per lm-eval task, per arm

Usage: python make_figures.py [--analysis analysis] [--out analysis/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent

# Okabe-Ito, assigned in FIXED order by arm identity (never cycled / rank-based).
ARM_STYLE = {
    "sft":         dict(color="#0072B2", marker="o", ls="-",  label="SFT"),
    "sdft_ema":    dict(color="#D55E00", marker="s", ls="-",  label="SDFT-EMA"),
    "sdft_frozen": dict(color="#009E73", marker="^", ls="--", label="SDFT-frozen"),
    "online_sft":  dict(color="#CC79A7", marker="D", ls=":",  label="online-SFT"),
}
SCALE_ORDER = {"3b": 0, "7b": 1, "14b": 2}


def arm_key(objective, teacher):
    if objective == "online_sft":
        return "online_sft"
    if objective == "sdft":
        return "sdft_ema" if teacher == "ema" else "sdft_frozen"
    return "sft"


# ---------------------------------------------------------------------------
# Data prep (pure pandas — testable without matplotlib)
# ---------------------------------------------------------------------------
def load_tables(adir):
    adir = Path(adir)
    names = ["results_long", "retention", "forgetting", "gap_closed", "aggregate",
             "significance", "runs_index"]
    return {n: (pd.read_csv(adir / f"{n}.csv") if (adir / f"{n}.csv").exists() else pd.DataFrame())
            for n in names}


def prep_forgetting_curve(t, eval_dataset="tooluse", eval_set="holdout"):
    """Per stage-2 run: (steps, accs, arm_key) for checkpoint-curve retention."""
    df = t["results_long"]
    if df.empty:
        return {}
    m = df[(df["metric"] == "accuracy") & (df["eval_dataset"] == eval_dataset)
           & (df["eval_set"] == eval_set) & (df["checkpoint"].astype(str).str.match(r"step\d+"))]
    out = {}
    for run, g in m.groupby("run"):
        g = g.copy()
        g["step"] = g["checkpoint"].str.slice(4).astype(int)
        g = g.sort_values("step")
        obj = g["objective"].iloc[0]
        tea = g["teacher"].iloc[0] if "teacher" in g else None
        out[run] = (g["step"].tolist(), g["value"].tolist(), arm_key(obj, tea))
    return out


def prep_tradeoff(t):
    """One point per stage-2 arm: (acquisition_x, retention_y, arm_key, label)."""
    ret, rl = t["retention"], t["results_long"]
    if ret.empty:
        return []
    pts = []
    for _, r in ret[ret["eval_set"] == "tooluse/holdout"].iterrows():
        # new-task (Science) acquisition for this stage-2 arm
        sci = rl[(rl["run"] == r["arm"]) & (rl["metric"] == "accuracy")
                 & (rl["eval_dataset"] == "science") & (rl["checkpoint"] == "final")]
        acq = float(sci["value"].iloc[0]) if not sci.empty else None
        obj = r.get("objective")
        pts.append({"x": acq, "y": r["retention_abs"], "arm": arm_key(obj, "ema" if obj == "sdft" else None),
                    "label": r["arm"]})
    return [p for p in pts if p["x"] is not None]


def prep_scale_trend(t, dataset="tooluse", eval_set="tooluse/holdout", stage=1):
    """Per arm: (scales, accs) from the seed-aggregated table."""
    agg = t["aggregate"]
    if agg.empty:
        return {}
    m = agg[(agg["dataset"] == dataset) & (agg["eval_set"] == eval_set) & (agg["stage"] == stage)]
    out = {}
    for _, r in m.iterrows():
        k = arm_key(r["objective"], r.get("teacher"))
        out.setdefault(k, []).append((r["scale"], r["mean_acc"], r.get("std_acc", 0.0)))
    for k in out:
        out[k] = sorted(out[k], key=lambda s: SCALE_ORDER.get(s[0], 9))
    return out


def prep_forgetting_bars(t):
    """DataFrame [task, arm_key, arm_label, forgetting] for grouped bars."""
    f = t["forgetting"]
    if f.empty or "forgetting" not in f.columns:
        return pd.DataFrame()
    f = f.dropna(subset=["forgetting"]).copy()
    f["arm"] = [arm_key(o, "ema") for o in f["objective"]]
    return f


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
    for run, (steps, accs, ak) in data.items():
        st = ARM_STYLE[ak]
        ax.plot(steps, accs, color=st["color"], marker=st["marker"], ls=st["ls"],
                lw=2, ms=6, label=st["label"], zorder=3)
    ax.set_xlabel("Stage-2 (Science) training step")
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
    for p in pts:
        st = ARM_STYLE[p["arm"]]
        ax.scatter(p["x"], p["y"], color=st["color"], marker=st["marker"], s=90,
                   label=st["label"], zorder=3, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="0.6", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("New-task (Science) acquisition")
    ax.set_ylabel("Skill-1 retention (Δ vs stage-1)")
    ax.set_title("Acquisition vs. forgetting")
    ax.legend(frameon=False)
    _save(fig, out, "fig_tradeoff")
    return True


def fig_scale_trend(t, out):
    data = prep_scale_trend(t)
    if not data:
        return False
    fig, ax = _new_ax()
    for ak, pts in data.items():
        st = ARM_STYLE[ak]
        xs = [SCALE_ORDER.get(s, 9) for s, _, _ in pts]
        ys = [a for _, a, _ in pts]
        es = [e for _, _, e in pts]
        ax.errorbar(xs, ys, yerr=es, color=st["color"], marker=st["marker"], ls=st["ls"],
                    lw=2, ms=7, capsize=3, label=st["label"], zorder=3)
    ax.set_xticks(list(SCALE_ORDER.values()))
    ax.set_xticklabels([s.upper() for s in SCALE_ORDER])
    ax.set_xlabel("Model scale")
    ax.set_ylabel("Tool-Use accuracy (holdout)")
    ax.set_title("Scaling trend")
    ax.legend(frameon=False)
    _save(fig, out, "fig_scale_trend")
    return True


def fig_forgetting_bars(t, out):
    import numpy as np
    f = prep_forgetting_bars(t)
    if f.empty:
        return False
    tasks = sorted(f["task"].unique())
    arms = [a for a in ARM_STYLE if a in set(f["arm"])]
    fig, ax = _new_ax()
    w = 0.8 / max(len(arms), 1)
    for i, ak in enumerate(arms):
        st = ARM_STYLE[ak]
        vals = [float(f[(f["task"] == tk) & (f["arm"] == ak)]["forgetting"].mean()) if
                not f[(f["task"] == tk) & (f["arm"] == ak)].empty else 0.0 for tk in tasks]
        x = np.arange(len(tasks)) + i * w
        ax.bar(x, vals, width=w * 0.92, color=st["color"], label=st["label"], zorder=3)
    ax.set_xticks(np.arange(len(tasks)) + w * (len(arms) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("Forgetting (base − adapted)")
    ax.set_title("General-capability forgetting, per task")
    ax.legend(frameon=False)
    _save(fig, out, "fig_forgetting_bars")
    return True


def main():
    ap = argparse.ArgumentParser(description="Render paper figures from analysis/*.csv")
    ap.add_argument("--analysis", default=str(REPO / "analysis"))
    ap.add_argument("--out", default=None, help="default: <analysis>/figures")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(args.analysis) / "figures"

    t = load_tables(args.analysis)
    if all(df.empty for df in t.values()):
        print(f"[figures] no analysis tables under {args.analysis}. Run collect_results.py first.")
        return

    made = {
        "forgetting_curve": fig_forgetting_curve(t, out),
        "tradeoff": fig_tradeoff(t, out),
        "scale_trend": fig_scale_trend(t, out),
        "forgetting_bars": fig_forgetting_bars(t, out),
    }
    done = [k for k, v in made.items() if v]
    skipped = [k for k, v in made.items() if not v]
    print(f"[figures] rendered: {done or 'none'}" + (f" | skipped (no data): {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
