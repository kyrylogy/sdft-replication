"""Paper figures from the analysis/*.csv tables (produced by collect_results.py).

Colour + marker + linestyle are fixed PER ARM (identity, not rank; never colour-alone,
so the figures survive greyscale printing). Palette is Okabe-Ito. Data-prep (pandas) is
separate from plotting (matplotlib) so prep is testable without a plot backend.

Figures (each skips cleanly if its data isn't there):
  fig_forgetting_curve  Tool-Use retention over stage-2 checkpoints (Fig-4 analogue)
  fig_tradeoff          new-task acquisition vs. skill-1 retention (BWT), one point per arm
  fig_scale_trend       accuracy vs scale, line = seed mean, individual SEED POINTS overlaid
  fig_forgetting_bars   general forgetting (base-adapted) per lm-eval task, per arm, with stderr

Usage: python make_figures.py [--analysis analysis] [--out analysis/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent

ARM_STYLE = {
    "sft":         dict(color="#0072B2", marker="o", ls="-",  label="SFT"),
    "sdft_ema":    dict(color="#D55E00", marker="s", ls="-",  label="SDFT-EMA"),
    "sdft_frozen": dict(color="#009E73", marker="^", ls="--", label="SDFT-frozen"),
    "online_sft":  dict(color="#CC79A7", marker="D", ls=":",  label="online-SFT"),
}
SCALE_ORDER = {"3b": 0, "7b": 1, "14b": 2}
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
        out[run] = (g["step"].tolist(), g["value"].tolist(),
                    _arm_of(t, g["objective"].iloc[0], g.get("teacher").iloc[0] if "teacher" in g else None))
    return out


def _arm_of(t, objective, teacher):
    if objective == "online_sft":
        return "online_sft"
    if objective == "sdft":
        return "sdft_ema" if teacher == "ema" else "sdft_frozen"
    return "sft"


def prep_tradeoff(t):
    """One point per stage-2 arm: (science acquisition, tool-use BWT, arm_id)."""
    c = t["continual_metrics"]
    if c.empty:
        return []
    return [{"x": r["science_acc"], "y": r["bwt_tooluse"], "arm": r["arm_id"], "label": r["arm"]}
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
            ak = _arm_of(t, r["objective"], r.get("teacher"))
            if ak in out:
                out[ak]["points"].append((r["scale"], r["value"]))
    for k in out:
        out[k]["line"] = sorted(out[k]["line"], key=lambda s: SCALE_ORDER.get(s[0], 9))
    return out


def prep_forgetting_bars(t):
    f = t["forgetting"]
    if f.empty or "forgetting" not in f.columns:
        return pd.DataFrame()
    f = f.dropna(subset=["forgetting"]).copy()
    # delta stderr = sqrt(base^2 + adapted^2) when both present
    def _se(row):
        b, a = row.get("base_stderr"), row.get("adapted_stderr")
        return (float(b) ** 2 + float(a) ** 2) ** 0.5 if pd.notna(b) and pd.notna(a) else 0.0
    f["delta_stderr"] = f.apply(_se, axis=1)
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
        s = ARM_STYLE[ak]
        ax.plot(steps, accs, color=s["color"], marker=s["marker"], ls=s["ls"], lw=2, ms=6, label=s["label"], zorder=3)
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
    import numpy as np
    f = prep_forgetting_bars(t)
    if f.empty:
        return False
    tasks = sorted(f["task"].unique())
    arms = [a for a in ARM_STYLE if a in set(f["arm_id"])]
    fig, ax = _new_ax()
    w = 0.8 / max(len(arms), 1)
    for i, ak in enumerate(arms):
        s = ARM_STYLE[ak]
        sub = f[f["arm_id"] == ak]
        vals = [float(sub[sub["task"] == tk]["forgetting"].mean()) if not sub[sub["task"] == tk].empty else 0.0
                for tk in tasks]
        errs = [float(sub[sub["task"] == tk]["delta_stderr"].mean()) if not sub[sub["task"] == tk].empty else 0.0
                for tk in tasks]
        x = np.arange(len(tasks)) + i * w
        ax.bar(x, vals, width=w * 0.92, color=s["color"], label=s["label"], zorder=3,
               yerr=errs, capsize=2, error_kw=dict(lw=0.8))
    ax.axhline(0, color="0.4", lw=0.7)
    ax.set_xticks(np.arange(len(tasks)) + w * (len(arms) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("Forgetting (base − adapted)")
    ax.set_title("General-capability forgetting, per task (±stderr)")
    ax.legend(frameon=False)
    _save(fig, out, "fig_forgetting_bars")
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
            "scale_trend": fig_scale_trend(t, out), "forgetting_bars": fig_forgetting_bars(t, out)}
    done = [k for k, v in made.items() if v]
    skipped = [k for k, v in made.items() if not v]
    print(f"[figures] rendered: {done or 'none'}" + (f" | skipped (no data): {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
