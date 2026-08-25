"""Final per-sample inference for the thesis retention chapter. No GPU.

Three analyses the aggregate CSVs cannot deliver (they need the per_sample_scores
lists inside eval_results.json, which are index-aligned across runs because
eval sets load deterministically):

  1. Per-chain McNemar (stage-1 vs stage-2, same items) — is any single arm's
     BWT distinguishable from noise? Esp. SDFT's eval_data drops (-7.2/-10.3/-3.1).
  2. Seed-blocked paired permutation on per-item BWT differences between arms —
     the honest replacement for pooled-Fisher inference (seeds are blocks, items
     are the paired unit; sign-flip the per-item arm difference).
  3. Attractor check — regress BWT on stage-1 accuracy across all chains and
     tabulate across-arm spread at stage 1 vs stage 2. If stage-2 training pulls
     every arm to a common level, BWT ~ (attractor - stage1): slope -> -1.

Writes analysis/final_stats.txt (and prints the same).

  .venv/bin/python stats_final.py
"""
from __future__ import annotations

import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from collect_results import REPO, load_runs, mcnemar_exact

OUT = REPO / "analysis" / "final_stats.txt"
_lines = []


def emit(s=""):
    print(s)
    _lines.append(s)


def group_of(meta, run):
    """Arm identity incl. the acq50 control (which shares arm=sft with full SFT)."""
    return meta["arm"] + ("_acq50" if "acq50" in run else "")


def paired_bwt_items(s1, s2):
    """Per-item BWT contribution (s2_i - s1_i), None where either side ungradeable."""
    return [None if (a is None or b is None) else (b - a) for a, b in zip(s1, s2)]


def perm_test(diffs, iters=20000, seed=0):
    """Two-sided sign-flip permutation on paired per-item differences."""
    diffs = [d for d in diffs if d is not None]
    if not diffs:
        return None, 0
    obs = abs(st.mean(diffs))
    rng = random.Random(seed)
    hits = sum(1 for _ in range(iters)
               if abs(st.mean([d if rng.random() < 0.5 else -d for d in diffs])) >= obs - 1e-12)
    return (hits + 1) / (iters + 1), len(diffs)


def main():
    runs, acc, _ = load_runs(str(REPO / "runs"))

    # chains[group][seed] = {"s1": stage1_run, "s2": stage2_run}
    chains = defaultdict(dict)
    for run_dir, meta in runs:
        if meta["scale"] == "7b" and meta["stage"] == 2 and meta["dataset"] == "science" and meta["stage1_run"]:
            chains[group_of(meta, meta["run"])][meta["seed"]] = {"s1": meta["stage1_run"], "s2": meta["run"]}

    def per(run, eset):
        rec = acc.get((run, "final", "tooluse", eset))
        return rec["per_sample"] if rec else None

    emit(f"chains: " + ", ".join(f"{g}({len(s)} seeds)" for g, s in sorted(chains.items())))

    for eset in ["holdout", "eval_data"]:
        emit(f"\n================ tooluse/{eset} ================")

        # ---- 1. per-chain McNemar: stage-1 vs stage-2 on identical items ----
        emit(f"{'group':12} {'seed':>5} {'acc1':>7} {'acc2':>7} {'bwt':>8} {'0->1':>5} {'1->0':>5} {'disc':>5} {'p_mcnemar':>10}")
        items = {}  # (group, seed) -> per-item BWT list, for the arm comparisons below
        for g in sorted(chains):
            for seed in sorted(chains[g]):
                c = chains[g][seed]
                s1, s2 = per(c["s1"], eset), per(c["s2"], eset)
                if s1 is None or s2 is None:
                    emit(f"{g:12} {seed:>5}  MISSING per-sample ({'s1' if s1 is None else 's2'})")
                    continue
                b01, c10, n, p = mcnemar_exact(s1, s2)  # 0->1 gains, 1->0 losses
                graded = [(a, b) for a, b in zip(s1, s2) if a is not None and b is not None]
                a1 = st.mean(a for a, _ in graded)
                a2 = st.mean(b for _, b in graded)
                items[(g, seed)] = paired_bwt_items(s1, s2)
                emit(f"{g:12} {seed:>5} {a1:>7.4f} {a2:>7.4f} {a2-a1:>+8.4f} {b01:>5} {c10:>5} {n:>5} {p:>10.4f}")

        # ---- 2. seed-blocked paired permutation between arms on per-item BWT ----
        emit("\narm comparisons (per-item BWT diff, seeds as blocks, sign-flip permutation):")
        for ga, gb in [("sdft_ema", "sft"), ("sdft_ema", "sft_acq50"), ("sft", "sft_acq50")]:
            diffs = []
            for seed in sorted(set(chains.get(ga, {})) & set(chains.get(gb, {}))):
                da, db = items.get((ga, seed)), items.get((gb, seed))
                if da is None or db is None:
                    continue
                diffs += [None if (x is None or y is None) else x - y for x, y in zip(da, db)]
            p, n = perm_test(diffs)
            if p is None:
                emit(f"  {ga} vs {gb}: no paired items")
            else:
                graded = [d for d in diffs if d is not None]
                emit(f"  {ga:9} vs {gb:9}: mean_bwt_diff={st.mean(graded):+.4f}  n_item_pairs={n}  p={p:.4f}")

        # ---- 3. attractor: BWT vs stage-1 accuracy across chains ----
        pts = []  # (stage1_acc, bwt, group, seed)
        for (g, seed), d in items.items():
            c = chains[g][seed]
            s1, s2 = per(c["s1"], eset), per(c["s2"], eset)
            graded = [(a, b) for a, b in zip(s1, s2) if a is not None and b is not None]
            a1 = st.mean(a for a, _ in graded)
            pts.append((a1, st.mean(b for _, b in graded) - a1, g, seed))
        if len(pts) >= 3:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            mx, my = st.mean(xs), st.mean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            slope = sxy / sxx if sxx else float("nan")
            r = sxy / (sxx ** 0.5 * sum((y - my) ** 2 for y in ys) ** 0.5) if sxx else float("nan")
            emit(f"\nattractor check across {len(pts)} chains: BWT = a + b*stage1_acc  ->  slope={slope:+.3f}  pearson_r={r:+.3f}")
            emit("  (slope near -1, strongly negative r  =>  BWT tracks starting altitude above a common")
            emit("   post-stage-2 level, not a method-specific protection)")
            by_stage = defaultdict(list)
            for a1, bwt, g, _ in pts:
                by_stage[g].append((a1, a1 + bwt))
            emit(f"  {'group':12} {'stage1_mean':>12} {'stage2_mean':>12}")
            g_means = {}
            for g, vals in sorted(by_stage.items()):
                m1, m2 = st.mean(v[0] for v in vals), st.mean(v[1] for v in vals)
                g_means[g] = (m1, m2)
                emit(f"  {g:12} {m1:>12.4f} {m2:>12.4f}")
            s1v, s2v = [m[0] for m in g_means.values()], [m[1] for m in g_means.values()]
            emit(f"  across-arm spread (max-min): stage1={max(s1v)-min(s1v):.4f}  stage2={max(s2v)-min(s2v):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_lines) + "\n")
    print(f"\n[write] {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
