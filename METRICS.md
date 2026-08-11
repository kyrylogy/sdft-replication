# Metrics — definitions

Every number in the paper comes from `collect_results.py` reading the on-disk run records.
This file pins down what each metric means, and — crucially — **which statistic supports
which claim**, so a reviewer can't accuse you of laundering a per-run p-value into a
method-level conclusion.

## Pre-registered primary endpoint

Write this down *before* the sweep: the **confirmatory** claim is **7B SDFT-EMA vs SFT on
`retention_abs` (BWT), 3 seeds**. Everything else — ablations (SDFT-frozen, online-SFT), the
3B/14B scale curve, the per-task forgetting battery — is **exploratory**. The matrix has
enough cells to turn up a "significant" result by chance; naming the one confirmatory test
up front is a one-sentence, large-credibility move.

## Reference points (per scale × dataset × eval set)

- **Base anchor** — the bare model (`--base`). The zero point for general forgetting, and
  the lm-eval anchor (`--mode forgetting --base`), which MUST use the same `num_fewshot` as
  every arm (`collect_results` **excludes** any arm whose fewshot differs — see below).
- **Teacher ceiling** — base + golden demo in-context (`eval.teacher_ceiling=true`). Use the
  **holdout** ceiling (real demo); `collect_results` only ever uses the holdout ceiling for
  gap-closed, because the eval_data ceiling is a different (Thought-less) condition.

## Primary metrics

| Metric | Definition | Table |
|---|---|---|
| **Acquisition** | new-task accuracy after learning it (stage-1 `final` on the task test set) | `results_long` |
| **BWT** (`retention_abs`) | backward transfer = `acc(stage-2 on Tool Use) − acc(stage-1 on Tool Use)`. Negative = forgetting. (Lopez-Paz & Ranzato 2017) | `retention` |
| **ACC** | mean task accuracy after the last stage = `mean(Science acquisition, Tool-Use retained)`. The standard CL "who wins overall" number. | `continual_metrics` |
| **General forgetting** | `base − adapted`, **per lm-eval task** (never averaged) | `forgetting` |
| **Gap-closed** | `(arm − base)/(ceiling − base)` on holdout. **Descriptive only** — never attach significance (ratio of noisy differences). | `gap_closed` |

## Which statistic for which claim

- **Retention is a PAIRED comparison** (stage-1 vs stage-2 on the *same* holdout items), so
  `retention.csv` carries a **McNemar exact test** (`forgot`/`gained` discordant counts,
  `mcnemar_p`) — **not** two separate Wilson bands, which overstate uncertainty on a paired
  design. This is the within-seed evidence that a given run forgot.
- **The METHOD-level claim** (SDFT forgets less than SFT) is **not** a McNemar p-value — that
  fixes the training seed, and training-run variance dominates (your repro landed a ~6pp paper
  gap at ~1pp). It is the **per-seed paired difference** SDFT−SFT at 42/1234/2024, reported
  individually with **sign consistency** in `method_effect.csv`. State in the thesis: McNemar =
  within-seed; `method_effect` = method-level.
- **`retention_pct` rewards weak learners** (an arm that acquired less has less to lose). Keep
  **`retention_abs` (BWT) as the headline, always displayed next to `acquisition`** — which
  `retention.csv` does. The `fig_tradeoff` scatter (acquisition vs BWT) is the honest figure.
- **Underpowered**: at n≈100 the Wilson band is ~±10pt for a 1–6pp expected effect. `n` is
  printed in every table — quote it. Lean on the **checkpoint curve + seeds** (a consistent
  trend across checkpoints and seeds argues better than any single endpoint).
- **Multiple comparisons**: the forgetting battery is 6 tasks × several arms, and
  `significance.csv` runs an all-pairs within-stage McNemar per cell — both are **exploratory**
  and its `sig_05_uncorrected` flag is exactly that, **uncorrected**. If you cite any single
  cell as significant, apply Holm/BH and say so. IFEval is the sensitive probe (it tanks 20+pt
  while MMLU barely moves) — give it its own panel; carry `*_stderr` (done).
- **`significance.csv` is not the confirmatory test.** It is arm-vs-arm at a *fixed* stage and
  seed (never cross-stage — a stage effect must not masquerade as a method effect). The method
  claim is `method_effect.csv` (per-seed sign consistency); the within-run forgetting claim is
  `retention.csv`'s paired McNemar.

## Continual retention — measured against stage-1, NOT base

A model that reached **70** on Tool Use and sits at **48** after Science has forgotten 22
points — even though 48 beats the base's 41. `bwt`/`retention_abs` are always `stage-2 −
stage-1`, never `− base`. Reporting "48 > base" would hide the forgetting.

## Provenance & enforced invariants

Each `resolved_config.yaml` stamps the resolved config + git sha + libs + GPU + a **dataset
fingerprint** (arrow-file hash). `collect_results` doesn't just flag mismatches — it
**excludes** a derived row and logs it to `pairing_issues.csv` when the two sides disagree
on scale, seed, scorer, eval set, or (for forgetting) `num_fewshot`. A silent mismatch here
is exactly the ~10pt scorer swing you got burned by once. `runs_index.csv` shows matrix
coverage. Plot **all seed points**, not just mean±std (a std over 3 points is meaningless).
