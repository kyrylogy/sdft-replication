# Metrics — definitions

Every number in the paper comes from `collect_results.py` reading the on-disk run records.
This file pins down exactly what each metric means so they can't be conflated.

## Reference points (per scale × dataset × eval set)

- **Base anchor** — the bare, un-fine-tuned model. `./run.sh eval <cfg> --base`. The zero
  point: forgetting is defined as a drop *from here*. Also the lm-eval anchor
  (`--mode forgetting --base`), which MUST use the same `num_fewshot` as every arm.
- **Teacher ceiling** — base model + the golden demonstration in-context (`eval.teacher_ceiling=true`).
  The soft upper bound the distillation targets. Use the **holdout** ceiling (real
  `golden_response` demo) as the faithful target; the eval_data ceiling uses a synthesized,
  Thought-less demo and is a *different* condition.

## Primary metrics

| Metric | Definition | Table |
|---|---|---|
| **Acquisition** | new-task accuracy after learning it (stage-1 `final` on the task's test set) | `results_long.csv` |
| **Gap-closed** | `(arm − base) / (ceiling − base)` — fraction of the demo-conditioning gap the model internalized on its own | `gap_closed.csv` |
| **General forgetting** | `base_anchor − adapted`, **per lm-eval task** (hellaswag, mmlu, …). Never averaged into one number. | `forgetting.csv` |
| **Continual retention** | stage-1 skill accuracy **after** learning skill-2, measured **against the stage-1 level** | `retention.csv` |

## Continual retention — measured against stage-1, NOT base

This is the thesis's core number and the easiest to get wrong.

```
retention_abs = acc(stage-2 model on Tool Use) − acc(stage-1 model on Tool Use)
retention_pct = 100 × acc(stage-2 on Tool Use) / acc(stage-1 on Tool Use)
```

**Headline = `retention_abs` vs the stage-1 level.** A model that reached **70** on Tool Use
and sits at **48** after Science has forgotten 22 points — even though 48 still beats the base's
41. Reporting "48 > base" would hide the forgetting. `retention.csv` reports both the absolute
drop and the % retained, always relative to that model's own stage-1 number.

## Uncertainty & significance

- **Wilson 95% CI** on every accuracy (`ci_lo`, `ci_hi` in `results_long.csv`). At n≈100 the
  interval is ~±10 points — report it; a raw point estimate over-claims.
- **McNemar exact test** (`significance.csv`) between two arms on the *same* eval set, from the
  saved per-sample scores. This is the right test for "is SDFT's gap over SFT real?" — it uses
  the discordant pairs (where the arms disagree), not the marginal accuracies. `significant_05`
  flags p < 0.05.
- **Seed spread** (`aggregate.csv`) — mean ± std across seed runs. A single-seed gap smaller than
  the seed std is not a result; run headline cells at ≥3 seeds (`SEEDS="42 1234 2024"`).

## Provenance (what makes a number trustworthy)

Each `runs/<name>/resolved_config.yaml` stamps the full resolved config + git sha + library
versions + GPU + timestamp + a **dataset fingerprint** (arrow-file hash + byte size). If the
tool-use set is silently regenerated, the fingerprint changes and cross-run comparisons flag it.
`runs_index.csv` shows matrix coverage — which (arm × scale × protocol) cells are trained /
evaluated / forgetting-done, so gaps are visible at a glance.
