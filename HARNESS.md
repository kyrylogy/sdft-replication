# SDFT experiment harness

Config-driven system for the SDFT-vs-SFT continual-learning study. One YAML fully
describes a run; you pick a GPU and fire one command; every run stamps a permanent,
reproducible record. Supersedes the per-script pipeline now in [`legacy/`](legacy/).

## The one command

```bash
GPU=0 ./run.sh train configs/experiments/<arm>.yaml      # train
GPU=0 ./run.sh eval  configs/experiments/<arm>.yaml      # accuracy eval
GPU=0 ./run.sh eval  configs/experiments/<arm>.yaml --base            # base anchor
GPU=0 ./run.sh eval  configs/experiments/<arm>.yaml --mode forgetting # lm-eval battery
SEEDS="42 1234 2024" GPU=0 ./run.sh train configs/experiments/<arm>.yaml   # seed sweep
./run.sh train configs/experiments/<arm>.yaml --dry_run  # validate plan, no weights/GPU
```

## The pieces

| File | Role |
|---|---|
| `configs/base.yaml` | every parameter + default; the tracking surface |
| `configs/shared/*.yaml` | reusable fragments (`model_7b`, `model_14b`, `stage2_science_lora`, `smoke`) pulled in via `extends:` |
| `configs/experiments/*.yaml` | one file per arm; overrides base + shared |
| `exp_config.py` | merge `base → extends → experiment → --set`, validate, **stamp** |
| `datasets_sdft.py` | single source of tooluse+science formatting (student/teacher prompts, response mask) |
| `train.py` | unified trainer: objective × tuning × teacher × stage |
| `eval_runner.py` | adapter-aware eval: strict scorer, Wilson CIs, checkpoint curve, lm-eval battery |
| `run.sh` | GPU selection + dispatch + seed sweep |
| `distil_trainer.py`, `distil_config.py` | upstream SDFT loss/trainer (unmodified); `train.py` builds on them |

## The experiment cube (config axes)

```
method.objective : sft | sdft | online_sft
method.tuning    : lora | full
method.teacher   : ema | frozen | none        # none <=> sft
model.scale      : 3b | 7b | 14b              # 14b is lora-only
data.dataset     : tooluse | science
data.stage       : 1 (fresh) | 2 (continue from a stage-1 adapter)
```

An arm is a point in this cube. The provided configs cover Tier-1 (7B) and Tier-2 (14B).

## Sequential protocol (the forgetting measurement)

Stage 2 continues the *same* adapter onto a second task. Run order matters:

```bash
GPU=0 ./run.sh train configs/experiments/sdft_ema_lora_7b_tooluse_s1.yaml   # 1. learn Tool Use
GPU=0 ./run.sh eval  configs/experiments/sdft_ema_lora_7b_tooluse_s1.yaml   # 2. record the gate baseline
GPU=0 ./run.sh train configs/experiments/sdft_ema_lora_7b_science_s2.yaml   # 3. continue onto Science
GPU=0 ./run.sh eval  configs/experiments/sdft_ema_lora_7b_science_s2.yaml   # 4. Science acquisition
GPU=0 ./run.sh eval  configs/experiments/sdft_ema_lora_7b_science_s2.yaml \
     --set data.dataset=tooluse --set 'eval.sets=[holdout]'                 # 5. Tool-Use RETENTION
```

Hardening guarantees baked in:
- **Teacher carries stage-1 init** — the EMA teacher continues from the stage-1 adapter, not a bare base.
- **Step-0 retention gate** — before any stage-2 gradient, the loaded model is re-scored on the stage-1 task; it **fails closed** if the number doesn't match (so run step 2 first, or set `retention_gate.expected_accuracy`).
- **Stage-2 config shared** — every stage-2 arm `extends` `stage2_science_lora.yaml`, so SFT and SDFT get an identical schedule by construction.

## What each run produces (the record)

```
runs/<name>/
  resolved_config.yaml        # fully-resolved config + git sha + libs + GPU + timestamp
  lora_adapter/               # (lora) final adapter    | final_model/ (full)
  checkpoint-<N>/             # per-save-step checkpoints (the forgetting curve)
  eval/<label>/<dataset>_<set>/
    eval_results.json         # accuracy, n_correct, Wilson 95% CI, scorer, per-sample scores
    eval_responses.json       # per-sample prompt/response/correct (qualitative analysis)
  eval/base_anchor/...        # the base-model reference
  eval/forgetting[/_base]/    # lm-eval per-task outputs
```

`resolved_config.yaml` is the single source of truth for "what ran". Never hand-edit it.

## Analysis (results → paper)

After runs exist, turn scattered JSON into paper tables and figures:

```bash
python collect_results.py                 # runs/ -> analysis/*.csv
python make_figures.py                    # analysis/*.csv -> analysis/figures/*.{png,pdf}
```

`collect_results.py` emits `results_long.csv` (raw), `retention.csv` (continual forgetting,
vs stage-1), `forgetting.csv` (general, per lm-eval task), `gap_closed.csv`, `significance.csv`
(McNemar between arms), `aggregate.csv` (seed mean±std), and `runs_index.csv` (matrix coverage).
`make_figures.py` renders the forgetting curve, acquisition-vs-forgetting tradeoff, scale trend,
and per-task forgetting bars — each arm a fixed colour+marker+linestyle (colourblind- and
greyscale-safe). **Metric definitions are in [`METRICS.md`](METRICS.md)** — read it before quoting
a number (continual retention is measured vs the stage-1 level, never base).

## Smoke-testing locally (no GPU)

`configs/experiments/smoke_*` run the whole loop on `Qwen2.5-0.5B` (8 rows, CPU/MPS) in minutes.
Use `--dry_run` on any config first for a zero-cost plan/data/gate check.

## Scale notes

- **7B** — primary. vLLM on (student rollouts) fits one 80GB card.
- **14B** — LoRA only; vLLM **off** (HF-generate rollouts) so student base + teacher base fit ~60GB
  on one 80GB card. Aim at a big card with `GPU=<id>`. Full-FT 14B is rejected (~250GB).
- **full-FT** — set `method.tuning=full`; needs its own lower `learning_rate` (there is no default).
