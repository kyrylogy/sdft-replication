# SDFT under LoRA: a replication

A controlled replication of Self-Distillation Fine-Tuning (SDFT; Shenfeld et al., 2026,
[arXiv:2601.19897](https://arxiv.org/abs/2601.19897)) with rank-16 LoRA on Qwen2.5-7B-Instruct.
Two tasks are learned in sequence (Tool-Use, then Science, same adapter), with three seeds for SFT and
SDFT-EMA and an SFT checkpoint matched to SDFT's stage-1 accuracy as a control.

- **Results page:** <https://kyrylogy.github.io/sdft-replication/>
- **All results, by arm, stage and seed:** [analysis/SUMMARY.md](analysis/SUMMARY.md)
- **Where each number comes from:** [RESULTS.md](RESULTS.md)

## Key results

- On the Tool-Use holdout (APIs seen in training), SDFT's smaller forgetting (BWT −1.0 vs −3.3 pp) is not
  significant (p = 0.30), and SFT stopped early at a similar accuracy matches it (−1.3 pp). Both lose about
  11% of the items they answered correctly after stage 1. On APIs never seen in training the order
  reverses (SDFT −6.9, SFT +2.7 pp).
- SDFT protects general capabilities after the first task (IFEval −6.8 vs −10.4 pp, TruthfulQA −1.5 vs
  −7.8 pp against the base model, on every seed), but not after the second: once Science is learned,
  SDFT is the worst arm on IFEval (−14.7 vs −9.1 pp) and its TruthfulQA edge shrinks to 1 pp.
- SDFT costs 3.2× SFT's training compute. Plain SFT has the best continual-learning average (ACC), within
  0.6 pp of training on both tasks at once.

## Layout

| Path | What |
|---|---|
| `train.py`, `eval_runner.py`, `exp_config.py`, `datasets_sdft.py`, `run.sh`, `run_tier.sh`, `away_final.sh` | config-driven experiment harness ([HARNESS.md](HARNESS.md)) |
| `distil_trainer.py`, `distil_config.py` | SDFT trainer from the reference implementation, unmodified |
| `configs/` | `base.yaml`, shared fragments, and one YAML per run configuration (arm × scale × stage) |
| `data/` | Tool-Use and Science data from the SDFT release, plus the holdout indices |
| `collect_results.py`, `stats_final.py`, `make_figures.py`, `make_summary.py`, `adapter_distance.py`, `battery_audit.py`, `status.sh` | analysis: run records → tables, statistics, figures, `analysis/SUMMARY.md` |
| `export_runs.py`, `export_wandb.py` | publishable copies of the run records and W&B logs |
| `analysis/`, `runs_export/`, `wandb_export/` | results ([RESULTS.md](RESULTS.md)) |
| `docs/` | results page (GitHub Pages) |
| `tests/` | regression tests, no GPU needed |
| `legacy/`, `baselines/`, `scorer_audit.py`, `report_extract.py`, `verify_training_targets.py`, `requirements-laptop.txt` | the original per-script pipeline and an earlier 3B pilot, not used for these results ([legacy/README.md](legacy/README.md)) |

## Reproduce

Tables and figures, from the committed run records (CPU, a couple of minutes):

```bash
pip install pyyaml pandas matplotlib
python collect_results.py --root runs_export
python stats_final.py --root runs_export
python make_figures.py
python make_summary.py
```

Training and evaluation need one 94 GB GPU and Python 3.12. One YAML describes each run, and
`run_tier.sh` runs the 7B matrix in order and resumes after a crash:

```bash
pip install -r requirements.txt
pip install "git+https://github.com/EleutherAI/lm-evaluation-harness@03c44adc0586f88bb343a74da1a1c602103536dd" langdetect immutabledict
wandb login                          # or add --set runtime.report_to=none to every run
GPU=0 SCALE=7b SEEDS="42 1234 2024" ./run_tier.sh run
```

Two parts of the results are not in `run_tier.sh`, shown here for seed 42 (repeat for 1234 and 2024):

```bash
# unseen-API retention of each stage-2 adapter (likewise for sdft_ema_lora_7b_science_s2.yaml)
GPU=0 ./run.sh eval configs/experiments/sft_lora_7b_science_s2.yaml --set train.seed=42 --set experiment.tag=seed42 \
    --set data.dataset=tooluse --set 'eval.sets=[eval_data]' --set eval.checkpoint_curve=false

# SFT-acq50: SFT's checkpoint-50 becomes its own stage-1 run, then continues on Science.
# (Reconstructed from the run records; the original promotion step was not scripted.)
A=runs/sft_lora_7b_tooluse_s1_acq50_seed42
mkdir -p $A/eval/final && cp -r runs/sft_lora_7b_tooluse_s1_seed42/checkpoint-50 $A/lora_adapter
cp runs/sft_lora_7b_tooluse_s1_seed42/resolved_config.yaml $A/
cp -r runs/sft_lora_7b_tooluse_s1_seed42/eval/step50/* $A/eval/final/
GPU=0 ./run.sh eval configs/experiments/sft_lora_7b_tooluse_s1.yaml --set train.seed=42 \
    --set experiment.tag=acq50_seed42 --mode forgetting --set eval.forgetting.enabled=true
GPU=0 ./run.sh train configs/experiments/sft_lora_7b_science_s2.yaml --set train.seed=42 \
    --set experiment.tag=acq50_seed42 --set data.init_adapter=$A/lora_adapter
# then evaluate it like the other stage-2 runs, with --set experiment.tag=acq50_seed42
```

[HARNESS.md](HARNESS.md) explains the harness, [EXPERIMENTS.md](EXPERIMENTS.md) the run plan and
[METRICS.md](METRICS.md) the metrics. Tests: `python -m pytest tests/` and `bash tests/test_run_tier.sh`.

## Credit

Built on the SDFT reference implementation,
[idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation), which also provides the
Tool-Use and Science data. Its README has the original full-fine-tuning instructions.
