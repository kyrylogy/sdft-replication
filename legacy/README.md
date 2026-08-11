# legacy/ — pre-config-system pipeline (archived)

These are the original per-script entrypoints, superseded by the config-driven harness
at the repo root (see [`../HARNESS.md`](../HARNESS.md)). Kept for reference and because
the already-produced results in `../pod-results/` and `../baselines/` were generated with
them. **Not maintained; not guaranteed to run from this subdirectory** (relative `data/`
paths assume the repo root as CWD).

| File | Was | Replaced by |
|---|---|---|
| `main.py` | full-FT SDFT reference (paper regime, `sync_ref_model` EMA) | `train.py` (`method.tuning=full, teacher=ema`) |
| `train_sdft_lora.py` | LoRA SDFT trainer (+ `WORK_LOG.md`) | `train.py` (`objective=sdft`) |
| `train_sft_lora.py` | classic LoRA SFT baseline | `train.py` (`objective=sft`) |
| `eval_tooluse.py` | tool-use eval + teacher-ceiling | `eval_runner.py` |
| `eval_science.py` | science eval (vLLM-only, no adapters) | `eval_runner.py` |
| `experiment.py` | tiny smoke driver | `configs/experiments/smoke_*` |
| `run_cluster.sh` | phase1–7 orchestrator | `run.sh` + `configs/` |

Note: the strict tool-use scorer and the SFT/SDFT data formatting these files contained
now live canonically in `../eval_runner.py` and `../datasets_sdft.py`. `run_cluster.sh` still
documents useful eval phases the new harness has not yet fully ported (calibration
checkpoints, the phase-7 summary table) — mine it when wiring those into `eval_runner.py`.
