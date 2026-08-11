# Thesis experiments — what to run

The runnable plan. Every arm is a config in `configs/experiments/`; run it with
`GPU=<id> ./run.sh train|eval <cfg>`. Metric definitions: [`METRICS.md`](METRICS.md).
Harness reference: [`HARNESS.md`](HARNESS.md).

**Run Tier 1 and you have a defensible thesis.** Everything below it strengthens the
scaling / interaction story but isn't load-bearing for the core claim.

---

## 0. Prerequisites (once)

```bash
export HF_HOME=$HOME/hf_cache      # writable model cache (run.sh also defaults this)
wandb login                        # or set WANDB_API_KEY (report_to=wandb by default)
```
Pick a free GPU each launch (`nvidia-smi`); the examples use GPU 3.

## Cluster reality (shared box — read before launching)

You can't kill other users' jobs, so treat this as a **one-GPU, serialized** run, not the
parallel plan the tiers imply. As of last check: GPU 0/1 are full (other services), GPU 2 has
an active job, **GPU 3 has ~76 GB free** (a parked vLLM engine holds the rest) and is the
workhorse. Consequences:

- **Cap vLLM memory** so it reserves against what's *free*, not a whole empty card. `base.yaml`
  defaults `vllm.gpu_memory_utilization: 0.3` (≈28 GB of 94) — fine at 76 free. If a run dies at
  vLLM init with "less than desired", drop it: `--set vllm.gpu_memory_utilization=0.25`.
- **The driver guards free memory** at launch (`MIN_FREE_MIB`), so another user landing on GPU 3
  mid-run can't silently OOM you — but a run already in flight is still at risk; the cap is your
  protection.
- **Rollout-backend confound (optional cleanup):** the scale curve runs 3B/7B with vLLM but 14B
  with vLLM *off* (HF-generate), so the backend changes with scale. For clean science either
  (a) state it as a caveat, or (b) run everything vLLM-off for a uniform backend — add
  `--set vllm.enabled=false` to the 3B/7B SDFT/online arms. Time one 7B epoch both ways and decide.
- **Full-FT stays out of scope**: it needs multi-GPU sharding across cards that other people own.

## The run cycle (per arm)

Order matters because the stage-2 retention gate **fails closed** without a stage-1 baseline:

```
base anchor ──►  stage-1 train ──►  stage-1 eval ──►  stage-2 train ──►  stage-2 eval ──►  collect + figures
 (once/scale)     (the skill)      (feeds the gate)   (continue+gate)    (retention)
```

---

## Tier 1 — 7B spine (the defensible thesis)

Primary scale. Two arms (SFT vs SDFT-EMA) across both protocols, plus two ablations.
Run the headline cells at **3 seeds** (`SEEDS="42 1234 2024"`); ablations at 1 seed.

### 1a. Base anchors (7B, once)
```bash
GPU=3 ./run.sh eval configs/experiments/sft_lora_7b_tooluse_s1.yaml --base                      # Tool-Use floor
GPU=3 ./run.sh eval configs/experiments/sft_lora_7b_tooluse_s1.yaml --base --set eval.teacher_ceiling=true  # ceiling (target)
GPU=3 ./run.sh eval configs/experiments/sft_lora_7b_tooluse_s1.yaml --base --mode forgetting --set eval.forgetting.enabled=true   # lm-eval anchor
```

### 1b. Single-task (Tool Use): train → eval → forgetting
```bash
# headline arms, 3 seeds each
SEEDS="42 1234 2024" GPU=3 ./run.sh train configs/experiments/sft_lora_7b_tooluse_s1.yaml
SEEDS="42 1234 2024" GPU=3 ./run.sh train configs/experiments/sdft_ema_lora_7b_tooluse_s1.yaml
# ablations, 1 seed
GPU=3 ./run.sh train configs/experiments/sdft_frozen_lora_7b_tooluse_s1.yaml
GPU=3 ./run.sh train configs/experiments/online_sft_lora_7b_tooluse_s1.yaml
# then eval each (accuracy + forgetting). Example for one:
GPU=3 ./run.sh eval configs/experiments/sdft_ema_lora_7b_tooluse_s1.yaml
GPU=3 ./run.sh eval configs/experiments/sdft_ema_lora_7b_tooluse_s1.yaml --mode forgetting --set eval.forgetting.enabled=true
```
Point 1 answers: does it learn the skill (acquisition), and does it damage general
ability (forgetting vs the base anchor). Under LoRA the arm-to-arm forgetting gap is
expected to be *small* — that near-zero gap is a finding.

### 1c. Sequential (Tool Use → Science): the continual-forgetting number
```bash
# continue the SAME adapter onto Science (gate re-checks Tool Use first)
GPU=3 ./run.sh train configs/experiments/sft_lora_7b_science_s2.yaml
GPU=3 ./run.sh train configs/experiments/sdft_ema_lora_7b_science_s2.yaml
# Science acquisition:
GPU=3 ./run.sh eval configs/experiments/sdft_ema_lora_7b_science_s2.yaml
# Tool-Use RETENTION (re-eval the stage-2 adapter on Tool Use):
GPU=3 ./run.sh eval configs/experiments/sdft_ema_lora_7b_science_s2.yaml \
     --set data.dataset=tooluse --set 'eval.sets=[holdout]'
```
Point 2 answers: did learning Science erase Tool Use — measured **against the stage-1
level** (see METRICS.md). This is the thesis's core number.

> **Seed sweeps + sequential:** `SEEDS=...` tags stage-1 as `..._seed42/`, so for each
> seed's stage-2 pass the init: `--set data.init_adapter=runs/sdft_ema_lora_7b_tooluse_s1_seed42/lora_adapter`.

---

## Tier 2 — LoRA scale curve (3B + 14B)

Same two arms, both protocols, at the ends of the scale axis. SDFT is expected to trail
at 3B and lead at 14B (bigger teacher → better in-context demo → better target). Do the
7B middle from Tier 1.

```bash
# 3B (vLLM fits easily)
GPU=3 ./run.sh train configs/experiments/sft_lora_3b_tooluse_s1.yaml
GPU=3 ./run.sh train configs/experiments/sdft_ema_lora_3b_tooluse_s1.yaml
#   ...eval each, then the _science_s2 continuations, same cycle as Tier 1.

# 14B (LoRA only; vLLM OFF -> HF-generate rollouts fit ~60GB on one 94GB card)
GPU=3 ./run.sh train configs/experiments/sft_lora_14b_tooluse_s1.yaml
GPU=3 ./run.sh train configs/experiments/sdft_ema_lora_14b_tooluse_s1.yaml
#   ...eval, then sft_lora_14b_science_s2 / sdft_ema_lora_14b_science_s2.
```

---

## Analysis (after any batch of runs)

```bash
.venv/bin/python collect_results.py     # runs/ -> analysis/*.csv
.venv/bin/python make_figures.py        # analysis/*.csv -> analysis/figures/*.{png,pdf}
```
`runs_index.csv` shows coverage (which cells are done). `retention.csv`, `forgetting.csv`,
`significance.csv` (McNemar), `aggregate.csv` (seed mean±std) are your result tables.

---

## The matrix at a glance

| Arm | 3B | 7B | 14B | protocols |
|---|:--:|:--:|:--:|---|
| SFT-LoRA            | ✓ | **✓** | ✓ | single + sequential |
| SDFT-EMA-LoRA       | ✓ | **✓** | ✓ | single + sequential |
| SDFT-frozen (abl.)  |   | ✓ |   | single |
| online-SFT (abl.)   |   | ✓ |   | single |

Bold = Tier-1 spine. All arms above are wired and single-GPU runnable.

## Deferred — full fine-tuning (a separate milestone)

The paper's regime where forgetting is largest. Configs would be `method.tuning=full`,
but two things aren't ready: (1) the full-FT path is written but untested end-to-end, and
(2) 7B full-FT (~126 GB with the teacher) **exceeds one 94 GB H100** — it needs multi-GPU
sharding (DeepSpeed ZeRO-3 / FSDP across your 4 H100s), which `run.sh` doesn't launch yet.
Full-FT 14B is out of budget entirely. Treat this as a follow-up once the LoRA matrix is in;
it needs an `accelerate launch --multi_gpu` wrapper + a smoke on the full-FT path first.

## Rough cost

Per 7B SDFT arm: single-task train (vLLM rollouts, 2 epochs) is the long pole; eval + the
lm-eval battery add on top. Budget generously and run arms in parallel across your idle GPUs
(2 and 3). Start with **Tier-1 1a→1b for SFT + SDFT-EMA at one seed** to shake out the full
pipeline before committing the 3-seed sweep.
