#!/usr/bin/env bash
# ============================================================================
# run_cluster.sh — Single orchestrator for the cluster A100 run.
#
# Hardware: ONE A100 40GB. Previously had a ~15GB orphan eating headroom; that
# has cleared. The script's memory budgeting assumes the full ~40GB is now
# available; if a co-tenant returns, drop VLLM_MEM_EVAL / VLLM_MEM_TRAIN to
# compensate.
#
# What fits:
#   - 3B anything (LoRA train, eval, ceiling): comfortable.
#   - 7B eval, 7B LoRA in bf16: fine.
#   - 7B SDFT FULL-FT (student+teacher resident ~28GB + grads/opt/KV): does
#     NOT fit; intentionally NOT in this script.
#
# Phases (LoRA-focused after Simon's scope change; eval phases retained as
# the practical-work artifact + eval-chain validation):
#   evals  — calibration, 7B ceiling, 3B grid (eval-only, validates pipeline)
#   train  — LoRA training matrix (the thesis core)
#
# Training matrix per size:
#   classic_sft     classic offline SFT-LoRA, no teacher, no rollouts (CE on
#                   golden_response). The canonical SFT baseline.
#   sdft            SDFT-LoRA with frozen-base teacher (A.3 underperforming
#                   arm — lower-bound for SDFT under LoRA).
#   sdft_ema        SDFT-LoRA with adapter-EMA teacher (A.3 recommended arm,
#                   paper-faithful). The teacher gets its own LoRA, EMA-mixed
#                   from the student's LoRA each step.
#   online_sft      "online SFT" — teacher rolls out completions (requires
#                   vLLM). On-policy-isolation ablation, not the SFT baseline.
#
# Comparisons:
#   within-size:  classic_sft vs sdft vs sdft_ema  → SDFT-vs-SFT thesis claim
#   across-size:  3B vs 7B per arm                 → does the gap scale?
#   teacher arm:  sdft vs sdft_ema                 → settles the A.3 confound
#
# Usage:
#   ./run_cluster.sh setup        # one-time: create venv + install deps
#   ./run_cluster.sh phase1       # calibration: eval 3 known 7B checkpoints
#   ./run_cluster.sh phase2       # 7B ceiling: base + teacher_ceiling
#   ./run_cluster.sh phase3       # 3B 4-cell grid re-run on CUDA
#   ./run_cluster.sh phase4       # 3B LoRA matrix (4 arms) + evals
#   ./run_cluster.sh phase5       # 7B LoRA matrix (3 arms; no online_sft) + evals
#   ./run_cluster.sh phase6       # post-hoc strict scorer audit
#   ./run_cluster.sh phase7       # summary table across all eval_results.json
#   ./run_cluster.sh evals        # phase1 + phase2 + phase3 + phase6 + phase7
#   ./run_cluster.sh training     # phase4 + phase5 + phase6 + phase7
#   ./run_cluster.sh all          # setup -> phase7
#
# Per-phase logs land in logs/cluster_<YYYYMMDD>/<phase>.log; wall times in
# logs/cluster_<YYYYMMDD>/wall_times.csv. Both should be copied off the pod
# to NFS for reproducibility.
# ============================================================================

set -euo pipefail
export HF_HOME=/var/nfs/hf-cache
export HF_HUB_CACHE=/var/nfs/hf-cache/hub

# ----------------------------------------------------------------------------
# Paths / config
# ----------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
LOG_DIR="logs/cluster_${RUN_TAG}"
WALL_TIMES_CSV="${LOG_DIR}/wall_times.csv"
mkdir -p "$LOG_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${VENV_DIR}/bin/python"

# Knobs — every one of these honors environment overrides at call time:
#   VLLM_MEM=0.25 NUM_TRAIN_EPOCHS=1.0 ./run_cluster.sh phase4
# Edit only the right-hand-side defaults; the `:-` pattern picks the env
# value if set, otherwise falls back here.

# Eval hyperparameters (reference protocol)
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"
EVAL_TEMP="${EVAL_TEMP:-0.0}"

# LoRA training hyperparameters
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_LR="${LORA_LR:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2.0}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"   # matches Khamis et al. 7B-FT reproduction's effective batch 32 (paper sweep was {16,32,64})

# Intermediate checkpointing — so a mid-arm kill (especially in the 5-8h 7B
# arms under HF generate) only loses up to SAVE_STEPS worth of work, not the
# whole run. Resume manually via:
#   ./run_cluster.sh — won't auto-resume; instead invoke the trainer directly:
#   .venv/bin/python train_sdft_lora.py --resume_from_checkpoint auto ... (same flags as the original)
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# EMA mixup rate for adapter-EMA teacher (sdft_ema arm only).
# Mixing rule: teacher = (1-α)·teacher + α·student. α is the teacher's
# tracking SPEED. Paper used 0.01 for full-FT EMA; under LoRA the dynamics
# differ (A/B scale, B=0 init, effective delta depends on rank+lora_alpha)
# so 0.01 is a starting guess, NOT a known-good.
#
# Reading the kl_approx curve vs the FROZEN arm — be careful: in the
# frozen arm the teacher is a stationary target, so kl_approx is
# student-moves-toward-fixed-point. In the EMA arm the teacher chases
# the student, so the same kl_approx value reflects a gap closing from
# BOTH sides. EMA's curve may sit at a LOWER absolute level (e.g.
# 0.05-0.15 vs frozen's 0.1-0.3) while being equally healthy. Compare
# SHAPE (slow drift, non-collapsing, non-exploding), not absolute band.
# A lower-but-stable EMA curve is NOT a failure signal.
#
# Failure → fix mapping (DON'T invert these):
#   kl_approx collapses to ~0 in <50 steps   →  α TOO HIGH (teacher catches
#                                                student before signal
#                                                accumulates).  LOWER α:
#       EMA_ALPHA=0.001 RUN_TAG=ema_a001 ./run_cluster.sh canary
#   entropy explodes (>~2.5 nats early)      →  α TOO LOW (teacher lags,
#                                                student drifts).  RAISE α:
#       EMA_ALPHA=0.05  RUN_TAG=ema_a05  ./run_cluster.sh canary
#       EMA_ALPHA=0.1   RUN_TAG=ema_a10  ./run_cluster.sh canary
EMA_ALPHA="${EMA_ALPHA:-0.01}"

# Metric tracking — default to wandb. Set REPORT_TO=none to skip.
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-sdft-replication}"

# vLLM memory utilization — SPLIT between eval (pure vLLM) and training
# (colocated with HF student+teacher resident).
#   VLLM_MEM_EVAL  = 0.5  → ~20 GB on A100; safe for 7B vLLM-only.
#   VLLM_MEM_TRAIN = 0.3  → paper-default (main.py); leaves headroom for
#                            colocated student+teacher in HF.
# The old single $VLLM_MEM alias still works for back-compat; if set, it
# overrides both. Otherwise the two split values apply per phase.
VLLM_MEM_EVAL="${VLLM_MEM_EVAL:-0.5}"
VLLM_MEM_TRAIN="${VLLM_MEM_TRAIN:-0.3}"
VLLM_MEM="${VLLM_MEM:-}"   # back-compat — if set, used for BOTH
if [[ -n "$VLLM_MEM" ]]; then
    VLLM_MEM_EVAL="$VLLM_MEM"
    VLLM_MEM_TRAIN="$VLLM_MEM"
fi
VLLM_MODE="${VLLM_MODE:-colocate}"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log_phase() {
    local title="$1"
    echo
    echo "============================================================"
    echo "  $title"
    echo "  $(date -Iseconds)"
    echo "============================================================"
}

# run_step <step_name> <log_subpath> <command...>
# Tees stdout+stderr to the log, records wall time to CSV.
run_step() {
    local step="$1"; shift
    local log_subpath="$1"; shift
    local logfile="${LOG_DIR}/${log_subpath}"
    mkdir -p "$(dirname "$logfile")"

    echo "[run] $step  -> $logfile"
    local start_ts end_ts duration
    start_ts=$(date +%s)
    if "$@" 2>&1 | tee "$logfile"; then
        local rc=0
    else
        local rc=$?
    fi
    end_ts=$(date +%s)
    duration=$(( end_ts - start_ts ))

    if [[ ! -f "$WALL_TIMES_CSV" ]]; then
        echo "step,log,start_iso,duration_seconds,duration_human,exit_code" > "$WALL_TIMES_CSV"
    fi
    printf "%s,%s,%s,%d,%dh%02dm%02ds,%d\n" \
        "$step" "$logfile" "$(date -Iseconds -d @${start_ts} 2>/dev/null || date -Iseconds)" \
        "$duration" $((duration/3600)) $(((duration%3600)/60)) $((duration%60)) "$rc" \
        >> "$WALL_TIMES_CSV"

    return "$rc"
}

require_venv() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: python venv not found at $PYTHON. Run './run_cluster.sh setup' first." >&2
        exit 1
    fi
}

# ----------------------------------------------------------------------------
# Phase 0 — setup
# ----------------------------------------------------------------------------
phase0_setup() {
    log_phase "Phase 0 — setup"

    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: 'uv' not on PATH. Install uv first (the Docker base image ships it)." >&2
        exit 1
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not found — are you in a GPU pod?" >&2
        exit 1
    fi

    echo "[gpu] baseline state:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap \
        --format=csv

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "[venv] creating $VENV_DIR with Python 3.12"
        uv venv --python 3.12 "$VENV_DIR"
    fi
    echo "[venv] installing requirements.txt"
    uv pip install --python "$PYTHON" -r requirements.txt

    # lm-evaluation-harness — pinned to the commit upstream README references.
    echo "[venv] installing lm-evaluation-harness for the forgetting suite"
    uv pip install --python "$PYTHON" \
        "git+https://github.com/EleutherAI/lm-evaluation-harness@03c44adc0586f88bb343a74da1a1c602103536dd"

    echo "[venv] dependency snapshot saved to ${LOG_DIR}/pip_freeze.txt"
    "$PYTHON" -m pip freeze > "${LOG_DIR}/pip_freeze.txt"

    echo "[ok] setup complete"
}

# ----------------------------------------------------------------------------
# Phase 1 — calibration (eval 3 known 7B checkpoints; expect ~70 / ~70 / ~42.2)
# ----------------------------------------------------------------------------
phase1_calibration() {
    log_phase "Phase 1 — calibration (3 known 7B checkpoints)"
    require_venv

    local calib=(
        "sdft-7b-improbable    baselines/calib_sdft_improbable_7b"
        "sdft-7b-kickit        baselines/calib_sdft_kickit_7b"
        "qwen2.5-7b            baselines/calib_qwen2.5_7b_base"
    )
    for entry in "${calib[@]}"; do
        # shellcheck disable=SC2086
        set -- $entry
        local model="$1"
        local outdir="$2"
        run_step "phase1.${model}" "phase1/${model}.log" \
            "$PYTHON" eval_tooluse.py \
                --model_path "$model" \
                --engine vllm \
                --gpu_memory_utilization "$VLLM_MEM_EVAL" \
                --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
                --temperature "$EVAL_TEMP" \
                --output_dir "$outdir"
    done
}

# ----------------------------------------------------------------------------
# Phase 2 — 7B base + ceiling on eval split (scale point)
# ----------------------------------------------------------------------------
phase2_7b_ceiling() {
    log_phase "Phase 2 — 7B base + ceiling on eval_data"
    require_venv

    run_step "phase2.7b_base" "phase2/7b_base.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-7b \
            --engine vllm \
            --gpu_memory_utilization "$VLLM_MEM_EVAL" \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
            --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-7b-instruct-cuda

    run_step "phase2.7b_ceiling" "phase2/7b_ceiling.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-7b \
            --engine vllm \
            --gpu_memory_utilization "$VLLM_MEM_EVAL" \
            --teacher_ceiling \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
            --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-7b-instruct-ceiling-cuda
}

# ----------------------------------------------------------------------------
# Phase 3 — 3B 4-cell grid re-run on CUDA (numerics check vs MPS)
# ----------------------------------------------------------------------------
phase3_3b_grid() {
    log_phase "Phase 3 — 3B 4-cell grid on CUDA"
    require_venv

    run_step "phase3.eval_base" "phase3/eval_base.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-3b \
            --engine vllm \
            --gpu_memory_utilization "$VLLM_MEM_EVAL" \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-3b-instruct-cuda

    run_step "phase3.eval_ceiling" "phase3/eval_ceiling.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-3b \
            --engine vllm \
            --gpu_memory_utilization "$VLLM_MEM_EVAL" \
            --teacher_ceiling \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-3b-instruct-teacher-ceiling-cuda

    run_step "phase3.holdout_base" "phase3/holdout_base.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-3b \
            --engine vllm \
            --gpu_memory_utilization "$VLLM_MEM_EVAL" \
            --eval_data data/tooluse_data/train_subset_holdout \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-3b-instruct-holdout-base-cuda

    run_step "phase3.holdout_ceiling" "phase3/holdout_ceiling.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-3b \
            --engine vllm \
            --gpu_memory_utilization "$VLLM_MEM_EVAL" \
            --eval_data data/tooluse_data/train_subset_holdout \
            --teacher_ceiling \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-3b-instruct-holdout-ceiling-cuda
}

# ----------------------------------------------------------------------------
# Phase 4 / 5 — LoRA training matrix (4 arms × 2 sizes)
# ----------------------------------------------------------------------------
# Arms: classic_sft | sdft | sdft_ema | online_sft
# - classic_sft : train_sft_lora.py (TRL SFTTrainer, CE on golden_response).
#                 No teacher, no vLLM. Fits 3B AND 7B comfortably.
# - sdft        : train_sdft_lora.py with frozen-base teacher (A.3 worse arm).
# - sdft_ema    : train_sdft_lora.py + --teacher_adapter_ema. Paper-faithful
#                 EMA-of-student teacher recovered under LoRA. Two LoRA wraps
#                 (student + teacher) but both small; fits 3B and 7B.
# - online_sft  : train_sdft_lora.py + --generate_from_teacher. On-policy SFT
#                 isolation ablation. Requires vLLM → skipped at 7B (vLLM
#                 student copy + resident student+teacher overflow 40 GB).

_model_id() {
    case "$1" in
        qwen2.5-3b) echo "Qwen/Qwen2.5-3B-Instruct" ;;
        qwen2.5-7b) echo "Qwen/Qwen2.5-7B-Instruct" ;;
        *) echo "unknown model short: $1" >&2; return 1 ;;
    esac
}

# Classic offline SFT-LoRA — separate trainer, no teacher, no rollouts.
_train_classic_sft() {
    local model_short="$1"
    local outdir="$2"

    run_step "train.classic_sft_lora_${model_short}" "phase_train/classic_sft_lora_${model_short}.log" \
        "$PYTHON" train_sft_lora.py \
            --model_name "$(_model_id "$model_short")" \
            --output_dir "$outdir" \
            --learning_rate "$LORA_LR" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --num_train_epochs "$NUM_TRAIN_EPOCHS" \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps "$GRAD_ACCUM" \
            --max_prompt_length 1024 \
            --max_completion_length 1024 \
            --bf16 \
            --save_steps "$SAVE_STEPS" \
            --save_total_limit "$SAVE_TOTAL_LIMIT" \
            --report_to "$REPORT_TO" \
            --wandb_project "$WANDB_PROJECT" \
            --run_name "classic_sft_lora_${model_short}_${RUN_TAG}"
}

# SDFT-LoRA variants — all share train_sdft_lora.py with mode-specific flags.
_train_lora() {
    local mode="$1"           # sdft | sdft_ema | online_sft
    local model_short="$2"    # qwen2.5-3b | qwen2.5-7b
    local outdir="$3"
    local extra_flags=()

    case "$mode" in
        sdft)
            ;;
        sdft_ema)
            extra_flags+=(--teacher_adapter_ema --ema_alpha "$EMA_ALPHA")
            ;;
        online_sft)
            # Teacher rolls out → online SFT (NOT the classic SFT baseline).
            extra_flags+=(--generate_from_teacher)
            ;;
        *) echo "unknown train mode: $mode" >&2; exit 1 ;;
    esac

    # vLLM rollouts: ON for 3B always; for 7B only when memory clearly fits
    # (sdft & sdft_ema can fit if VLLM_MEM_TRAIN is conservative; online_sft
    # at 7B requires vLLM but doesn't fit and is gated upstream).
    case "$model_short" in
        qwen2.5-3b)
            extra_flags+=(--use_vllm
                          --vllm_mode "$VLLM_MODE"
                          --vllm_gpu_memory_utilization "$VLLM_MEM_TRAIN"
                          --vllm_enable_sleep_mode
                          --vllm_importance_sampling_correction)
            ;;
        qwen2.5-7b)
            # 7B without vLLM. HF generate path; works for sdft / sdft_ema
            # (student samples). online_sft at 7B is gated by caller.
            :
            ;;
    esac

    run_step "train.${mode}_lora_${model_short}" "phase_train/${mode}_lora_${model_short}.log" \
        "$PYTHON" train_sdft_lora.py \
            --model_name "$(_model_id "$model_short")" \
            --output_dir "$outdir" \
            --learning_rate "$LORA_LR" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --num_train_epochs "$NUM_TRAIN_EPOCHS" \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps "$GRAD_ACCUM" \
            --max_prompt_length 1024 \
            --max_completion_length 1024 \
            --bf16 \
            --enable_input_require_grads \
            --save_steps "$SAVE_STEPS" \
            --save_total_limit "$SAVE_TOTAL_LIMIT" \
            --report_to "$REPORT_TO" \
            --wandb_project "$WANDB_PROJECT" \
            --run_name "${mode}_lora_${model_short}_${RUN_TAG}" \
            "${extra_flags[@]}"
}

_eval_lora_adapter() {
    local model_short="$1"
    local adapter_dir="$2"
    local out_prefix="$3"
    local base_id
    base_id="$(case "$model_short" in
        qwen2.5-3b) echo "Qwen/Qwen2.5-3B-Instruct" ;;
        qwen2.5-7b) echo "Qwen/Qwen2.5-7B-Instruct" ;;
    esac)"

    # vLLM rejects --adapter_path; use HF engine for adapter eval.
    # Tool-use, eval_data
    run_step "eval.${out_prefix}.eval" "eval_lora/${out_prefix}_eval.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path "$base_id" \
            --adapter_path "$adapter_dir" \
            --engine hf \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir "baselines/${out_prefix}_eval"

    # Tool-use, holdout
    run_step "eval.${out_prefix}.holdout" "eval_lora/${out_prefix}_holdout.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path "$base_id" \
            --adapter_path "$adapter_dir" \
            --engine hf \
            --eval_data data/tooluse_data/train_subset_holdout \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir "baselines/${out_prefix}_holdout"

    # Forgetting suite (lm-eval-harness) — peft= loads the adapter without merging.
    run_step "eval.${out_prefix}.forgetting" "eval_lora/${out_prefix}_forgetting.log" \
        "$VENV_DIR/bin/lm_eval" \
            --model hf \
            --model_args "pretrained=${base_id},peft=${adapter_dir},dtype=bfloat16" \
            --tasks hellaswag,mmlu,truthfulqa_mc2,winogrande,humaneval,ifeval \
            --batch_size 8 \
            --output_path "baselines/${out_prefix}_forgetting" \
            --confirm_run_unsafe_code
}

phase4_3b_lora() {
    log_phase "Phase 4 — 3B LoRA matrix (classic_sft, sdft_ema, sdft, online_sft) + evals"
    require_venv

    # CANARY-LED ORDER. classic_sft is the fastest path AND validates the
    # data pipeline + TRL assistant_only_loss feature. sdft_ema is second
    # because it validates the LoRAEMACallback (asserts on step 1 if LoRA
    # pairing is wrong). sdft (frozen) is known-good from earlier smoke.
    # online_sft is last because it's the most expensive and least critical.

    # 1) CANARY 1: classic offline SFT-LoRA — data path + TRL feature.
    _train_classic_sft qwen2.5-3b runs/classic_sft_lora_3b
    _eval_lora_adapter qwen2.5-3b runs/classic_sft_lora_3b/lora_adapter classic_sft_lora_3b

    # 2) CANARY 2: SDFT-LoRA with adapter-EMA teacher — validates EMA callback.
    _train_lora sdft_ema qwen2.5-3b runs/sdft_ema_lora_3b
    _eval_lora_adapter qwen2.5-3b runs/sdft_ema_lora_3b/lora_adapter sdft_ema_lora_3b

    # 3) SDFT-LoRA, frozen-base teacher (A.3 worse arm; safety net).
    _train_lora sdft qwen2.5-3b runs/sdft_lora_3b
    _eval_lora_adapter qwen2.5-3b runs/sdft_lora_3b/lora_adapter sdft_lora_3b

    # 4) Online SFT — on-policy-isolation ablation, not the SFT baseline.
    _train_lora online_sft qwen2.5-3b runs/online_sft_lora_3b
    _eval_lora_adapter qwen2.5-3b runs/online_sft_lora_3b/lora_adapter online_sft_lora_3b
}

# Run JUST the two canary trainings — no evals, no other arms. Use this to
# validate the matrix before committing the 16h+ full phase4.
phase4_canary() {
    log_phase "Phase 4 — CANARY (classic_sft + sdft_ema 3B training only, no evals)"
    require_venv

    echo "[canary] sdft_ema arm will use EMA_ALPHA=$EMA_ALPHA"
    echo "[canary] Mixing rule: teacher = (1-α)·teacher + α·student"
    echo "[canary]   kl_approx collapses → α TOO HIGH → LOWER:  EMA_ALPHA=0.001"
    echo "[canary]   entropy explodes    → α TOO LOW  → RAISE: EMA_ALPHA=0.05 or 0.1"
    echo

    _train_classic_sft qwen2.5-3b runs/classic_sft_lora_3b
    _train_lora sdft_ema qwen2.5-3b runs/sdft_ema_lora_3b

    echo
    echo "[canary] Both training canaries finished. If wandb shows healthy"
    echo "[canary]   curves for both runs, launch the full phase4."
    echo "[canary] Quick checks:"
    echo "[canary]   - [ema] line on canary 2: 'matched N LoRA parameter pairs' —"
    echo "[canary]     expected N = 7 target_modules × 2 (A+B) × num_layers."
    echo "[canary]     Qwen2.5-3B has 36 layers → N=504. Glance to confirm; a low"
    echo "[canary]     N would mean teacher LoRA isn't fully shadowing student."
    echo "[canary]   - classic_sft loss curve: should decrease from ~1-2 toward <0.5 by epoch 1"
    echo "[canary]   - sdft_ema kl_approx: compare SHAPE not LEVEL to sdft frozen."
    echo "[canary]     EMA arm may sit lower (e.g. 0.05-0.15 vs frozen's 0.1-0.3) and"
    echo "[canary]     still be healthy — the teacher chases the student, so the gap"
    echo "[canary]     closes from both sides. Failure is collapse-to-zero or NaN."
    echo "[canary]     Collapse-to-~0 in <50 steps => α too high. LOWER α:"
    echo "[canary]     EMA_ALPHA=0.001 RUN_TAG=ema_a001 ./run_cluster.sh canary"
    echo "[canary]   - sdft_ema entropy: should hover similarly to sdft — NOT exploding above"
    echo "[canary]     ~2.5 nats early (=> α too low). If it does, RAISE α:"
    echo "[canary]     EMA_ALPHA=0.05 RUN_TAG=ema_a05 ./run_cluster.sh canary"
    echo "[canary]   - both adapters saved at runs/<name>/lora_adapter/"
    echo "[canary]   - intermediate checkpoints at runs/<name>/checkpoint-{50,100,...}/"
}

phase5_7b_lora() {
    log_phase "Phase 5 — 7B LoRA matrix (classic_sft, sdft_ema, sdft) + evals"
    require_venv

    # Same canary-led order as phase4. classic_sft is fastest, sdft_ema
    # validates the EMA callback at 7B (different LoRA shapes, worth its
    # own canary), sdft is the safety net. 7B SDFT arms use HF generate
    # (no vLLM) so each is 5-8h — wandb + SAVE_STEPS=50 mean a kill at
    # hour 6 only loses <2h of work, not the whole arm.

    # 1) CANARY 1: classic offline SFT-LoRA — fits trivially at 7B.
    _train_classic_sft qwen2.5-7b runs/classic_sft_lora_7b
    _eval_lora_adapter qwen2.5-7b runs/classic_sft_lora_7b/lora_adapter classic_sft_lora_7b

    # 2) CANARY 2: SDFT-LoRA at 7B with adapter-EMA teacher.
    _train_lora sdft_ema qwen2.5-7b runs/sdft_ema_lora_7b
    _eval_lora_adapter qwen2.5-7b runs/sdft_ema_lora_7b/lora_adapter sdft_ema_lora_7b

    # 3) SDFT-LoRA at 7B, frozen-base teacher (safety net).
    _train_lora sdft qwen2.5-7b runs/sdft_lora_7b
    _eval_lora_adapter qwen2.5-7b runs/sdft_lora_7b/lora_adapter sdft_lora_7b

    # online_sft at 7B requires vLLM (HF generate ignores generate_from_teacher),
    # and vLLM-student-copy + student/teacher resident overflows 40 GB. The
    # classic_sft arm at 7B is the SFT baseline; the gap doesn't leave the
    # scaling claim empty.
    echo "[notice] Skipping online_sft 7B: requires vLLM colocate that overflows 40GB."
    echo "[notice] classic_sft_7b is the load-bearing SFT baseline at 7B."
}

# ----------------------------------------------------------------------------
# Phase 6 — strict-scorer audit pass over everything new
# ----------------------------------------------------------------------------
phase6_scorer_audit() {
    log_phase "Phase 6 — strict-scorer audit"
    require_venv

    run_step "phase6.audit" "phase6/audit.log" \
        "$PYTHON" scorer_audit.py
}

# ----------------------------------------------------------------------------
# Phase 7 — summary table
# ----------------------------------------------------------------------------
phase7_summary() {
    log_phase "Phase 7 — summary"
    require_venv

    "$PYTHON" - <<'PYEOF'
import json, pathlib
rows = []
for p in sorted(pathlib.Path("baselines").rglob("eval_results.json")):
    try:
        d = json.load(open(p))
        cfg = d.get("config", {})
        rows.append({
            "dir": str(p.parent),
            "acc": d.get("accuracy"),
            "n": d.get("num_total"),
            "correct": d.get("num_correct"),
            "tokens": cfg.get("max_new_tokens"),
            "ceiling": cfg.get("teacher_ceiling"),
            "eval_data": cfg.get("eval_data"),
            "adapter": cfg.get("adapter_path"),
        })
    except Exception as e:
        print(f"skip {p}: {e}")

if not rows:
    print("(no eval_results.json files found)")
else:
    width = max(len(r["dir"]) for r in rows)
    print(f"{'dir'.ljust(width)}  {'acc':>8} {'n':>5} {'tok':>5} {'ceil':>5} {'adapter':>5} {'data':>30}")
    print("-" * (width + 70))
    for r in rows:
        acc = f"{r['acc']*100:6.2f}%" if r["acc"] is not None else "  n/a "
        ceil = "Y" if r["ceiling"] else "N" if r["ceiling"] is False else "-"
        adapter = "Y" if r["adapter"] else "-"
        data = (r["eval_data"] or "eval_data").rsplit("/", 1)[-1][:30]
        print(f"{r['dir'].ljust(width)}  {acc:>8} {r['n']:>5} {r['tokens']:>5} {ceil:>5} {adapter:>5} {data:>30}")
PYEOF

    if [[ -f "$WALL_TIMES_CSV" ]]; then
        echo
        echo "Per-step wall times: $WALL_TIMES_CSV"
        column -t -s, "$WALL_TIMES_CSV"
    fi
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------
case "${1:-help}" in
    setup)     phase0_setup ;;
    phase1)    phase1_calibration ;;
    phase2)    phase2_7b_ceiling ;;
    phase3)    phase3_3b_grid ;;
    phase4)    phase4_3b_lora ;;
    phase5)    phase5_7b_lora ;;
    phase6)    phase6_scorer_audit ;;
    phase7)    phase7_summary ;;
    canary)    phase4_canary ;;
    verify_targets)
        require_venv
        "$PYTHON" verify_training_targets.py --row 0 --n 1
        ;;
    evals)     phase1_calibration; phase2_7b_ceiling; phase3_3b_grid; phase6_scorer_audit; phase7_summary ;;
    training)  phase4_3b_lora; phase5_7b_lora; phase6_scorer_audit; phase7_summary ;;
    all)
        phase0_setup
        phase1_calibration
        phase2_7b_ceiling
        phase3_3b_grid
        phase4_3b_lora
        phase5_7b_lora
        phase6_scorer_audit
        phase7_summary
        ;;
    help|--help|-h|"")
        cat <<EOF
run_cluster.sh — single orchestrator for the cluster A100 run.

Hardware assumption: ONE A100 40GB with ~7.5GB already used by another
process. Plan around ~33GB free. 7B FULL-FT SDFT does NOT fit and is not
in this script; the LoRA pair at 7B is.

Commands:
  setup     create venv (.venv, Python 3.12) and install requirements.txt
            plus lm-evaluation-harness (forgetting suite).
  phase1    calibration: eval 3 known 7B checkpoints (expect ~70/~70/~42.2).
  phase2    7B base + teacher_ceiling on eval_data (the scale point).
  phase3    re-run the 3B 4-cell grid on CUDA.
  phase4    3B LoRA matrix (4 arms):
              classic_sft / sdft / sdft_ema / online_sft
            + tool-use eval (eval_data + holdout) + forgetting suite per arm.
  phase5    7B LoRA matrix (3 arms; online_sft skipped at 7B):
              classic_sft / sdft / sdft_ema
            + same eval shape.
  phase6    strict-scorer audit pass over all new eval_responses.json.
  phase7    summary table across baselines/*/eval_results.json.

  canary          run JUST classic_sft_3b + sdft_ema_3b TRAINING (no evals)
                  Use this as a fail-fast check before committing 16h+ of
                  phase4. Catches: data-pipeline bugs, TRL feature mismatches,
                  LoRAEMACallback pairing failures — all at step 1.
  verify_targets  print one row through classic-SFT and SDFT pipelines side
                  by side to confirm both consume full golden_response.

  evals     phase1 + phase2 + phase3 + phase6 + phase7  (eval-only pipeline)
  training  phase4 + phase5 + phase6 + phase7           (training-only)
  all       setup -> phase1 -> ... -> phase7

Logs:       ${LOG_DIR}/<phase>/<step>.log
Wall times: ${WALL_TIMES_CSV}

Env overrides (prefix the command):
  RUN_TAG=<str>            log dir suffix (default: today's YYYYMMDD)
  VENV_DIR=<path>          venv location (default: .venv)
  EVAL_MAX_NEW_TOKENS=<n>  default 2048
  EVAL_TEMP=<f>            default 0.0 (greedy)
  LORA_R=<n>               default 16
  LORA_ALPHA=<n>           default 32
  LORA_LR=<f>              default 1e-4
  NUM_TRAIN_EPOCHS=<f>     default 2.0
  GRAD_ACCUM=<n>           default 32 (Khamis et al. effective batch; paper sweep {16,32,64})
  SAVE_STEPS=<n>           default 50 (mid-arm checkpoints; bumps survive kills)
  SAVE_TOTAL_LIMIT=<n>     default 3 (keep last N checkpoints; older GC'd)
  EMA_ALPHA=<f>            default 0.01 (sdft_ema teacher mixup; sweep if canary 2 misbehaves)
  VLLM_MEM_EVAL=<f>        default 0.5 (pure-vLLM evals in phase 1/2/3)
  VLLM_MEM_TRAIN=<f>       default 0.3 (colocated with student+teacher in phase 4)
  VLLM_MEM=<f>             back-compat alias — if set, overrides BOTH above
  VLLM_MODE=<colocate|server>  default colocate
  REPORT_TO=<wandb|tensorboard|none>  default wandb (training metrics)
  WANDB_PROJECT=<str>      default sdft-replication
  WANDB_API_KEY=<str>      required for wandb; or run \`wandb login\` once

Examples:
  VLLM_MEM=0.25 ./run_cluster.sh phase4
  RUN_TAG=v2 NUM_TRAIN_EPOCHS=1.0 LORA_R=32 LORA_ALPHA=64 ./run_cluster.sh phase4
  REPORT_TO=wandb WANDB_PROJECT=sdft-thesis ./run_cluster.sh phase4
EOF
        ;;
    *)
        echo "unknown command: $1" >&2
        echo "try: $0 help" >&2
        exit 2
        ;;
esac
