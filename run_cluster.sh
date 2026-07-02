#!/usr/bin/env bash
# run_cluster.sh — orchestrator for the cluster A100 run.
# Usage: ./run_cluster.sh <command>   (see: ./run_cluster.sh help)
# Logs:  logs/cluster_<YYYYMMDD>/<phase>/<step>.log  +  wall_times.csv

set -euo pipefail
export HF_HOME=/var/nfs/hf-cache
export HF_HUB_CACHE=/var/nfs/hf-cache/hub

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
LOG_DIR="logs/cluster_${RUN_TAG}"
WALL_TIMES_CSV="${LOG_DIR}/wall_times.csv"
# LOG_DIR is created lazily by run_step (via mkdir -p on the log's parent), so
# a bare `./run_cluster.sh help` invocation doesn't leave an empty dir behind.

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${VENV_DIR}/bin/python"

EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2048}"
EVAL_TEMP="${EVAL_TEMP:-0.0}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_LR="${LORA_LR:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2.0}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"   # Khamis et al. effective batch 32

SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# teacher = (1-α)·teacher + α·student.
# kl_approx collapses → α too HIGH; entropy explodes → α too LOW.
EMA_ALPHA="${EMA_ALPHA:-0.01}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-sdft-replication}"

# EVAL is pure-vLLM (0.5); TRAIN is colocated with HF student+teacher (0.3).
VLLM_MEM_EVAL="${VLLM_MEM_EVAL:-0.5}"
VLLM_MEM_TRAIN="${VLLM_MEM_TRAIN:-0.3}"
VLLM_MEM="${VLLM_MEM:-}"
if [[ -n "$VLLM_MEM" ]]; then
    VLLM_MEM_EVAL="$VLLM_MEM"
    VLLM_MEM_TRAIN="$VLLM_MEM"
fi
VLLM_MODE="${VLLM_MODE:-colocate}"

REUSE_ADAPTERS="${REUSE_ADAPTERS:-1}"

# ============================================================================
# Helpers
# ============================================================================
log_phase() {
    local title="$1"
    echo
    echo "============================================================"
    echo "  $title"
    echo "  $(date -Iseconds)"
    echo "============================================================"
}

# run_step <name> <log_subpath> <cmd...>   Tees to log, records wall time.
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

# Fault-tolerance for phase4/phase5: one failing eval must not abort a 16-20h phase.
FAILED_STEPS=()

_safe_step() {
    local label="$1"; shift
    local rc=0
    "$@" || rc=$?
    if (( rc != 0 )); then
        echo
        echo "[phase] STEP FAILED: $label (exit $rc). Continuing to next step."
        FAILED_STEPS+=("$label (exit $rc)")
    fi
    return 0
}

_print_failed_summary() {
    local phase_label="$1"
    echo
    if (( ${#FAILED_STEPS[@]} == 0 )); then
        echo "[$phase_label] All steps completed cleanly."
        return 0
    fi
    echo "============================================================"
    echo "  [$phase_label] Completed with ${#FAILED_STEPS[@]} FAILED step(s):"
    for step in "${FAILED_STEPS[@]}"; do
        echo "    - $step"
    done
    echo "  Re-run with REUSE_ADAPTERS=1 (default) to skip successful arms."
    echo "============================================================"
}

_adapter_ready() {
    [[ -s "${1}/lora_adapter/adapter_config.json" ]]
}

_maybe_train_classic_sft() {
    local model_short="$1" outdir="$2"
    if [[ "$REUSE_ADAPTERS" == "1" ]] && _adapter_ready "$outdir"; then
        echo "[reuse] classic_sft ${model_short}: adapter at ${outdir}/lora_adapter — SKIPPING training"
        return 0
    fi
    _train_classic_sft "$model_short" "$outdir"
}

_maybe_train_lora() {
    local mode="$1" model_short="$2" outdir="$3"
    if [[ "$REUSE_ADAPTERS" == "1" ]] && _adapter_ready "$outdir"; then
        echo "[reuse] ${mode} ${model_short}: adapter at ${outdir}/lora_adapter — SKIPPING training"
        return 0
    fi
    _train_lora "$mode" "$model_short" "$outdir"
}

_eval_arm_safely() {
    local arm_label="$1" model_short="$2" outdir="$3" out_prefix="$4"
    if ! _adapter_ready "$outdir"; then
        echo "[phase] $arm_label: adapter missing at ${outdir}/lora_adapter — SKIPPING evals"
        FAILED_STEPS+=("$arm_label.eval (no adapter)")
        return 0
    fi
    _safe_step "$arm_label.eval" _eval_lora_adapter "$model_short" "${outdir}/lora_adapter" "$out_prefix"
}

# ============================================================================
# Phases
# ============================================================================
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

    echo "[venv] installing lm-evaluation-harness for the forgetting suite"
    uv pip install --python "$PYTHON" \
        "git+https://github.com/EleutherAI/lm-evaluation-harness@03c44adc0586f88bb343a74da1a1c602103536dd"

    "$PYTHON" -m pip freeze > "${LOG_DIR}/pip_freeze.txt"
    echo "[ok] setup complete"
}

phase1_calibration() {
    log_phase "Phase 1 — calibration (3 known 7B checkpoints; expect ~70 / ~70 / ~42.2)"
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

_model_id() {
    case "$1" in
        qwen2.5-3b) echo "Qwen/Qwen2.5-3B-Instruct" ;;
        qwen2.5-7b) echo "Qwen/Qwen2.5-7B-Instruct" ;;
        *) echo "unknown model short: $1" >&2; return 1 ;;
    esac
}

_train_classic_sft() {
    local model_short="$1" outdir="$2"
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

_train_lora() {
    local mode="$1" model_short="$2" outdir="$3"
    local extra_flags=()

    case "$mode" in
        sdft) ;;
        sdft_ema)    extra_flags+=(--teacher_adapter_ema --ema_alpha "$EMA_ALPHA") ;;
        online_sft)  extra_flags+=(--generate_from_teacher) ;;
        *) echo "unknown train mode: $mode" >&2; exit 1 ;;
    esac

    # 3B rollouts go through vLLM colocate; 7B uses HF generate (vLLM copy overflows 40GB).
    if [[ "$model_short" == "qwen2.5-3b" ]]; then
        extra_flags+=(--use_vllm
                      --vllm_mode "$VLLM_MODE"
                      --vllm_gpu_memory_utilization "$VLLM_MEM_TRAIN"
                      --vllm_enable_sleep_mode
                      --vllm_importance_sampling_correction)
    fi

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
    local model_short="$1" adapter_dir="$2" out_prefix="$3"
    local base_id
    base_id="$(_model_id "$model_short")"

    # wandb wiring: eval_tooluse.py uses --wandb flags, lm_eval uses --wandb_args.
    # Everything wandb-related is inside these guards; no-op when REPORT_TO != wandb.
    local wb_eval_split=() wb_holdout_split=() wb_lm=()
    if [[ "$REPORT_TO" == "wandb" ]]; then
        wb_eval_split=(--wandb --wandb_project "$WANDB_PROJECT"
                       --wandb_group "$out_prefix"
                       --wandb_run_name "eval_${out_prefix}_eval")
        wb_holdout_split=(--wandb --wandb_project "$WANDB_PROJECT"
                          --wandb_group "$out_prefix"
                          --wandb_run_name "eval_${out_prefix}_holdout")
        wb_lm=(--wandb_args "project=${WANDB_PROJECT},name=eval_${out_prefix}_forgetting,group=${out_prefix},job_type=eval")
    fi

    # vLLM rejects --adapter_path; adapter evals use HF engine.
    run_step "eval.${out_prefix}.eval" "eval_lora/${out_prefix}_eval.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path "$base_id" \
            --adapter_path "$adapter_dir" \
            --engine hf \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir "baselines/${out_prefix}_eval" \
            "${wb_eval_split[@]}"

    run_step "eval.${out_prefix}.holdout" "eval_lora/${out_prefix}_holdout.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path "$base_id" \
            --adapter_path "$adapter_dir" \
            --engine hf \
            --eval_data data/tooluse_data/train_subset_holdout \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir "baselines/${out_prefix}_holdout" \
            "${wb_holdout_split[@]}"

    # peft= loads the adapter without merging.
    # Invoked via `python -m lm_eval` (not $VENV_DIR/bin/lm_eval) so the module
    # import is the requirement, not the entry-point binary — more portable.
    run_step "eval.${out_prefix}.forgetting" "eval_lora/${out_prefix}_forgetting.log" \
        "$PYTHON" -m lm_eval \
            --model hf \
            --model_args "pretrained=${base_id},peft=${adapter_dir},dtype=bfloat16" \
            --tasks hellaswag,mmlu,truthfulqa_mc2,winogrande,humaneval,ifeval \
            --batch_size 8 \
            --output_path "baselines/${out_prefix}_forgetting" \
            --confirm_run_unsafe_code \
            "${wb_lm[@]}"
}

# RUN_TAG is threaded through runs/ and baselines/ so tagged experiments are
# fully independent. Same-day canary → phase4 reuse works because both use
# today's default tag. Cross-day reuse: pass RUN_TAG=<canary_day> explicitly.
_arm_dir() { echo "runs/${1}_${RUN_TAG}"; }        # $1 = arm short (classic_sft_lora_3b)
_arm_prefix() { echo "${1}_${RUN_TAG}"; }          # used as out_prefix → baselines/<prefix>_{eval,holdout,forgetting}

# Order: classic_sft → sdft_ema → sdft → online_sft. Fastest / most novel first.
phase4_3b_lora() {
    log_phase "Phase 4 — 3B LoRA matrix + evals   (tag=${RUN_TAG})"
    require_venv
    FAILED_STEPS=()

    echo "[phase4] REUSE_ADAPTERS=$REUSE_ADAPTERS   fault-tolerant across arms"
    echo

    local d
    d="$(_arm_dir classic_sft_lora_3b)"
    _safe_step "arm1.classic_sft.train" _maybe_train_classic_sft qwen2.5-3b "$d"
    _eval_arm_safely      "arm1.classic_sft" qwen2.5-3b "$d" "$(_arm_prefix classic_sft_lora_3b)"

    d="$(_arm_dir sdft_ema_lora_3b)"
    _safe_step "arm2.sdft_ema.train"   _maybe_train_lora sdft_ema qwen2.5-3b "$d"
    _eval_arm_safely      "arm2.sdft_ema"    qwen2.5-3b "$d" "$(_arm_prefix sdft_ema_lora_3b)"

    d="$(_arm_dir sdft_lora_3b)"
    _safe_step "arm3.sdft.train"       _maybe_train_lora sdft     qwen2.5-3b "$d"
    _eval_arm_safely      "arm3.sdft"        qwen2.5-3b "$d" "$(_arm_prefix sdft_lora_3b)"

    d="$(_arm_dir online_sft_lora_3b)"
    _safe_step "arm4.online_sft.train" _maybe_train_lora online_sft qwen2.5-3b "$d"
    _eval_arm_safely      "arm4.online_sft"  qwen2.5-3b "$d" "$(_arm_prefix online_sft_lora_3b)"

    _print_failed_summary "phase4"
}

# Runs only classic_sft + sdft_ema at 3B, no evals. Fail-fast for a full phase4.
# Writes to the same tagged paths as phase4 → same-tag phase4 will reuse them.
phase4_canary() {
    log_phase "Phase 4 — CANARY (classic_sft + sdft_ema 3B training only)   (tag=${RUN_TAG})"
    require_venv

    echo "[canary] EMA_ALPHA=$EMA_ALPHA   teacher = (1-α)·teacher + α·student"
    echo "[canary]   kl_approx collapses → α TOO HIGH → LOWER (EMA_ALPHA=0.001)"
    echo "[canary]   entropy explodes    → α TOO LOW  → RAISE (EMA_ALPHA=0.05 / 0.1)"
    echo

    _train_classic_sft qwen2.5-3b "$(_arm_dir classic_sft_lora_3b)"
    _train_lora sdft_ema qwen2.5-3b "$(_arm_dir sdft_ema_lora_3b)"

    cat <<EOF

[canary] Both trainings finished. Quick checks:
  - [ema] log line 'matched N LoRA parameter pairs' — Qwen2.5-3B → N=504
    (36 layers × 7 modules × 2 A+B). Lower N = teacher LoRA not fully shadowing student.
  - classic_sft loss: down from ~1-2 toward <0.5 by epoch 1.
  - sdft_ema kl_approx: compare SHAPE not LEVEL vs sdft frozen. EMA may sit
    lower (teacher chases) and still be healthy. Failure = collapse-to-0 or NaN.
  - Adapters at runs/<name>_${RUN_TAG}/lora_adapter/, intermediate ckpts at checkpoint-N/.
EOF
}

# online_sft skipped at 7B: --generate_from_teacher would require vLLM colocate
# for reasonable teacher rollout throughput, but the 7B branch of _train_lora
# does NOT enable vLLM (only 3B does). HF-generate teacher rollouts at 7B every
# step are prohibitively slow. classic_sft_7b remains the SFT baseline at 7B.
phase5_7b_lora() {
    log_phase "Phase 5 — 7B LoRA matrix + evals (online_sft skipped)   (tag=${RUN_TAG})"
    require_venv
    FAILED_STEPS=()

    echo "[phase5] REUSE_ADAPTERS=$REUSE_ADAPTERS   fault-tolerant across arms"
    echo

    local d
    d="$(_arm_dir classic_sft_lora_7b)"
    _safe_step "arm1.classic_sft.train" _maybe_train_classic_sft qwen2.5-7b "$d"
    _eval_arm_safely      "arm1.classic_sft" qwen2.5-7b "$d" "$(_arm_prefix classic_sft_lora_7b)"

    d="$(_arm_dir sdft_ema_lora_7b)"
    _safe_step "arm2.sdft_ema.train"   _maybe_train_lora sdft_ema qwen2.5-7b "$d"
    _eval_arm_safely      "arm2.sdft_ema"    qwen2.5-7b "$d" "$(_arm_prefix sdft_ema_lora_7b)"

    d="$(_arm_dir sdft_lora_7b)"
    _safe_step "arm3.sdft.train"       _maybe_train_lora sdft     qwen2.5-7b "$d"
    _eval_arm_safely      "arm3.sdft"        qwen2.5-7b "$d" "$(_arm_prefix sdft_lora_7b)"

    _print_failed_summary "phase5"
}

phase6_scorer_audit() {
    log_phase "Phase 6 — strict-scorer audit"
    require_venv
    run_step "phase6.audit" "phase6/audit.log" "$PYTHON" scorer_audit.py
}

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
    def fmt(x):
        return "-" if x is None else str(x)
    print(f"{'dir'.ljust(width)}  {'acc':>8} {'correct':>7} {'n':>5} {'tok':>5} {'ceil':>5} {'adapter':>5} {'data':>30}")
    print("-" * (width + 78))
    for r in rows:
        acc = f"{r['acc']*100:6.2f}%" if r["acc"] is not None else "  n/a "
        ceil = "Y" if r["ceiling"] else "N" if r["ceiling"] is False else "-"
        adapter = "Y" if r["adapter"] else "-"
        data = (r["eval_data"] or "eval_data").rsplit("/", 1)[-1][:30]
        print(f"{r['dir'].ljust(width)}  {acc:>8} {fmt(r['correct']):>7} {fmt(r['n']):>5} {fmt(r['tokens']):>5} {ceil:>5} {adapter:>5} {data:>30}")
PYEOF

    if [[ -f "$WALL_TIMES_CSV" ]]; then
        echo
        echo "Per-step wall times: $WALL_TIMES_CSV"
        column -t -s, "$WALL_TIMES_CSV"
    fi
}

# ============================================================================
# Dispatch
# ============================================================================
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
run_cluster.sh — orchestrator for the cluster A100 run.

Commands:
  setup     create venv + install requirements.txt + lm-eval-harness
  phase1    calibration: 3 known 7B checkpoints (expect ~70 / ~70 / ~42.2)
  phase2    7B base + teacher_ceiling on eval_data
  phase3    3B 4-cell grid (eval_data × {base, ceiling} × {eval, holdout})
  phase4    3B LoRA matrix (classic_sft, sdft_ema, sdft, online_sft) + evals
  phase5    7B LoRA matrix (classic_sft, sdft_ema, sdft) + evals
  phase6    strict-scorer audit over eval_responses.json
  phase7    summary table across baselines/*/eval_results.json

  canary          classic_sft_3b + sdft_ema_3b training only, no evals
  verify_targets  side-by-side classic-SFT vs SDFT format for one row

  evals     phase1 + phase2 + phase3 + phase6 + phase7
  training  phase4 + phase5 + phase6 + phase7
  all       phase0 → phase7

Logs:       ${LOG_DIR}/<phase>/<step>.log
Wall times: ${WALL_TIMES_CSV}

Env overrides (prefix the command):
  RUN_TAG=<str>                default: today's YYYYMMDD; scopes LOG_DIR,
                               wandb run names, runs/<arm>_<TAG>/ output paths,
                               AND baselines/<arm>_<TAG>_{eval,holdout,forgetting}/
                               → tagged runs are fully independent from each other.
  VENV_DIR=<path>              default: .venv
  EVAL_MAX_NEW_TOKENS=<n>      default: 2048
  EVAL_TEMP=<f>                default: 0.0 (greedy)
  LORA_R=<n>                   default: 16
  LORA_ALPHA=<n>               default: 32
  LORA_LR=<f>                  default: 1e-4
  NUM_TRAIN_EPOCHS=<f>         default: 2.0
  GRAD_ACCUM=<n>               default: 32 (Khamis et al.)
  SAVE_STEPS=<n>               default: 50
  SAVE_TOTAL_LIMIT=<n>         default: 3
  EMA_ALPHA=<f>                default: 0.01 (sdft_ema teacher mixup)
  REUSE_ADAPTERS=<0|1>         default: 1 (phase4/5 skip train if adapter exists)
  VLLM_MEM_EVAL=<f>            default: 0.5
  VLLM_MEM_TRAIN=<f>           default: 0.3
  VLLM_MEM=<f>                 back-compat alias for both above
  VLLM_MODE=<colocate|server>  default: colocate
  REPORT_TO=<wandb|tensorboard|none>  default: wandb
  WANDB_PROJECT=<str>          default: sdft-replication
  WANDB_API_KEY=<str>          required for wandb (or run \`wandb login\`)

Examples:
  VLLM_MEM=0.25 ./run_cluster.sh phase4
  # Independent v2 experiment with fresh hyperparams — writes to runs/<arm>_v2/
  # and baselines/<arm>_v2_*/, so v1 is untouched:
  RUN_TAG=v2 NUM_TRAIN_EPOCHS=1.0 LORA_R=32 LORA_ALPHA=64 ./run_cluster.sh phase4
  # Reuse a prior-day canary adapter (RUN_TAG must match the canary's tag):
  RUN_TAG=20260701 ./run_cluster.sh phase4
EOF
        ;;
    *)
        echo "unknown command: $1" >&2
        echo "try: $0 help" >&2
        exit 2
        ;;
esac
