#!/usr/bin/env bash
# ============================================================================
# run_cluster.sh — Single orchestrator for the cluster A100 run.
#
# Hardware assumption: ONE A100 40GB with ~7.5GB already used by another
# process. Plan around ~33GB free, not the full 40GB. This is what fits:
#   - 3B anything (eval, LoRA, ceiling): comfortable.
#   - 7B eval, 7B LoRA in bf16: fine.
#   - 7B SDFT FULL-FT (student+teacher resident ~28GB + grads/opt/KV): does
#     NOT fit; it's intentionally NOT in this script. Route that to a bigger
#     card or do CPU-offload of the teacher.
#
# Usage:
#   ./run_cluster.sh setup        # one-time: create venv + install deps
#   ./run_cluster.sh phase1       # calibration: eval 3 known 7B checkpoints
#   ./run_cluster.sh phase2       # 7B ceiling: base + teacher_ceiling
#   ./run_cluster.sh phase3       # 3B 4-cell grid re-run on CUDA
#   ./run_cluster.sh phase4       # 3B LoRA pair: SDFT-LoRA + SFT-LoRA + evals
#   ./run_cluster.sh phase5       # 7B LoRA pair: SDFT-LoRA + SFT-LoRA + evals
#   ./run_cluster.sh phase6       # post-hoc strict scorer audit
#   ./run_cluster.sh phase7       # summary table across all eval_results.json
#   ./run_cluster.sh evals        # phase1 + phase2 + phase3
#   ./run_cluster.sh training     # phase4 + phase5
#   ./run_cluster.sh all          # setup -> phase7
#
# Per-phase logs land in logs/cluster_<YYYYMMDD>/<phase>.log; wall times are
# appended to logs/cluster_<YYYYMMDD>/wall_times.csv.
# ============================================================================

set -euo pipefail

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
GRAD_ACCUM="${GRAD_ACCUM:-8}"

# vLLM memory utilization — matches main.py's paper-default (0.3 leaves room
# for the training process + teacher in colocate mode).
VLLM_MEM="${VLLM_MEM:-0.3}"
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
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
            --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-7b-instruct-cuda

    run_step "phase2.7b_ceiling" "phase2/7b_ceiling.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-7b \
            --engine vllm \
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
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-3b-instruct-cuda

    run_step "phase3.eval_ceiling" "phase3/eval_ceiling.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-3b \
            --engine vllm \
            --teacher_ceiling \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-3b-instruct-teacher-ceiling-cuda

    run_step "phase3.holdout_base" "phase3/holdout_base.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-3b \
            --engine vllm \
            --eval_data data/tooluse_data/train_subset_holdout \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-3b-instruct-holdout-base-cuda

    run_step "phase3.holdout_ceiling" "phase3/holdout_ceiling.log" \
        "$PYTHON" eval_tooluse.py \
            --model_path qwen2.5-3b \
            --engine vllm \
            --eval_data data/tooluse_data/train_subset_holdout \
            --teacher_ceiling \
            --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --temperature "$EVAL_TEMP" \
            --output_dir baselines/qwen2.5-3b-instruct-holdout-ceiling-cuda
}

# ----------------------------------------------------------------------------
# Phase 4 — 3B LoRA pair: SDFT-LoRA + SFT-LoRA + tool-use eval + forgetting
# ----------------------------------------------------------------------------
# 3B leaves enough headroom to run vLLM rollouts during training.
_train_lora() {
    local mode="$1"           # sdft | sft
    local model_short="$2"    # qwen2.5-3b | qwen2.5-7b
    local outdir="$3"
    local extra_flags=()

    case "$mode" in
        sft)
            # SFT-LoRA = teacher rolls out (requires --use_vllm).
            extra_flags+=(--generate_from_teacher)
            ;;
        sdft) ;;
        *) echo "unknown train mode: $mode" >&2; exit 1 ;;
    esac

    # vLLM rollouts: ON for 3B (fits), OFF for 7B (student+teacher resident
    # already eats ~28GB, an extra vLLM copy of student pushes us over the
    # ~33GB shared-GPU budget).
    case "$model_short" in
        qwen2.5-3b)
            extra_flags+=(--use_vllm
                          --vllm_mode "$VLLM_MODE"
                          --vllm_gpu_memory_utilization "$VLLM_MEM"
                          --vllm_enable_sleep_mode
                          --vllm_importance_sampling_correction)
            ;;
        qwen2.5-7b)
            # NO vLLM. SFT-LoRA without vLLM is impossible (generate_from_teacher
            # would silently use the student). Skip the SFT-LoRA 7B pair member
            # in that case — caller handles this.
            :
            ;;
    esac

    run_step "train.${mode}_lora_${model_short}" "phase_train/${mode}_lora_${model_short}.log" \
        "$PYTHON" train_sdft_lora.py \
            --model_name "$(case "$model_short" in
                qwen2.5-3b) echo "Qwen/Qwen2.5-3B-Instruct" ;;
                qwen2.5-7b) echo "Qwen/Qwen2.5-7B-Instruct" ;;
            esac)" \
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
            --tasks hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval \
            --batch_size 8 \
            --output_path "baselines/${out_prefix}_forgetting" \
            --confirm_run_unsafe_code
}

phase4_3b_lora() {
    log_phase "Phase 4 — 3B LoRA pair (SDFT + SFT) + evals"
    require_venv

    _train_lora sdft qwen2.5-3b runs/sdft_lora_3b
    _eval_lora_adapter qwen2.5-3b runs/sdft_lora_3b/lora_adapter sdft_lora_3b

    _train_lora sft qwen2.5-3b runs/sft_lora_3b
    _eval_lora_adapter qwen2.5-3b runs/sft_lora_3b/lora_adapter sft_lora_3b
}

phase5_7b_lora() {
    log_phase "Phase 5 — 7B LoRA pair (SDFT + SFT) + evals"
    require_venv

    # SDFT-LoRA on 7B without vLLM: HF generate path on student. Slower than
    # vLLM but fits the shared-GPU budget.
    _train_lora sdft qwen2.5-7b runs/sdft_lora_7b
    _eval_lora_adapter qwen2.5-7b runs/sdft_lora_7b/lora_adapter sdft_lora_7b

    # SFT-LoRA on 7B requires vLLM (HF generate ignores generate_from_teacher).
    # On a shared 40GB A100 the vLLM-copy + student/teacher resident does not
    # fit. Skip with a notice and surface this as a routing decision for Simon.
    echo "[notice] Skipping SFT-LoRA 7B: requires --use_vllm which doesn't fit on 33GB shared."
    echo "[notice] Options: (a) merge-adapter-eval-only on a bigger card, (b) drop SFT-LoRA"
    echo "[notice]          7B from the pair and frame as future work, (c) ask for an 80GB node."
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
  phase4    3B LoRA pair: SDFT-LoRA + SFT-LoRA + tool-use eval + forgetting.
  phase5    7B LoRA pair: SDFT-LoRA (HF-generate) + skipped SFT-LoRA-7B
            (vLLM required, doesn't fit shared 33GB).
  phase6    strict-scorer audit pass over all new eval_responses.json.
  phase7    summary table across baselines/*/eval_results.json.

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
  GRAD_ACCUM=<n>           default 8
  VLLM_MEM=<f>             default 0.3 (paper-aligned)
  VLLM_MODE=<colocate|server>  default colocate

Examples:
  VLLM_MEM=0.25 ./run_cluster.sh phase4
  RUN_TAG=v2 NUM_TRAIN_EPOCHS=1.0 LORA_R=32 LORA_ALPHA=64 ./run_cluster.sh phase4
EOF
        ;;
    *)
        echo "unknown command: $1" >&2
        echo "try: $0 help" >&2
        exit 2
        ;;
esac
