#!/usr/bin/env bash
# run.sh — one command to train or evaluate a fully-config-described run on a chosen GPU.
#
#   GPU=0 ./run.sh train configs/experiments/sdft_ema_lora_7b_tooluse_s1.yaml
#   GPU=0 ./run.sh eval  configs/experiments/sdft_ema_lora_7b_tooluse_s1.yaml
#   GPU=0 ./run.sh eval  configs/experiments/sdft_ema_lora_7b_tooluse_s1.yaml --base   # base anchor
#   GPU=0 ./run.sh eval  <cfg> --mode forgetting                                       # lm-eval battery
#
# GPU selection: `GPU=0` env, or `--gpu 0`. Multi-GPU: GPU=0,1.
# Seed sweep (train only): `SEEDS="42 1234 2024" GPU=0 ./run.sh train <cfg>` runs one
#   tagged run per seed (experiment.tag=seed<N>), so headline cells get their spread.
# Extra `--set key=value` overrides pass straight through to the Python entrypoint.

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PY="${PYTHON:-.venv/bin/python}"

# Default the HF cache to a writable dir if the environment hasn't set one (shared
# cluster caches are often read-only to you). Override by exporting HF_HOME yourself,
# e.g. HF_HOME=/mnt/data/$USER/hf_cache for big models on a quota-limited home.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME" 2>/dev/null || true

if [[ $# -lt 2 ]]; then
    echo "usage: [GPU=id] [SEEDS='42 ...'] ./run.sh <train|eval> <config.yaml> [--gpu id] [--set k=v ...] [eval flags]" >&2
    exit 2
fi

MODE="$1"; CONFIG="$2"; shift 2

# Pull an optional `--gpu N` out of the passthrough args.
GPU="${GPU:-}"
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        *) PASS+=("$1"); shift ;;
    esac
done

if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "[run] CUDA_VISIBLE_DEVICES=$GPU"
else
    echo "[run] GPU not set (CUDA_VISIBLE_DEVICES unchanged)"
fi

[[ -x "$PY" ]] || { echo "ERROR: python not found at $PY (set PYTHON=...)" >&2; exit 1; }

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"
base="$(basename "${CONFIG%.yaml}")"

_run_one() {
    local extra=("$@")
    local logfile="logs/${base}_${MODE}_${ts}.log"
    echo "[run] $MODE $CONFIG ${extra[*]:-}  -> $logfile"
    case "$MODE" in
        train) "$PY" train.py       --config "$CONFIG" "${extra[@]}" 2>&1 | tee "$logfile" ;;
        eval)  "$PY" eval_runner.py --config "$CONFIG" "${extra[@]}" 2>&1 | tee "$logfile" ;;
        *) echo "unknown mode: $MODE (train|eval)" >&2; exit 2 ;;
    esac
}

if [[ "$MODE" == "train" && -n "${SEEDS:-}" ]]; then
    for s in $SEEDS; do
        echo "== seed $s =="
        _run_one --set "train.seed=$s" --set "experiment.tag=seed$s" "${PASS[@]}"
    done
else
    _run_one "${PASS[@]}"
fi
