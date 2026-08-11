#!/usr/bin/env bash
# run_tier.sh - drive a whole scale's matrix on ONE GPU, in gate-safe order, RESUMABLE.
# Survives disconnection (run under tmux or nohup). Skips already-finished steps, so just
# re-run it after a crash/kill and it continues where it stopped.
#
#   GPU=3 SCALE=7b SEEDS="42 1234 2024" ./run_tier.sh preflight   # dry-run every config, no GPU
#   GPU=3 SCALE=7b SEEDS="42 1234 2024" ./run_tier.sh run         # the whole scale
#   GPU=3 SCALE=14b SEEDS="42" ./run_tier.sh run                  # 14B, one seed
#
# Persist it (see below): tmux new -s tier ; then the run line ; detach with Ctrl-b d.
# Or:  nohup env GPU=3 SCALE=7b SEEDS="42 1234 2024" ./run_tier.sh run > /dev/null 2>&1 &
# (the script tees its own timestamped log to logs/ regardless).
#
# Before running: export HF_HOME=$HOME/hf_cache ; export WANDB_API_KEY=<key>  (see notes)

set -uo pipefail                     # NOT -e: one failed arm must not abort the whole tier
cd "$(dirname "$0")"

GPU="${GPU:-3}"
SCALE="${SCALE:-7b}"
SEEDS="${SEEDS:-42}"
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
MODE="${1:-run}"
E=configs/experiments
FG=(--mode forgetting --set eval.forgetting.enabled=true)

mkdir -p logs
LOG="logs/tier_${SCALE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1        # everything below is logged AND shown
echo "[tier] SCALE=$SCALE GPU=$GPU SEEDS='$SEEDS' log=$LOG"

# Headline arms run at every seed with a stage-2 continuation; ablations (7B only) are
# single-seed, single-stage.
HEADLINE=(sft_lora sdft_ema_lora)
ABLATION=(); [ "$SCALE" = "7b" ] && ABLATION=(sdft_frozen_lora online_sft_lora)

FAILED=()
r() { echo "  + GPU=$GPU ./run.sh $*"; if ! GPU="$GPU" ./run.sh "$@"; then FAILED+=("$*"); echo "  [FAIL] $*"; fi; }
seedset() { echo "--set train.seed=$1 --set experiment.tag=seed$1"; }

gpu_guard() {
  command -v nvidia-smi >/dev/null || return 0
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 999999)
  if [ "$free" -lt "$MIN_FREE_MIB" ]; then
    echo "[tier] GPU $GPU only ${free} MiB free (< ${MIN_FREE_MIB}). Another job is resident; pick a freer GPU or lower vllm mem." >&2
    exit 1
  fi
  echo "[tier] GPU $GPU free=${free} MiB — ok"
}

anchors() {   # once per scale; live in the untagged sft dir. collect_results keys them by scale.
  local base="$E/sft_lora_${SCALE}_tooluse_s1.yaml" ad="runs/sft_lora_${SCALE}_tooluse_s1"
  echo "[tier] === anchors ($SCALE) ==="
  [ -d "$ad/eval/base_anchor" ]     || r eval "$base" --base
  [ -d "$ad/eval/ceiling" ]         || r eval "$base" --base --set eval.teacher_ceiling=true --set 'eval.sets=[holdout]'
  [ -d "$ad/eval/forgetting_base" ] || r eval "$base" --base "${FG[@]}"
}

chain() {   # chain <arm> <seed>
  local arm=$1 seed=$2; local tag; tag=$(seedset "$seed")
  local s1="$E/${arm}_${SCALE}_tooluse_s1.yaml" s2="$E/${arm}_${SCALE}_science_s2.yaml"
  local s1d="runs/${arm}_${SCALE}_tooluse_s1_seed${seed}" s2d="runs/${arm}_${SCALE}_science_s2_seed${seed}"
  echo "[tier] === $arm seed=$seed ==="
  # --- stage 1: train, accuracy eval, forgetting ---
  [ -f "$s1d/lora_adapter/adapter_config.json" ] || r train "$s1" $tag
  [ -d "$s1d/eval/final" ]        || r eval "$s1" $tag
  [ -d "$s1d/eval/forgetting" ]   || r eval "$s1" $tag "${FG[@]}"
  # --- stage 2 (headline arms only; ablations have no _science_s2 config) ---
  [ -f "$s2" ] || return 0
  [ -f "$s2d/lora_adapter/adapter_config.json" ] || r train "$s2" $tag --set data.init_adapter="$s1d/lora_adapter"
  [ -d "$s2d/eval/final/science_eval_data" ] || r eval "$s2" $tag
  [ -d "$s2d/eval/final/tooluse_holdout" ]   || r eval "$s2" $tag --set data.dataset=tooluse --set 'eval.sets=[holdout]'
}

analyze() { echo "[tier] === collect + figures ==="; .venv/bin/python collect_results.py; .venv/bin/python make_figures.py; }

preflight() {
  for arm in "${HEADLINE[@]}" "${ABLATION[@]}"; do
    echo "[preflight] $arm"; ./run.sh train "$E/${arm}_${SCALE}_tooluse_s1.yaml" --dry_run || true
  done
}

run_all() {
  gpu_guard
  anchors
  # first seed = shakeout: headline + ablations, then a collect so you can sanity-check early
  local first; first=$(echo "$SEEDS" | awk '{print $1}')
  for arm in "${HEADLINE[@]}"; do chain "$arm" "$first"; done
  for arm in "${ABLATION[@]}"; do chain "$arm" "$first"; done
  analyze
  # remaining seeds: headline only
  for s in $SEEDS; do
    [ "$s" = "$first" ] && continue
    for arm in "${HEADLINE[@]}"; do chain "$arm" "$s"; done
  done
  analyze
  echo "[tier] DONE. failed steps: ${#FAILED[@]}"
  for f in "${FAILED[@]}"; do echo "   - $f"; done
}

case "$MODE" in
  preflight) preflight ;;
  run)       run_all ;;
  *) echo "usage: GPU=<id> SCALE=<3b|7b|14b> SEEDS='42 ...' $0 {preflight|run}" >&2; exit 2 ;;
esac
