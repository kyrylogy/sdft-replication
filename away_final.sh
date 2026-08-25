#!/usr/bin/env bash
# away_final.sh — the LAST away-block: everything still standing between here and the
# freeze, as one resumable tmux command. Re-run after any crash; finished steps skip.
#
#   tmux new -s final
#   GPU=2 bash away_final.sh          # Ctrl-b d to detach
#   tmux attach -t final              # peek later
#
# Phases (in payoff order — free reads first, 11 GPU-h of batteries last-but-one):
#   0. provenance   C1 battery audit + C2 full dirty-tree diff + acq50 gate lines   (free)
#   1. stats        collect_results + per-sample inference (McNemar, seed-blocked    (free)
#                   permutation, attractor check) on everything already on disk
#   2. batteries    9 stage-2 lm-eval batteries: sft/sdft_ema x3 seeds + acq50 x3   (~11 GPU-h)
#   3. joint        joint-training ceiling: SFT on tooluse+science, seed 42          (~1 GPU-h)
#   4. wrap         re-collect + re-stats + figures + status dump + freeze checklist (free)
#
# Env knobs: GPU (default 3), MIN_FREE_MIB (default 40000), SKIP_BATTERIES=1, SKIP_JOINT=1.

set -uo pipefail                       # NOT -e: one failed step must not kill the block
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

GPU="${GPU:-3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
E=configs/experiments
PY=.venv/bin/python
FG=(--mode forgetting --set eval.forgetting.enabled=true)
DIRTY_REF="${DIRTY_REF:-bc810cbe}"     # the pre-dirty-tree commit C2 diffs against

mkdir -p logs analysis/provenance
LOG="logs/final_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[final] GPU=$GPU log=$LOG"

FAILED=()
r() { echo "  + GPU=$GPU ./run.sh $*"; if ! GPU="$GPU" SEEDS='' ./run.sh "$@"; then FAILED+=("$*"); echo "  [FAIL] $*"; fi; }
fg_done() { [ -n "$(find "$1" -name 'results*.json' 2>/dev/null | head -1)" ]; }

gpu_guard() {
  command -v nvidia-smi >/dev/null || return 0
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 999999)
  if [ "$free" -lt "$MIN_FREE_MIB" ]; then
    echo "[final] GPU $GPU only ${free} MiB free (< ${MIN_FREE_MIB}). Pick a freer card: GPU=<id>." >&2
    exit 1
  fi
  echo "[final] GPU $GPU free=${free} MiB — ok"
}

echo "########## 0. PROVENANCE (free) ##########"
# C2: the FULL dirty-tree diff (the earlier block only captured --stat). eval_runner.py
# is the file that matters — if the hunk touches scoring, reported numbers inherit it.
if git rev-parse --verify -q "$DIRTY_REF" >/dev/null; then
  git diff "$DIRTY_REF" -- eval_runner.py run_tier.sh > analysis/provenance/dirty_diff.patch
  echo "[C2] full diff vs $DIRTY_REF -> analysis/provenance/dirty_diff.patch ($(wc -l < analysis/provenance/dirty_diff.patch) lines):"
  cat analysis/provenance/dirty_diff.patch
else
  echo "[C2] ref $DIRTY_REF not found — set DIRTY_REF=<sha> and re-run"
fi
# C1: settings + metric-key consistency across every battery (base anchor included).
"$PY" battery_audit.py | tee analysis/provenance/battery_audit.txt
C1_STATUS=${PIPESTATUS[0]}
echo "[C1] battery audit exit=$C1_STATUS (0=clean)"
# acq50 stage-2 gate provenance (expected_accuracy was null -> runs trained ungated; record it).
{ grep -hA6 "retention_gate:" runs/sft_lora_7b_science_s2_acq50_seed*/resolved_config.yaml
  grep -h "\[gate\]" logs/*science_s2*train*.log 2>/dev/null | tail -20
} > analysis/provenance/acq50_gate.txt 2>/dev/null
echo "[gate] -> analysis/provenance/acq50_gate.txt"

echo "########## 1. STATS ON WHAT EXISTS (free) ##########"
"$PY" collect_results.py
"$PY" stats_final.py

if [ "${SKIP_BATTERIES:-0}" != "1" ]; then
  echo "########## 2. STAGE-2 BATTERIES (9 x ~75 min) ##########"
  gpu_guard
  # Headline arms first (the thesis claim), acq50 control after.
  for arm in sft_lora sdft_ema_lora; do
    for s in 42 1234 2024; do
      d="runs/${arm}_7b_science_s2_seed${s}"
      fg_done "$d/eval/forgetting" && { echo "[skip] $d battery done"; continue; }
      r eval "$E/${arm}_7b_science_s2.yaml" --set train.seed=$s --set experiment.tag=seed$s "${FG[@]}"
    done
  done
  for s in 42 1234 2024; do
    d="runs/sft_lora_7b_science_s2_acq50_seed${s}"
    fg_done "$d/eval/forgetting" && { echo "[skip] $d battery done"; continue; }
    r eval "$E/sft_lora_7b_science_s2.yaml" --set train.seed=$s --set experiment.tag=acq50_seed$s "${FG[@]}"
  done
fi

if [ "${SKIP_JOINT:-0}" != "1" ]; then
  echo "########## 3. JOINT-TRAINING CEILING (seed 42, ~1 GPU-h) ##########"
  gpu_guard
  J="$E/sft_lora_7b_joint_s1.yaml"; JD="runs/sft_lora_7b_joint_s1_seed42"
  TAG=(--set train.seed=42 --set experiment.tag=seed42)
  [ -f "$JD/lora_adapter/adapter_config.json" ] || r train "$J" "${TAG[@]}"
  if [ -f "$JD/lora_adapter/adapter_config.json" ]; then
    [ -d "$JD/eval/final/tooluse_holdout" ]    || r eval "$J" "${TAG[@]}" --set data.dataset=tooluse
    [ -d "$JD/eval/final/science_eval_data" ]  || r eval "$J" "${TAG[@]}" --set data.dataset=science --set 'eval.sets=[eval_data]'
  else
    echo "[joint] training did not produce an adapter — skipping its evals"
  fi
fi

echo "########## 4. WRAP ##########"
"$PY" collect_results.py
"$PY" stats_final.py
"$PY" adapter_distance.py --glob 'runs/*/*' --csv analysis/adapter_distance.csv || true   # picks up the joint adapter too
"$PY" make_figures.py
bash status.sh --wide

echo "########## DONE ##########"
echo "failed steps: ${#FAILED[@]}"; for f in "${FAILED[@]}"; do echo "   - $f"; done
cat <<'EOF'

Next (MANUAL — the freeze):
  1. read analysis/provenance/dirty_diff.patch — if the eval_runner.py hunk touches
     scoring/generation, the two re-run chains are back on the table; if it's plumbing
     (adapter override / step-eval support), write the one-sentence disclosure and move on.
  2. read analysis/provenance/battery_audit.txt — headline the 14pp HumanEval gap ONLY if clean.
  3. git add analysis/ && git add -f analysis/provenance/ ; git add configs data-stamps as needed
     git commit -m "freeze: final tables, provenance, stage-2 batteries, joint ceiling"
     git tag thesis-freeze-$(date +%Y%m%d)
  4. After the tag: no new analyses that could change the conclusion. Declared tables only.
EOF
