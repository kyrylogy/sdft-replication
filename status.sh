#!/usr/bin/env bash
# status.sh - compact dump of every thesis-relevant number, sized to paste into a chat.
#
#   ./status.sh            # the tables
#   ./status.sh --wide     # + per-seed battery rows and gap_closed
#
# Reads analysis/*.csv (run collect_results.py first). Never touches the GPU.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
A=analysis
WIDE=0; [ "${1:-}" = "--wide" ] && WIDE=1

hdr() { printf '\n=== %s ===\n' "$1"; }

[ -d "$A" ] || { echo "no $A/ — run: .venv/bin/python collect_results.py" >&2; exit 1; }

hdr "COVERAGE (runs_index)"
column -t -s, "$A/runs_index.csv" 2>/dev/null | cut -c1-150

hdr "RETENTION / BWT"
column -t -s, "$A/retention.csv" 2>/dev/null | cut -c1-190

hdr "CONTINUAL (ACC)"
column -t -s, "$A/continual_metrics.csv" 2>/dev/null

hdr "METHOD EFFECT"
column -t -s, "$A/method_effect.csv" 2>/dev/null

hdr "AGGREGATE (seed mean+/-sd)"
column -t -s, "$A/aggregate.csv" 2>/dev/null

hdr "FORGETTING - 6 top-level tasks only (MMLU subtasks suppressed)"
awk -F, 'NR==1 || $4 ~ /^(mmlu|hellaswag|truthfulqa_mc2|winogrande|humaneval|ifeval)$/' \
    "$A/forgetting.csv" 2>/dev/null \
  | awk -F, 'BEGIN{OFS=","}{print $1,$4,$5,$7,$9}' | column -t -s,

if [ -f "$A/adapter_distance.csv" ]; then
  hdr "ADAPTER DISTANCE ||dW||_F (final checkpoints only)"
  awk -F, 'NR==1 || $1 ~ /(lora_adapter|checkpoint-(166|168|246|248))$/' "$A/adapter_distance.csv" \
    | awk -F, 'BEGIN{OFS=","}{print $1,$2}' | column -t -s,
fi

if [ "$WIDE" = "1" ]; then
  hdr "GAP CLOSED"
  column -t -s, "$A/gap_closed.csv" 2>/dev/null
  hdr "SIGNIFICANCE"
  column -t -s, "$A/significance.csv" 2>/dev/null | cut -c1-190
fi

hdr "PAIRING ISSUES"
if [ -f "$A/pairing_issues.csv" ]; then column -t -s, "$A/pairing_issues.csv"; else echo "(none - file absent)"; fi

hdr "GIT"
git log --oneline -1
git status --porcelain | head -10
