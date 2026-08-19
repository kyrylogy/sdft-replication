#!/usr/bin/env bash
# Regression tests for run_tier.sh's resumability/skip logic -- no GPU, <1s.
# Sources the real run_tier.sh (safe: its GPU-launching code is guarded behind
# `[[ "${BASH_SOURCE[0]}" == "${0}" ]]`, see the bottom of that file) so these
# tests exercise the exact functions the tier driver runs on the cluster,
# not a copy that can drift out of sync.
#
# Run: bash tests/test_run_tier.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PASS=0 FAIL=0
ok()   { PASS=$((PASS+1)); echo "  ok   - $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL - $1"; }

# --- fg_done(): the fix for the "stale directory looks done forever" trap ---
# (chain()/anchors() used to test `[ -d eval/forgetting ]`, which is true the
# moment the dir is mkdir'd -- well before the ~75min lm_eval subprocess
# finishes. A crash in between left a directory that silently blocked every
# future retry. fg_done() requires the actual results file instead.)
test_fg_done() {
  echo "[test_fg_done]"
  GPU=x SCALE=7b SEEDS=42 source ./run_tier.sh

  local d; d=$(mktemp -d)

  [ -d "$d/empty" ] || mkdir "$d/empty"
  if fg_done "$d/empty"; then bad "empty dir should not be 'done'"; else ok "empty dir is not 'done'"; fi

  mkdir -p "$d/stale"; echo "x" > "$d/stale/resolved_config.yaml"
  if fg_done "$d/stale"; then bad "stale dir (stamp only, no results) should not be 'done'"
  else ok "stale dir (resolved_config.yaml only) is not 'done' -- this is the exact bug"; fi

  mkdir -p "$d/real/modelhash"; echo "{}" > "$d/real/modelhash/results_2026-01-01T00-00-00.json"
  if fg_done "$d/real"; then ok "dir with a real results*.json is 'done'"
  else bad "dir with a real results*.json should be 'done'"; fi

  if fg_done "$d/does_not_exist"; then bad "nonexistent dir should not be 'done'"
  else ok "nonexistent dir is not 'done'"; fi

  rm -rf "$d"
}

# --- r(): must clear SEEDS before calling run.sh, or run.sh's own SEEDS loop ---
# (run.sh independently loops over $SEEDS for `train` mode. run_tier.sh already
# seeds each call itself via --set train.seed=N; if SEEDS leaks through, every
# training step silently re-runs once per seed in $SEEDS -- ~3x GPU cost.)
#
# run_tier.sh unconditionally `cd`s to its own directory on source, so the
# stub run.sh has to live NEXT TO a copy of run_tier.sh, not just in a temp
# dir we cd into first -- otherwise the cd lands back in the real repo and
# the stub is never reached.
test_r_clears_seeds() {
  echo "[test_r_clears_seeds]"
  local d; d=$(mktemp -d)
  cp run_tier.sh "$d/run_tier.sh"
  cat > "$d/run.sh" <<'STUB'
#!/usr/bin/env bash
echo "SEEDS='${SEEDS:-<unset>}'" >> "$STUB_LOG"
exit 0
STUB
  chmod +x "$d/run.sh"
  local log="$d/calls.log"; : > "$log"

  ( export STUB_LOG="$log" SEEDS="42 1234 2024" GPU=x SCALE=7b
    source "$d/run_tier.sh"
    r train fake_config.yaml )

  local n; n=$(wc -l < "$log")
  if [ "$n" -ne 1 ]; then
    bad "r() should invoke run.sh exactly once regardless of outer SEEDS (got $n calls)"
  else
    ok "r() invokes run.sh exactly once"
  fi
  if grep -q "SEEDS='<unset>'" "$log"; then
    ok "r() clears SEEDS before calling run.sh (outer SEEDS='42 1234 2024' did not leak through)"
  else
    bad "r() leaked SEEDS into run.sh: $(cat "$log")"
  fi

  rm -rf "$d"
}

# --- sourcing must be side-effect-free (no logs/ dir, no stdout redirect) ---
# The tier driver's real dispatch (mkdir logs/, redirect stdout, run_all) is
# guarded behind `[[ "${BASH_SOURCE[0]}" == "${0}" ]]` so tests can source it
# safely -- confirm sourcing never creates a new tier log.
test_sourcing_has_no_side_effects() {
  echo "[test_sourcing_has_no_side_effects]"
  local before after
  before=$(find logs -name 'tier_*.log' 2>/dev/null | wc -l)
  ( GPU=x SCALE=7b SEEDS=42 source ./run_tier.sh )
  after=$(find logs -name 'tier_*.log' 2>/dev/null | wc -l)
  if [ "$before" -eq "$after" ]; then
    ok "sourcing run_tier.sh does not create a new tier log"
  else
    bad "sourcing run_tier.sh created a new tier log (before=$before after=$after) -- the source guard regressed"
  fi
}

test_fg_done
test_r_clears_seeds
test_sourcing_has_no_side_effects

echo
echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
