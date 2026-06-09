#!/usr/bin/env bash
# nifty_next_50 sweep runner — post-path-leak-cleanup validation sweep.
#
# Iterates configs/gbdt/experiments/nifty_next_50_up_*pct_*d_dd*pct.yaml
# alphabetically, skipping any cell whose artifact dir already exists and is
# non-empty. Each cell runs via `uv run python -m gbdt experiment <spec.yaml>`.
# Failures are logged and the sweep continues.
#
# Per-cell timing + outcome is appended to logs/nifty_next_50_sweep.log as:
#   [RUN ] <cell> <start-iso-ts>
#   [DONE] <cell> exit=<code> elapsed=<seconds>s
#   [FAIL] <cell> exit=<code> elapsed=<seconds>s
#
# Mirror of scripts/gbdt/run_nasdaq100_sweep.sh, with:
#   - the sweep-log `[SWEEP] start` line no longer echoes $REPO_ROOT (post-#150
#     path-leak fix);
#   - the runner invocation uses the documented `python -m gbdt experiment`
#     (subcommand form) per CLAUDE.md.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/nifty_next_50_sweep.log"
mkdir -p "$LOG_DIR"

SWEEP_START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SWEEP_START_EPOCH=$(date +%s)

{
  echo "[SWEEP] start $SWEEP_START_TS"
  echo "[SWEEP] python -m gbdt experiment via uv run"
} >> "$LOG_FILE"

CELLS_TOTAL=0
CELLS_RUN=0
CELLS_DONE=0
CELLS_FAIL=0
CELLS_SKIP=0

shopt -s nullglob
for spec in "$REPO_ROOT"/configs/gbdt/experiments/nifty_next_50_up_*pct_*d_dd*pct.yaml; do
  cell="$(basename "$spec" .yaml)"

  CELLS_TOTAL=$((CELLS_TOTAL + 1))
  out_dir="$REPO_ROOT/results/gbdt/experiments/$cell"

  if [[ -d "$out_dir" ]] && [[ -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
    echo "[SKIP] $cell exists (non-empty)" >> "$LOG_FILE"
    CELLS_SKIP=$((CELLS_SKIP + 1))
    continue
  fi

  start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_epoch=$(date +%s)
  echo "[RUN ] $cell $start_iso" >> "$LOG_FILE"
  CELLS_RUN=$((CELLS_RUN + 1))

  cell_log="$LOG_DIR/nifty_next_50_${cell}.cell.log"
  # shellcheck disable=SC2024
  uv run python -m gbdt experiment "$spec" > "$cell_log" 2>&1
  ec=$?
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - start_epoch))

  if [[ $ec -eq 0 ]]; then
    CELLS_DONE=$((CELLS_DONE + 1))
    echo "[DONE] $cell exit=$ec elapsed=${elapsed}s" >> "$LOG_FILE"
  else
    CELLS_FAIL=$((CELLS_FAIL + 1))
    echo "[FAIL] $cell exit=$ec elapsed=${elapsed}s log=$cell_log" >> "$LOG_FILE"
  fi
done
shopt -u nullglob

SWEEP_END_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SWEEP_END_EPOCH=$(date +%s)
SWEEP_ELAPSED=$((SWEEP_END_EPOCH - SWEEP_START_EPOCH))

{
  echo "[SWEEP] end $SWEEP_END_TS elapsed=${SWEEP_ELAPSED}s"
  echo "[SWEEP] totals total=$CELLS_TOTAL run=$CELLS_RUN done=$CELLS_DONE fail=$CELLS_FAIL skip=$CELLS_SKIP"
} >> "$LOG_FILE"

# Exit 0 even if some cells failed — the orchestrator inspects the log.
exit 0
