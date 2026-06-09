#!/usr/bin/env bash
# sp500 sweep runner (task #193 / post-#183 shared feature cache + #190 source-hash cache key).
#
# Iterates configs/gbdt/experiments/sp500_up_*pct_*d_dd*pct.yaml alphabetically
# (EXCLUDING sp500_smoketest.yaml — only the 17 canonical sweep specs), skipping
# any cell whose artifact dir already exists and is non-empty. Each cell runs
# via `uv run python -m gbdt.experiment <spec.yaml>`. Failures are logged and
# the sweep continues (one bad cell does not halt the rest).
#
# Per-cell timing + outcome is appended to logs/sp500_sweep.log as:
#   [RUN ] <cell> <start-iso-ts>
#   [DONE] <cell> exit=<code> elapsed=<seconds>s
#   [FAIL] <cell> exit=<code> elapsed=<seconds>s
#
# Cache discipline: cells must run sequentially (single-writer-per-data_root
# SQLite contract + universe_feature_cache write race). Do NOT parallelise.
#
# Mirror of scripts/gbdt/run_nasdaq100_sweep.sh (#192) and
# scripts/gbdt/run_russell1000_sweep.sh (#188); same skip-non-empty semantics,
# same log format. Glob pattern is the canonical sweep one — the smoketest is
# filtered out explicitly.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/sp500_sweep.log"
mkdir -p "$LOG_DIR"

SWEEP_START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SWEEP_START_EPOCH=$(date +%s)

{
  echo "[SWEEP] start $SWEEP_START_TS"
  echo "[SWEEP] python -m gbdt.experiment via uv run"
} >> "$LOG_FILE"

CELLS_TOTAL=0
CELLS_RUN=0
CELLS_DONE=0
CELLS_FAIL=0
CELLS_SKIP=0

shopt -s nullglob
# Canonical sweep glob: sp500_up_<thr>pct_<H>d_dd<dd>pct.yaml only.
# sp500_smoketest.yaml is a non-sweep companion and must NOT be included.
for spec in "$REPO_ROOT"/configs/gbdt/experiments/sp500_up_*pct_*d_dd*pct.yaml; do
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

  cell_log="$LOG_DIR/sp500_${cell}.cell.log"
  # shellcheck disable=SC2024
  uv run python -m gbdt.experiment "$spec" > "$cell_log" 2>&1
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
