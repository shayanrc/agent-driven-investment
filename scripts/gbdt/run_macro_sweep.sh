#!/usr/bin/env bash
# Macro-lattice sweep runner (memo _263).
#
# Runs the matched base-vs-+macro A/B across a universe's canonical cells. The
# per-cell spec pairs (<cell>_swbase.yaml / <cell>_swmacro.yaml) are produced by
# scripts/gbdt/gen_macro_sweep_specs.py; both arms share an identical fixed config
# (xgboost, min_child_weight=10, single fit) so the only difference is the macro
# feature family -> a clean per-cell macro delta (avoids the _260 default-auto
# per-arm HP confound).
#
# IMPORTANT: the entry point is `python -m gbdt experiment <spec> --snapshot-end
# <DATE>` (the CLI subcommand) — NOT `python -m gbdt.experiment <spec>` (the module
# form used by run_<uni>_sweep.sh), which does not accept --snapshot-end and exits 2
# with "expected one positional spec path".
#
# Sequential only (single-writer-per-data_root SQLite + universe feature-cache write
# race — do NOT parallelise). Skips any cell whose test.csv already exists, so it is
# resumable. Per-cell timing + outcome is appended to logs/macro_sweep_run.log.
#
# Usage:
#   scripts/gbdt/run_macro_sweep.sh <universe> [snapshot-end]
#   e.g. scripts/gbdt/run_macro_sweep.sh sp500 2026-06-20
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

UNIVERSE="${1:?usage: run_macro_sweep.sh <universe> [snapshot-end]}"
SNAPSHOT="${2:-$(date -u +%F)}"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/macro_sweep_run.log"
mkdir -p "$LOG_DIR"

echo "[SWEEP] start $(date -u +%FT%TZ) universe=$UNIVERSE snapshot=$SNAPSHOT" >> "$LOG_FILE"

shopt -s nullglob
n=0
for spec in "$REPO_ROOT"/configs/gbdt/experiments/"${UNIVERSE}"_up_*pct_*d_dd*pct_sw*.yaml; do
  cell="$(basename "$spec" .yaml)"
  n=$((n + 1))
  if [[ -f "$REPO_ROOT/results/gbdt/experiments/$cell/predictions/test.csv" ]]; then
    echo "[$n] $cell SKIP (test.csv exists)" >> "$LOG_FILE"
    continue
  fi
  t0=$(date +%s)
  timeout 1200 uv run python -m gbdt experiment "$spec" --snapshot-end "$SNAPSHOT" \
    > "$LOG_DIR/$cell.log" 2>&1
  rc=$?
  el=$(( $(date +%s) - t0 ))
  if [[ -f "$REPO_ROOT/results/gbdt/experiments/$cell/predictions/test.csv" ]]; then
    echo "[$n] $cell exit=$rc ${el}s test_csv=yes" >> "$LOG_FILE"
  else
    echo "[$n] $cell exit=$rc ${el}s test_csv=NO" >> "$LOG_FILE"
  fi
done
echo "[SWEEP] done $(date -u +%FT%TZ)" >> "$LOG_FILE"
