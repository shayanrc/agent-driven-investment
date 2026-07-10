#!/usr/bin/env bash
# Universe-parameterized F18 fundamentals sweep: <universe> cells × {fbase, ffund}.
#
# The universe-general sibling of run_fund_sweep.sh (which hardcodes sp500). Runs
# the matched base-vs-+fund A/B (all vs all_fundamentals, default HP, single fit
# — the _272/_273 protocol) across every generated <universe>_up_*_{fbase,ffund}
# spec. base arms run FIRST so the shared 'all' feature matrix is built once and
# reused by every later base cell (features are target-independent); then the
# fund arms warm 'all_fundamentals'. ~2 matrix builds + N cheap fits.
#
# For NSE universes (nifty*) the runner auto-routes F18 to the INR in_fundamentals
# valuation panel (results/valuation/data/valuation_panel_nse.parquet — build it
# first via `build_valuation_panel --domain nse`). Repo-root is derived from the
# script location, so this runs correctly from a worktree.
#
# Sequential only (single-writer SQLite + universe feature-cache write race).
# Resumable: skips any cell whose predictions/test.csv already exists.
#
# Usage: scripts/gbdt/run_fund_sweep_uni.sh <universe> [snapshot-end]
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

UNIVERSE="${1:?usage: run_fund_sweep_uni.sh <universe> [snapshot-end]}"
SNAP="${2:-$(date -u +%F)}"
LOG="/tmp/fund_sweep_${UNIVERSE}"
mkdir -p "$LOG"

# Disk pre-flight (long run).
FREE_G=$(df --output=avail "$REPO_ROOT" | tail -1 | awk '{print int($1/1024/1024)}')
if [ "$FREE_G" -lt 10 ]; then
  echo "[FUND:$UNIVERSE] ABORT: only ${FREE_G}G free (<10G)"; exit 1
fi

# Distinct cells = the fbase specs' stems (drop the _fbase suffix).
shopt -s nullglob
CELLS=()
for f in configs/gbdt/experiments/"${UNIVERSE}"_up_*pct_*d_dd*pct_fbase.yaml; do
  b="$(basename "$f" .yaml)"
  CELLS+=("${b%_fbase}")
done
if [ "${#CELLS[@]}" -eq 0 ]; then
  echo "[FUND:$UNIVERSE] no *_fbase specs — generate them first"; exit 1
fi
echo "[FUND:$UNIVERSE] ${#CELLS[@]} cells × {fbase,ffund}, snapshot=$SNAP"

run() {
  local spec="$1"
  if [ -f "results/gbdt/experiments/$spec/predictions/test.csv" ]; then
    echo "[FUND:$UNIVERSE] SKIP  $spec (test.csv exists)"; return
  fi
  local t0; t0=$(date +%s)
  echo "[FUND:$UNIVERSE] START $spec $(date -u +%FT%TZ)"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 3000 uv run python -m gbdt experiment \
    "configs/gbdt/experiments/$spec.yaml" --snapshot-end "$SNAP" --overwrite \
    > "$LOG/$spec.log" 2>&1
  local rc=$?
  echo "[FUND:$UNIVERSE] DONE  $spec rc=$rc elapsed=$(( $(date +%s) - t0 ))s free=$(df --output=avail "$REPO_ROOT" | tail -1 | awk '{print int($1/1024/1024)"G"}')"
}

# base arms first (warm 'all'), then fund arms (warm 'all_fundamentals').
for arm in fbase ffund; do
  echo "[FUND:$UNIVERSE] === arm: $arm ==="
  for cell in "${CELLS[@]}"; do
    run "${cell}_${arm}"
  done
done
echo "[FUND:$UNIVERSE] ALL COMPLETE $(date -u +%FT%TZ)"
