#!/usr/bin/env bash
# V1.10 nifty500 canonical scan (task #55): 20 cells x {fbase, ffund} = 40 single fits.
# fbase arm first (panel-independent, warms the all_calendar2 matrix), then ffund
# (needs the backfilled NSE valuation panel; warms all_fundamentals_calendar2).
# snapshot-end at the panel max so long-horizon (200d) test labels are computable.
set -u
cd /mnt/Workspace/Workspace/wt-nse-valuation
SNAP="2026-07-06"
LOG=/tmp/nifty_canon_scan
mkdir -p "$LOG"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

CELLS=(
  nifty500_up_10pct_10d_dd5pct  nifty500_up_10pct_25d_dd5pct  nifty500_up_10pct_50d_dd5pct
  nifty500_up_10pct_100d_dd5pct nifty500_up_10pct_200d_dd5pct
  nifty500_up_20pct_10d_dd10pct nifty500_up_20pct_25d_dd10pct nifty500_up_20pct_50d_dd10pct
  nifty500_up_20pct_100d_dd10pct nifty500_up_20pct_200d_dd10pct
  nifty500_up_30pct_10d_dd15pct nifty500_up_30pct_25d_dd15pct nifty500_up_30pct_50d_dd15pct
  nifty500_up_30pct_100d_dd15pct nifty500_up_30pct_200d_dd15pct
  nifty500_up_50pct_10d_dd25pct nifty500_up_50pct_25d_dd25pct nifty500_up_50pct_50d_dd25pct
  nifty500_up_50pct_100d_dd25pct nifty500_up_50pct_200d_dd25pct
)

run_arm () {
  local arm="$1"
  for cell in "${CELLS[@]}"; do
    local spec="${cell}_${arm}_canon"
    local t0=$(date +%s)
    echo "[SCAN] START $spec $(date -u +%FT%TZ)"
    timeout 3000 uv run python -m gbdt experiment \
      "configs/gbdt/experiments/$spec.yaml" --snapshot-end "$SNAP" --overwrite \
      > "$LOG/$spec.log" 2>&1
    echo "[SCAN] DONE  $spec rc=$? elapsed=$(( $(date +%s) - t0 ))s"
  done
}

echo "[SCAN] === fbase arm (20 cells) $(date -u +%FT%TZ) ==="
run_arm fbase

# Gate: ffund needs the pre-2019-backfilled panel. Wait up to 30 min for the concat.
PANEL="results/valuation/data/valuation_panel_nse.parquet"
echo "[SCAN] gating ffund on backfilled panel (train_start<2019 coverage) ..."
for i in $(seq 1 90); do
  minyr=$(uv run python -c "import pandas as pd;print(pd.read_parquet('$PANEL',columns=['date'])['date'].min().year)" 2>/dev/null)
  if [ "${minyr:-9999}" -le 2016 ] 2>/dev/null; then
    echo "[SCAN] panel back-extended to $minyr — proceeding with ffund."
    break
  fi
  sleep 20
done

echo "[SCAN] === ffund arm (20 cells) $(date -u +%FT%TZ) ==="
run_arm ffund
echo "[SCAN] ALL COMPLETE $(date -u +%FT%TZ)"
