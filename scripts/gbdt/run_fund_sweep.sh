#!/usr/bin/env bash
# F18 fundamentals horizon×target sweep: 17 sp500 cells × {fbase, ffund}.
# base arms first (warm the shared 'all' feature matrix — features don't depend
# on the target, so all cells reuse it), then the fund arms (warm
# 'all_fundamentals'). ~2 matrix builds + 34 cheap fits. Memo _274.
set -u
cd /mnt/Workspace/Workspace/agent-driven-investment
SNAP="2026-07-02"
LOG=/tmp/fund_sweep
mkdir -p "$LOG"
CELLS=(
  sp500_up_10pct_5d_dd5pct sp500_up_10pct_10d_dd5pct
  sp500_up_10pct_25d_dd5pct sp500_up_10pct_50d_dd5pct
  sp500_up_20pct_5d_dd10pct sp500_up_20pct_10d_dd10pct
  sp500_up_20pct_25d_dd10pct sp500_up_20pct_50d_dd10pct
  sp500_up_20pct_100d_dd10pct
  sp500_up_40pct_25d_dd20pct sp500_up_40pct_50d_dd20pct
  sp500_up_40pct_100d_dd20pct sp500_up_40pct_200d_dd20pct
  sp500_up_50pct_25d_dd25pct sp500_up_50pct_50d_dd25pct
  sp500_up_50pct_100d_dd25pct sp500_up_50pct_200d_dd25pct
)
# Per-cell (base then fund): the first cell builds BOTH the 'all' and
# 'all_fundamentals' matrices, so any base- or fund-path (panel load / 13-col
# F18) error surfaces in the first ~30 min. Every later cell reuses both matrices.
for cell in "${CELLS[@]}"; do
  for arm in fbase ffund; do
    spec="${cell}_${arm}"
    t0=$(date +%s)
    echo "[SWEEP] START $spec $(date -u +%FT%TZ)"
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 3000 uv run python -m gbdt experiment \
      "configs/gbdt/experiments/$spec.yaml" --snapshot-end "$SNAP" --overwrite \
      > "$LOG/$spec.log" 2>&1
    echo "[SWEEP] DONE  $spec rc=$? elapsed=$(( $(date +%s) - t0 ))s"
  done
done
echo "[SWEEP] ALL COMPLETE $(date -u +%FT%TZ)"
