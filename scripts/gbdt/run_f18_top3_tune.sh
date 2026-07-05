#!/usr/bin/env bash
# Full FS+HP loop tune of the top-3 F18 fundamentals models from _274
# (+20%/100d, +20%/50d, +40%/200d — by R-p@3 among skill-bearing cells).
# Sequential (single-writer contract + CPU-bound xgboost fits). Feature
# matrices are warm from the _274 sweep, so each iteration ~7 min; 8-iter
# cap => worst case ~1h/cell, ~3h total; plateau usually stops earlier.
# Memo _275.
set -u
cd /mnt/Workspace/Workspace/agent-driven-investment
SNAP="2026-07-02"
LOG=/tmp/f18_top3_tune
mkdir -p "$LOG"
CELLS=(
  sp500_up_20pct_100d_dd10pct
  sp500_up_20pct_50d_dd10pct
  sp500_up_40pct_200d_dd20pct
)
for cell in "${CELLS[@]}"; do
  spec="${cell}_ffundtune"
  t0=$(date +%s)
  echo "[TUNE] START $spec $(date -u +%FT%TZ)"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 7200 uv run python -m gbdt experiment \
    "configs/gbdt/experiments/$spec.yaml" --snapshot-end "$SNAP" --overwrite \
    > "$LOG/$spec.log" 2>&1
  echo "[TUNE] DONE  $spec rc=$? elapsed=$(( $(date +%s) - t0 ))s"
done
echo "[TUNE] ALL COMPLETE $(date -u +%FT%TZ)"
