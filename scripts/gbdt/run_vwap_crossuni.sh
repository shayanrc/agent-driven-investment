#!/usr/bin/env bash
# V1.8 VWAP (F20) cross-universe orchestrator: after the sp500 sweep finishes,
# run the nasdaq100 then russell1000 4-arm xgboost sweeps (base/vwap/fund/
# fundvwap). Sequential across universes (single-writer SQLite contract). Waits
# on the sp500 driver's completion marker so it never contends with it.
#
# Usage: scripts/gbdt/run_vwap_crossuni.sh [SNAPSHOT_END]   (default 2026-07-06)
set -u
cd /mnt/Workspace/Workspace/agent-driven-investment
SNAP="${1:-2026-07-06}"

echo "[CROSSUNI] waiting for sp500 sweep to finish $(date -u +%FT%TZ)"
# Exit the wait when sp500 writes its completion marker OR its driver vanishes.
until grep -q "ALL COMPLETE" /tmp/vwap_sweep_xgb.log 2>/dev/null \
      || ! pgrep -f "run_vwap_sweep\.sh 2026" >/dev/null 2>&1; do
  sleep 180
done
echo "[CROSSUNI] sp500 done — starting cross-universe sweeps $(date -u +%FT%TZ)"

for U in nasdaq100 russell1000; do
  echo "[CROSSUNI] === $U START $(date -u +%FT%TZ) ==="
  bash scripts/gbdt/run_vwap_sweep_uni.sh "$U" "$SNAP"
  echo "[CROSSUNI] === $U DONE $(date -u +%FT%TZ) ==="
done
echo "[CROSSUNI] ALL COMPLETE $(date -u +%FT%TZ)"
