#!/usr/bin/env bash
# V1.9 calendar2 (F21) matched A/B driver: 6 single fits — 3 nasdaq100 cells ×
# {base=all, cal2=all_calendar2}, xgboost, default HP, max_iterations 1,
# date_aligned train_start 2019-01-01. Within a cell the arms differ ONLY in
# features.candidates, so cal2-minus-base is a clean F21 read.
#
# Order: all three base arms first (one cold `all` universe-cache build, two
# warm), then all three cal2 arms (one cold `all_calendar2` build, two warm).
# Sequential (single-writer SQLite contract). Snapshot pinned so the universe
# feature-cache key is stable across arms. Resumable: skips arms whose
# metrics.json already exists.
#
# MUST run from this worktree (its src/gbdt carries F21). Usage:
#   scripts/gbdt/run_cal2_ab.sh [SNAPSHOT_END]   (default: 2026-07-06)
set -u
cd /mnt/Workspace/Workspace/agent-driven-investment/.claude/worktrees/agent-a7fe9269d958fdc34
SNAP="${1:-2026-07-06}"
LOG=/tmp/cal2_ab
mkdir -p "$LOG"

FREE_G=$(df --output=avail "$(pwd)" | tail -1 | awk '{print int($1/1024/1024)}')
if [ "$FREE_G" -lt 15 ]; then
  echo "[CAL2] ABORT: only ${FREE_G}G free (<15G) — FS-wedge guard." >&2
  exit 1
fi

run() {
  local spec=$1
  if [ -f "results/gbdt/experiments/$spec/metrics.json" ]; then
    echo "[CAL2] SKIP  $spec (already done)"
    return
  fi
  local t0=$(date +%s)
  echo "[CAL2] START $spec $(date -u +%FT%TZ)"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 3600 uv run python -m gbdt experiment \
    "configs/gbdt/experiments/cal2_ab/$spec.yaml" --snapshot-end "$SNAP" --overwrite \
    >> "$LOG/$spec.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    rm -f "results/gbdt/experiments/$spec/_feature_matrix_cache.parquet" \
          "results/gbdt/experiments/$spec/_feature_matrix_cache.key.json"
  fi
  echo "[CAL2] DONE  $spec rc=$rc elapsed=$(( $(date +%s) - t0 ))s free=$(df --output=avail $(pwd) | tail -1 | awk '{print int($1/1024/1024)"G"}')"
}

for cell in nasdaq100_up_50pct_25d_dd25pct nasdaq100_up_20pct_50d_dd10pct nasdaq100_up_40pct_200d_dd20pct; do
  run "${cell}_cal2ab_base"
done
for cell in nasdaq100_up_50pct_25d_dd25pct nasdaq100_up_20pct_50d_dd10pct nasdaq100_up_40pct_200d_dd20pct; do
  run "${cell}_cal2ab_cal2"
done
echo "[CAL2] ALL DONE $(date -u +%FT%TZ)"
