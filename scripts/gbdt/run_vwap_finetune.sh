#!/usr/bin/env bash
# V1.8 VWAP/fund finetune driver (_282): auto-loop FS+HP on the strong-absolute
# nasdaq100 cells (R-p@3>0.5 & AUC>0.5). PARALLEL — unlike the sweeps, these are
# cache_only reads against WARM universe caches writing to independent experiment
# dirs, so there is no single-writer/cache-build contention. Runs MAXJOBS wide,
# thread-capped (OMP_NUM_THREADS) so MAXJOBS×threads = cores (no oversubscription).
# Resumable (skip-if-done). Specs: the *_vwaptune / *_fundtune / *_fvwaptune
# finetune specs ONLY (NOT the broad *tune.yaml glob, which would sweep up
# pre-existing specs like w2ffundtune — the deployed daily-predictions candidate).
#
# Usage: scripts/gbdt/run_vwap_finetune.sh [SNAPSHOT_END] [MAXJOBS] [THREADS]
set -u
cd /mnt/Workspace/Workspace/agent-driven-investment
SNAP="${1:-2026-07-06}"
MAXJOBS="${2:-4}"
export OMP_NUM_THREADS="${3:-2}"
LOG=/tmp/vwap_finetune
mkdir -p "$LOG"

FREE_G=$(df --output=avail "$(pwd)" | tail -1 | awk '{print int($1/1024/1024)}')
[ "$FREE_G" -lt 15 ] && { echo "[FT] ABORT: ${FREE_G}G free (<15G)"; exit 1; }

run() {
  local spec=$1
  [ -f "results/gbdt/experiments/$spec/metrics.json" ] && { echo "[FT] SKIP $spec (done)"; return; }
  local t0=$(date +%s)
  echo "[FT] START $spec $(date -u +%FT%TZ)"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 3600 uv run python -m gbdt experiment \
    "configs/gbdt/experiments/$spec.yaml" --snapshot-end "$SNAP" --overwrite \
    > "$LOG/$spec.log" 2>&1
  local rc=$?
  [ "$rc" -eq 0 ] && rm -f "results/gbdt/experiments/$spec/_feature_matrix_cache.parquet" \
                           "results/gbdt/experiments/$spec/_feature_matrix_cache.key.json"
  echo "[FT] DONE  $spec rc=$rc elapsed=$(( $(date +%s)-t0 ))s"
}

SPECS=$(ls configs/gbdt/experiments/*_vwaptune.yaml \
           configs/gbdt/experiments/*_fundtune.yaml \
           configs/gbdt/experiments/*_fvwaptune.yaml 2>/dev/null | sed 's|.*/||; s|\.yaml$||')
[ -z "$SPECS" ] && { echo "[FT] no finetune specs found"; exit 1; }

echo "[FT] launching $(echo "$SPECS" | wc -l) finetunes, ${MAXJOBS}-wide, ${OMP_NUM_THREADS} threads each $(date -u +%FT%TZ)"
for spec in $SPECS; do
  run "$spec" &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 3; done
done
wait
echo "[FT] ALL COMPLETE $(date -u +%FT%TZ)"
