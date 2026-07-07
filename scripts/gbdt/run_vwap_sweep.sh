#!/usr/bin/env bash
# V1.8 VWAP (F20) lattice sweep driver: 136 matched single fits — 17 sp500 cells
# × {xgboost, catboost} × {all, all_vwap, all_fundamentals, all_fundamentals_vwap}.
# Order: within each backend, base → vwap → fund → fundvwap; xgboost group before
# catboost (faster → early signal). Sequential (single-writer SQLite contract).
# Snapshot pinned so the universe feature-cache key is stable across all arms
# (one cold build per universe-view, warm after).
#
# Usage: scripts/gbdt/run_vwap_sweep.sh [SNAPSHOT_END]   (default: today)
set -u
cd /mnt/Workspace/Workspace/agent-driven-investment
SNAP="${1:-$(date -I)}"
LOG=/tmp/vwap_sweep
mkdir -p "$LOG"

# Disk pre-flight (FS-wedge guard): refuse to start under 15G free.
FREE_G=$(df --output=avail "$(pwd)" | tail -1 | awk '{print int($1/1024/1024)}')
if [ "$FREE_G" -lt 15 ]; then
  echo "[VWAP] ABORT: only ${FREE_G}G free (<15G) — FS-wedge guard." >&2
  exit 1
fi

run() {
  local spec=$1
  local t0=$(date +%s)
  echo "[VWAP] START $spec $(date -u +%FT%TZ)"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 3600 uv run python -m gbdt experiment \
    "configs/gbdt/experiments/$spec.yaml" --snapshot-end "$SNAP" --overwrite \
    >> "$LOG/$spec.log" 2>&1
  local rc=$?
  # Drop the regenerable per-experiment matrix cache (~3.8G) on a clean fit so
  # 136 arms don't accumulate; the shared universe cache is untouched.
  if [ "$rc" -eq 0 ]; then
    rm -f "results/gbdt/experiments/$spec/_feature_matrix_cache.parquet" \
          "results/gbdt/experiments/$spec/_feature_matrix_cache.key.json"
  fi
  echo "[VWAP] DONE  $spec rc=$rc elapsed=$(( $(date +%s) - t0 ))s free=$(df --output=avail $(pwd) | tail -1 | awk '{print int($1/1024/1024)"G"}')"
}

CELLS=$(uv run python - <<'PY'
import glob, os, re
EXCLUDE = re.compile(r"agentloop|base_v2|macro|champ|resnap|bear2022|smoketest|fund|trail|_(da)?sw(base|macro)|_fbase|_ffund|_w2|cbbase|cbagent|ffundagent|ffundtune|_f18|_f19|daswmacro|aligned_mixmatch|vwap|_base(xgb|cb)")
d = "configs/gbdt/experiments"
for f in sorted(glob.glob(f"{d}/sp500_up_*pct_*d_dd*pct.yaml")):
    b = os.path.basename(f)[:-5]
    if not EXCLUDE.search(os.path.basename(f)):
        print(b)
PY
)

for suffix in basexgb vwapxgb fundxgb fundvwapxgb basecb vwapcb fundcb fundvwapcb; do
  echo "[VWAP] === arm group: $suffix ==="
  for cell in $CELLS; do
    run "${cell}_${suffix}"
  done
done
echo "[VWAP] ALL COMPLETE $(date -u +%FT%TZ)"
