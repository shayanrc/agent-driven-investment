#!/usr/bin/env bash
# V1.8 VWAP (F20) lattice sweep driver — universe-parameterized, xgboost-only.
# <UNIVERSE> cells × xgboost × {all, all_vwap, all_fundamentals, all_fundamentals_vwap}.
# Same design as run_vwap_sweep.sh (the sp500-hardcoded original) but takes the
# universe as $1. No fundamentals-reuse shortcut is available off sp500 (the whole
# F18/F19 arc was sp500-only), so the fund/fundvwap arms build from scratch here.
# Sequential (single-writer SQLite contract); resumable (skips arms whose
# metrics.json already exists); one cold universe-build per feature-token.
#
# Usage: scripts/gbdt/run_vwap_sweep_uni.sh <UNIVERSE> [SNAPSHOT_END]
set -u
cd /mnt/Workspace/Workspace/agent-driven-investment
UNIVERSE="${1:?usage: run_vwap_sweep_uni.sh <UNIVERSE> [SNAPSHOT_END]}"
SNAP="${2:-$(date -I)}"
LOG="/tmp/vwap_sweep_${UNIVERSE}"
mkdir -p "$LOG"

FREE_G=$(df --output=avail "$(pwd)" | tail -1 | awk '{print int($1/1024/1024)}')
if [ "$FREE_G" -lt 15 ]; then
  echo "[VWAP:$UNIVERSE] ABORT: only ${FREE_G}G free (<15G) — FS-wedge guard." >&2
  exit 1
fi

run() {
  local spec=$1
  if [ -f "results/gbdt/experiments/$spec/metrics.json" ]; then
    echo "[VWAP:$UNIVERSE] SKIP  $spec (already done)"; return
  fi
  local t0=$(date +%s)
  echo "[VWAP:$UNIVERSE] START $spec $(date -u +%FT%TZ)"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 5400 uv run python -m gbdt experiment \
    "configs/gbdt/experiments/$spec.yaml" --snapshot-end "$SNAP" --overwrite \
    >> "$LOG/$spec.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    rm -f "results/gbdt/experiments/$spec/_feature_matrix_cache.parquet" \
          "results/gbdt/experiments/$spec/_feature_matrix_cache.key.json"
  fi
  echo "[VWAP:$UNIVERSE] DONE  $spec rc=$rc elapsed=$(( $(date +%s) - t0 ))s free=$(df --output=avail $(pwd) | tail -1 | awk '{print int($1/1024/1024)"G"}')"
}

CELLS=$(UNIVERSE="$UNIVERSE" uv run python - <<'PY'
import glob, os, re
U=os.environ["UNIVERSE"]
EXCLUDE = re.compile(r"agentloop|base_v2|macro|champ|resnap|bear2022|smoketest|fund|trail|_(da)?sw(base|macro)|_fbase|_ffund|_w2|cbbase|cbagent|ffundagent|ffundtune|_f18|_f19|daswmacro|aligned_mixmatch|vwap|_base(xgb|cb)")
for f in sorted(glob.glob(f"configs/gbdt/experiments/{U}_up_*pct_*d_dd*pct.yaml")):
    b = os.path.basename(f)[:-5]
    if not EXCLUDE.search(os.path.basename(f)):
        print(b)
PY
)

for suffix in basexgb vwapxgb fundxgb fundvwapxgb; do
  echo "[VWAP:$UNIVERSE] === arm group: $suffix ==="
  for cell in $CELLS; do
    run "${cell}_${suffix}"
  done
done
echo "[VWAP:$UNIVERSE] ALL COMPLETE $(date -u +%FT%TZ)"
