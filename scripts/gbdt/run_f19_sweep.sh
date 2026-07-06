#!/usr/bin/env bash
# V1.8 F19 lattice sweep driver (memo _279): 68 matched single fits —
# 17 sp500 cells × {xgboost, catboost} × {all_fundamentals, all_fundamentals2}.
# xgboost arms first (faster → early signal), then catboost. Sequential
# (single-writer contract). Snapshot pinned so the universe feature-cache key
# is stable across all 68 fits (one cold build per universe-view, warm after).
set -u
cd /mnt/Workspace/Workspace/agent-driven-investment
SNAP="${1:-$(date -I)}"
LOG=/tmp/f19_sweep
mkdir -p "$LOG"

run() {
  local spec=$1
  local t0=$(date +%s)
  echo "[F19] START $spec $(date -u +%FT%TZ)"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 3600 uv run python -m gbdt experiment \
    "configs/gbdt/experiments/$spec.yaml" --snapshot-end "$SNAP" --overwrite \
    >> "$LOG/$spec.log" 2>&1
  local rc=$?
  # Post-fit cleanup: the durable outputs (metrics.json, predictions/,
  # model, report) are what Phase E aggregation reads; the per-experiment
  # matrix cache (~3.8G) is a regenerable slice of the shared universe cache
  # (data/gbdt_feature_cache, untouched). Drop it so 68 arms don't re-
  # accumulate ~260G. Only on a clean fit (rc=0) so a failure stays debuggable.
  if [ "$rc" -eq 0 ]; then
    rm -f "results/gbdt/experiments/$spec/_feature_matrix_cache.parquet" \
          "results/gbdt/experiments/$spec/_feature_matrix_cache.key.json"
  fi
  echo "[F19] DONE  $spec rc=$rc elapsed=$(( $(date +%s) - t0 ))s free=$(df --output=avail $(pwd) | tail -1 | awk '{print int($1/1024/1024)"G"}')"
}

CELLS=$(uv run python - <<'PY'
import glob, os, re
EXCLUDE = re.compile(r"agentloop|base_v2|macro|champ|resnap|bear2022|smoketest|fund|trail|_(da)?sw(base|macro)|_fbase|_ffund|_w2|cbbase|cbagent|ffundagent|ffundtune|_f18|_f19|daswmacro|aligned_mixmatch")
d = "configs/gbdt/experiments"
for f in sorted(glob.glob(f"{d}/sp500_up_*pct_*d_dd*pct.yaml")):
    b = os.path.basename(f)[:-5]
    if not EXCLUDE.search(os.path.basename(f)):
        print(b)
PY
)

# xgboost arms first (both feature tokens), then catboost.
for suffix in f18xgb f19xgb f18cb f19cb; do
  echo "[F19] === arm group: $suffix ==="
  for cell in $CELLS; do
    run "${cell}_${suffix}"
  done
done
echo "[F19] ALL COMPLETE $(date -u +%FT%TZ)"
