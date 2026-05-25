#!/usr/bin/env bash
# Sequential deep-history fetch for NIFTY 50 (2015-01-01 → 2026-05-25).
# Uses --back-extend to bypass the dispatcher's cache-first cap so providers
# get asked for the pre-cache range. Runs one ticker at a time (sequential)
# to avoid contending with the parallel total-market seeder writing to the
# same data/processed.db. Hard 120s subprocess timeout per ticker so any
# stuck call cannot block the batch.
set -u

cd "$(dirname "$0")/.."

LOG_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="logs/nifty50_deep_${LOG_TS}.log"
START="2015-01-01"
END="2026-05-25"
UNIVERSE_YAML="configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml"
PER_TICKER_TIMEOUT="120"  # seconds; mirrors total-market seeder finding

mkdir -p logs

# Extract tickers (preserve order)
TICKERS=$(uv run python -c "
import yaml
with open('${UNIVERSE_YAML}') as f:
    u = yaml.safe_load(f)
for t in u['tickers']:
    print(t)
")

echo "Deep-history seed for NIFTY 50: ${START} -> ${END}  (--back-extend)" | tee -a "${LOG_FILE}"
echo "Per-ticker timeout: ${PER_TICKER_TIMEOUT}s" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

i=0
n_total=$(echo "${TICKERS}" | wc -l)
n_ok=0
n_fail=0
n_timeout=0
fails=""

count_rows() {
    uv run python -c "
import sqlite3, sys
con = sqlite3.connect('data/processed.db')
try:
    r = con.execute('SELECT COUNT(*) FROM nse_equities_data WHERE ticker = ?', (sys.argv[1],)).fetchone()
    print(r[0] if r else 0)
except Exception:
    print(0)
" "$1" 2>/dev/null
}

for T in ${TICKERS}; do
    i=$((i+1))
    echo "" | tee -a "${LOG_FILE}"
    echo "[${i}/${n_total}] ${T}  $(date -u +%H:%M:%SZ)" | tee -a "${LOG_FILE}"

    PRE_N=$(count_rows "${T}")

    attempt=0
    rc=1
    while [ ${attempt} -lt 3 ] && [ ${rc} -ne 0 ]; do
        attempt=$((attempt+1))
        echo "  attempt ${attempt}/3" | tee -a "${LOG_FILE}"
        # `timeout` returns 124 on hard kill; treat that as a non-retryable
        # signal for THIS attempt but let the outer loop retry once more in
        # case it was a one-off slow provider.
        timeout --kill-after=10 ${PER_TICKER_TIMEOUT} \
            uv run python -m data_pipelines fetch "${T}" \
                --start "${START}" --end "${END}" --back-extend \
                >>"${LOG_FILE}" 2>&1
        rc=$?
        if [ ${rc} -eq 124 ]; then
            echo "  TIMEOUT after ${PER_TICKER_TIMEOUT}s" | tee -a "${LOG_FILE}"
        fi
        if [ ${rc} -ne 0 ]; then
            echo "  rc=${rc}, sleeping 5s before retry" | tee -a "${LOG_FILE}"
            sleep 5
        fi
    done

    POST_N=$(count_rows "${T}")
    DELTA=$((POST_N - PRE_N))

    if [ ${rc} -eq 0 ]; then
        n_ok=$((n_ok+1))
        echo "  OK rows: ${PRE_N} -> ${POST_N}  (+${DELTA})" | tee -a "${LOG_FILE}"
    elif [ ${rc} -eq 124 ]; then
        n_timeout=$((n_timeout+1))
        n_fail=$((n_fail+1))
        fails="${fails} ${T}(timeout)"
        echo "  FAILED (timeout) rows: ${PRE_N} -> ${POST_N}" | tee -a "${LOG_FILE}"
    else
        n_fail=$((n_fail+1))
        fails="${fails} ${T}(rc=${rc})"
        echo "  FAILED rows: ${PRE_N} -> ${POST_N}" | tee -a "${LOG_FILE}"
    fi
done

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_FILE}"
echo "OK: ${n_ok}  FAIL: ${n_fail}  (timeouts: ${n_timeout})" | tee -a "${LOG_FILE}"
if [ -n "${fails}" ]; then
    echo "Failed tickers:${fails}" | tee -a "${LOG_FILE}"
fi
