#!/usr/bin/env bash
# bug #228 — H>=100 US-sweep rerun under V1.4 date-aligned splits +
# bug #226 --snapshot-end pin.
#
# Launches 19 cells sequentially. NO `-e`: a per-cell failure does not abort
# the sweep; the next cell still runs. Logs to /tmp/v1_228_sweep.log;
# heartbeat / progress status to /tmp/v1_228_sweep_status.json.
#
# Designed to be invoked via setsid so it survives session exit:
#   setsid bash scripts/gbdt/_228_sweep_h100.sh &
#   disown
set -uo pipefail

END=$(date -I)
LOG=/tmp/v1_228_sweep.log
STATUS=/tmp/v1_228_sweep_status.json

SPECS=(
  configs/gbdt/experiments/nasdaq100_up_10pct_100d_dd5pct_aligned.yaml
  configs/gbdt/experiments/nasdaq100_up_10pct_200d_dd5pct_aligned.yaml
  configs/gbdt/experiments/nasdaq100_up_20pct_100d_dd10pct_aligned.yaml
  configs/gbdt/experiments/nasdaq100_up_40pct_100d_dd20pct_aligned.yaml
  configs/gbdt/experiments/nasdaq100_up_40pct_200d_dd20pct_aligned.yaml
  configs/gbdt/experiments/nasdaq100_up_50pct_100d_dd25pct_aligned.yaml
  configs/gbdt/experiments/nasdaq100_up_50pct_200d_dd25pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_10pct_100d_dd5pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_10pct_200d_dd5pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_20pct_100d_dd10pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_40pct_100d_dd20pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_40pct_200d_dd20pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_50pct_100d_dd25pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_50pct_200d_dd25pct_aligned.yaml
  configs/gbdt/experiments/sp500_up_20pct_100d_dd10pct_aligned.yaml
  configs/gbdt/experiments/sp500_up_40pct_100d_dd20pct_aligned.yaml
  configs/gbdt/experiments/sp500_up_40pct_200d_dd20pct_aligned.yaml
  configs/gbdt/experiments/sp500_up_50pct_100d_dd25pct_aligned.yaml
  configs/gbdt/experiments/sp500_up_50pct_200d_dd25pct_aligned.yaml
)

N=${#SPECS[@]}

{
  echo "=== bug #228 H>=100 sweep launched at $(date -Is) ==="
  echo "Pinned --snapshot-end $END"
  echo "Total cells: $N"
  echo "Worktree: $(pwd)"
  echo
  i=0
  for spec in "${SPECS[@]}"; do
    i=$((i+1))
    cell=$(basename "$spec" .yaml)
    echo
    echo "--- cell $i/$N: $cell ($(date -Is)) ---"
    jq -n \
      --arg t "$(date -Is)" \
      --argjson i "$i" \
      --argjson n "$N" \
      --arg s "$spec" \
      --arg c "$cell" \
      '{ts:$t, cell_idx:$i, cell_total:$n, current_spec:$s, current_cell:$c, status:"running"}' \
      > "$STATUS"
    uv run python -m gbdt experiment "$spec" --snapshot-end "$END" --callback-mode default 2>&1
    rc=$?
    echo "--- cell $i exit $rc ($(date -Is)) ---"
  done
  jq -n --arg t "$(date -Is)" --argjson n "$N" \
    '{ts:$t, cell_idx:$n, cell_total:$n, status:"done"}' > "$STATUS"
  echo
  echo "=== sweep complete at $(date -Is) ==="
} > "$LOG" 2>&1
