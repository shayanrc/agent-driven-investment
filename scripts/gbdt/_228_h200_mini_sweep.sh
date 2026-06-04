#!/usr/bin/env bash
# bug #228 — H=200 mini-sweep: 8 *_200d_*_aligned.yaml cells that the
# main H>=100 sweep (PID 578608, script: _228_sweep_h100.sh) would have
# crashed at _validate_spec before commit 28601eb landed.
#
# This script POLLS the main sweep's status file every 30s; once the
# main sweep reports status="done", it sequentially runs the 8 H=200
# cells with the same --snapshot-end value the main sweep used (2026-06-04).
#
# If the main sweep PID dies WITHOUT writing status="done", we bail out
# with non-zero (don't run on top of a crashed main sweep — the crash
# may have left the SQLite cache in a state that needs human review).
#
# NO `-e`: a per-cell failure does not abort the mini-sweep; the next
# cell still runs. Logs to /tmp/v1_228_sweep_h200.log; heartbeat /
# progress status to /tmp/v1_228_sweep_h200_status.json.
#
# Designed to be invoked via setsid so it survives session exit:
#   setsid bash scripts/gbdt/_228_h200_mini_sweep.sh &
#   disown
set -uo pipefail

END=2026-06-04
LOG=/tmp/v1_228_sweep_h200.log
STATUS=/tmp/v1_228_sweep_h200_status.json
MAIN_PID=578608
MAIN_STATUS=/tmp/v1_228_sweep_status.json

SPECS=(
  configs/gbdt/experiments/nasdaq100_up_10pct_200d_dd5pct_aligned.yaml
  configs/gbdt/experiments/nasdaq100_up_40pct_200d_dd20pct_aligned.yaml
  configs/gbdt/experiments/nasdaq100_up_50pct_200d_dd25pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_10pct_200d_dd5pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_40pct_200d_dd20pct_aligned.yaml
  configs/gbdt/experiments/russell1000_up_50pct_200d_dd25pct_aligned.yaml
  configs/gbdt/experiments/sp500_up_40pct_200d_dd20pct_aligned.yaml
  configs/gbdt/experiments/sp500_up_50pct_200d_dd25pct_aligned.yaml
)

N=${#SPECS[@]}

{
  echo "=== bug #228 H=200 mini-sweep launched at $(date -Is) ==="
  echo "Waiting on main sweep PID $MAIN_PID"
  echo "Pinned --snapshot-end $END"
  echo "Total cells: $N"
  echo "Worktree: $(pwd)"
  echo

  # Initial status: waiting
  jq -n --arg t "$(date -Is)" --argjson n "$N" --argjson p "$MAIN_PID" \
    '{ts:$t, cell_idx:0, cell_total:$n, current_spec:null, current_cell:null, status:"waiting", waiting_on_pid:$p}' \
    > "$STATUS"

  # Poll loop
  while true; do
    if [ ! -d "/proc/$MAIN_PID" ]; then
      st=$(jq -r '.status // "missing"' "$MAIN_STATUS" 2>/dev/null)
      if [ "$st" = "done" ]; then
        echo "Main sweep PID $MAIN_PID finished and status=done at $(date -Is) — starting mini-sweep"
        break
      else
        echo "Main sweep PID $MAIN_PID gone but status=$st at $(date -Is) — aborting mini-sweep"
        jq -n --arg t "$(date -Is)" --arg s "$st" \
          '{ts:$t, status:"aborted_main_crashed", main_last_status:$s}' > "$STATUS"
        exit 1
      fi
    fi
    st=$(jq -r '.status // ""' "$MAIN_STATUS" 2>/dev/null)
    if [ "$st" = "done" ]; then
      echo "Main sweep reports status=done at $(date -Is) — starting mini-sweep"
      break
    fi
    sleep 30
  done

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
  echo "=== H=200 mini-sweep complete at $(date -Is) ==="
} > "$LOG" 2>&1
