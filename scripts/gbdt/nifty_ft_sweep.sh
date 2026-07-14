#!/usr/bin/env bash
# V1.10 Phase 2 finetune HP sweep (task #55): deep+bagging grid on FEATS=all per
# the canonical recipe §3. hp_one_canon fits each config on val+eval (test untouched)
# and appends val/eval R-p@K to the cell's FT jsonl. Select on VAL (eval unreliable).
# Usage: CELL=<id> bash nifty_ft_sweep.sh
set -u
cd /mnt/Workspace/Workspace/wt-nse-valuation
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
CELL="${CELL:?set CELL}"
L=/tmp/nifty_canon_scan
OUT="$L/ftsweep_${CELL}.log"
: > "$OUT"

run () {  # depth mcw ss cs
  local d=$1 mcw=$2 ss=$3 cs=$4
  local hp="{\"max_depth\":$d,\"min_child_weight\":$mcw,\"subsample\":$ss,\"colsample_bytree\":$cs,\"gamma\":0,\"eta\":0.05}"
  local lab="d${d}_mcw${mcw}_ss${ss}_cs${cs}"
  echo "[SWEEP] $lab $(date -u +%TZ)" | tee -a "$OUT"
  CELL=$CELL FEATS=all HP="$hp" LABEL="$lab" timeout 1200 uv run python -m scripts.gbdt.hp_one_canon >> "$OUT" 2>&1
}

# deep+bagging grid: depth{6,8,10} x ss{0.7,0.85} x cs{0.7,1.0}, mcw1 (recipe §3)
for d in 6 8 10; do for ss in 0.7 0.85; do for cs in 0.7 1.0; do run $d 1 $ss $cs; done; done; done
# regularized variants for the rare/high-@1 profile: mcw{5,10} at d6/d8, light bagging
for d in 6 8; do for mcw in 5 10; do run $d $mcw 0.85 1.0; done; done
echo "[SWEEP] ALL COMPLETE $CELL $(date -u +%FT%TZ)" | tee -a "$OUT"
