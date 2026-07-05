#!/usr/bin/env bash
# Window-2 confirmation driver (V1.7_TBD par.5, memo _278): 10 single fits +
# the scripted _275-ffundtune-config replica, all date_aligned train_start
# 2019-07-01 (test ~2025-H1), snapshot 2026-07-02 (matrices warm from _274+).
# Sequential (single-writer contract). The ffundtune mini's decisions are
# deterministic (prune to the _275 141-col list + lambda 4.5, then stop), so
# the whole driver is unattended.
set -u
cd /mnt/Workspace/Workspace/agent-driven-investment
SNAP="2026-07-02"
LOG=/tmp/f18_w2
mkdir -p "$LOG"
run() {
  local spec=$1; shift
  local t0=$(date +%s)
  echo "[W2] START $spec $(date -u +%FT%TZ)"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 3600 uv run python -m gbdt experiment \
    "configs/gbdt/experiments/$spec.yaml" --snapshot-end "$SNAP" --overwrite "$@" \
    >> "$LOG/$spec.log" 2>&1
  echo "[W2] DONE  $spec rc=$? elapsed=$(( $(date +%s) - t0 ))s"
}
resume() {
  local spec=$1
  local t0=$(date +%s)
  echo "[W2] RESUME $spec $(date -u +%FT%TZ)"
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt timeout 3600 uv run python -m gbdt experiment \
    "configs/gbdt/experiments/$spec.yaml" --resume "$spec" --snapshot-end "$SNAP" \
    >> "$LOG/$spec.log" 2>&1
  echo "[W2] DONE  $spec rc=$? elapsed=$(( $(date +%s) - t0 ))s"
}

# 10 single fits
for spec in \
  sp500_up_40pct_200d_dd20pct_w2fbase sp500_up_40pct_200d_dd20pct_w2ffund \
  sp500_up_40pct_200d_dd20pct_w2cbdef \
  sp500_up_20pct_100d_dd10pct_w2fbase sp500_up_20pct_100d_dd10pct_w2ffund \
  sp500_up_20pct_100d_dd10pct_w2cbdef sp500_up_20pct_100d_dd10pct_w2cbd4 \
  sp500_up_20pct_50d_dd10pct_w2fbase sp500_up_20pct_50d_dd10pct_w2ffund \
  sp500_up_20pct_50d_dd10pct_w2cbdef; do
  run "$spec"
done

# the _275 ffundtune-config replica (scripted agent-protocol mini)
T=sp500_up_40pct_200d_dd20pct_w2ffundtune
run "$T" --callback-mode agent_file_protocol
uv run python - <<'PY'
import json, yaml
keep = set(yaml.safe_load(open("results/gbdt/experiments/sp500_up_40pct_200d_dd20pct_ffundtune/features.yaml"))["features"])
D = "results/gbdt/experiments/sp500_up_40pct_200d_dd20pct_w2ffundtune/loop"
d = json.load(open(f"{D}/iter_0_request.json"))
prune = [f for f in d["available_features"] if f not in keep]
assert len(d["available_features"]) - len(prune) == len(keep) == 141, (len(d["available_features"]), len(prune), len(keep))
json.dump({"iter": 0, "prune_features": prune, "hp_changes": {"lambda": 4.5}, "should_stop": False,
  "rationale": "scripted replica: prune to the _275 ffundtune 141-col FS list + lambda 4.5 - iter_1 = the exact config under test on window 2."},
  open(f"{D}/iter_0_decision.json", "w"), indent=1)
print(f"[W2] decision 0 written: prune {len(prune)} -> keep {len(keep)}")
PY
resume "$T"
uv run python - <<'PY'
import json
D = "results/gbdt/experiments/sp500_up_40pct_200d_dd20pct_w2ffundtune/loop"
r1 = json.load(open(f"{D}/iter_1_request.json"))["diagnostics"]["metrics"]["val_brier"]
r0 = json.load(open(f"{D}/iter_0_request.json"))["diagnostics"]["metrics"]["val_brier"]
json.dump({"iter": 1, "should_stop": True,
  "rationale": f"scripted replica complete (iter_0 val {r0:.5f}, iter_1 val {r1:.5f}); finalize. If iter_0 holds the argmin the artifact is NOT the ffundtune config - the driver log records both vals so the aggregation step can detect that and fall back."},
  open(f"{D}/iter_1_decision.json", "w"), indent=1)
print(f"[W2] decision 1 written: val iter0={r0:.5f} iter1={r1:.5f} argmin={'iter1 (config under test)' if r1 < r0 else 'ITER0 - REPLICA NOT EMITTED, handle in aggregation'}")
PY
resume "$T"
echo "[W2] ALL COMPLETE $(date -u +%FT%TZ)"
