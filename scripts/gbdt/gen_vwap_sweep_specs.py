#!/usr/bin/env python
"""Generate the V1.8 VWAP (F20) lattice sweep specs (memo _NNN, plan
docs/gbdt/V1.8_vwap_deviation_features_plan.md).

For each canonical sp500 lattice cell, emit an 8-arm backend × features
factorial — matched single fits (default HP, ``max_iterations: 1``, date_aligned
``train_start: 2019-01-01``) so the only difference within a backend is
``features.candidates``:

    <cell>_basexgb      xgboost  · all                    (F1–F16)
    <cell>_vwapxgb      xgboost  · all_vwap               (F1–F16 + F20)
    <cell>_fundxgb      xgboost  · all_fundamentals       (F1–F16 + F18)
    <cell>_fundvwapxgb  xgboost  · all_fundamentals_vwap  (F1–F16 + F18 + F20)
    <cell>_basecb  … _vwapcb … _fundcb … _fundvwapcb      catboost mirror

This reads BOTH the VWAP marginal (``vwap − base``) AND whether VWAP stacks on
fundamentals (``fundvwap − fund``), on both backends. Pin ``--snapshot-end`` so
the universe feature-cache key is stable across all arms.

Usage:
    uv run python -m scripts.gbdt.gen_vwap_sweep_specs [universe]
"""
import glob
import os
import re
import sys

import yaml

UNIVERSE = sys.argv[1] if len(sys.argv) > 1 else "sp500"
# Keep only the canonical (threshold, horizon, drawdown) lattice cells — skip
# every champion / variant / previously-generated arm (incl. this sweep's own
# base/vwap/fund arms so a re-run doesn't recurse).
EXCLUDE = re.compile(
    r"agentloop|base_v2|macro|champ|resnap|bear2022|smoketest|fund|trail"
    r"|_(da)?sw(base|macro)|_fbase|_ffund|_w2|cbbase|cbagent|ffundagent"
    r"|ffundtune|_f18|_f19|daswmacro|aligned_mixmatch|vwap|_base(xgb|cb)"
)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(ROOT, "configs", "gbdt", "experiments")

# (suffix, backend, candidates)
ARMS = (
    ("basexgb", "xgboost", "all"),
    ("vwapxgb", "xgboost", "all_vwap"),
    ("fundxgb", "xgboost", "all_fundamentals"),
    ("fundvwapxgb", "xgboost", "all_fundamentals_vwap"),
    ("basecb", "catboost", "all"),
    ("vwapcb", "catboost", "all_vwap"),
    ("fundcb", "catboost", "all_fundamentals"),
    ("fundvwapcb", "catboost", "all_fundamentals_vwap"),
)

_FAM = {
    "all": "technical", "all_vwap": "technical+VWAP",
    "all_fundamentals": "technical+F18", "all_fundamentals_vwap": "technical+F18+VWAP",
}


def render(cell: str, tgt: dict, backend: str, candidates: str) -> str:
    return f"""# Experiment: {cell}
# V1.8 VWAP (F20) lattice sweep ({backend} · {_FAM[candidates]}). Matched single
# fit (default HP, max_iterations 1, date_aligned train_start 2019-01-01): within
# a backend the arms differ ONLY in features.candidates, so vwap-minus-base and
# fundvwap-minus-fund are clean reads of the F20 VWAP-deviation contribution.
# --snapshot-end set at runtime.
target:
  universe: {tgt['universe']}
  direction: {tgt['direction']}
  threshold_pct: {tgt['threshold_pct']}
  horizon_days: {tgt['horizon_days']}
  max_drawdown: {tgt['max_drawdown']}

split:
  mode: date_aligned
  train_start: 2019-01-01

features:
  candidates: {candidates}

backend:
  library: {backend}
  calibration_method: conditional_isotonic
  fs_hp_loop:
    callback_mode: default
    max_iterations: 1
    plateau_threshold: 0.005
    degradation_gate: 0.01

random_seed: 42
"""


def main() -> None:
    canon = [
        f for f in sorted(
            glob.glob(os.path.join(EXP_DIR, f"{UNIVERSE}_up_*pct_*d_dd*pct.yaml")))
        if not EXCLUDE.search(os.path.basename(f))
    ]
    if not canon:
        raise SystemExit(f"no canonical sweep specs found for universe {UNIVERSE!r}")
    n = 0
    for f in canon:
        tgt = yaml.safe_load(open(f))["target"]
        base = os.path.basename(f)[:-5]
        for suffix, backend, candidates in ARMS:
            cell = f"{base}_{suffix}"
            with open(os.path.join(EXP_DIR, f"{cell}.yaml"), "w") as fh:
                fh.write(render(cell, tgt, backend, candidates))
            n += 1
    print(f"generated {n} specs across {len(canon)} {UNIVERSE} cells "
          f"({len(ARMS)} arms each)")


if __name__ == "__main__":
    main()
