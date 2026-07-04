#!/usr/bin/env python
"""Generate matched base-vs-fundamentals sweep specs across a universe's lattice.

For each canonical sweep spec
(``configs/gbdt/experiments/<uni>_up_<thr>pct_<H>d_dd<dd>pct.yaml``, excluding the
champion / variant / previously-generated arms), emit two specs:

    <cell>_fbase.yaml   features.candidates: all              (F1–F16)
    <cell>_ffund.yaml   features.candidates: all_fundamentals (F1–F16 + F18)

Both carry the IDENTICAL config used by the `_272`/`_273` fundamentals A/B —
xgboost, conditional_isotonic, **default HP** (no ``hp_starting``), a single fit
(``max_iterations: 1``), date_aligned split (train_start 2019-01-01). So the ONLY
difference between the two arms is the F18 fundamentals family, and each cell's
``fund − base`` R-Precision/AUC/Brier delta is a clean read of the F18
contribution at that (threshold, horizon). This maps WHERE fundamentals help
across the lattice — the follow-up to the heterogeneous 2-cell `_272`/`_273`
result. Memo _274.

Matches the `_272` config exactly (default HP, NOT ``mcw=10`` like the macro
sweep) so the +50/50 and +20/25 cells re-measure the champions with the current
13-col F18 (a strict extension of the `_272` date-aligned arms).

Usage:
    uv run python -m scripts.gbdt.gen_fund_sweep_specs [universe]
"""
import glob
import os
import re
import sys

import yaml

UNIVERSE = sys.argv[1] if len(sys.argv) > 1 else "sp500"
# Skip champion / variant / macro / previously-generated arms.
EXCLUDE = re.compile(
    r"agentloop|base_v2|macro|champ|resnap|bear2022|smoketest|fund|trail"
    r"|_(da)?sw(base|macro)|_fbase|_ffund"
)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(ROOT, "configs", "gbdt", "experiments")


def render(cell: str, tgt: dict, candidates: str) -> str:
    arm = "+fund" if candidates == "all_fundamentals" else "base"
    return f"""# Experiment: {cell}
# Fundamentals horizon×target sweep ({arm} arm, date_aligned), memo _274.
# IDENTICAL config to the _272/_273 A/B (xgboost, default HP, single fit): base
# and +fund differ ONLY in features.candidates, so each cell's fund-minus-base
# delta is a clean read of the F18 contribution at this (threshold, horizon).
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
  library: xgboost
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
        f
        for f in sorted(glob.glob(os.path.join(EXP_DIR, f"{UNIVERSE}_up_*pct_*d_dd*pct.yaml")))
        if not EXCLUDE.search(os.path.basename(f))
    ]
    if not canon:
        raise SystemExit(f"no canonical sweep specs found for universe {UNIVERSE!r}")
    n = 0
    for f in canon:
        tgt = yaml.safe_load(open(f))["target"]
        base = os.path.basename(f)[:-5]
        for suffix, candidates in (("fbase", "all"), ("ffund", "all_fundamentals")):
            cell = f"{base}_{suffix}"
            with open(os.path.join(EXP_DIR, f"{cell}.yaml"), "w") as fh:
                fh.write(render(cell, tgt, candidates))
            n += 1
    print(f"generated {n} specs across {len(canon)} {UNIVERSE} cells")


if __name__ == "__main__":
    main()
