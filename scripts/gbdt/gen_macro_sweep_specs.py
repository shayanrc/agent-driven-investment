#!/usr/bin/env python
"""Generate matched base-vs-+macro sweep specs across a universe's canonical cells.

For each canonical sweep spec
(``configs/gbdt/experiments/<uni>_up_<thr>pct_<H>d_dd<dd>pct.yaml``, excluding the
champion / macro / variant specs), emit two specs:

    <cell>_swbase.yaml    features.candidates: all
    <cell>_swmacro.yaml   features.candidates: all_macro

Both carry an IDENTICAL fixed config — xgboost, ``min_child_weight=10``, a single
fit (``max_iterations: 1``) — so the ONLY difference between the two arms is the
macro feature family. That makes each cell's ``macro - base`` R-Precision delta a
clean read of the macro contribution, and deliberately avoids the ``_260``
default-auto per-arm HP-divergence confound (see ``docs/gbdt/_262_*`` / ``_263_*``).

The fixed ``mcw=10`` is the two sp500 champions' converged choice; it is held
constant across the lattice so the *delta* is matched (it is NOT each cell's tuned
optimum — the sweep answers "does macro add signal at a fixed model", not "macro at
each cell's best tuning").

Usage:
    uv run python -m scripts.gbdt.gen_macro_sweep_specs [universe]   # default: sp500
"""
import glob
import os
import re
import sys

import yaml

UNIVERSE = sys.argv[1] if len(sys.argv) > 1 else "sp500"
# Skip champion/macro/variant specs and any previously-generated sweep arms.
EXCLUDE = re.compile(r"agentloop|base_v2|macro|champ|resnap|bear2022|smoketest|_sw(base|macro)")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(ROOT, "configs", "gbdt", "experiments")


def render(cell: str, tgt: dict, candidates: str) -> str:
    arm = "+macro" if candidates == "all_macro" else "base"
    return f"""# Experiment: {cell}
# Macro-lattice sweep ({arm} arm), memo _263.
# Matched fixed config (xgboost, min_child_weight=10, single fit): the base and
# +macro arms differ ONLY in features.candidates, so each cell's macro-minus-base
# R-Precision delta is a clean read of the macro contribution (avoids the _260
# default-auto per-arm HP confound). trailing split; --snapshot-end set at runtime.
target:
  universe: {tgt['universe']}
  direction: {tgt['direction']}
  threshold_pct: {tgt['threshold_pct']}
  horizon_days: {tgt['horizon_days']}
  max_drawdown: {tgt['max_drawdown']}

features:
  candidates: {candidates}

backend:
  library: xgboost
  calibration_method: conditional_isotonic
  hp_starting:
    min_child_weight: 10
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
        base = os.path.basename(f)[:-5]  # strip .yaml
        for suffix, candidates in (("swbase", "all"), ("swmacro", "all_macro")):
            cell = f"{base}_{suffix}"
            with open(os.path.join(EXP_DIR, f"{cell}.yaml"), "w") as fh:
                fh.write(render(cell, tgt, candidates))
            n += 1
    print(f"generated {n} specs across {len(canon)} {UNIVERSE} cells")


if __name__ == "__main__":
    main()
