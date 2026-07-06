#!/usr/bin/env python
"""Generate the V1.8 F19 lattice sweep specs (memo _279).

For each of the 17 canonical sp500 lattice cells, emit a 4-arm
backend × features factorial — matched single fits (default HP,
``max_iterations: 1``, date_aligned ``train_start: 2019-01-01``) so the only
difference within a backend is ``features.candidates``:

    <cell>_f18xgb.yaml   xgboost  · all_fundamentals   (F1–F16 + F18)
    <cell>_f19xgb.yaml   xgboost  · all_fundamentals2  (F1–F16 + F18 + F19)
    <cell>_f18cb.yaml    catboost · all_fundamentals
    <cell>_f19cb.yaml    catboost · all_fundamentals2

The per-cell, per-backend ``f19 − f18`` R-Precision/AUC/Brier delta is a clean
read of the F19 revenue-growth contribution on top of F18. All four arms are
fit fresh on the regenerated valuation panel (now carrying ``revenue_q``), so
the factorial is uniform — no reuse of the _274 rows (which measured F18 on the
pre-revenue_q parquet). The _274 fbase (xgb technical) + _278 cb technical rows
are pulled as context in the memo, not recomputed here.

Usage:
    uv run python -m scripts.gbdt.gen_f19_sweep_specs [universe]
"""
import glob
import os
import re
import sys

import yaml

UNIVERSE = sys.argv[1] if len(sys.argv) > 1 else "sp500"
# Skip every champion / variant / previously-generated arm — keep only the
# 17 canonical (threshold, horizon, drawdown) lattice cells.
EXCLUDE = re.compile(
    r"agentloop|base_v2|macro|champ|resnap|bear2022|smoketest|fund|trail"
    r"|_(da)?sw(base|macro)|_fbase|_ffund|_w2|cbbase|cbagent|ffundagent"
    r"|ffundtune|_f18|_f19|daswmacro|aligned_mixmatch"
)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(ROOT, "configs", "gbdt", "experiments")

# (suffix, backend, candidates)
ARMS = (
    ("f18xgb", "xgboost", "all_fundamentals"),
    ("f19xgb", "xgboost", "all_fundamentals2"),
    ("f18cb", "catboost", "all_fundamentals"),
    ("f19cb", "catboost", "all_fundamentals2"),
)


def render(cell: str, tgt: dict, backend: str, candidates: str) -> str:
    fam = "F18+F19" if candidates == "all_fundamentals2" else "F18"
    return f"""# Experiment: {cell}
# V1.8 F19 lattice sweep ({backend} · {fam}), memo _279. Matched single fit
# (default HP, max_iterations 1, date_aligned train_start 2019-01-01): within a
# backend the f18/f19 arms differ ONLY in features.candidates, so the per-cell
# f19-minus-f18 delta is a clean read of the F19 revenue-growth contribution.
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
