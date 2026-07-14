"""Generate the nifty500 canonical-period fbase/ffund lattice specs (V1.10, task #55).

Re-runs the `_285` base-vs-fund A/B lattice on the CANONICAL evaluation periods
(explicit-boundary date_aligned split) after the screener.in pre-2019 backfill
de-confounds F18-IN. 20 cells (4 thresholds x 5 horizons, dd = threshold/2) x 2 arms
= 40 single-fit specs; identical HP, only features.candidates differs.

Writes to configs/gbdt/experiments/nifty500_up_<T>pct_<H>d_dd<D>pct_{fbase,ffund}_canon.yaml
"""
from __future__ import annotations

from pathlib import Path

OUT = Path("configs/gbdt/experiments")

THRESHOLDS = [10, 20, 30, 50]          # dd = threshold // 2
HORIZONS = [10, 25, 50, 100, 200]
ARMS = {"fbase": "all_calendar2", "ffund": "all_fundamentals_calendar2"}

# Canonical evaluation periods (NSE cells land on nearest NSE trading days).
SPLIT = """split:
  mode: date_aligned
  train_start: '2015-01-01'
  val_start: '2022-03-30'
  eval_start: '2023-07-01'
  test_start: '2024-07-01'
  test_end: '2025-06-30'
  min_rows_per_ticker: 2591
"""

TEMPLATE = """# Experiment: {name}
# V1.10 nifty500 canonical scan (task #55) — supersedes the INVALIDATED _285 sweep.
# Canonical explicit-boundary date_aligned split; F18-IN de-confounded by the
# screener.in pre-2019 backfill. Matched base-vs-fund A/B: {arm} arm differs from
# its sibling ONLY in features.candidates (xgboost default HP, single fit), so each
# cell's fund-minus-base delta is a clean read of the F18 contribution.
target:
  universe: nifty500
  direction: up
  threshold_pct: {thr}
  horizon_days: {hor}
  max_drawdown: {dd:.2f}

{split}
features:
  candidates: {token}

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
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    names = []
    for thr in THRESHOLDS:
        dd = thr // 2
        for hor in HORIZONS:
            for arm, token in ARMS.items():
                name = f"nifty500_up_{thr}pct_{hor}d_dd{dd}pct_{arm}_canon"
                (OUT / f"{name}.yaml").write_text(
                    TEMPLATE.format(name=name, arm=arm, thr=thr, hor=hor,
                                    dd=dd / 100.0, split=SPLIT, token=token)
                )
                names.append(name)
                n += 1
    print(f"wrote {n} specs to {OUT}/")
    for nm in names:
        print(" ", nm)


if __name__ == "__main__":
    main()
