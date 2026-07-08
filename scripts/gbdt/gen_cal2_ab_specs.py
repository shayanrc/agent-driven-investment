"""Generate the V1.9 calendar2 (F21) matched single-fit A/B specs.

Three representative nasdaq100 cells (the VWAP-lattice cells) × two arms that
differ ONLY in ``features.candidates``:

    base  -> all            (F1-F16)
    cal2  -> all_calendar2   (F1-F16 + F21)

Matched single-fit (``max_iterations 1``, default HP, ``date_aligned``
``train_start 2019-01-01``, xgboost) so the per-cell delta is a clean read of
the F21 contribution (the ``[[project-gbdt-macro-features-f17]]`` matched-HP
rule — never compare feature arms via the auto-loop). ``--snapshot-end`` is
set at runtime by ``run_cal2_ab.sh``.

Writes to ``configs/gbdt/experiments/cal2_ab/<cell>_cal2ab_<arm>.yaml``.
"""

from __future__ import annotations

from pathlib import Path

# (cell stem, threshold_pct, horizon_days, max_drawdown)
CELLS = [
    ("nasdaq100_up_50pct_25d_dd25pct", 50, 25, 0.25),
    ("nasdaq100_up_20pct_50d_dd10pct", 20, 50, 0.10),
    ("nasdaq100_up_40pct_200d_dd20pct", 40, 200, 0.20),
]

ARMS = {"base": "all", "cal2": "all_calendar2"}

TEMPLATE = """\
# Experiment: {name}
# V1.9 calendar2 (F21) matched single-fit A/B (xgboost, technical). Default HP,
# max_iterations 1, date_aligned train_start 2019-01-01: the two arms differ
# ONLY in features.candidates, so cal2-minus-base is a clean read of the F21
# month-of-quarter / quarter-of-year contribution. --snapshot-end set at runtime.
target:
  universe: nasdaq100
  direction: up
  threshold_pct: {threshold}
  horizon_days: {horizon}
  max_drawdown: {mdd}

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
    out_dir = Path(__file__).resolve().parents[2] / "configs/gbdt/experiments/cal2_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cell, thr, hor, mdd in CELLS:
        for arm, token in ARMS.items():
            name = f"{cell}_cal2ab_{arm}"
            text = TEMPLATE.format(
                name=name, threshold=thr, horizon=hor, mdd=mdd, candidates=token,
            )
            path = out_dir / f"{name}.yaml"
            path.write_text(text)
            written.append(path.name)
    for w in written:
        print(w)


if __name__ == "__main__":
    main()
