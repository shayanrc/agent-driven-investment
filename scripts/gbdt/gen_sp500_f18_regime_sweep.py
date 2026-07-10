#!/usr/bin/env python
"""Generate the sp500 F18 second-window A/B sweep, regime-corrected to MIRROR the
nifty500 `_285` setup (train_start 2015 + calendar2 tokens).

For each canonical sp500 sweep cell
(``configs/gbdt/experiments/sp500_up_<thr>pct_<H>d_dd<dd>pct.yaml``, excluding
generated / champion / variant arms), emit two specs:

    <cell>_rfbase.yaml   features.candidates: all_calendar2               (F1-F16 + F21)
    <cell>_rffund.yaml   features.candidates: all_fundamentals_calendar2  (F1-F16 + F21 + F18)

Both carry the IDENTICAL config the nifty500 `_285` sweep used — xgboost, default
HP, single fit (``max_iterations: 1``), date_aligned split train_start 2015-01-01
+ calendar2 tokens — so the ONLY difference between the two arms is the F18
fundamentals family. Each cell's ``fund - base`` R-Precision/AUC/Brier delta is a
clean read of the F18 contribution at that (threshold, horizon) on sp500 — the
independent second-market replication of the nifty500 "F18 helps at 100d"
hypothesis (`_285` follow-up / task #28). Memo _287.

The ``_r`` prefix (regime-corrected) distinguishes these from the pre-existing
`_fbase`/`_ffund` arms (the `_274` sweep: train_start 2019, non-calendar2 tokens).

Usage:
    uv run python -m scripts.gbdt.gen_sp500_f18_regime_sweep
"""
import glob
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(ROOT, "configs", "gbdt", "experiments")

# Mirror nifty500 _285 exactly (date_aligned, 2015 anchor). date_aligned anchors
# segments to the universe calendar from train_start, so the same row counts give
# comparable NYSE windows (train -> ~2022-03, test -> ~2025-07).
SPLIT = {
    "mode": "date_aligned",
    "train_start": "2015-01-01",
    "train_rows": 1787,
    "min_rows_per_ticker": 2591,
    "test_rows": 204,
}

EXCLUDE = re.compile(
    r"_(rfbase|rffund|fbase|ffund|champion|maxtune|macro|w2|tune|resnap"
    r"|bear\d*|smoketest|trail|base_v2|agentloop|dasw|swbase|swmacro)"
)


def spec_for(tgt: dict, token: str) -> dict:
    return {
        "target": tgt,
        "split": dict(SPLIT),
        "features": {"candidates": token},
        "backend": {
            "library": "xgboost",
            "calibration_method": "conditional_isotonic",
            "fs_hp_loop": {
                "callback_mode": "default",
                "max_iterations": 1,
                "plateau_threshold": 0.005,
                "degradation_gate": 0.01,
            },
        },
        "random_seed": 42,
    }


def emit(stem: str, tgt: dict, token: str, suffix: str) -> str:
    path = os.path.join(EXP_DIR, f"{stem}_{suffix}.yaml")
    arm = "+fund (F18)" if "fundamentals" in token else "base"
    with open(path, "w") as f:
        f.write(f"# {stem}_{suffix} — sp500 F18 second-window sweep ({arm} arm), memo _287.\n")
        f.write("# Regime-corrected mirror of nifty500 _285: date_aligned train_start 2015,\n")
        f.write("# calendar2 tokens, xgboost default HP, single fit. base and +fund differ\n")
        f.write("# ONLY in features.candidates -> fund-minus-base delta = clean F18 read.\n")
        f.write("# --snapshot-end 2025-07-01 set at runtime.\n")
        yaml.safe_dump(spec_for(tgt, token), f, sort_keys=False)
    return path


def main() -> None:
    cells = sorted(glob.glob(os.path.join(EXP_DIR, "sp500_up_*pct_*d_dd*pct.yaml")))
    cells = [c for c in cells if not EXCLUDE.search(os.path.basename(c))]
    n = 0
    for c in cells:
        tgt = yaml.safe_load(open(c))["target"]
        stem = os.path.basename(c)[:-5]
        emit(stem, tgt, "all_calendar2", "rfbase")
        emit(stem, tgt, "all_fundamentals_calendar2", "rffund")
        n += 2
    print(f"generated {n} specs ({len(cells)} cells x 2 arms) under {EXP_DIR}")


if __name__ == "__main__":
    main()
