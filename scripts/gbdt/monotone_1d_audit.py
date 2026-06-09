"""Decisive check: is the UNCONSTRAINED model already monotone in each of the
17 constrained features? If yes, the iter-5 monotone constraints were redundant
(the model learned the relationship from data) and the harm is structural cost,
not destruction of useful non-monotonicity.

1D partial dependence: sweep each feature over its 5..95pct deciles, hold all
other features at their in-sample values (averaged over a subsample), record
mean predicted P(+breakout). Report whether the curve is non-decreasing.

Also caches the in-sample feature matrix to parquet to avoid future rebuilds.

**FROZEN ONE-SHOT.** Results live in the nifty50 H=25 monotone-constraint
memo. To re-run, set ``WORKSPACE_ROOT`` per per-user memory
``scratch-cache-path``; the referenced ``wt-exp-nifty50-up10-25d/`` worktree
may have been pruned.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from gbdt import data as gbdt_data
from gbdt import features as gbdt_features

UNIVERSE = "nifty50"
TEST_TAIL_ROWS = 100
# WORKSPACE_ROOT = parent dir where ``wt-*/`` worktrees live; per-machine,
# see per-user memory ``scratch-cache-path`` for the literal.
BEFORE = f"{os.environ.get('WORKSPACE_ROOT', '<SET-WORKSPACE_ROOT>')}/wt-exp-nifty50-up10-25d/results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct/model.cbm"
AFTER = "results/gbdt/experiments/nifty50_manualloop_iter5/model.cbm"
CACHE = "results/gbdt/experiments/_nifty50_insample_matrix.parquet"
GRID, SUBSAMPLE = 12, 4000

CONSTRAINED = (
    [f"garman_klass_{w}" for w in (5, 10, 20, 50, 100, 200)]
    + [f"parkinson_{w}" for w in (5, 10, 20, 50, 100, 200)]
    + [f"realized_vol_{w}" for w in (10, 20, 50, 100, 200)]
)


def build_insample() -> pd.DataFrame:
    po = gbdt_data.load_panel(UNIVERSE, min_rows=1600)
    X = gbdt_features.build_feature_matrix(
        po.panel, po.index_series, annualization=po.annualization_factor,
    ).dropna(axis=1, how="all")
    keep = []
    for _, g in X.groupby(level="ticker"):
        keep.append(g.iloc[:-TEST_TAIL_ROWS] if len(g) > TEST_TAIL_ROWS else g)
    ins = pd.concat(keep)
    ins.to_parquet(CACHE)
    return ins


def pdp_1d(model, sub_vals: np.ndarray, names: list[str], feat: str, grid: np.ndarray) -> np.ndarray:
    fi = names.index(feat)
    out = np.zeros(len(grid))
    for k, gv in enumerate(grid):
        g = sub_vals.copy()
        g[:, fi] = gv
        out[k] = model.predict_proba(g)[:, 1].mean()
    return out


def main() -> None:
    ins = build_insample()
    m_b = CatBoostClassifier(); m_b.load_model(BEFORE)
    m_a = CatBoostClassifier(); m_a.load_model(AFTER)
    names = list(m_b.feature_names_)
    sub = ins.sample(min(SUBSAMPLE, len(ins)), random_state=42)[names].values

    print(f"in-sample rows={len(ins)}  auditing {len(CONSTRAINED)} constrained feats\n")
    print(f"{'feature':<22}{'BEFORE monotone?':>18}{'max_dip':>10}{'AFTER monotone?':>18}")
    print("-" * 70)
    n_already_mono = 0
    for f in CONSTRAINED:
        if f not in names:
            continue
        grid = np.quantile(ins[f].dropna(), np.linspace(0.05, 0.95, GRID))
        cb = pdp_1d(m_b, sub, names, f, grid)
        ca = pdp_1d(m_a, sub, names, f, grid)
        db = np.diff(cb); da = np.diff(ca)
        # max downward step (negative => non-monotone dip) relative to total range
        rng_b = max(cb.max() - cb.min(), 1e-9)
        max_dip_b = float(min(db.min(), 0.0)) / rng_b  # fraction of range
        mono_b = bool(np.all(db >= -1e-4))
        mono_a = bool(np.all(da >= -1e-4))
        n_already_mono += int(mono_b)
        print(f"{f:<22}{('YES' if mono_b else 'no'):>18}{max_dip_b:>10.2%}{('YES' if mono_a else 'no'):>18}")
    print("-" * 70)
    print(f"{n_already_mono}/{len(CONSTRAINED)} constrained features were ALREADY monotone in the unconstrained model.")
    print("(If all/most YES -> the iter-5 constraints were redundant, not information-removing.)")


if __name__ == "__main__":
    main()
