"""Bit-identical parity gate for feature-build optimizations (GBDTPERF).

Builds the feature matrix on the daily ``--since`` slice and either SAVES it as a
baseline or COMPARES against a saved baseline, asserting value-for-value equality
(NaN-aware, exact). This is the correctness gate for every vectorize/parallelize
change to ``src/gbdt/features.py``: capture a baseline on the unmodified code, then
re-run after each edit and require ``max|Δ| == 0`` on every column.

    # capture baseline (run on unmodified code FIRST):
    uv run python -m scripts.gbdt.check_feature_parity --save /tmp/feat_base.parquet --max-tickers 60
    # after an edit, compare:
    uv run python -m scripts.gbdt.check_feature_parity --baseline /tmp/feat_base.parquet --max-tickers 60

``--max-tickers`` subsets the panel (same subset both runs → cross-sectional
features stay consistent) for a fast loop; omit it for the full-slice confirmation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gbdt import features as gbdt_features
from gbdt.data import load_panel


def build(cell: str, since: str, end: str, max_tickers: int | None) -> pd.DataFrame:
    universe = yaml.safe_load((Path(cell) / "spec.yaml").read_text())["target"]["universe"]
    warmup_start = str((pd.Timestamp(since) - pd.Timedelta(days=2700)).date())
    po = load_panel(universe, start=warmup_start, end=end, cache_only=True)
    panel, idx, ann = po.panel, po.index_series, po.annualization_factor
    if max_tickers is not None:
        keep = sorted(panel.index.get_level_values("ticker").unique())[:max_tickers]
        panel = panel[panel.index.get_level_values("ticker").isin(keep)]
    t0 = time.time()
    X = gbdt_features.build_feature_matrix(panel, idx, annualization=ann)
    print(f"[parity] built {X.shape[1]} cols × {len(X):,} rows in {time.time()-t0:.1f}s")
    return X


def compare(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns):
        only_a = set(a.columns) - set(b.columns)
        only_b = set(b.columns) - set(a.columns)
        print(f"[parity] COLUMN MISMATCH — only_baseline={sorted(only_a)[:8]} "
              f"only_new={sorted(only_b)[:8]}")
        return False
    a, b = a.align(b, axis=0)  # align row index
    worst = 0.0
    bad_cols = []
    for c in a.columns:
        av, bv = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(av), np.isnan(bv)):
            bad_cols.append((c, "NaN-structure")); continue
        m = ~np.isnan(av)
        d = np.abs(av[m] - bv[m]).max() if m.any() else 0.0
        worst = max(worst, d)
        if d > 0:
            bad_cols.append((c, f"max|Δ|={d:.3e}"))
    if bad_cols:
        print(f"[parity] DIVERGES on {len(bad_cols)}/{len(a.columns)} cols (worst |Δ|={worst:.3e}):")
        for c, why in bad_cols[:15]:
            print(f"          {c}: {why}")
        return False
    print(f"[parity] ✓ BIT-IDENTICAL — all {len(a.columns)} cols match exactly (max|Δ|=0, NaN-structure identical)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="results/gbdt/experiments/sp500_up_50pct_50d_dd25pct_agentloop")
    ap.add_argument("--since", default="2026-06-15")
    ap.add_argument("--end", default=str(pd.Timestamp.today().date()))
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--save", default=None, help="build + save baseline to this parquet")
    ap.add_argument("--baseline", default=None, help="build + compare against this baseline parquet")
    args = ap.parse_args()

    X = build(args.cell, args.since, args.end, args.max_tickers)
    if args.save:
        X.to_parquet(args.save)
        print(f"[parity] saved baseline → {args.save}")
    elif args.baseline:
        base = pd.read_parquet(args.baseline)
        sys.exit(0 if compare(base, X) else 1)
    else:
        print("[parity] (no --save/--baseline; built only)")


if __name__ == "__main__":
    main()
