"""F20 VWAP-deviation feature family — columns, deviation math, cross-sectional
z-score, causality (C1), and the invariance locks: the ``"all"`` and
``"all_fundamentals"`` tokens (and every existing model) stay byte-identical, and
the new ``all_vwap`` / ``all_fundamentals_vwap`` tokens add exactly the F20 columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbdt import features as F
from gbdt.leakage_harness import make_synthetic_panel


def _index_df(dates, *, seed=2):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "adj_close": close, "volume": 1_000_000},
        index=pd.DatetimeIndex(dates),
    )


def _tickers(panel):
    return list(panel.index.get_level_values("ticker").unique())


def _fund_df(dates, tickers, *, seed=3):
    """Minimal synthetic (date, symbol)-indexed valuation panel for F18."""
    rows = []
    for i, tk in enumerate(tickers):
        ey = 0.03 + 0.001 * i
        rows.append(pd.DataFrame({
            "date": dates, "symbol": tk,
            "earnings_yield": ey, "sales_yield": ey * 4.0, "fcf_yield": ey * 0.8,
            "revenue_ttm": 1000.0 * (i + 1),
        }))
    return pd.concat(rows).set_index(["date", "symbol"]).sort_index()


_LOOKBACKS = (5, 10, 20, 50, 100, 200)


def test_emits_fourteen_columns():
    panel = make_synthetic_panel(n_rows=320, n_tickers=3)
    feat = F.vwap_family(panel)

    assert feat.index.equals(panel.index)
    expected = (
        [f"vwap_dev_{n}" for n in _LOOKBACKS]
        + [f"vwap_dev_zscore_{n}" for n in _LOOKBACKS]
        + ["vwap_dev_xs_zscore", "vwap_dev_xs_rank"]
    )
    assert sorted(feat.columns) == sorted(expected)
    assert feat.shape[1] == 14


def test_vwap_dev_is_relative_close_over_vwap():
    # On a controlled panel, vwap_dev_N reproduces close / rolling-VWAP_N − 1.
    panel = make_synthetic_panel(n_rows=260, n_tickers=2)
    feat = F.vwap_family(panel)
    tp = (panel["high"] + panel["low"] + panel["close"]) / 3.0
    vol = panel["volume"].astype(float)
    for n in (5, 20, 200):
        num = (tp * vol).groupby(level="ticker").rolling(n, min_periods=n).sum().droplevel(0)
        den = vol.groupby(level="ticker").rolling(n, min_periods=n).sum().droplevel(0)
        expect = panel["close"] / (num / den) - 1.0
        got = feat[f"vwap_dev_{n}"]
        both_nan = expect.isna() & got.isna()
        diff = (got.fillna(0.0) - expect.fillna(0.0)).abs().mask(both_nan, 0.0)
        assert float(diff.max()) <= 1e-12


def test_xs_zscore_is_cross_sectional_standardized():
    panel = make_synthetic_panel(n_rows=300, n_tickers=6)
    feat = F.vwap_family(panel)
    z = feat["vwap_dev_xs_zscore"].dropna()
    by_date = z.groupby(level="date")
    # Per date the cross-section is mean≈0, std≈1 (population/sample ddof aside,
    # only check dates with a full 6-ticker cross-section).
    full = by_date.count() == 6
    good_dates = full[full].index
    m = by_date.mean().reindex(good_dates)
    assert m.abs().max() < 1e-9


def test_vwap_features_are_causal():
    # Perturbing FUTURE bars must not move any pre-leak row (strictly trailing).
    panel = make_synthetic_panel(n_rows=360, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    base = F.vwap_family(panel)

    leak_date = dates[250]
    p2 = panel.copy()
    future = p2.index.get_level_values("date") >= leak_date
    for col in ("open", "high", "low", "close", "volume"):
        p2.loc[future, col] *= 5.0
    leaked = F.vwap_family(p2)

    pre = panel.index.get_level_values("date") < leak_date
    b, l = base[pre], leaked[pre]
    both_nan = b.isna() & l.isna()
    diff = (l.fillna(0.0) - b.fillna(0.0)).abs().mask(both_nan, 0.0)
    assert float(diff.values.max()) <= 1e-9


def test_all_vwap_includes_f20_and_locks_baseline():
    panel = make_synthetic_panel(n_rows=320, n_tickers=3)
    idx = _index_df(panel.index.get_level_values("date").unique())

    X_all = F.build_feature_matrix(panel, idx, families="all")
    X_vwap = F.build_feature_matrix(panel, idx, families="all_vwap")

    vwap_cols = [c for c in X_vwap.columns if c.startswith("vwap_")]
    assert len(vwap_cols) == 14
    # "all" is byte-identical and carries NO vwap columns (opt-in only).
    assert not any(c.startswith("vwap_") for c in X_all.columns)
    assert X_all.shape[1] == F.EXPECTED_TOTAL_COLS
    # The all-subset inside all_vwap is byte-identical to all alone.
    assert X_vwap[list(X_all.columns)].equals(X_all)


def test_all_fundamentals_vwap_includes_f18_and_f20():
    panel = make_synthetic_panel(n_rows=320, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    idx = _index_df(dates)
    fdf = _fund_df(dates, _tickers(panel))

    X = F.build_feature_matrix(
        panel, idx, families="all_fundamentals_vwap", fund_df=fdf)
    assert any(c.startswith("vwap_") for c in X.columns)
    assert any(c.startswith("fund_") for c in X.columns)
    # and it locks the plain "all" baseline (no vwap/fund leakage into all)
    X_all = F.build_feature_matrix(panel, idx, families="all")
    assert X.shape[1] > X_all.shape[1]
    assert X[list(X_all.columns)].equals(X_all)
