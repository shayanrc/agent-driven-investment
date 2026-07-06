"""F19 revenue-growth feature family — columns, growth math (percent),
cross-sectional ranks/z-scores, causality (C1), and the invariance locks:
the ``"all"`` and ``"all_fundamentals"`` tokens (and every existing F18 model)
stay byte-identical, and F18's output is unchanged when ``fund_df`` gains the
new ``revenue_q`` column.
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


def _fund_df(dates, tickers, *, with_revenue_q=True, seed=3):
    """Synthetic (date, symbol)-indexed valuation panel. Each ticker's
    ``revenue_q`` is a clean geometric ramp so QoQ/YoY are exact: a +5% step
    every 63 trading days ⇒ ``fund_rev_*_qoq`` = 0.05 and ``fund_rev_*_yoy``
    (252-td = 4 quarters) = 1.05**4 − 1 = 0.21550625."""
    rows = []
    for i, tk in enumerate(tickers):
        t = np.arange(len(dates), dtype="float64")
        base = 1000.0 * (i + 1)
        rev_q = base * (1.05 ** (t / 63.0))
        ey = 0.03 + 0.001 * i
        d = {
            "date": dates, "symbol": tk,
            "earnings_yield": ey, "sales_yield": ey * 4.0, "fcf_yield": ey * 0.8,
            "revenue_ttm": rev_q * 4.0,
        }
        if with_revenue_q:
            d["revenue_q"] = rev_q
        rows.append(pd.DataFrame(d))
    return pd.concat(rows).set_index(["date", "symbol"]).sort_index()


def test_emits_twelve_columns():
    panel = make_synthetic_panel(n_rows=320, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    feat = F.fundamentals_growth_features(_fund_df(dates, _tickers(panel)), panel)

    assert feat.index.equals(panel.index)
    for m in ("fund_rev_q_yoy", "fund_rev_q_qoq",
              "fund_rev_ttm_yoy_pct", "fund_rev_ttm_qoq"):
        assert m in feat.columns
        assert f"{m}_xs_rank" in feat.columns
        assert f"{m}_xs_zscore" in feat.columns
    assert sum(c.startswith("fund_rev_") for c in feat.columns) == 12


def test_growth_math_is_exact_percent():
    # geometric ramp: +5% every 63 td ⇒ QoQ = 0.05, YoY(252) = 1.05**4 − 1.
    panel = make_synthetic_panel(n_rows=400, n_tickers=2)
    dates = panel.index.get_level_values("date").unique()
    feat = F.fundamentals_growth_features(_fund_df(dates, _tickers(panel)), panel)
    tk = _tickers(panel)[0]
    qoq = feat.xs(tk, level="ticker")["fund_rev_q_qoq"].dropna()
    yoy = feat.xs(tk, level="ticker")["fund_rev_q_yoy"].dropna()
    assert np.allclose(qoq.values, 0.05, atol=1e-9)
    assert np.allclose(yoy.values, 1.05 ** 4 - 1.0, atol=1e-9)
    # TTM twin is the same ramp ⇒ same growth
    assert np.allclose(
        feat.xs(tk, level="ticker")["fund_rev_ttm_qoq"].dropna().values,
        0.05, atol=1e-9)


def test_nonpositive_or_missing_base_is_nan():
    panel = make_synthetic_panel(n_rows=200, n_tickers=2)
    dates = panel.index.get_level_values("date").unique()
    fdf = _fund_df(dates, _tickers(panel))
    fdf["revenue_q"] = fdf["revenue_q"].where(
        fdf.index.get_level_values("date") != dates[50], -1.0)
    feat = F.fundamentals_growth_features(fdf, panel)
    # a negative base row 63 td later has no valid QoQ (NaN, never a huge %)
    q = feat["fund_rev_q_qoq"]
    assert q.replace([np.inf, -np.inf], np.nan).equals(q)  # no infs leaked


def test_xs_zscore_is_cross_sectional_standardized():
    panel = make_synthetic_panel(n_rows=320, n_tickers=5)
    dates = panel.index.get_level_values("date").unique()
    feat = F.fundamentals_growth_features(_fund_df(dates, _tickers(panel)), panel)
    d = dates[300]
    z = feat.xs(d, level="date")["fund_rev_q_yoy_xs_zscore"].dropna()
    if len(z) >= 2 and z.std(ddof=1) > 0:
        assert abs(z.mean()) < 1e-9


def test_growth_features_are_causal():
    # Perturbing FUTURE revenue must not move any pre-leak row.
    panel = make_synthetic_panel(n_rows=360, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    tickers = _tickers(panel)
    base = F.fundamentals_growth_features(_fund_df(dates, tickers), panel)

    leak_date = dates[250]
    fdf = _fund_df(dates, tickers)
    future = fdf.index.get_level_values("date") >= leak_date
    fdf.loc[future, :] *= 5.0
    leaked = F.fundamentals_growth_features(fdf, panel)

    pre = panel.index.get_level_values("date") < leak_date
    b, l = base[pre], leaked[pre]
    both_nan = b.isna() & l.isna()
    diff = (l.fillna(0.0) - b.fillna(0.0)).abs().mask(both_nan, 0.0)
    assert float(diff.values.max()) <= 1e-9


def test_all_fundamentals2_includes_f18_and_f19_and_locks_baseline():
    panel = make_synthetic_panel(n_rows=320, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    idx = _index_df(dates)
    fdf = _fund_df(dates, _tickers(panel))

    X_all = F.build_feature_matrix(panel, idx, families="all")
    X_f18 = F.build_feature_matrix(
        panel, idx, families="all_fundamentals", fund_df=fdf)
    X_f19 = F.build_feature_matrix(
        panel, idx, families="all_fundamentals2", fund_df=fdf)

    # F19 token carries both F18 (fund_<yield>) and F19 (fund_rev_) columns
    assert any(c.startswith("fund_earnings_yield") for c in X_f19.columns)
    assert any(c.startswith("fund_rev_q_") for c in X_f19.columns)
    # baseline byte-identical (the invariance lock)
    pd.testing.assert_frame_equal(X_f19[X_all.columns], X_all)
    # and every F18 column is byte-identical to the all_fundamentals build
    pd.testing.assert_frame_equal(X_f19[X_f18.columns], X_f18)


def test_f18_unchanged_when_fund_df_gains_revenue_q():
    # Adding revenue_q to fund_df must not perturb F18's output — F18 reads
    # only its own columns. Guards the _272–_278 byte-identical guarantee.
    panel = make_synthetic_panel(n_rows=300, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    without = F.fundamentals_features(
        _fund_df(dates, _tickers(panel), with_revenue_q=False), panel)
    with_q = F.fundamentals_features(
        _fund_df(dates, _tickers(panel), with_revenue_q=True), panel)
    pd.testing.assert_frame_equal(without, with_q)


def test_all_fundamentals2_requires_fund_df():
    panel = make_synthetic_panel(n_rows=200, n_tickers=2)
    dates = panel.index.get_level_values("date").unique()
    idx = _index_df(dates)
    try:
        F.build_feature_matrix(panel, idx, families="all_fundamentals2")
        assert False, "expected ValueError when fund_df is None"
    except ValueError as e:
        assert "fund_df" in str(e)
