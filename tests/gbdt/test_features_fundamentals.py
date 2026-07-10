"""F18 fundamentals feature family — columns, cross-sectional ranks, causality
(C1), and the invariance that the ``"all"`` token (and every existing model) is
unaffected.

Fundamentals vary per (date, ticker), so — unlike the macro family — F18 joins
on the panel index by symbol; these tests plant the leak in the *fundamentals*
input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbdt import features as F
from gbdt.leakage_harness import make_synthetic_panel


def _fund_df(dates, tickers, *, seed=3):
    """Synthetic (date, symbol)-indexed valuation panel: yields + revenue_ttm."""
    rng = np.random.default_rng(seed)
    rows = []
    for tk in tickers:
        ey = 0.03 + np.cumsum(rng.normal(0.0, 0.001, len(dates)))
        rev = 1000.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.002, len(dates))))
        rows.append(pd.DataFrame({
            "date": dates, "symbol": tk,
            "earnings_yield": ey,
            "sales_yield": ey * 4.0,
            "fcf_yield": ey * 0.8,
            "revenue_ttm": rev,
        }))
    return pd.concat(rows).set_index(["date", "symbol"]).sort_index()


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


def test_emits_expected_columns():
    panel = make_synthetic_panel(n_rows=300, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    feat = F.fundamentals_features(_fund_df(dates, _tickers(panel)), panel)

    assert feat.index.equals(panel.index)
    for y in ("earnings_yield", "sales_yield", "fcf_yield"):
        assert f"fund_{y}" in feat.columns
        assert f"fund_{y}_xs_rank" in feat.columns
        assert f"fund_{y}_xs_zscore" in feat.columns
    assert "fund_rev_ttm_yoy" in feat.columns
    assert "fund_rev_ttm_yoy_xs_rank" in feat.columns
    assert "fund_rev_ttm_yoy_xs_zscore" in feat.columns
    assert "fund_earnings_yield_chg_63" in feat.columns
    # F18 now emits 13 columns (3 yields × {level, rank, zscore} + rev_yoy ×
    # {level, rank, zscore} + earnings_yield_chg_63)
    assert sum(c.startswith("fund_") for c in feat.columns) == 13


def test_xs_zscore_is_cross_sectional_standardized():
    panel = make_synthetic_panel(n_rows=200, n_tickers=5)
    dates = panel.index.get_level_values("date").unique()
    feat = F.fundamentals_features(_fund_df(dates, _tickers(panel)), panel)
    d = dates[150]
    z = feat.xs(d, level="date")["fund_earnings_yield_xs_zscore"].dropna()
    assert len(z) == 5
    # per-date z-scores are mean≈0 and (sample) std≈1 across the cross-section
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=1) - 1.0) < 1e-9
    # and they preserve the ordering of the underlying yield
    y = feat.xs(d, level="date")["fund_earnings_yield"].dropna()
    assert (z.rank().values == y.rank().values).all()


def test_xs_rank_is_cross_sectional_percentile():
    panel = make_synthetic_panel(n_rows=200, n_tickers=4)
    dates = panel.index.get_level_values("date").unique()
    feat = F.fundamentals_features(_fund_df(dates, _tickers(panel)), panel)
    # on any date the xs_rank is a percentile in (0, 1], one per ticker
    d = dates[150]
    ranks = feat.xs(d, level="date")["fund_earnings_yield_xs_rank"].dropna()
    assert len(ranks) == 4
    assert ranks.min() > 0.0 and ranks.max() <= 1.0
    assert ranks.is_unique  # 4 distinct tickers → 4 distinct ranks


def test_level_join_matches_source():
    panel = make_synthetic_panel(n_rows=150, n_tickers=2)
    dates = panel.index.get_level_values("date").unique()
    fdf = _fund_df(dates, _tickers(panel))
    feat = F.fundamentals_features(fdf, panel)
    tk = _tickers(panel)[0]
    d = dates[100]
    assert feat.loc[(d, tk), "fund_earnings_yield"] == fdf.loc[(d, tk), "earnings_yield"]


def test_fundamentals_features_are_causal():
    # Perturbing FUTURE fundamentals must not move any pre-leak row.
    panel = make_synthetic_panel(n_rows=300, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    tickers = _tickers(panel)
    base = F.fundamentals_features(_fund_df(dates, tickers), panel)

    leak_row = 200
    leak_date = dates[leak_row]
    fdf = _fund_df(dates, tickers)
    future = fdf.index.get_level_values("date") >= leak_date
    fdf.loc[future, :] *= 5.0  # spike all future fundamentals
    leaked = F.fundamentals_features(fdf, panel)

    pre = panel.index.get_level_values("date") < leak_date
    b, l = base[pre], leaked[pre]
    both_nan = b.isna() & l.isna()
    diff = (l.fillna(0.0) - b.fillna(0.0)).abs().mask(both_nan, 0.0)
    assert float(diff.values.max()) <= 1e-9


def test_all_token_excludes_f18_but_all_fundamentals_includes_it():
    panel = make_synthetic_panel(n_rows=260, n_tickers=3)
    dates = panel.index.get_level_values("date").unique()
    idx = _index_df(dates)
    fdf = _fund_df(dates, _tickers(panel))

    X_all = F.build_feature_matrix(panel, idx, families="all")
    X_fund = F.build_feature_matrix(
        panel, idx, families="all_fundamentals", fund_df=fdf)

    # "all" carries NO fundamentals columns
    assert not any(c.startswith("fund_") for c in X_all.columns)
    # "all_fundamentals" adds them
    assert any(c.startswith("fund_") for c in X_fund.columns)
    # and leaves every baseline column byte-identical (the invariance lock)
    pd.testing.assert_frame_equal(X_fund[X_all.columns], X_all)


def test_all_fundamentals_requires_fund_df():
    panel = make_synthetic_panel(n_rows=200, n_tickers=2)
    dates = panel.index.get_level_values("date").unique()
    idx = _index_df(dates)
    try:
        F.build_feature_matrix(panel, idx, families="all_fundamentals")
        assert False, "expected ValueError when fund_df is None"
    except ValueError as e:
        assert "fund_df" in str(e)


# --- F18-IN: NSE valuation-panel routing (task #27) -------------------------

def test_nse_universe_routes_to_nse_panel_path(tmp_path):
    """The runner picks VALUATION_PANEL_NSE_PATH for NSE-calendar universes and
    the US default (None → VALUATION_PANEL_PATH) otherwise. Guards the routing
    predicate + that load_fundamentals_panel honors an explicit path."""
    import pandas as pd
    from gbdt import data as gbdt_data
    from gbdt.universe_calendar import resolve_calendar_name

    # routing predicate
    assert resolve_calendar_name("nifty500", None) == "NSE"
    assert resolve_calendar_name("nifty50", None) == "NSE"
    assert resolve_calendar_name("sp500", None) == "NYSE"
    assert resolve_calendar_name("russell1000", None) == "NYSE"

    # load_fundamentals_panel honors an explicit path (the NSE parquet)
    nse_dir = tmp_path / "results" / "valuation" / "data"
    nse_dir.mkdir(parents=True)
    panel = pd.DataFrame({
        "ticker": ["INFUND:RELIANCE", "INFUND:TCS"],
        "date": pd.to_datetime(["2024-06-28", "2024-06-28"]),
        "earnings_yield": [0.04, 0.05], "sales_yield": [0.5, 0.3],
        "fcf_yield": [float("nan"), float("nan")],  # all-NaN in India
        "revenue_ttm": [900000.0, 240000.0], "revenue_q": [230000.0, 63000.0],
    })
    panel.to_parquet(nse_dir / "valuation_panel_nse.parquet")
    out = gbdt_data.load_fundamentals_panel(
        None, None, repo_root=tmp_path,
        path=gbdt_data.VALUATION_PANEL_NSE_PATH,
    )
    assert set(out.index.get_level_values("symbol")) == {"RELIANCE", "TCS"}
    assert out["fcf_yield"].isna().all()  # India: no fcf
