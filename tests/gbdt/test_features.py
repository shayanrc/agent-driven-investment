"""Stage 2 — feature pipeline tests.

Per-family causality (via the leakage harness from Stage 1), hand-computed
spot checks, and the headline 279-column-count assertion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt import features as F
from gbdt.leakage_harness import make_synthetic_panel, LeakageHarness


def _synth_panel_with_index(n_rows: int = 300, n_tickers: int = 3, seed: int = 0):
    panel = make_synthetic_panel(n_rows, n_tickers, seed=seed)
    # The index has its own random walk so it's not perfectly correlated.
    rng = np.random.default_rng(seed + 99)
    dates = panel.index.get_level_values("date").unique()
    rets = rng.normal(0, 0.008, size=len(dates))
    close = 1000.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0.0, 0.005, len(dates)))
    low = close * (1.0 - rng.uniform(0.0, 0.005, len(dates)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, len(dates)))
    index_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": np.ones(len(dates))},
        index=pd.Index(dates, name="date"),
    )
    return panel, index_df


# ---------------------------------------------------------------------------
# Hand-computed spot checks
# ---------------------------------------------------------------------------


def test_stock_return_5_matches_hand_formula():
    panel, _ = _synth_panel_with_index(50, 1, seed=1)
    sr = F.stock_return_N(panel, lookbacks=(5,))
    close = panel["close"]
    # At row 10, expect close[10]/close[5] - 1
    t = panel.index[10]
    t_minus_5 = panel.index[5]
    expected = close.loc[t] / close.loc[t_minus_5] - 1.0
    assert sr.loc[t, "stock_return_5"] == pytest.approx(expected, rel=1e-9)


def test_realized_vol_annualization():
    panel, _ = _synth_panel_with_index(60, 1, seed=2)
    rv = F.realized_vol_N(panel, lookbacks=(10,), annualization=250)
    # Last row's value should equal np.std of last 10 log returns * sqrt(250)
    close = panel["close"].droplevel("ticker")
    rets = np.log(close).diff()
    expected = rets.iloc[-10:].std() * np.sqrt(250)
    actual = rv["realized_vol_10"].iloc[-1]
    assert actual == pytest.approx(expected, rel=1e-9)


def test_drawdown_uses_high_not_close():
    panel, _ = _synth_panel_with_index(40, 1, seed=3)
    dd = F.drawdown_N(panel, lookbacks=(5,))
    close = panel["close"]
    high = panel["high"]
    t = panel.index[20]
    # Find the rolling max over rows [16..20] of HIGH
    window_idx = panel.index[16:21]
    expected = close.loc[t] / high.loc[window_idx].max() - 1.0
    assert dd.loc[t, "drawdown_5"] == pytest.approx(expected, rel=1e-9)


def test_sma_distance_matches_formula():
    panel, _ = _synth_panel_with_index(30, 1, seed=4)
    sd = F.sma_distance_N(panel, lookbacks=(5,))
    close = panel["close"]
    t = panel.index[15]
    window_idx = panel.index[11:16]
    expected = close.loc[t] / close.loc[window_idx].mean() - 1.0
    assert sd.loc[t, "sma_distance_5"] == pytest.approx(expected, rel=1e-9)


def test_signed_days_outside_band_streaks():
    # Construct a z series with known streak pattern
    import numpy as np
    z = pd.Series(
        [0.0, 0.5, 2.1, 2.2, 2.3, 0.4, -2.1, -2.2, 0.0, 2.5],
        index=pd.RangeIndex(10),
    )
    out = F._signed_days_outside_band_one(z.values, sigma=2.0)
    assert list(out) == [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, -1.0, -2.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Causality (leakage harness)
# ---------------------------------------------------------------------------


def _make_index_for(panel: pd.DataFrame) -> pd.DataFrame:
    """Stable index DataFrame for the harness panel — index_df can't depend on
    the panel's leaked row (it's not in the panel's ticker set)."""
    dates = panel.index.get_level_values("date").unique()
    close = np.linspace(100.0, 150.0, len(dates))
    high = close * 1.001
    low = close * 0.999
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": np.ones(len(dates))},
        index=pd.Index(dates, name="date"),
    )


def _check_family(family_fn, *args, name: str):
    h = LeakageHarness(n_rows=120, n_tickers=2, leak_row=80)
    def wrapper(panel):
        return family_fn(panel, *args)
    report = h.check(wrapper)
    assert report.causal, f"{name} leaks: {report}"


def test_causality_stock_return():
    _check_family(F.stock_return_N, (5, 20), name="stock_return_N")


def test_causality_realized_vol():
    _check_family(F.realized_vol_N, (5, 20), 250, name="realized_vol_N")


def test_causality_drawdown_runup():
    _check_family(F.drawdown_N, (5, 20), name="drawdown_N")
    _check_family(F.runup_N, (5, 20), name="runup_N")


def test_causality_higher_moments():
    _check_family(F.higher_moments, (10,), name="higher_moments")


def test_causality_sma_distance():
    _check_family(F.sma_distance_N, (5, 20), name="sma_distance_N")


def test_causality_vol_regime():
    _check_family(F.vol_regime, (5, 20), 250, name="vol_regime")


def test_causality_volume_family():
    _check_family(F.volume_family, (5, 20), name="volume_family")


def test_causality_range_vol():
    _check_family(F.range_vol, (5, 20), 250, name="range_vol")


def test_causality_cross_sectional_rank_z():
    _check_family(F.cross_sectional_rank_z, (5, 20), 250, name="cross_sectional_rank_z")


def test_causality_f16_underlying():
    _check_family(F.f16_underlying, (5, 20), 250, name="f16_underlying")


def test_causality_calendar():
    # Calendar is purely date-driven and trivially causal, but verify anyway.
    h = LeakageHarness(n_rows=120, n_tickers=2, leak_row=80)
    report = h.check(F.calendar_features)
    assert report.causal


def test_causality_signed_days_band_meta():
    """The band-meta layer composes z-scored underlyings; harness end-to-end."""
    h = LeakageHarness(n_rows=120, n_tickers=2, leak_row=80)
    def pipe(panel):
        underlyings = F.f16_meta_underlying_columns(panel, lookbacks=(5, 20), annualization=250)
        return F.signed_days_outside_band_meta(underlyings, sigmas=(1.0, 2.0))
    report = h.check(pipe)
    assert report.causal, f"band-meta leak: {report}"


# ---------------------------------------------------------------------------
# Cross-sectional shape sanity
# ---------------------------------------------------------------------------


def test_cross_sectional_rank_z_per_date_stats():
    panel, _ = _synth_panel_with_index(60, 4, seed=5)
    xs = F.cross_sectional_rank_z(panel, lookbacks=(5,), annualization=250)
    # Pick a date well past the lookback so ranks are populated
    a_date = panel.index.get_level_values("date").unique()[20]
    col = "return_xs_rank_5"
    vals = xs.xs(a_date, level="date")[col].dropna()
    assert len(vals) > 0
    # Ranks (pct=True) should be in (0, 1]
    assert vals.min() > 0.0
    assert vals.max() <= 1.0


# ---------------------------------------------------------------------------
# Total column count
# ---------------------------------------------------------------------------


def test_build_feature_matrix_total_col_count():
    panel, idx = _synth_panel_with_index(220, 3, seed=6)
    mat = F.build_feature_matrix(panel, idx, lookbacks=F.DEFAULT_LOOKBACKS,
                                  annualization=250, families="all")
    assert mat.shape[1] == F.EXPECTED_TOTAL_COLS, (
        f"expected {F.EXPECTED_TOTAL_COLS} cols; got {mat.shape[1]}: "
        f"{sorted(mat.columns.tolist())[:20]}..."
    )


def test_build_feature_matrix_exclude_glob():
    panel, idx = _synth_panel_with_index(220, 3, seed=7)
    mat = F.build_feature_matrix(panel, idx, lookbacks=F.DEFAULT_LOOKBACKS,
                                  annualization=250, exclude=["volume_ratio_*"])
    assert F.EXPECTED_TOTAL_COLS - mat.shape[1] == 6, (
        "expected 6 fewer cols after excluding volume_ratio_*"
    )
    assert not any(c.startswith("volume_ratio_") for c in mat.columns)


def test_build_feature_matrix_subset_family():
    panel, idx = _synth_panel_with_index(120, 3, seed=8)
    mat = F.build_feature_matrix(panel, idx, lookbacks=(5, 10), families=["F2"])
    # F2 alone = 2 lookbacks → 2 cols
    assert mat.shape[1] == 2
    assert "stock_return_5" in mat.columns
    assert "stock_return_10" in mat.columns
