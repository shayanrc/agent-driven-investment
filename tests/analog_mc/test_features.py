"""Causality tests for analog_mc.features (constraint C1).

The single most important test in the whole pipeline: verify that for every
index t, the value of a feature computed from the FULL series equals the value
of the same feature computed from series[: t + 1] alone. If this fails, the
feature is leaking future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analog_mc.features import (
    causal_ewma_vol, causal_trailing_mean, causal_zscore, compute_features,
)


@pytest.fixture
def returns_series() -> pd.Series:
    rng = np.random.default_rng(123)
    # A regime-switching series: low vol then high vol. Exposes any leakage of
    # later (high-vol) information into earlier indices.
    n = 400
    low = rng.normal(loc=0.0005, scale=0.005, size=n // 2)
    high = rng.normal(loc=-0.001, scale=0.02, size=n // 2)
    vals = np.concatenate([low, high])
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(vals, index=idx, name="log_return")


@pytest.mark.parametrize("t", [50, 100, 150, 199, 200, 201, 300, 399])
def test_causal_ewma_vol_no_lookahead(returns_series: pd.Series, t: int) -> None:
    """EWMA vol at index t must match the value computed from returns[: t + 1]."""
    full = causal_ewma_vol(returns_series, halflife=20.0)
    truncated = causal_ewma_vol(returns_series.iloc[: t + 1], halflife=20.0)
    assert not np.isnan(full.iloc[t])
    assert full.iloc[t] == pytest.approx(truncated.iloc[t], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("horizon", [20, 50, 200])
@pytest.mark.parametrize("t", [100, 199, 200, 300, 399])
def test_causal_zscore_no_lookahead(returns_series: pd.Series, horizon: int, t: int) -> None:
    """z-score at index t must match the value computed from returns[: t + 1]."""
    if t < horizon - 1:
        pytest.skip("not enough history for this horizon")
    full = causal_zscore(returns_series, horizon=horizon)
    truncated = causal_zscore(returns_series.iloc[: t + 1], horizon=horizon)
    assert not np.isnan(full.iloc[t])
    assert full.iloc[t] == pytest.approx(truncated.iloc[t], rel=1e-12, abs=1e-12)


def test_causal_zscore_matches_hand_computation(returns_series: pd.Series) -> None:
    """At index 100 with horizon 20, the z-score must equal the hand-computed
    mean/std over the 20 returns ending at index 100 inclusive (indices 81..100).
    """
    horizon = 20
    t = 100
    z = causal_zscore(returns_series, horizon=horizon).iloc[t]
    window = returns_series.iloc[t - horizon + 1 : t + 1].to_numpy()
    expected = window.mean() / window.std(ddof=1)
    assert z == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_causal_zscore_early_indices_are_nan(returns_series: pd.Series) -> None:
    z = causal_zscore(returns_series, horizon=20)
    # First horizon - 1 values lack enough history.
    assert z.iloc[:19].isna().all()
    assert not np.isnan(z.iloc[19])


def test_ewma_vol_early_index_is_nan(returns_series: pd.Series) -> None:
    v = causal_ewma_vol(returns_series, halflife=20.0)
    # ewm().std() needs >=2 observations.
    assert np.isnan(v.iloc[0])
    assert not np.isnan(v.iloc[1])


def test_compute_features_columns_and_causality(returns_series: pd.Series) -> None:
    horizons = (20, 50, 200)
    df = compute_features(returns_series, halflife=20.0, horizons=horizons)
    assert list(df.columns) == [
        "ewma_vol", "zscore_20", "zscore_50", "zscore_200", "trailing_mean_200",
    ]
    # Causality of the whole bundle at a representative index.
    t = 250
    truncated = compute_features(returns_series.iloc[: t + 1], halflife=20.0, horizons=horizons)
    for col in df.columns:
        assert df[col].iloc[t] == pytest.approx(truncated[col].iloc[t], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("horizon", [20, 50, 200])
@pytest.mark.parametrize("t", [100, 199, 200, 300, 399])
def test_causal_trailing_mean_no_lookahead(returns_series: pd.Series, horizon: int, t: int) -> None:
    """Trailing mean at index t must match the value computed from returns[: t + 1]."""
    if t < horizon - 1:
        pytest.skip("not enough history for this horizon")
    full = causal_trailing_mean(returns_series, horizon=horizon)
    truncated = causal_trailing_mean(returns_series.iloc[: t + 1], horizon=horizon)
    assert not np.isnan(full.iloc[t])
    assert full.iloc[t] == pytest.approx(truncated.iloc[t], rel=1e-12, abs=1e-12)


def test_compute_features_adds_momentum_column_when_requested(returns_series: pd.Series) -> None:
    """v2.1: passing momentum_lookback adds a second trailing-mean column,
    distinct from the mu_origin long-horizon column."""
    horizons = (20, 50, 200)
    df = compute_features(returns_series, halflife=20.0, horizons=horizons, momentum_lookback=20)
    assert "trailing_mean_200" in df.columns  # mu_origin (C3)
    assert "trailing_mean_20" in df.columns  # momentum source (v2.1)
    # Causality at a representative index.
    t = 250
    truncated = compute_features(
        returns_series.iloc[: t + 1], halflife=20.0, horizons=horizons, momentum_lookback=20,
    )
    assert df["trailing_mean_20"].iloc[t] == pytest.approx(
        truncated["trailing_mean_20"].iloc[t], rel=1e-12, abs=1e-12,
    )


def test_compute_features_skips_momentum_column_when_redundant(returns_series: pd.Series) -> None:
    """If momentum_lookback equals max(horizons), the column already exists as
    mu_origin; we should not duplicate it."""
    horizons = (20, 50, 200)
    df = compute_features(returns_series, halflife=20.0, horizons=horizons, momentum_lookback=200)
    # Exactly one trailing_mean column.
    tm_cols = [c for c in df.columns if c.startswith("trailing_mean_")]
    assert tm_cols == ["trailing_mean_200"]


def test_compute_features_no_momentum_column_when_none(returns_series: pd.Series) -> None:
    """v1 default: momentum_lookback=None means no extra column."""
    df = compute_features(returns_series, halflife=20.0, horizons=(20, 50, 200))
    assert [c for c in df.columns if c.startswith("trailing_mean_")] == ["trailing_mean_200"]


def test_causal_trailing_mean_matches_hand_computation(returns_series: pd.Series) -> None:
    """At index 100 with horizon 20, mean must equal mean of returns[81..100]."""
    horizon = 20
    t = 100
    m = causal_trailing_mean(returns_series, horizon=horizon).iloc[t]
    expected = returns_series.iloc[t - horizon + 1 : t + 1].mean()
    assert m == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_causal_trailing_mean_uses_same_window_as_zscore(returns_series: pd.Series) -> None:
    """The mean used here must equal the numerator of causal_zscore at the same horizon."""
    horizon = 50
    m = causal_trailing_mean(returns_series, horizon=horizon)
    # z * std should equal mean for the same window.
    z = causal_zscore(returns_series, horizon=horizon)
    std = returns_series.rolling(window=horizon, min_periods=horizon).std(ddof=1)
    reconstructed_mean = z * std
    # Compare at indices where both are defined.
    mask = ~m.isna() & ~reconstructed_mean.isna()
    np.testing.assert_allclose(m[mask], reconstructed_mean[mask], rtol=1e-12, atol=1e-12)


def test_causal_zscore_handles_constant_window() -> None:
    """A perfectly flat window has std=0; z-score should be NaN, not inf."""
    s = pd.Series([0.01] * 30, index=pd.date_range("2024-01-01", periods=30, freq="B"))
    z = causal_zscore(s, horizon=10)
    assert z.iloc[15:].isna().all()


def test_causal_ewma_vol_rejects_invalid_halflife() -> None:
    s = pd.Series([0.0, 0.01, -0.005])
    with pytest.raises(ValueError):
        causal_ewma_vol(s, halflife=0.0)
    with pytest.raises(ValueError):
        causal_ewma_vol(s, halflife=-1.0)


def test_causal_zscore_rejects_short_horizon() -> None:
    s = pd.Series([0.0, 0.01, -0.005])
    with pytest.raises(ValueError):
        causal_zscore(s, horizon=1)
