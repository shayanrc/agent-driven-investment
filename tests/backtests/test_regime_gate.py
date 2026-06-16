"""Tests for the _017 regime-gate trend mask (scripts.backtests.run_rolling_validation).

The gate is harness-level preprocessing (universe-market logic lives outside the
backend-agnostic strategy, per docs/trading_strategies/goal.md). These tests pin the
causal trend-regime computation: price>SMA, optional MA-rising slope, and zero
look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtests.run_rolling_validation import compute_risk_on


def _series(vals: list[float]) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)


def test_price_above_sma_is_risk_on() -> None:
    # Monotone-rising series: close is always above its trailing SMA → risk-ON.
    s = _series([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0])
    on = compute_risk_on(s, ma=3)
    assert bool(on.iloc[-1]) is True
    assert on.iloc[2:].all()  # every post-warmup day is risk-on


def test_price_below_sma_is_risk_off() -> None:
    # Monotone decline: close is always below its trailing mean → risk-OFF.
    s = _series([200.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0])
    on = compute_risk_on(s, ma=3)
    assert bool(on.iloc[-1]) is False
    assert not on.iloc[2:].any()  # every post-warmup day is risk-off


def test_slope_condition_blocks_falling_ma_cross() -> None:
    # A late up-tick that pokes above the SMA while the SMA is still FALLING
    # (a bear-rally cross): price>SMA alone says ON, but the slope condition
    # must veto it. This is the _017 Jan-2022-style whipsaw guard.
    s = _series([200, 190, 180, 170, 160, 150, 140, 130, 120, 145, 150])
    on_plain = compute_risk_on(s, ma=3, slope=0)
    on_slope = compute_risk_on(s, ma=3, slope=3)
    # The 145/150 up-tick crosses above the trailing SMA on the last day:
    assert bool(on_plain.iloc[-1]) is True
    # ...but the 3-day-ago SMA is still higher (MA falling) → slope vetoes it:
    assert bool(on_slope.iloc[-1]) is False


def test_no_lookahead() -> None:
    # The mask at t must not change when FUTURE values are appended.
    base = [100, 102, 101, 105, 110, 108, 112, 120, 95, 90]
    on_full = compute_risk_on(_series(base), ma=3, slope=2)
    on_trunc = compute_risk_on(_series(base[:7]), ma=3, slope=2)
    # Overlapping region must be identical (NaN-safe compare).
    a = on_full.iloc[:7].to_numpy(dtype=float)
    b = on_trunc.to_numpy(dtype=float)
    assert np.allclose(np.nan_to_num(a, nan=-1.0), np.nan_to_num(b, nan=-1.0))


def test_warmup_is_nan() -> None:
    s = _series([100, 101, 102, 103, 104])
    on = compute_risk_on(s, ma=3)
    assert on.iloc[:2].isna().all()  # first ma-1 rows are warmup NaN
    assert not on.iloc[2:].isna().any()
