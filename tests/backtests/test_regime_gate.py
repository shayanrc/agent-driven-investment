"""Tests for the _017 regime-gate trend mask (scripts.backtests.run_rolling_validation).

The gate is harness-level preprocessing (universe-market logic lives outside the
backend-agnostic strategy, per docs/trading_strategies/goal.md). These tests pin the
causal trend-regime computation: price>SMA, optional MA-rising slope, and zero
look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtests.regime_signals import (
    compute_risk_on,
    risk_on_breadth,
    risk_on_drawdown,
    risk_on_sma,
    risk_on_vol,
)

# back-compat alias: the _017 trend tests below were written against the SMA signal
compute_risk_on_sma = risk_on_sma


def _series(vals: list[float]) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)


def test_price_above_sma_is_risk_on() -> None:
    # Monotone-rising series: close is always above its trailing SMA → risk-ON.
    s = _series([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0])
    on = risk_on_sma(s, ma=3)
    assert bool(on.iloc[-1]) is True
    assert on.iloc[2:].all()  # every post-warmup day is risk-on


def test_price_below_sma_is_risk_off() -> None:
    # Monotone decline: close is always below its trailing mean → risk-OFF.
    s = _series([200.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0])
    on = risk_on_sma(s, ma=3)
    assert bool(on.iloc[-1]) is False
    assert not on.iloc[2:].any()  # every post-warmup day is risk-off


def test_slope_condition_blocks_falling_ma_cross() -> None:
    # A late up-tick that pokes above the SMA while the SMA is still FALLING
    # (a bear-rally cross): price>SMA alone says ON, but the slope condition
    # must veto it. This is the _017 Jan-2022-style whipsaw guard.
    s = _series([200, 190, 180, 170, 160, 150, 140, 130, 120, 145, 150])
    on_plain = risk_on_sma(s, ma=3, slope=0)
    on_slope = risk_on_sma(s, ma=3, slope=3)
    # The 145/150 up-tick crosses above the trailing SMA on the last day:
    assert bool(on_plain.iloc[-1]) is True
    # ...but the 3-day-ago SMA is still higher (MA falling) → slope vetoes it:
    assert bool(on_slope.iloc[-1]) is False


def test_no_lookahead() -> None:
    # The mask at t must not change when FUTURE values are appended.
    base = [100, 102, 101, 105, 110, 108, 112, 120, 95, 90]
    on_full = risk_on_sma(_series(base), ma=3, slope=2)
    on_trunc = risk_on_sma(_series(base[:7]), ma=3, slope=2)
    # Overlapping region must be identical (NaN-safe compare).
    a = on_full.iloc[:7].to_numpy(dtype=float)
    b = on_trunc.to_numpy(dtype=float)
    assert np.allclose(np.nan_to_num(a, nan=-1.0), np.nan_to_num(b, nan=-1.0))


def test_warmup_is_nan() -> None:
    s = _series([100, 101, 102, 103, 104])
    on = risk_on_sma(s, ma=3)
    assert on.iloc[:2].isna().all()  # first ma-1 rows are warmup NaN
    assert not on.iloc[2:].isna().any()


# ---- _018 forward-looking signals ----

def test_vol_gate_off_when_vol_high() -> None:
    # Calm series (tiny wiggles) → low realized vol → risk-ON; then a violent
    # whipsaw → vol spikes → risk-OFF.
    calm = [100 + 0.1 * (i % 2) for i in range(30)]
    wild = [100, 130, 80, 140, 70, 150, 60]
    on = risk_on_vol(_series(calm + wild), window=5, thresh=0.20)
    assert bool(on.iloc[28]) is True       # calm region: vol below threshold
    assert bool(on.iloc[-1]) is False      # wild tail: vol above threshold


def test_drawdown_gate_off_below_high() -> None:
    # Rise to a peak then fall >10% off it → risk-OFF; near the high → risk-ON.
    s = _series([100, 105, 110, 115, 120, 118, 116, 100, 95, 90])
    on = risk_on_drawdown(s, window=5, thresh=0.10)
    assert bool(on.iloc[4]) is True        # at the high
    assert bool(on.iloc[-1]) is False      # 90 vs trailing-high 120 → -25% < -10%


def test_breadth_gate_off_when_few_above_ma() -> None:
    # Two names: when both are rising (above their MA) breadth=100%>50% → ON;
    # when both roll over (below MA) breadth=0%<50% → OFF.
    idx = pd.bdate_range("2020-01-01", periods=12)
    up = pd.Series(range(100, 112), index=idx, dtype=float)
    df = pd.DataFrame({"A": up, "B": up * 1.01})
    on_up = risk_on_breadth(df, ma=3, thresh=0.5)
    assert bool(on_up.iloc[-1]) is True
    down = pd.Series(range(112, 100, -1), index=idx, dtype=float)
    dfd = pd.DataFrame({"A": down, "B": down * 0.99})
    on_dn = risk_on_breadth(dfd, ma=3, thresh=0.5)
    assert bool(on_dn.iloc[-1]) is False


def test_breadth_is_causal() -> None:
    # Truncating the future must not change the breadth mask in the past.
    idx = pd.bdate_range("2020-01-01", periods=12)
    a = pd.Series([100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 98, 97], index=idx, dtype=float)
    df = pd.DataFrame({"A": a, "B": a + 5})
    full = risk_on_breadth(df, ma=3)
    trunc = risk_on_breadth(df.iloc[:8], ma=3)
    x = np.nan_to_num(full.iloc[:8].to_numpy(dtype=float), nan=-1.0)
    y = np.nan_to_num(trunc.to_numpy(dtype=float), nan=-1.0)
    assert np.allclose(x, y)


def test_dispatch_unknown_signal_raises() -> None:
    s = _series([100.0, 101.0, 102.0])
    try:
        compute_risk_on("momentum", s, ma=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown signal")


def test_dispatch_breadth_requires_roster() -> None:
    s = _series([100.0, 101.0, 102.0])
    try:
        compute_risk_on("breadth", s, roster_closes=None, ma=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError when breadth has no roster")
