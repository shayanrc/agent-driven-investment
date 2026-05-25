"""Stage 3 — target builder tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt.targets import build_target


def _panel_from_close(close: list[float], ticker: str = "A",
                       high: list[float] | None = None,
                       low: list[float] | None = None) -> pd.DataFrame:
    n = len(close)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    if high is None:
        high = close
    if low is None:
        low = close
    df = pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": np.ones(n, dtype=int),
    })
    return df.set_index(["date", "ticker"]).sort_index()


# ---------------------------------------------------------------------------
# Simple binary mode — UP
# ---------------------------------------------------------------------------


def test_up_simple_breach_in_window():
    """Close 100; +10% target = 110; breach on day 3 via HIGH=115. Label = 1."""
    close = [100.0] * 10
    high = list(close)
    high[3] = 115.0  # breach
    panel = _panel_from_close(close, high=high)
    y = build_target(panel, direction="up", threshold_pct=10, horizon_days=5)
    # Origin t=0: forward window days 1..5, breach at day 3
    assert y.iloc[0] == 1.0


def test_up_simple_no_breach():
    """No high ever clears +10%; label 0 on all eligible rows."""
    close = [100.0] * 30
    panel = _panel_from_close(close)
    y = build_target(panel, direction="up", threshold_pct=10, horizon_days=5)
    # All eligible rows (n - horizon = 25) should be 0
    assert (y.iloc[:25] == 0.0).all()


def test_up_breach_on_last_day_of_horizon():
    close = [100.0] * 10
    high = list(close)
    high[5] = 115.0  # exactly t+5 with horizon=5 → breach lands at edge
    panel = _panel_from_close(close, high=high)
    y = build_target(panel, direction="up", threshold_pct=10, horizon_days=5)
    assert y.iloc[0] == 1.0


def test_last_horizon_rows_are_nan():
    close = [100.0] * 10
    panel = _panel_from_close(close)
    y = build_target(panel, direction="up", threshold_pct=10, horizon_days=5)
    assert y.iloc[-5:].isna().all()


# ---------------------------------------------------------------------------
# Simple binary mode — DOWN
# ---------------------------------------------------------------------------


def test_down_simple_breach_via_low():
    close = [100.0] * 10
    low = list(close)
    low[2] = 88.0  # -10% target = 90; breach at 88
    panel = _panel_from_close(close, low=low)
    y = build_target(panel, direction="down", threshold_pct=10, horizon_days=5)
    assert y.iloc[0] == 1.0


def test_down_simple_no_breach():
    close = [100.0] * 20
    panel = _panel_from_close(close)
    y = build_target(panel, direction="down", threshold_pct=10, horizon_days=5)
    assert (y.iloc[:15] == 0.0).all()


# ---------------------------------------------------------------------------
# Path-honesty UP
# ---------------------------------------------------------------------------


def test_up_path_honesty_clean_breach_label_1():
    # close grows smoothly to 110+ in horizon, no drawdown
    close = [100.0, 101.0, 103.0, 106.0, 110.5, 112.0]
    panel = _panel_from_close(close)
    y = build_target(panel, direction="up", threshold_pct=10, horizon_days=5,
                     max_drawdown=0.05)
    assert y.iloc[0] == 1.0


def test_up_path_honesty_breach_after_drawdown_label_0():
    # close goes 100 → 94 (6% dd) → 105 → 112 (breach via CLOSE).
    # Drawdown 6% > 5% → label 0 even though breach occurred.
    close = [100.0, 94.0, 102.0, 108.0, 112.0, 112.0]
    panel = _panel_from_close(close)
    y = build_target(panel, direction="up", threshold_pct=10, horizon_days=5,
                     max_drawdown=0.05)
    assert y.iloc[0] == 0.0


def test_up_path_honesty_shallow_drawdown_label_1():
    # close goes 100 → 97 (3% dd) → 105 → 112. Drawdown < 5% → label 1.
    close = [100.0, 97.0, 102.0, 108.0, 112.0, 112.0]
    panel = _panel_from_close(close)
    y = build_target(panel, direction="up", threshold_pct=10, horizon_days=5,
                     max_drawdown=0.05)
    assert y.iloc[0] == 1.0


def test_up_path_honesty_uses_close_not_high():
    """High touches 112 but CLOSE never clears 110 in window → label 0."""
    close = [100.0, 101.0, 102.0, 103.0, 104.0, 104.0]
    high = [100.0, 101.0, 112.0, 103.0, 104.0, 104.0]  # spike on day 2
    panel = _panel_from_close(close, high=high)
    y = build_target(panel, direction="up", threshold_pct=10, horizon_days=5,
                     max_drawdown=0.05)
    assert y.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Path-honesty DOWN
# ---------------------------------------------------------------------------


def test_down_path_honesty_clean_breach_label_1():
    close = [100.0, 99.0, 97.0, 94.0, 89.0, 88.0]  # smooth fall to <-10%
    panel = _panel_from_close(close)
    y = build_target(panel, direction="down", threshold_pct=10, horizon_days=5,
                     max_drawdown=0.05)
    assert y.iloc[0] == 1.0


def test_down_path_honesty_rally_then_drop_label_0():
    """Close rallies to 106 (>5% adverse for a short) then drops to 89."""
    close = [100.0, 106.0, 98.0, 95.0, 89.0, 89.0]
    panel = _panel_from_close(close)
    y = build_target(panel, direction="down", threshold_pct=10, horizon_days=5,
                     max_drawdown=0.05)
    assert y.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_direction_raises():
    panel = _panel_from_close([100.0] * 10)
    with pytest.raises(ValueError, match="direction"):
        build_target(panel, direction="sideways", threshold_pct=10, horizon_days=5)


def test_invalid_threshold_raises():
    panel = _panel_from_close([100.0] * 10)
    with pytest.raises(ValueError, match="threshold_pct"):
        build_target(panel, direction="up", threshold_pct=-1, horizon_days=5)


def test_invalid_max_drawdown_raises():
    panel = _panel_from_close([100.0] * 10)
    with pytest.raises(ValueError, match="max_drawdown"):
        build_target(panel, direction="up", threshold_pct=10, horizon_days=5,
                     max_drawdown=1.5)


# ---------------------------------------------------------------------------
# Multi-ticker
# ---------------------------------------------------------------------------


def test_multi_ticker_targets_are_independent():
    n = 12
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    a_close = [100.0] * n
    a_high = list(a_close); a_high[3] = 115.0   # A: breach +15% via high
    b_close = [100.0] * n  # B: no breach
    panel_a = pd.DataFrame({
        "date": dates, "ticker": "A",
        "open": a_close, "high": a_high, "low": a_close,
        "close": a_close, "adj_close": a_close, "volume": [1]*n,
    })
    panel_b = pd.DataFrame({
        "date": dates, "ticker": "B",
        "open": b_close, "high": b_close, "low": b_close,
        "close": b_close, "adj_close": b_close, "volume": [1]*n,
    })
    panel = pd.concat([panel_a, panel_b]).set_index(["date", "ticker"]).sort_index()
    y = build_target(panel, direction="up", threshold_pct=10, horizon_days=5)
    first_date = dates[0]
    assert y.loc[(first_date, "A")] == 1.0
    assert y.loc[(first_date, "B")] == 0.0
