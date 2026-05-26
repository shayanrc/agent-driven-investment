"""Stage 6 tests for the Strategy protocol + example strategies + run_strategy.

The Strategy protocol is a typing convention for callers, not an engine
hook — these tests verify the contract holds for the three example
implementations and that run_strategy drives a Backtest from reset() to
done with each.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import Backtest
from backtesting.strategy import (
    FixedWeightStrategy,
    HoldStrategy,
    ScriptedActionStrategy,
    Strategy,
    run_strategy,
)


def _ohlcv(dates: pd.DatetimeIndex, start: float = 100.0) -> pd.DataFrame:
    n = len(dates)
    base = np.arange(n, dtype=float) + start
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": np.full(n, 1_000.0),
        },
        index=dates,
    )


def _make_backtest(n_dates: int = 20, **kw) -> Backtest:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    feeds = {
        "equities": {
            "AAPL": _ohlcv(dates, 100.0),
            "MSFT": _ohlcv(dates, 200.0),
        }
    }
    return Backtest(feeds, lookback=3, **kw)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------
def test_hold_strategy_is_a_strategy():
    assert isinstance(HoldStrategy(), Strategy)


def test_fixed_weight_strategy_is_a_strategy():
    s = FixedWeightStrategy({"AAPL": 0.4})
    assert isinstance(s, Strategy)


def test_scripted_action_strategy_is_a_strategy():
    s = ScriptedActionStrategy([None, None])
    assert isinstance(s, Strategy)


def test_plain_function_is_a_strategy():
    def my_strategy(state: dict[str, Any], info: dict[str, Any]) -> None:
        return None

    # runtime_checkable Protocol with one method matches plain callables.
    assert isinstance(my_strategy, Strategy)


def test_lambda_is_a_strategy():
    s = lambda state, info: None  # noqa: E731
    assert isinstance(s, Strategy)


def test_non_callable_is_not_a_strategy():
    assert not isinstance(42, Strategy)
    assert not isinstance("hello", Strategy)


# ---------------------------------------------------------------------------
# FixedWeightStrategy
# ---------------------------------------------------------------------------
def test_fixed_weight_strategy_rejects_over_allocation_at_construction():
    """Mirrors the engine's Q1 rule, fails fast at strategy construction
    rather than at the first step() call."""
    with pytest.raises(ValueError, match="1.0"):
        FixedWeightStrategy({"AAPL": 0.6, "MSFT": 0.6})


def test_fixed_weight_strategy_submits_once_then_holds():
    s = FixedWeightStrategy({"AAPL": 0.3, "MSFT": 0.3})
    state = {"step": 3}
    info: dict[str, Any] = {}
    a1 = s(state, info)
    assert a1 == {"type": "weight", "target_weights": {"AAPL": 0.3, "MSFT": 0.3}}
    # Subsequent calls return None.
    for _ in range(5):
        assert s(state, info) is None


def test_fixed_weight_strategy_reset_re_arms_submission():
    s = FixedWeightStrategy({"AAPL": 0.5})
    state = {"step": 3}
    info: dict[str, Any] = {}
    a1 = s(state, info)
    assert a1 is not None
    assert s(state, info) is None
    s.reset()
    a2 = s(state, info)
    assert a2 is not None


# ---------------------------------------------------------------------------
# ScriptedActionStrategy
# ---------------------------------------------------------------------------
def test_scripted_action_strategy_plays_then_holds():
    script = [
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 5}]},
        None,
        {"type": "weight", "target_weights": {"AAPL": 0.5}},
    ]
    s = ScriptedActionStrategy(script)
    state, info = {"step": 3}, {}
    assert s(state, info) == script[0]
    assert s(state, info) is None
    assert s(state, info) == script[2]
    # Exhausted; returns None.
    assert s(state, info) is None
    assert s(state, info) is None


# ---------------------------------------------------------------------------
# run_strategy
# ---------------------------------------------------------------------------
def test_run_strategy_with_hold_strategy_reaches_done():
    bt = _make_backtest(n_dates=15)
    history = run_strategy(bt, HoldStrategy())
    # First entry is the reset state, last is the terminal step.
    assert history[0][1] is False  # done==False at reset
    assert history[-1][1] is True  # done==True at terminus
    # All info dicts should be empty (no orders submitted).
    for _state, _done, info in history:
        assert info == {}


def test_run_strategy_with_fixed_weight_strategy_submits_one_order_batch():
    bt = _make_backtest(n_dates=15, lot_sizes={"AAPL": 0, "MSFT": 0})
    history = run_strategy(bt, FixedWeightStrategy({"AAPL": 0.3, "MSFT": 0.3}))
    # Count steps where info["fills"] appeared.
    fill_steps = [i for i, (_s, _d, info) in enumerate(history) if "fills" in info]
    # Exactly one step has fills (the first step after the initial reset).
    assert len(fill_steps) == 1
    # And it's the step right after reset (history[1]).
    assert fill_steps == [1]


def test_run_strategy_with_max_steps_cap_stops_early():
    bt = _make_backtest(n_dates=100)
    history = run_strategy(bt, HoldStrategy(), max_steps=5)
    # reset + 5 steps = 6 entries.
    assert len(history) == 6
    # done must NOT be True (we capped early on a long timeline).
    assert history[-1][1] is False


def test_run_strategy_rejects_non_callable():
    bt = _make_backtest()
    with pytest.raises(TypeError, match="callable"):
        run_strategy(bt, 42)  # type: ignore[arg-type]


def test_run_strategy_scripted_drives_known_actions():
    """End-to-end: scripted strategy drives a known order sequence, and
    run_strategy returns the trace including every fill in info."""
    bt = _make_backtest(n_dates=20, fill_mode="current_close")
    script = [
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 10}]},
        None,
        {"type": "order", "orders": [{"asset": "AAPL", "qty": -5}]},
    ]
    history = run_strategy(bt, ScriptedActionStrategy(script))
    fill_records = []
    for _s, _d, info in history:
        for f in info.get("fills", []):
            fill_records.append(f)
    # Two fills: the buy and the sell. The middle step has None / no fills.
    assert len(fill_records) == 2
    assert fill_records[0]["asset"] == "AAPL"
    assert fill_records[0]["qty"] == 10
    assert fill_records[1]["asset"] == "AAPL"
    assert fill_records[1]["qty"] == -5
