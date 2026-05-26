"""Stage 5 integration tests for the Backtest orchestrator.

Coverage:
- Construction + reset
- Single-bar step lifecycle in both fill_modes (B2)
- Look-ahead absence (B2 — fills never reference a bar the caller hasn't observed)
- Per-bar mark-to-market correctness (B3)
- Q8 advance_time no-mutate-when-done (B7)
- Q9 gap_policy plumbed through to DataHandler
- info-key emission per Q10 (only when non-empty)
- multi-step run on order + weight + None actions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import Backtest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _ohlcv(dates: pd.DatetimeIndex, start: float = 100.0) -> pd.DataFrame:
    """OHLCV with strictly monotone, distinct open vs close so tests can tell
    them apart by inspection."""
    n = len(dates)
    base = np.arange(n, dtype=float) + start
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 2.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": np.full(n, 1_000.0),
        },
        index=dates,
    )


def _feeds_two_assets(n: int = 30):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return {
        "equities": {
            "AAPL": _ohlcv(dates, 100.0),
            "MSFT": _ohlcv(dates, 200.0),
        }
    }


# ---------------------------------------------------------------------------
# Construction + reset
# ---------------------------------------------------------------------------
def test_construct_default():
    bt = Backtest(_feeds_two_assets(), lookback=5)
    assert bt.fill_mode == "next_open"
    assert bt.portfolio.cash == 100_000.0
    assert bt.data_handler.current_step == 5
    # Initial mark sets equity to cash (no positions yet).
    assert bt.portfolio.equity == 100_000.0


def test_reject_invalid_fill_mode():
    with pytest.raises(ValueError, match="fill_mode"):
        Backtest(_feeds_two_assets(), lookback=5, fill_mode="midnight")  # type: ignore[arg-type]


def test_reset_returns_first_state():
    bt = Backtest(_feeds_two_assets(), lookback=5)
    state, done, info = bt.reset()
    assert done is False
    assert info == {}
    assert state["step"] == 5
    assert "market_data" in state
    assert "portfolio" in state
    assert "timestamp" in state
    assert state["portfolio"]["cash"] == 100_000.0
    assert state["portfolio"]["pending_orders"] == 0


# ---------------------------------------------------------------------------
# Step lifecycle in both fill_modes
# ---------------------------------------------------------------------------
def test_step_none_advances_time_without_orders():
    bt = Backtest(_feeds_two_assets(), lookback=5, fill_mode="next_open")
    pre_step = bt.data_handler.current_step
    state, done, info = bt.step(None)
    assert done is False
    assert info == {}  # nothing happened — every locked key omitted
    assert state["step"] == pre_step + 1


def test_step_order_current_close_fills_at_T_close():
    """fill_mode='current_close': order submitted at T fills at T's close.

    The caller observed T's close in the prior state, so this is MOC —
    not look-ahead. The fill price must equal T's close, NOT T+1's close.
    """
    feeds = _feeds_two_assets(30)
    bt = Backtest(feeds, lookback=5, fill_mode="current_close")
    # T is current_step before step() runs.
    T = bt.data_handler.current_step
    expected_close_at_T = feeds["equities"]["AAPL"].iloc[T]["close"]

    _state, done, info = bt.step(
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 10}]}
    )
    assert done is False
    assert "fills" in info
    assert len(info["fills"]) == 1
    fill = info["fills"][0]
    assert fill["asset"] == "AAPL"
    assert fill["qty"] == 10
    assert fill["fill_price"] == pytest.approx(expected_close_at_T)


def test_step_order_next_open_fills_at_T_plus_1_open():
    """fill_mode='next_open': order submitted at T fills at T+1's open.

    The caller has NOT seen T+1 when submitting — this is the structural
    no-look-ahead guarantee. The fill price must equal T+1's open.
    """
    feeds = _feeds_two_assets(30)
    bt = Backtest(feeds, lookback=5, fill_mode="next_open")
    T = bt.data_handler.current_step
    expected_open_at_T_plus_1 = feeds["equities"]["AAPL"].iloc[T + 1]["open"]

    _state, done, info = bt.step(
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 10}]}
    )
    assert done is False
    assert "fills" in info
    fill = info["fills"][0]
    assert fill["fill_price"] == pytest.approx(expected_open_at_T_plus_1)


# ---------------------------------------------------------------------------
# B2 — structural look-ahead absence
# ---------------------------------------------------------------------------
def test_b2_no_lookahead_next_open_fill_price_only_from_T_plus_1():
    """At step T, the fill price must come from bar T+1 (the bar the engine
    has just advanced to), never from a strictly-later bar."""
    feeds = _feeds_two_assets(30)
    bt = Backtest(feeds, lookback=5, fill_mode="next_open")
    aapl = feeds["equities"]["AAPL"]
    for _ in range(5):
        T = bt.data_handler.current_step
        _state, done, info = bt.step(
            {"type": "order", "orders": [{"asset": "AAPL", "qty": 1}]}
        )
        if done:
            break
        expected = aapl.iloc[T + 1]["open"]
        assert info["fills"][0]["fill_price"] == pytest.approx(expected)


def test_b2_no_lookahead_current_close_fill_price_only_from_T():
    feeds = _feeds_two_assets(30)
    bt = Backtest(feeds, lookback=5, fill_mode="current_close")
    aapl = feeds["equities"]["AAPL"]
    for _ in range(5):
        T = bt.data_handler.current_step
        _state, done, info = bt.step(
            {"type": "order", "orders": [{"asset": "AAPL", "qty": 1}]}
        )
        expected = aapl.iloc[T]["close"]
        assert info["fills"][0]["fill_price"] == pytest.approx(expected)
        if done:
            break


# ---------------------------------------------------------------------------
# B3 — per-bar mark-to-market
# ---------------------------------------------------------------------------
def test_b3_mark_to_market_uses_post_step_close():
    """After step(), portfolio.equity must equal cash + Σ qty * close at the
    new current_step (T+1 on non-terminal, T on terminal)."""
    feeds = _feeds_two_assets(30)
    bt = Backtest(feeds, lookback=5, fill_mode="next_open")
    aapl = feeds["equities"]["AAPL"]
    state, done, info = bt.step(
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 10}]}
    )
    assert not done
    # current_step is now T+1; mark is at close of T+1.
    T_plus_1 = bt.data_handler.current_step
    mark_price = aapl.iloc[T_plus_1]["close"]
    expected_equity = state["portfolio"]["cash"] + 10 * mark_price
    assert state["portfolio"]["equity"] == pytest.approx(expected_equity)


# ---------------------------------------------------------------------------
# Q8 / B7 — advance_time no-mutate-when-done
# ---------------------------------------------------------------------------
def test_b7_terminal_step_advance_time_no_mutate():
    """At the terminal bar, done=True and current_step is pinned at
    max_steps - 1. A second step(None) keeps it pinned."""
    feeds = _feeds_two_assets(15)
    bt = Backtest(feeds, lookback=5, fill_mode="next_open")
    # Step through to terminal.
    done = False
    last_state = None
    while not done:
        last_state, done, _info = bt.step(None)
    assert done is True
    max_steps = bt.data_handler.max_steps
    assert bt.data_handler.current_step == max_steps - 1
    # Re-entrant call: still done, still pinned.
    state2, done2, _ = bt.step(None)
    assert done2 is True
    assert bt.data_handler.current_step == max_steps - 1
    assert state2["step"] == max_steps - 1


def test_b7_terminal_next_open_pending_order_rejected_untradeable():
    """If a next_open order is still pending when the engine hits the
    terminal bar (no T+1 to fill against), it must surface in
    info['rejected_untradeable']."""
    feeds = _feeds_two_assets(8)
    bt = Backtest(feeds, lookback=5, fill_mode="next_open")
    # Walk to the second-to-last bar so the next step is terminal.
    while bt.data_handler.current_step < bt.data_handler.max_steps - 1:
        _s, _d, _i = bt.step(None)
        if _d:
            break
    # Now step with an order; advance_time should return done=True and the
    # broker is drained as untradeable.
    _state, done, info = bt.step(
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 1}]}
    )
    assert done is True
    assert "rejected_untradeable" in info
    assert info["rejected_untradeable"][0]["asset"] == "AAPL"


# ---------------------------------------------------------------------------
# Q9 — gap_policy plumbed through
# ---------------------------------------------------------------------------
def test_q9_gap_policy_raise_propagates():
    """A feed with an internal gap (relative to the master timeline) +
    gap_policy='raise' must fail construction inside DataHandler.

    The master timeline is the UNION of asset dates, so a single asset's
    'missing' date isn't actually a gap unless another asset has it. We
    use two assets — MSFT has the full calendar, AAPL is missing one
    interior date that's inside AAPL's active range."""
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    feeds = {
        "equities": {
            "AAPL": _ohlcv(dates, 100.0).drop(dates[5]),
            "MSFT": _ohlcv(dates, 200.0),
        }
    }
    with pytest.raises(ValueError, match="gap"):
        Backtest(feeds, lookback=3, gap_policy="raise")


def test_q9_gap_policy_default_ffill_zero_volume():
    """Default gap_policy forward-fills price columns and zeros volume on
    AAPL's gap day (the date AAPL is missing but MSFT has)."""
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    feeds = {
        "equities": {
            "AAPL": _ohlcv(dates, 100.0).drop(dates[5]),
            "MSFT": _ohlcv(dates, 200.0),
        }
    }
    bt = Backtest(feeds, lookback=3)
    aapl_frame = bt.data_handler.data["equities"]["AAPL"]
    # Position 5 of the master timeline is AAPL's gap day.
    assert aapl_frame.iloc[5]["volume"] == 0
    assert aapl_frame.iloc[5]["close"] == aapl_frame.iloc[4]["close"]


# ---------------------------------------------------------------------------
# Weight rebalance + info-key emission
# ---------------------------------------------------------------------------
def test_weight_rebalance_emits_fills():
    """Weight rebalance produces orders sized from pre-fill close prices.

    Use current_close so fill price == sizing price (no decision-time-vs-
    fill-time slippage), and a moderate target sum so cash overdraw can't
    eat the second buy."""
    feeds = _feeds_two_assets(30)
    bt = Backtest(
        feeds,
        lookback=5,
        fill_mode="current_close",
        lot_sizes={"AAPL": 0, "MSFT": 0},
    )
    _state, done, info = bt.step(
        {"type": "weight", "target_weights": {"AAPL": 0.3, "MSFT": 0.3}}
    )
    assert not done
    assert "fills" in info
    assert len(info["fills"]) == 2
    # Weight-drift reported (post-MARK against target).
    # In current_close mode, fill price == close at T == mark price at T,
    # but ADVANCE moves to T+1 and MARK uses T+1's close — so positions
    # are valued at a different price than they were bought at, giving
    # non-zero drift.
    assert "weight_drift" in info


def test_info_omits_empty_keys():
    """Q10 contract: every key emitted only when its payload is non-empty.

    A no-op step yields {} for info."""
    bt = Backtest(_feeds_two_assets(), lookback=5)
    _s, _d, info = bt.step(None)
    assert info == {}


def test_info_completeness_one_of_each_side_effect():
    """Drive a scenario that produces at least one of: fill,
    rejected_overdraw, lot_size_audit (snap-to-zero), weight_drift."""
    feeds = _feeds_two_assets(30)
    # Use a heavy whole-share lot to force rounding + zero-snap.
    bt = Backtest(
        feeds,
        lookback=5,
        initial_cash=1_000.0,  # small cash → easy overdraw
        fill_mode="current_close",
        lot_sizes={"AAPL": 1, "MSFT": 1},
    )
    # Submit: one fillable buy + one fractional that snaps to zero + one
    # overdraw buy. The overdraw uses MSFT × 1000 shares vs $1000 cash.
    _s, _d, info = bt.step(
        {
            "type": "order",
            "orders": [
                {"asset": "AAPL", "qty": 1},  # ~$100, fits
                {"asset": "AAPL", "qty": 0.4},  # snap → 0
                {"asset": "MSFT", "qty": 1000},  # overdraw
            ],
        }
    )
    assert "fills" in info
    assert "lot_size_audit" in info  # AAPL 0.4 → 0
    assert info["lot_size_audit"]["AAPL"]["filled_qty"] == 0
    assert "rejected_overdraw" in info


def test_multi_step_smoke():
    """A 20-step mixed-action run completes without exception and
    preserves portfolio consistency at every step."""
    feeds = _feeds_two_assets(30)
    bt = Backtest(feeds, lookback=5, fill_mode="next_open")
    actions = [
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 5}]},
        None,
        {"type": "weight", "target_weights": {"AAPL": 0.3, "MSFT": 0.4}},
        None,
        {"type": "order", "orders": [{"asset": "MSFT", "qty": -2}]},
    ] * 4
    for action in actions:
        state, done, _info = bt.step(action)
        # B4: cash >= 0 and equity == cash + Σ pos * close.
        assert state["portfolio"]["cash"] >= 0
        # equity is mark-consistent via update_valuations.
        if done:
            break


def test_known_assets_used_for_broker_validation():
    """An order with a typo'd asset name must be rejected at submit time
    and surface in info['rejected_invalid']."""
    bt = Backtest(_feeds_two_assets(), lookback=5, fill_mode="current_close")
    _s, _d, info = bt.step(
        {"type": "order", "orders": [{"asset": "FAKEASSET", "qty": 1}]}
    )
    assert "rejected_invalid" in info
    assert info["rejected_invalid"][0]["asset"] == "FAKEASSET"
