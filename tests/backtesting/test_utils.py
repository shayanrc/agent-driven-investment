"""Stage 4 unit tests for snap_to_lot + parse_action."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.data_handler import DataHandler
from backtesting.portfolio import Portfolio
from backtesting.utils import parse_action, snap_to_lot


# ---------------------------------------------------------------------------
# snap_to_lot
# ---------------------------------------------------------------------------
def test_snap_to_lot_zero_is_fractional_passthrough():
    assert snap_to_lot(2.7, 0) == 2.7
    assert snap_to_lot(-3.14, 0) == -3.14
    assert snap_to_lot(0.001, 0) == 0.001


def test_snap_to_lot_one_truncates_toward_zero():
    assert snap_to_lot(2.7, 1) == 2.0
    assert snap_to_lot(-2.7, 1) == -2.0
    assert snap_to_lot(0.5, 1) == 0.0
    assert snap_to_lot(-0.5, 1) == 0.0


def test_snap_to_lot_round_lot_n_truncates_to_nearest_multiple():
    assert snap_to_lot(57.0, 25) == 50.0
    assert snap_to_lot(-57.0, 25) == -50.0
    assert snap_to_lot(24.9, 25) == 0.0
    assert snap_to_lot(25.0, 25) == 25.0
    assert snap_to_lot(75.0, 25) == 75.0


def test_snap_to_lot_zero_input_returns_zero():
    for lot in (0, 1, 10, 100):
        assert snap_to_lot(0, lot) == 0


def test_snap_to_lot_negative_lot_raises():
    with pytest.raises(ValueError, match="lot_size"):
        snap_to_lot(10.0, -1)


# ---------------------------------------------------------------------------
# parse_action setup
# ---------------------------------------------------------------------------
def _ohlcv(dates, base=100.0):
    n = len(dates)
    base_arr = np.arange(n, dtype=float) + base
    return pd.DataFrame(
        {
            "open": base_arr,
            "high": base_arr + 1.0,
            "low": base_arr - 1.0,
            "close": base_arr + 0.5,
            "volume": np.full(n, 1_000.0),
        },
        index=dates,
    )


def _setup():
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    feeds = {
        "equities": {
            "AAPL": _ohlcv(dates, 100.0),
            "MSFT": _ohlcv(dates, 200.0),
            "NIFTY_FUT": _ohlcv(dates, 18_000.0),
        }
    }
    dh = DataHandler(feeds, lookback=3)
    pf = Portfolio(100_000.0)
    return dh, pf


# ---------------------------------------------------------------------------
# parse_action — None
# ---------------------------------------------------------------------------
def test_parse_action_none_returns_empty():
    dh, pf = _setup()
    orders, audit = parse_action(None, pf, dh, {}, 1)
    assert orders == []
    assert audit == {}


# ---------------------------------------------------------------------------
# parse_action — order-type
# ---------------------------------------------------------------------------
def test_parse_order_action_passes_through():
    dh, pf = _setup()
    action = {"type": "order", "orders": [{"asset": "AAPL", "qty": 10}]}
    orders, audit = parse_action(action, pf, dh, {}, 1)
    assert orders == [{"asset": "AAPL", "qty": 10.0}]
    assert audit == {}


def test_parse_order_action_snaps_fractional_to_whole_with_default_lot_1():
    dh, pf = _setup()
    action = {"type": "order", "orders": [{"asset": "AAPL", "qty": 10.7}]}
    orders, audit = parse_action(action, pf, dh, {}, 1)
    assert orders == [{"asset": "AAPL", "qty": 10.0}]
    assert audit == {
        "AAPL": {"requested_qty": 10.7, "filled_qty": 10.0}
    }


def test_parse_order_action_with_fractional_lot_size_zero_keeps_fractional():
    dh, pf = _setup()
    action = {"type": "order", "orders": [{"asset": "AAPL", "qty": 1.234}]}
    orders, audit = parse_action(action, pf, dh, {"AAPL": 0}, 1)
    assert orders == [{"asset": "AAPL", "qty": 1.234}]
    assert audit == {}


def test_parse_order_action_with_round_lot_N():
    dh, pf = _setup()
    action = {"type": "order", "orders": [{"asset": "NIFTY_FUT", "qty": 60}]}
    orders, audit = parse_action(action, pf, dh, {"NIFTY_FUT": 25}, 1)
    assert orders == [{"asset": "NIFTY_FUT", "qty": 50.0}]
    assert audit == {
        "NIFTY_FUT": {"requested_qty": 60.0, "filled_qty": 50.0}
    }


def test_parse_order_action_snap_to_zero_recorded_but_not_emitted():
    """Q10: a snap-to-zero appears in audit with filled_qty: 0."""
    dh, pf = _setup()
    action = {"type": "order", "orders": [{"asset": "AAPL", "qty": 0.5}]}
    orders, audit = parse_action(action, pf, dh, {}, 1)
    assert orders == []  # not emitted
    assert audit == {"AAPL": {"requested_qty": 0.5, "filled_qty": 0.0}}


def test_parse_order_action_rejects_extra_fields():
    """Q6: order schema is exactly {asset, qty}."""
    dh, pf = _setup()
    action = {
        "type": "order",
        "orders": [{"asset": "AAPL", "qty": 10, "execution": "limit"}],
    }
    with pytest.raises(ValueError, match="execution"):
        parse_action(action, pf, dh, {}, 1)


def test_parse_order_action_rejects_malformed_entry():
    dh, pf = _setup()
    with pytest.raises(ValueError, match="malformed order"):
        parse_action(
            {"type": "order", "orders": [{"asset": "AAPL"}]}, pf, dh, {}, 1
        )


# ---------------------------------------------------------------------------
# parse_action — weight-type
# ---------------------------------------------------------------------------
def test_parse_weight_action_basic_allocation():
    """100k equity, target 50% AAPL @ 103.5 ⇒ 50000/103.5 ≈ 483.09 ⇒ 483
    shares with default lot 1."""
    dh, pf = _setup()
    action = {"type": "weight", "target_weights": {"AAPL": 0.5}}
    orders, audit = parse_action(action, pf, dh, {}, 1)
    assert len(orders) == 1
    assert orders[0]["asset"] == "AAPL"
    assert orders[0]["qty"] == 483.0
    assert "AAPL" in audit  # fractional residual was snapped


def test_parse_weight_action_drops_asset_emits_sell_to_zero():
    dh, pf = _setup()
    pf.positions["AAPL"] = 100.0
    # target_weights has no AAPL ⇒ target 0 ⇒ sell-to-zero.
    action = {"type": "weight", "target_weights": {"MSFT": 0.2}}
    orders, _ = parse_action(action, pf, dh, {}, 1)
    sells = [o for o in orders if o["qty"] < 0]
    assert any(o["asset"] == "AAPL" and o["qty"] == -100.0 for o in sells)


def test_parse_weight_action_sells_before_buys():
    dh, pf = _setup()
    pf.positions["AAPL"] = 100.0  # pre-existing position to sell
    action = {
        "type": "weight",
        "target_weights": {"MSFT": 0.3, "AAPL": 0.0},
    }
    orders, _ = parse_action(action, pf, dh, {}, 1)
    # First emitted order should be the sell.
    assert orders[0]["qty"] < 0


def test_parse_weight_action_over_allocation_raises():
    """Q1: sum(target_weights) > 1.0 raises ValueError."""
    dh, pf = _setup()
    action = {
        "type": "weight",
        "target_weights": {"AAPL": 0.6, "MSFT": 0.5},
    }
    with pytest.raises(ValueError, match="exceeds 1.0"):
        parse_action(action, pf, dh, {}, 1)


def test_parse_weight_action_at_exactly_one_is_allowed():
    dh, pf = _setup()
    action = {
        "type": "weight",
        "target_weights": {"AAPL": 0.5, "MSFT": 0.5},
    }
    orders, _ = parse_action(action, pf, dh, {}, 1)
    assert all(o["qty"] != 0 for o in orders)


def test_parse_weight_action_lot_size_audit_populated_on_snap():
    dh, pf = _setup()
    action = {
        "type": "weight",
        "target_weights": {"NIFTY_FUT": 0.5},
    }
    orders, audit = parse_action(action, pf, dh, {"NIFTY_FUT": 25}, 1)
    # NIFTY_FUT close at step 3 = 18_003.5. ideal_qty = 50000/18003.5 ≈ 2.78
    # snapped to lot 25 ⇒ 0. Order not emitted; audit records the snap.
    assert "NIFTY_FUT" in audit
    assert audit["NIFTY_FUT"]["filled_qty"] == 0


# ---------------------------------------------------------------------------
# parse_action — type validation
# ---------------------------------------------------------------------------
def test_parse_action_unknown_type_raises():
    dh, pf = _setup()
    with pytest.raises(ValueError, match="type"):
        parse_action({"type": "magic"}, pf, dh, {}, 1)


def test_parse_action_non_dict_raises():
    dh, pf = _setup()
    with pytest.raises(ValueError, match="action must be dict"):
        parse_action("hello", pf, dh, {}, 1)


def test_parse_weight_action_uses_pre_fill_equity_not_post_fill():
    """Weight target is calculated against current portfolio equity at
    the call time, not against some post-fill projection."""
    dh, pf = _setup()
    pf.cash = 100_000.0  # known equity
    # Sanity: 100% MSFT @ 203.5 ⇒ 491.40 ⇒ 491 shares snapped.
    action = {"type": "weight", "target_weights": {"MSFT": 1.0}}
    orders, _ = parse_action(action, pf, dh, {}, 1)
    assert orders[0]["qty"] == 491.0
