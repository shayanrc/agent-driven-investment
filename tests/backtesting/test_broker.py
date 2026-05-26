"""Stage 3 unit tests for ExecutionBroker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.broker import ExecutionBroker
from backtesting.data_handler import DataHandler
from backtesting.portfolio import Portfolio


def _ohlcv(dates: pd.DatetimeIndex, base_price: float = 100.0) -> pd.DataFrame:
    n = len(dates)
    base = np.arange(n, dtype=float) + base_price
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


def _make_setup(initial_cash: float = 100_000.0, commission_fn=None):
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    feeds = {
        "equities": {
            "AAPL": _ohlcv(dates, 100.0),
            "MSFT": _ohlcv(dates, 200.0),
        }
    }
    dh = DataHandler(feeds, lookback=3)
    pf = Portfolio(initial_cash)
    broker = ExecutionBroker(commission_fn)
    return broker, dh, pf


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_submit_orders_rejects_missing_asset():
    broker, dh, _ = _make_setup()
    rejected = broker.submit_orders(
        [{"qty": 10}], known_assets=dh.get_known_assets()
    )
    assert len(rejected) == 1
    assert "missing" in rejected[0]["reason"]
    assert broker.get_pending_count() == 0


def test_submit_orders_rejects_missing_qty():
    broker, dh, _ = _make_setup()
    rejected = broker.submit_orders(
        [{"asset": "AAPL"}], known_assets=dh.get_known_assets()
    )
    assert len(rejected) == 1
    assert "missing" in rejected[0]["reason"]


def test_submit_orders_rejects_zero_qty():
    broker, dh, _ = _make_setup()
    rejected = broker.submit_orders(
        [{"asset": "AAPL", "qty": 0}], known_assets=dh.get_known_assets()
    )
    assert len(rejected) == 1
    assert "qty == 0" in rejected[0]["reason"]


def test_submit_orders_rejects_extra_fields():
    """Q6 contract: execution / limit_price / TIF are v1.1 features;
    submitting them in v1 is a deliberate API misuse."""
    broker, dh, _ = _make_setup()
    rejected = broker.submit_orders(
        [{"asset": "AAPL", "qty": 10, "execution": "limit"}],
        known_assets=dh.get_known_assets(),
    )
    assert len(rejected) == 1
    assert "execution" in rejected[0]["reason"]


def test_submit_orders_rejects_unknown_asset():
    broker, dh, _ = _make_setup()
    rejected = broker.submit_orders(
        [{"asset": "GOOG", "qty": 5}], known_assets=dh.get_known_assets()
    )
    assert len(rejected) == 1
    assert "unknown asset" in rejected[0]["reason"]


def test_submit_orders_normalizes_int_qty_to_float():
    broker, dh, _ = _make_setup()
    broker.submit_orders([{"asset": "AAPL", "qty": 10}], known_assets=dh.get_known_assets())
    assert broker.pending_orders[0]["qty"] == 10.0
    assert isinstance(broker.pending_orders[0]["qty"], float)


# ---------------------------------------------------------------------------
# Process queue — happy paths
# ---------------------------------------------------------------------------
def test_process_queue_fills_at_close_when_given_close_bar():
    broker, dh, pf = _make_setup()
    broker.submit_orders([{"asset": "AAPL", "qty": 10}], known_assets=dh.get_known_assets())
    bar = dh.get_current_bar()
    log = broker.process_queue(bar, pf, dh)
    assert len(log["filled"]) == 1
    fill = log["filled"][0]
    # At step 3, close = 100 + 3 + 0.5 = 103.5
    assert fill["fill_price"] == pytest.approx(103.5)
    assert pf.positions == {"AAPL": 10.0}


def test_process_queue_drains_queue():
    broker, dh, pf = _make_setup()
    broker.submit_orders(
        [{"asset": "AAPL", "qty": 5}, {"asset": "MSFT", "qty": 3}],
        known_assets=dh.get_known_assets(),
    )
    broker.process_queue(dh.get_current_bar(), pf, dh)
    assert broker.get_pending_count() == 0


# ---------------------------------------------------------------------------
# Sells before buys
# ---------------------------------------------------------------------------
def test_sells_before_buys_frees_cash_for_buys():
    """Scenario: portfolio has cash X and existing long position. A
    submitted sell + buy would overdraw if buy ran first, but sell-first
    keeps both fills clean."""
    broker, dh, pf = _make_setup(initial_cash=1_000.0)
    # Pre-load a long position so we have something to sell — bypass
    # the broker/execute_trade overdraw guard since we're seeding state
    # directly for the scenario.
    pf.positions["MSFT"] = 10.0

    broker.submit_orders(
        # Buy submitted first, sell second; broker reorders to sell first.
        [
            {"asset": "AAPL", "qty": 10},   # buy ~103.5 * 10 = 1035
            {"asset": "MSFT", "qty": -5},   # sell ~203.5 * 5 = 1017.5 → cash freed
        ],
        known_assets=dh.get_known_assets(),
    )
    log = broker.process_queue(dh.get_current_bar(), pf, dh)
    # Both should fill (sell-first frees enough cash for the buy).
    assert len(log["filled"]) == 2
    # Sell processed first ⇒ first filled record is the MSFT sell.
    assert log["filled"][0]["asset"] == "MSFT"
    assert log["filled"][0]["qty"] == -5
    assert log["filled"][1]["asset"] == "AAPL"


# ---------------------------------------------------------------------------
# Overdraw rejection
# ---------------------------------------------------------------------------
def test_buy_overdraw_rejected_no_state_change():
    broker, dh, pf = _make_setup(initial_cash=500.0)
    broker.submit_orders([{"asset": "AAPL", "qty": 10}], known_assets=dh.get_known_assets())
    log = broker.process_queue(dh.get_current_bar(), pf, dh)
    assert len(log["filled"]) == 0
    assert len(log["rejected_overdraw"]) == 1
    assert pf.cash == 500.0
    assert pf.positions == {}


def test_buy_overdraw_with_commission_just_too_high():
    broker, dh, pf = _make_setup(
        initial_cash=1035.0, commission_fn=lambda a, q, p: 5.0
    )
    # AAPL close=103.5; 10 shares = 1035 + 5 commission = 1040 > 1035.
    broker.submit_orders([{"asset": "AAPL", "qty": 10}], known_assets=dh.get_known_assets())
    log = broker.process_queue(dh.get_current_bar(), pf, dh)
    assert len(log["rejected_overdraw"]) == 1
    assert pf.cash == 1035.0


# ---------------------------------------------------------------------------
# Untradeable handling
# ---------------------------------------------------------------------------
def _setup_with_delisted_asset():
    dates_a = pd.date_range("2024-01-01", periods=20, freq="B")
    dates_b = pd.date_range("2024-01-01", periods=10, freq="B")  # delists earlier
    feeds = {
        "equities": {
            "A": _ohlcv(dates_a, 100.0),
            "B": _ohlcv(dates_b, 200.0),
        }
    }
    dh = DataHandler(feeds, lookback=3)
    pf = Portfolio(50_000.0)
    broker = ExecutionBroker()
    return broker, dh, pf


def test_sell_of_delisted_asset_permitted_at_last_known_price():
    broker, dh, pf = _setup_with_delisted_asset()
    # Pre-load a position in B.
    pf.positions["B"] = 5.0
    pf.cash = 50_000.0
    # Advance well past B's delisting.
    for _ in range(15):
        dh.advance_time()
    assert not dh.is_active("B")
    broker.submit_orders([{"asset": "B", "qty": -5}], known_assets=dh.get_known_assets())
    log = broker.process_queue(dh.get_current_bar(), pf, dh)
    assert len(log["filled"]) == 1
    assert log["filled"][0]["asset"] == "B"


def test_buy_of_delisted_asset_rejected():
    broker, dh, pf = _setup_with_delisted_asset()
    for _ in range(15):
        dh.advance_time()
    broker.submit_orders([{"asset": "B", "qty": 5}], known_assets=dh.get_known_assets())
    log = broker.process_queue(dh.get_current_bar(), pf, dh)
    assert len(log["rejected_untradeable"]) == 1


def test_buy_of_pre_ipo_asset_rejected():
    dates_a = pd.date_range("2024-01-01", periods=20, freq="B")
    dates_b = pd.date_range("2024-01-15", periods=10, freq="B")  # starts late
    feeds = {"equities": {"A": _ohlcv(dates_a), "B": _ohlcv(dates_b)}}
    dh = DataHandler(feeds, lookback=3)
    pf = Portfolio(50_000.0)
    broker = ExecutionBroker()
    # At step 3, B hasn't started yet.
    assert not dh.is_active("B")
    broker.submit_orders([{"asset": "B", "qty": 5}], known_assets=dh.get_known_assets())
    log = broker.process_queue(dh.get_current_bar(), pf, dh)
    assert len(log["rejected_untradeable"]) == 1


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------
def test_commission_deducted_post_fill():
    broker, dh, pf = _make_setup(
        initial_cash=10_000.0, commission_fn=lambda a, q, p: 7.0
    )
    broker.submit_orders([{"asset": "AAPL", "qty": 5}], known_assets=dh.get_known_assets())
    log = broker.process_queue(dh.get_current_bar(), pf, dh)
    fill = log["filled"][0]
    assert fill["commission"] == 7.0
    # Cash = 10000 - 5 * 103.5 - 7 = 9475.5
    assert pf.cash == pytest.approx(10_000.0 - 5 * 103.5 - 7.0)


# ---------------------------------------------------------------------------
# Audit-trail invariant
# ---------------------------------------------------------------------------
def test_fill_log_audit_trail():
    """Every submitted order ends up in exactly one bucket: filled,
    rejected_overdraw, rejected_untradeable, or rejected_invalid."""
    broker, dh, pf = _make_setup(initial_cash=300.0)
    # Submit a mix: valid filled, valid overdraw, invalid (extra field),
    # invalid (unknown asset).
    orders = [
        {"asset": "AAPL", "qty": 1},       # fills
        {"asset": "AAPL", "qty": 100},     # overdraw
        {"asset": "AAPL", "qty": 5, "execution": "limit"},  # invalid
        {"asset": "UNKNOWN", "qty": 5},    # invalid (unknown)
    ]
    broker.submit_orders(orders, known_assets=dh.get_known_assets())
    log = broker.process_queue(dh.get_current_bar(), pf, dh)

    total_in_buckets = (
        len(log["filled"])
        + len(log["rejected_overdraw"])
        + len(log["rejected_untradeable"])
        + len(log["rejected_invalid"])
    )
    assert total_in_buckets == 4
    assert len(log["filled"]) == 1
    assert len(log["rejected_overdraw"]) == 1
    assert len(log["rejected_invalid"]) == 2


# ---------------------------------------------------------------------------
# drain_pending_as_untradeable (B7 / Q6 terminal-step path)
# ---------------------------------------------------------------------------
def test_drain_pending_as_untradeable_moves_all_pending_to_untradeable():
    broker, dh, pf = _make_setup()
    broker.submit_orders(
        [{"asset": "AAPL", "qty": 1}, {"asset": "MSFT", "qty": 2}],
        known_assets=dh.get_known_assets(),
    )
    log = broker.drain_pending_as_untradeable()
    assert len(log["rejected_untradeable"]) == 2
    assert broker.get_pending_count() == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
def test_reset_clears_pending_and_invalid_buffer():
    broker, dh, _ = _make_setup()
    broker.submit_orders([{"asset": "AAPL", "qty": 1}], known_assets=dh.get_known_assets())
    broker.submit_orders([{"asset": "AAPL", "qty": 0}], known_assets=dh.get_known_assets())
    broker.reset()
    assert broker.get_pending_count() == 0
    assert broker._invalid_buffer == []
