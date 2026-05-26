"""Stage 1 unit tests for Portfolio (per V1_PLAN § Stage 1)."""

from __future__ import annotations

import pytest

from backtesting.portfolio import Portfolio


def test_construction_state_matches_initial_cash():
    pf = Portfolio(initial_cash=100_000.0)
    state = pf.get_state()
    assert state["cash"] == 100_000.0
    assert state["equity"] == 100_000.0
    assert state["positions"] == {}
    assert state["unrealized_pnl"] == 0.0


def test_construction_rejects_negative_initial_cash():
    with pytest.raises(ValueError, match="initial_cash"):
        Portfolio(initial_cash=-1.0)


def test_single_buy_updates_cash_and_position():
    pf = Portfolio(100_000.0)
    pf.execute_trade("AAPL", qty=100, price=150.0)
    assert pf.cash == 100_000.0 - 100 * 150.0
    assert pf.positions == {"AAPL": 100}


def test_single_sell_updates_cash_and_position():
    pf = Portfolio(100_000.0)
    pf.execute_trade("AAPL", qty=-50, price=150.0)
    # Selling without prior position = short; cash inflow.
    assert pf.cash == 100_000.0 + 50 * 150.0
    assert pf.positions == {"AAPL": -50}


def test_multiple_trades_accumulate_signed_position():
    pf = Portfolio(1_000_000.0)
    pf.execute_trade("MSFT", qty=100, price=300.0)
    pf.execute_trade("MSFT", qty=50, price=305.0)
    pf.execute_trade("MSFT", qty=-30, price=310.0)
    assert pf.positions == {"MSFT": 120}
    expected_cash = 1_000_000.0 - 100 * 300.0 - 50 * 305.0 + 30 * 310.0
    assert pf.cash == pytest.approx(expected_cash)


def test_position_dropped_when_zeroed_out():
    pf = Portfolio(1_000_000.0)
    pf.execute_trade("GOOG", qty=10, price=2000.0)
    pf.execute_trade("GOOG", qty=-10, price=2100.0)
    assert "GOOG" not in pf.positions


def test_update_valuations_equity_consistency():
    pf = Portfolio(100_000.0)
    pf.execute_trade("AAPL", qty=100, price=150.0)
    pf.execute_trade("MSFT", qty=50, price=300.0)
    pf.update_valuations({"AAPL": 160.0, "MSFT": 290.0})
    expected_cash = 100_000.0 - 100 * 150.0 - 50 * 300.0
    expected_equity = expected_cash + 100 * 160.0 + 50 * 290.0
    assert pf.cash == pytest.approx(expected_cash)
    assert pf.equity == pytest.approx(expected_equity)


def test_update_valuations_missing_price_for_held_asset_raises():
    pf = Portfolio(100_000.0)
    pf.execute_trade("AAPL", qty=10, price=150.0)
    with pytest.raises(KeyError, match="AAPL"):
        pf.update_valuations({"MSFT": 300.0})


def test_overdraw_raises():
    pf = Portfolio(1_000.0)
    with pytest.raises(ValueError, match="overdraw"):
        pf.execute_trade("AAPL", qty=10, price=200.0)
    # State unchanged after failed trade.
    assert pf.cash == 1_000.0
    assert pf.positions == {}


def test_short_sale_does_not_raise_overdraw():
    pf = Portfolio(1_000.0)
    pf.execute_trade("AAPL", qty=-100, price=200.0)  # short — cash inflow
    assert pf.cash == 21_000.0
    assert pf.positions == {"AAPL": -100}


def test_reset_restores_initial_state():
    pf = Portfolio(50_000.0)
    pf.execute_trade("AAPL", qty=10, price=100.0)
    pf.update_valuations({"AAPL": 120.0})
    assert pf.equity != pf.initial_cash

    pf.reset()
    assert pf.cash == 50_000.0
    assert pf.positions == {}
    assert pf.equity == 50_000.0


def test_get_state_returns_independent_positions_copy():
    pf = Portfolio(100_000.0)
    pf.execute_trade("AAPL", qty=10, price=100.0)
    state = pf.get_state()
    state["positions"]["AAPL"] = 9999  # caller mutation
    assert pf.positions == {"AAPL": 10}  # internal state untouched


def test_zero_qty_trade_is_noop():
    pf = Portfolio(100_000.0)
    pf.execute_trade("AAPL", qty=0, price=150.0)
    assert pf.cash == 100_000.0
    assert pf.positions == {}
