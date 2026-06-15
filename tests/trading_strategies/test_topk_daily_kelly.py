"""Tests for TopKDailyKellyLabelExit (plan §6.5, D11-D14, D21-D23)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import Backtest
from backtesting.strategy import run_strategy
from trading_strategies.sizing import DiscreteBoundedLossKelly
from trading_strategies.topk_daily_kelly_label_exit import (
    TopKDailyKellyLabelExit,
)

WIN, LOSS = 0.10, 0.05
BREAKEVEN = 1.0 / 3.0  # 1/(1 + win/loss), win/loss = 2


def _mk_state(step, ts, equity, positions, closes, lookback=5):
    """Build a minimal engine-shaped state dict. close at OHLCV index 3."""
    md = {}
    for tk, c in closes.items():
        arr = np.zeros((lookback, 5), dtype=float)
        arr[-1, 3] = c
        md[tk] = arr
    return {
        "market_data": {"equities": md},
        "portfolio": {
            "equity": equity,
            "positions": dict(positions),
            "cash": equity,
            "pending_orders": 0,
        },
        "step": step,
        "timestamp": pd.Timestamp(ts),
    }


def _strategy(preds, **kw):
    defaults = dict(
        K=3,
        target_return=WIN,
        stop_drawdown=LOSS,
        horizon_days=50,
        sizer=DiscreteBoundedLossKelly(),
        sizer_payoffs=(WIN, LOSS),
        breakeven_p=BREAKEVEN,
        fractional_c=0.5,
    )
    defaults.update(kw)
    return TopKDailyKellyLabelExit(preds, **defaults)


# -- entry + sizing ----------------------------------------------------------
def test_entry_on_signal_day():
    d = pd.Timestamp("2025-01-02")
    s = _strategy({d: [("AAPL", 0.40, 0.3, 0.5)]})
    action = s(_mk_state(5, d, 100_000, {}, {"AAPL": 100.0}), {})
    assert action["type"] == "order"
    assert len(action["orders"]) == 1
    o = action["orders"][0]
    assert o["asset"] == "AAPL"
    # notional_f = c * f_risk / loss; f_risk(0.40) = (2*0.4-0.6)/2 = 0.10
    # notional_f = 0.5 * 0.10 / 0.05 = 1.0 → capped at room=1.0. Cash buffer
    # 2% → spendable 98_000 → shares = 98_000 / 100 = 980.
    assert o["qty"] == pytest.approx(980.0)
    assert "AAPL" in s._open
    assert s._open["AAPL"]["anchor_close"] == 100.0


def test_below_breakeven_not_entered():
    d = pd.Timestamp("2025-01-02")
    s = _strategy({d: [("AAPL", 0.30, 0.2, 0.4)]})  # below breakeven 1/3
    action = s(_mk_state(5, d, 100_000, {}, {"AAPL": 100.0}), {})
    assert action is None
    assert not s._open


def test_plow_selection_rejects_wide_band_pick():
    """selection_bound='low': p_mean clears breakeven but p_low doesn't → rejected."""
    d = pd.Timestamp("2025-01-02")
    # p_mean 0.40 > breakeven, but p_low 0.30 < breakeven (1/3) → wide band.
    s = _strategy({d: [("AAPL", 0.40, 0.30, 0.55)]}, selection_bound="low")
    action = s(_mk_state(5, d, 100_000, {}, {"AAPL": 100.0}), {})
    assert action is None
    assert not s._open


def test_plow_selection_accepts_tight_band_pick():
    """selection_bound='low': p_low above breakeven → accepted."""
    d = pd.Timestamp("2025-01-02")
    s = _strategy({d: [("AAPL", 0.40, 0.35, 0.45)]}, selection_bound="low")
    action = s(_mk_state(5, d, 100_000, {}, {"AAPL": 100.0}), {})
    assert action is not None
    assert "AAPL" in s._open


def test_mean_selection_default_ignores_plow():
    """Default selection_bound='mean': wide band still entered on p_mean."""
    d = pd.Timestamp("2025-01-02")
    s = _strategy({d: [("AAPL", 0.40, 0.10, 0.55)]})  # very wide band
    action = s(_mk_state(5, d, 100_000, {}, {"AAPL": 100.0}), {})
    assert "AAPL" in s._open


def test_invalid_selection_bound_raises():
    with pytest.raises(ValueError, match="selection_bound"):
        _strategy({}, selection_bound="median")


# -- V1.2 rank-sizing modes --------------------------------------------------
def test_rank_mode_trades_below_breakeven():
    """rank selection: a sub-breakeven pick (p<1/3) IS entered (gate dropped)."""
    d = pd.Timestamp("2025-01-02")
    s = _strategy({d: [("AAPL", 0.05, 0.02, 0.10)]},
                  selection_mode="rank", sizing_mode="equal", fractional_c=1.0)
    action = s(_mk_state(5, d, 100_000, {}, {"AAPL": 100.0}), {})
    assert action is not None and "AAPL" in s._open


def test_breakeven_mode_rejects_below_breakeven():
    """Contrast: default breakeven mode rejects the same sub-breakeven pick."""
    d = pd.Timestamp("2025-01-02")
    s = _strategy({d: [("AAPL", 0.05, 0.02, 0.10)]})
    assert s(_mk_state(5, d, 100_000, {}, {"AAPL": 100.0}), {}) is None


def test_equal_sizing_fraction():
    """equal sizing: per-position notional_f = fractional_c · gross_cap / K."""
    s = _strategy({}, selection_mode="rank", sizing_mode="equal",
                  fractional_c=1.0, K=3, gross_cap=1.0)
    assert s._notional_f(0.05) == pytest.approx(1.0 / 3.0)
    s2 = _strategy({}, selection_mode="rank", sizing_mode="equal",
                   fractional_c=0.5, K=4, gross_cap=1.0)
    assert s2._notional_f(0.99) == pytest.approx(0.5 * 1.0 / 4.0)  # p-independent


def test_equal_sizing_wide_k_enters():
    """_012 regression: equal sizing at wide K must enter.

    The old K-independent dust floor (max(5% equity, 10% room)) rejected every
    legitimate 1/K slice at K>=20 (5% slice < 10%-of-room floor on the first
    entry) → 0 entries forever. The equal-mode floor (half the intended slice)
    must let a full slice through.
    """
    d = pd.Timestamp("2025-01-02")
    picks = [(f"T{i:02d}", 0.05, 0.02, 0.10) for i in range(25)]  # 25 candidates
    closes = {f"T{i:02d}": 100.0 for i in range(25)}
    s = _strategy({d: picks}, selection_mode="rank", sizing_mode="equal",
                  fractional_c=1.0, K=20, gross_cap=1.0)
    action = s(_mk_state(5, d, 100_000, {}, closes), {})
    assert action is not None and action["type"] == "order"
    # All 20 top-K equal slices (5% each) clear the half-slice floor (2.5%).
    assert len(action["orders"]) == 20
    assert len(s._open) == 20


def test_equal_sizing_floor_still_drops_dust():
    """The equal-mode floor still drops a slice squeezed below half by tiny room."""
    d = pd.Timestamp("2025-01-02")
    # K=3 → intended slice 1/3; floor = 1/6. Pre-fill room so only ~10% remains,
    # squeezing actual_f (=min(1/3, room)) to 0.10 < floor 1/6 → dropped.
    s = _strategy({d: [("AAPL", 0.05, 0.02, 0.10)]},
                  selection_mode="rank", sizing_mode="equal", fractional_c=1.0,
                  K=3, gross_cap=1.0)
    s._open = {f"H{i}": {"entry_step": 0, "anchor_close": 100.0, "f": 0.30}
               for i in range(3)}  # exposure 0.90 → room 0.10
    action = s(_mk_state(6, d, 100_000, {}, {"AAPL": 100.0}), {})
    assert action is None or not any(o["asset"] == "AAPL"
                                     for o in (action or {}).get("orders", []))


def test_prob_weight_sizes_proportional_to_p():
    """_013 prob_weight: top-K positions sized ∝ calibrated p, summing to
    fractional_c·gross_cap; higher-p picks get bigger books (auto-concentration).
    """
    d = pd.Timestamp("2025-01-02")
    # three picks, p = 0.6 / 0.3 / 0.1 → weights 0.6 / 0.3 / 0.1 of gross_cap.
    picks = [("AAA", 0.60, 0.5, 0.7), ("BBB", 0.30, 0.2, 0.4), ("CCC", 0.10, 0.05, 0.15)]
    closes = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}
    s = _strategy({d: picks}, selection_mode="rank", sizing_mode="prob_weight",
                  fractional_c=1.0, K=3, gross_cap=1.0)
    s(_mk_state(5, d, 100_000, {}, closes), {})
    f = {tk: st["f"] for tk, st in s._open.items()}
    # AAA twice BBB, BBB thrice CCC; book sums to ~gross_cap (CCC may hit the
    # half-equal-slice dust floor at 0.10 < 0.5/3=0.167 → dropped).
    assert f["AAA"] == pytest.approx(0.60, abs=1e-6)
    assert f["BBB"] == pytest.approx(0.30, abs=1e-6)
    assert "CCC" not in f  # 0.10 share < 0.167 dust floor → dropped
    assert f["AAA"] > f["BBB"]  # higher p → bigger book (precision tilt)


def test_prob_weight_flat_p_approximates_equal():
    """prob_weight with equal p → equal weights (degenerates to equal sizing)."""
    d = pd.Timestamp("2025-01-02")
    picks = [(f"T{i}", 0.40, 0.3, 0.5) for i in range(4)]
    closes = {f"T{i}": 100.0 for i in range(4)}
    s = _strategy({d: picks}, selection_mode="rank", sizing_mode="prob_weight",
                  fractional_c=1.0, K=4, gross_cap=1.0)
    s(_mk_state(5, d, 100_000, {}, closes), {})
    fs = [st["f"] for st in s._open.values()]
    assert len(fs) == 4
    for fv in fs:
        assert fv == pytest.approx(0.25, abs=1e-6)  # flat p → equal 1/K slices


def test_rank_kelly_sizes_on_bucket_hitrate():
    """rank_kelly sizing uses rank_kelly_p (eval hit-rate), not the per-row p."""
    s = _strategy({}, selection_mode="rank", sizing_mode="rank_kelly",
                  rank_kelly_p=0.60, fractional_c=1.0)
    assert s._notional_f(0.05) == pytest.approx(s._notional_f(0.99))  # p-independent
    # Kelly on p_eff=0.60: f_risk=(b·p−q)/b, b=2 → (2·.6−.4)/2=0.4; /loss(.05)=8.0
    assert s._notional_f(0.05) == pytest.approx(1.0 * 0.4 / 0.05)


def test_rank_mode_disables_breakeven_exit():
    """rank mode: a held position with sub-breakeven p_today is NOT breakeven-exited."""
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    s = _strategy({d0: [("AAPL", 0.05, 0.02, 0.10)], d1: [("AAPL", 0.04, 0.01, 0.08)]},
                  selection_mode="rank", sizing_mode="equal", fractional_c=1.0)
    s(_mk_state(5, d0, 100_000, {}, {"AAPL": 100.0}), {})
    assert "AAPL" in s._open
    s(_mk_state(6, d1, 100_000, {"AAPL": 300.0}, {"AAPL": 100.0}), {})  # flat price
    assert not [e for e in s.events if e.kind == "exit"]
    assert "AAPL" in s._open


def test_rank_mode_still_honors_dd_exit():
    """rank mode keeps DD/target/horizon exits."""
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    s = _strategy({d0: [("AAPL", 0.05, 0.02, 0.10)], d1: [("AAPL", 0.05, 0.02, 0.10)]},
                  selection_mode="rank", sizing_mode="equal", fractional_c=1.0)
    s(_mk_state(5, d0, 100_000, {}, {"AAPL": 100.0}), {})
    s(_mk_state(6, d1, 100_000, {"AAPL": 300.0}, {"AAPL": 94.0}), {})  # below 95 → DD
    assert any(e.kind == "exit" and e.trigger == "DD" for e in s.events)


def test_invalid_rank_modes_raise():
    with pytest.raises(ValueError, match="selection_mode"):
        _strategy({}, selection_mode="topk")
    with pytest.raises(ValueError, match="sizing_mode"):
        _strategy({}, sizing_mode="kelly2")
    with pytest.raises(ValueError, match="rank_kelly_p"):
        _strategy({}, sizing_mode="rank_kelly")


def test_tie_break_alphabetical(monkeypatch):
    """D21: equal p → alphabetically-lower ticker selected first."""
    d = pd.Timestamp("2025-01-02")
    # K=1, two tied candidates; ZZZZ vs AAAA at equal p → AAAA wins.
    s = _strategy(
        {d: [("ZZZZ", 0.40, 0.3, 0.5), ("AAAA", 0.40, 0.3, 0.5)]}, K=1
    )
    action = s(_mk_state(5, d, 100_000, {}, {"ZZZZ": 100.0, "AAAA": 50.0}), {})
    assert [o["asset"] for o in action["orders"]] == ["AAAA"]


# -- exits -------------------------------------------------------------------
def _enter(s, d, equity=100_000, close=100.0, p=0.40):
    s(_mk_state(5, d, equity, {}, {"AAPL": close}), {})
    return s._open["AAPL"]["anchor_close"]


def test_dd_exit():
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    s = _strategy({d0: [("AAPL", 0.40, 0.3, 0.5)], d1: [("AAPL", 0.40, 0.3, 0.5)]})
    anchor = _enter(s, d0)  # anchor 100
    # next day close at 94.9 = below 0.95*100 = 95 → DD exit
    action = s(_mk_state(6, d1, 100_000, {"AAPL": 1000}, {"AAPL": 94.9}), {})
    assert action["orders"][0]["qty"] == -1000  # sell all
    assert "AAPL" not in s._open
    assert s.events[-1].trigger == "DD"


def test_target_exit():
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    s = _strategy({d0: [("AAPL", 0.40, 0.3, 0.5)], d1: [("AAPL", 0.40, 0.3, 0.5)]})
    _enter(s, d0)  # anchor 100
    action = s(_mk_state(6, d1, 100_000, {"AAPL": 1000}, {"AAPL": 110.1}), {})
    assert action["orders"][0]["qty"] == -1000
    assert s.events[-1].trigger == "target"


def test_horizon_exit():
    d0 = pd.Timestamp("2025-01-02")
    dh = pd.Timestamp("2025-04-01")
    s = _strategy({d0: [("AAPL", 0.40, 0.3, 0.5)], dh: [("AAPL", 0.40, 0.3, 0.5)]})
    _enter(s, d0)  # entry_step 5
    # step 55 = 50 BD held → horizon exit (close flat, no DD/target)
    action = s(_mk_state(55, dh, 100_000, {"AAPL": 1000}, {"AAPL": 100.0}), {})
    assert action["orders"][0]["qty"] == -1000
    assert s.events[-1].trigger == "horizon"


def test_breakeven_exit():
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    s = _strategy(
        {d0: [("AAPL", 0.40, 0.3, 0.5)], d1: [("AAPL", 0.30, 0.2, 0.4)]}
    )
    _enter(s, d0)
    # close flat (no DD/target), but p_today 0.30 <= breakeven → exit
    action = s(_mk_state(6, d1, 100_000, {"AAPL": 1000}, {"AAPL": 100.0}), {})
    assert action["orders"][0]["qty"] == -1000
    assert s.events[-1].trigger == "breakeven"


def test_missing_p_today_skips_breakeven_and_trim():
    """D22: ticker absent today → no breakeven exit, no trim; position held."""
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    # d1 has NO prediction for AAPL.
    s = _strategy({d0: [("AAPL", 0.40, 0.3, 0.5)], d1: [("OTHER", 0.40, 0.3, 0.5)]})
    _enter(s, d0)
    # close flat → no DD/target/horizon; AAPL absent → held, no order for it.
    action = s(_mk_state(6, d1, 100_000, {"AAPL": 1000}, {"AAPL": 100.0, "OTHER": 100.0}), {})
    assert "AAPL" in s._open  # still held
    aapl_orders = [o for o in (action["orders"] if action else []) if o["asset"] == "AAPL"]
    assert aapl_orders == []  # no exit, no trim


def test_dd_still_fires_when_p_today_missing():
    """D22 only skips breakeven/trim — DD/target/horizon still fire."""
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    s = _strategy({d0: [("AAPL", 0.40, 0.3, 0.5)], d1: []})
    _enter(s, d0)
    action = s(_mk_state(6, d1, 100_000, {"AAPL": 1000}, {"AAPL": 94.0}), {})
    assert action["orders"][0] == {"asset": "AAPL", "qty": -1000}
    assert s.events[-1].trigger == "DD"


# -- trim --------------------------------------------------------------------
def test_trim_ratchet_down():
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    # Enter at p=0.50 (high f), then p drops to 0.36 (lower f, still > breakeven).
    s = _strategy(
        {d0: [("AAPL", 0.50, 0.4, 0.6)], d1: [("AAPL", 0.36, 0.3, 0.45)]}
    )
    s(_mk_state(5, d0, 100_000, {}, {"AAPL": 100.0}), {})
    booked_f = s._open["AAPL"]["f"]
    shares0 = s.events[-1].shares_after
    # Day 2: p lower → new_f < cur_f → trim (sell delta). Close flat at 100.
    action = s(_mk_state(6, d1, 100_000, {"AAPL": shares0}, {"AAPL": 100.0}), {})
    trim_orders = [o for o in action["orders"] if o["asset"] == "AAPL"]
    assert len(trim_orders) == 1
    assert trim_orders[0]["qty"] < 0  # a sell (ratchet down)
    assert s._open["AAPL"]["f"] < booked_f
    assert s._open["AAPL"]["anchor_close"] == 100.0  # D12: anchor unchanged
    assert s.events[-1].kind == "trim"


def test_no_add_up_when_p_rises():
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    # Enter at p=0.36 (small f), then p rises to 0.50 → must NOT add up.
    s = _strategy(
        {d0: [("AAPL", 0.36, 0.3, 0.45)], d1: [("AAPL", 0.50, 0.4, 0.6)]}, K=1
    )
    s(_mk_state(5, d0, 100_000, {}, {"AAPL": 100.0}), {})
    shares0 = s.events[-1].shares_after
    action = s(_mk_state(6, d1, 100_000, {"AAPL": shares0}, {"AAPL": 100.0}), {})
    # AAPL already held → not re-entered; p rose → no trim → no order for AAPL.
    aapl_orders = [o for o in (action["orders"] if action else []) if o["asset"] == "AAPL"]
    assert aapl_orders == []


def test_rebalance_off_holds_until_label_exit():
    """§7 counterfactual: enable_rebalance=False → no breakeven/trim."""
    d0, d1 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    s = _strategy(
        {d0: [("AAPL", 0.40, 0.3, 0.5)], d1: [("AAPL", 0.30, 0.2, 0.4)]},
        enable_rebalance=False,
    )
    _enter(s, d0)
    # p_today below breakeven would exit if rebalance ON; here it must hold.
    action = s(_mk_state(6, d1, 100_000, {"AAPL": 1000}, {"AAPL": 100.0}), {})
    assert "AAPL" in s._open
    aapl = [o for o in (action["orders"] if action else []) if o["asset"] == "AAPL"]
    assert aapl == []


# -- floor + cap -------------------------------------------------------------
def test_room_cap_drops_later_picks():
    """Gross cap binds: first high-p pick consumes ~all room, next dropped."""
    d = pd.Timestamp("2025-01-02")
    s = _strategy(
        {d: [("AAA", 0.40, 0.3, 0.5), ("BBB", 0.40, 0.3, 0.5)]}, K=3
    )
    action = s(_mk_state(5, d, 100_000, {}, {"AAA": 100.0, "BBB": 100.0}), {})
    # AAA notional_f = 1.0 → room exhausted → BBB dropped.
    assert [o["asset"] for o in action["orders"]] == ["AAA"]
    assert "BBB" not in s._open


# -- integration through the real engine ------------------------------------
def _ohlcv(dates, path):
    """Build OHLCV where close follows `path`; open = prior close (flat-ish)."""
    close = np.asarray(path, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(close), 1e6),
        },
        index=dates,
    )


def test_integration_through_engine_runs_and_trades():
    dates = pd.date_range("2025-01-01", periods=20, freq="B")
    # AAPL climbs to +10% then we expect a target exit; MSFT flat.
    aapl_path = np.linspace(100, 130, 20)
    msft_path = np.full(20, 200.0)
    feeds = {"equities": {"AAPL": _ohlcv(dates, aapl_path), "MSFT": _ohlcv(dates, msft_path)}}
    # Signal on day index 5 (the first decision step at lookback=5).
    preds = {dates[5]: [("AAPL", 0.45, 0.35, 0.55)]}
    # also keep AAPL present on later days so rebalance can evaluate it
    for i in range(6, 20):
        preds[dates[i]] = [("AAPL", 0.45, 0.35, 0.55)]
    strat = _strategy(preds, horizon_days=50)
    bt = Backtest(feeds, lookback=5, initial_cash=100_000.0, fill_mode="next_open")
    history = run_strategy(bt, strat)
    assert len(history) > 0
    # An entry happened and a target exit eventually fired.
    kinds = [e.kind for e in strat.events]
    assert "entry" in kinds
    triggers = [e.trigger for e in strat.events if e.kind == "exit"]
    assert "target" in triggers
    # Final equity should have grown (rode AAPL up ~10%).
    final_equity = history[-1][0]["portfolio"]["equity"]
    assert final_equity > 100_000.0
