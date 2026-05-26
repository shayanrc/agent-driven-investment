"""Stage 7 tests for the result-aggregation surface + info-schema lock.

Coverage:
- validate_info_schema: rejects unknown keys, rejects empty list/dict
  containers (omit-when-empty rule from spec.md § 3.2).
- summarize_run: builds equity_curve / fills / rejections / final_state.
- weight_drift omission rules (a no-drift weight action must NOT emit
  the key).
- rebalance_shortfall omission rules.
- lot_size_audit omission rules (no audit entry when requested == filled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import Backtest
from backtesting.results import (
    ALLOWED_INFO_KEYS,
    DICT_VALUED_INFO_KEYS,
    LIST_VALUED_INFO_KEYS,
    RunSummary,
    summarize_run,
    validate_info_schema,
)
from backtesting.strategy import (
    HoldStrategy,
    ScriptedActionStrategy,
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


def _bt(n: int = 20, **kw) -> Backtest:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    feeds = {
        "equities": {
            "AAPL": _ohlcv(dates, 100.0),
            "MSFT": _ohlcv(dates, 200.0),
        }
    }
    return Backtest(feeds, lookback=3, **kw)


# ---------------------------------------------------------------------------
# validate_info_schema
# ---------------------------------------------------------------------------
def test_empty_info_passes_validation():
    validate_info_schema({})  # no raise


def test_info_with_all_locked_keys_passes():
    validate_info_schema(
        {
            "fills": [{"asset": "AAPL", "qty": 1, "fill_price": 100, "commission": 0}],
            "rejected_overdraw": [{"asset": "X", "qty": 1}],
            "rejected_untradeable": [{"asset": "X", "qty": 1}],
            "rejected_invalid": [{"asset": "X"}],
            "weight_drift": {"AAPL": 0.01},
            "rebalance_shortfall": {"AAPL": 1.0},
            "lot_size_audit": {"AAPL": {"requested_qty": 0.4, "filled_qty": 0}},
        }
    )


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown key"):
        validate_info_schema({"unexpected": []})


def test_list_keys_must_be_non_empty():
    """List-valued keys (fills / rejected_*) must omit when empty."""
    for k in LIST_VALUED_INFO_KEYS:
        with pytest.raises(ValueError, match="empty"):
            validate_info_schema({k: []})


def test_dict_keys_must_be_non_empty():
    """Dict-valued keys (weight_drift / rebalance_shortfall / lot_size_audit)
    must omit when empty."""
    for k in DICT_VALUED_INFO_KEYS:
        with pytest.raises(ValueError, match="empty"):
            validate_info_schema({k: {}})


def test_list_key_must_be_list_not_dict():
    with pytest.raises(ValueError, match="list"):
        validate_info_schema({"fills": {"AAPL": 1}})


def test_dict_key_must_be_dict_not_list():
    with pytest.raises(ValueError, match="dict"):
        validate_info_schema({"weight_drift": [1.0]})


def test_allowed_info_keys_count_is_seven():
    """Sanity: the schema lock pins seven keys total. Any change is a
    spec amendment, not a quiet bug."""
    assert len(ALLOWED_INFO_KEYS) == 7


# ---------------------------------------------------------------------------
# Engine-level: weight_drift / rebalance_shortfall / lot_size_audit omission
# ---------------------------------------------------------------------------
def test_engine_omits_weight_drift_when_no_weight_action():
    """A pure order action must NOT populate info['weight_drift']."""
    bt = _bt()
    _s, _d, info = bt.step(
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 1}]}
    )
    assert "weight_drift" not in info


def test_engine_omits_rebalance_shortfall_when_no_overdraw():
    """A successful weight rebalance with sufficient cash must NOT
    populate info['rebalance_shortfall']."""
    bt = _bt(lot_sizes={"AAPL": 0, "MSFT": 0})
    _s, _d, info = bt.step(
        {"type": "weight", "target_weights": {"AAPL": 0.2, "MSFT": 0.2}}
    )
    assert "rebalance_shortfall" not in info


def test_engine_omits_lot_size_audit_when_no_rounding():
    """An order whose qty is already a valid multiple of the lot size
    must NOT populate info['lot_size_audit']."""
    bt = _bt(lot_sizes={"AAPL": 1})
    _s, _d, info = bt.step(
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 5}]}  # whole share, no snap
    )
    assert "lot_size_audit" not in info


def test_engine_emits_lot_size_audit_on_snap_to_zero():
    """An order whose snapped qty is zero is reported in lot_size_audit
    with filled_qty: 0 (Q10 unified-key rule)."""
    bt = _bt(lot_sizes={"AAPL": 1})
    _s, _d, info = bt.step(
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 0.3}]}
    )
    assert "lot_size_audit" in info
    assert info["lot_size_audit"]["AAPL"]["filled_qty"] == 0
    assert "fills" not in info  # nothing to fill


def test_engine_emits_lot_size_audit_on_truncation():
    """An order whose snapped qty truncates (but is non-zero) is also
    reported."""
    bt = _bt(lot_sizes={"AAPL": 10})  # round lots of 10
    _s, _d, info = bt.step(
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 57}]}  # snaps to 50
    )
    assert "lot_size_audit" in info
    assert info["lot_size_audit"]["AAPL"]["requested_qty"] == 57
    assert info["lot_size_audit"]["AAPL"]["filled_qty"] == 50


# ---------------------------------------------------------------------------
# summarize_run
# ---------------------------------------------------------------------------
def test_summarize_run_on_hold_strategy():
    bt = _bt(n=12)
    history = run_strategy(bt, HoldStrategy())
    summary = summarize_run(history)
    assert isinstance(summary, RunSummary)
    # initial == final == initial_cash (no trades).
    assert summary.initial_equity == 100_000.0
    assert summary.final_equity == 100_000.0
    assert summary.total_return == 0.0
    assert summary.fills == []
    assert summary.terminal_done is True
    # n_steps = total entries - 1 (the reset doesn't count as a step).
    assert summary.n_steps == len(history) - 1


def test_summarize_run_records_fills_and_equity_curve():
    bt = _bt(n=10, fill_mode="current_close")
    script = [
        {"type": "order", "orders": [{"asset": "AAPL", "qty": 10}]},
        None,
        None,
        {"type": "order", "orders": [{"asset": "AAPL", "qty": -10}]},
    ]
    history = run_strategy(bt, ScriptedActionStrategy(script))
    summary = summarize_run(history)
    assert len(summary.fills) == 2
    assert summary.fills[0]["qty"] == 10
    assert summary.fills[1]["qty"] == -10
    # Equity curve length == history length.
    assert len(summary.equity_curve) == len(history)


def test_summarize_run_emits_validates_each_step():
    """If the engine ever emits an unknown info key, summarize_run must
    raise immediately. We simulate this by post-hoc poisoning history."""
    bt = _bt(n=5)
    history = run_strategy(bt, HoldStrategy())
    poisoned = list(history)
    bad_state, bad_done, bad_info = poisoned[2]
    poisoned[2] = (bad_state, bad_done, {"unexpected_key": ["bad"]})
    with pytest.raises(ValueError, match="unknown key"):
        summarize_run(poisoned)


def test_summarize_run_empty_history_returns_empty_summary():
    summary = summarize_run([])
    assert summary.n_steps == 0
    assert summary.equity_curve == []
    assert summary.initial_equity is None
    assert summary.final_equity is None
    assert summary.total_return is None
    assert summary.final_state is None


def test_summarize_run_with_max_steps_cap_terminal_done_false():
    bt = _bt(n=50)
    history = run_strategy(bt, HoldStrategy(), max_steps=5)
    summary = summarize_run(history)
    assert summary.terminal_done is False
    assert summary.n_steps == 5
