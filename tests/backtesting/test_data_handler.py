"""Stage 2 unit tests for DataHandler + timeline correctness harness.

Includes the row-index-sentinel B1 harness (the canonical look-ahead-leak
detector). Every later stage that touches data access must keep that test
green.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.data_handler import DataHandler


def _make_ohlcv(dates: pd.DatetimeIndex, start_price: float = 100.0) -> pd.DataFrame:
    n = len(dates)
    base = np.arange(n, dtype=float) + start_price
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


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_single_feed_single_asset_construction():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = _make_ohlcv(dates)
    dh = DataHandler({"equities": {"AAPL": df}}, lookback=5)
    assert dh.max_steps == 30
    assert dh.current_step == 5
    assert len(dh.timeline) == 30


def test_multi_asset_union_timeline():
    a = _make_ohlcv(pd.date_range("2024-01-01", periods=10, freq="B"))
    b = _make_ohlcv(pd.date_range("2024-01-08", periods=10, freq="B"))
    dh = DataHandler({"equities": {"A": a, "B": b}}, lookback=3)
    # Union should be the merged range; both inputs are 10 business days,
    # overlapping from 2024-01-08; result is dates 2024-01-01 .. last(b).
    assert len(dh.timeline) > 10  # strictly more than either individually
    # A is active 2024-01-01 .. 2024-01-12 (10 business days).
    a_mask = dh.active["equities"]["A"]
    b_mask = dh.active["equities"]["B"]
    # First date: A active, B inactive (pre-IPO).
    assert a_mask[0] and not b_mask[0]
    # Last date: B active, A inactive (delisted).
    assert b_mask[-1] and not a_mask[-1]


def test_mid_series_start_asset_has_nan_before_first_obs():
    a = _make_ohlcv(pd.date_range("2024-01-01", periods=20, freq="B"))
    b = _make_ohlcv(pd.date_range("2024-01-15", periods=10, freq="B"))
    dh = DataHandler({"equities": {"A": a, "B": b}}, lookback=3)
    b_data = dh.data["equities"]["B"]
    b_mask = dh.active["equities"]["B"]
    # Rows before B's first observation should be NaN and inactive.
    pre_b_mask = ~b_mask
    if pre_b_mask.any():
        first_pre_idx = int(np.where(pre_b_mask)[0][0])
        assert pd.isna(b_data.iloc[first_pre_idx]["close"])


def test_mid_series_end_asset_ffilled_after_last_obs():
    a = _make_ohlcv(pd.date_range("2024-01-01", periods=20, freq="B"))
    b = _make_ohlcv(pd.date_range("2024-01-01", periods=10, freq="B"))
    dh = DataHandler({"equities": {"A": a, "B": b}}, lookback=3)
    b_data = dh.data["equities"]["B"]
    b_mask = dh.active["equities"]["B"]
    last_b_close = b.iloc[-1]["close"]
    # All rows past B's last obs should be forward-filled to that value
    # and inactive (untradeable for buys; sells permitted at last-known).
    post_idx = int(np.where(~b_mask)[0][-1])
    assert b_data.iloc[post_idx]["close"] == last_b_close
    assert not b_mask[post_idx]


# ---------------------------------------------------------------------------
# Window slicing
# ---------------------------------------------------------------------------
def test_get_window_at_first_valid_step_returns_full_length_window():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = _make_ohlcv(dates)
    dh = DataHandler({"equities": {"AAPL": df}}, lookback=5)
    w = dh.get_window()
    arr = w["equities"]["AAPL"]
    assert arr.shape == (5, 5)  # 5 lookback rows × 5 OHLCV columns
    assert not np.isnan(arr).any()


def test_get_current_bar_returns_dict_of_column_values():
    dates = pd.date_range("2024-01-01", periods=15, freq="B")
    df = _make_ohlcv(dates)
    dh = DataHandler({"equities": {"AAPL": df}}, lookback=3)
    bar = dh.get_current_bar()
    assert "equities" in bar
    assert "AAPL" in bar["equities"]
    assert set(bar["equities"]["AAPL"].keys()) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }


def test_get_price_returns_close_by_default():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = _make_ohlcv(dates, start_price=100.0)
    dh = DataHandler({"equities": {"AAPL": df}}, lookback=3)
    # current_step starts at 3, so index 3 close = 100 + 3 + 0.5 = 103.5
    assert dh.get_price("AAPL") == pytest.approx(103.5)


def test_get_price_unknown_asset_raises():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    dh = DataHandler({"equities": {"AAPL": _make_ohlcv(dates)}}, lookback=3)
    with pytest.raises(KeyError, match="MSFT"):
        dh.get_price("MSFT")


# ---------------------------------------------------------------------------
# advance_time semantics (Q8 / B7)
# ---------------------------------------------------------------------------
def test_advance_time_increments_until_last_step():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    dh = DataHandler({"equities": {"AAPL": _make_ohlcv(dates)}}, lookback=3)
    # 10 dates, lookback=3 → start at 3; final valid step = 9.
    done_log = []
    for _ in range(20):
        done_log.append(dh.advance_time())
    # After enough advances, done flips True and stays True; current_step
    # pinned to max_steps - 1.
    assert any(done_log)
    assert dh.current_step == dh.max_steps - 1
    # Q8 no-mutate: repeated advance after done stays pinned.
    for _ in range(5):
        assert dh.advance_time() is True
        assert dh.current_step == dh.max_steps - 1


def test_advance_time_invariant_holds_throughout_lifetime():
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    dh = DataHandler({"equities": {"AAPL": _make_ohlcv(dates)}}, lookback=5)
    assert dh.lookback <= dh.current_step <= dh.max_steps - 1
    for _ in range(50):
        dh.advance_time()
        assert dh.lookback <= dh.current_step <= dh.max_steps - 1


def test_reset_returns_to_first_valid_step():
    dates = pd.date_range("2024-01-01", periods=15, freq="B")
    dh = DataHandler({"equities": {"AAPL": _make_ohlcv(dates)}}, lookback=4)
    for _ in range(10):
        dh.advance_time()
    dh.reset()
    assert dh.current_step == 4


# ---------------------------------------------------------------------------
# Q9 — internal-gap policy
# ---------------------------------------------------------------------------
def _frame_with_internal_gap() -> pd.DataFrame:
    dates_full = pd.date_range("2024-01-01", periods=12, freq="B")
    # Drop one date inside the active range (index 5).
    keep = list(dates_full)
    dropped = keep.pop(5)
    df = _make_ohlcv(pd.DatetimeIndex(keep))
    return df, dropped


def test_internal_gap_raises_when_gap_policy_is_raise():
    df, _ = _frame_with_internal_gap()
    # Build a sibling feed with the full timeline so the gap becomes
    # *internal* (the master timeline includes the missing date because
    # another asset has it).
    full_dates = pd.date_range("2024-01-01", periods=12, freq="B")
    sibling = _make_ohlcv(full_dates)
    with pytest.raises(ValueError, match="Internal gap"):
        DataHandler(
            {"equities": {"gappy": df, "full": sibling}},
            lookback=3,
            gap_policy="raise",
        )


def test_internal_gap_ffills_when_gap_policy_is_default():
    df, dropped_date = _frame_with_internal_gap()
    full_dates = pd.date_range("2024-01-01", periods=12, freq="B")
    sibling = _make_ohlcv(full_dates)
    dh = DataHandler(
        {"equities": {"gappy": df, "full": sibling}},
        lookback=3,
        gap_policy="ffill_zero_volume",
    )
    reindexed = dh.data["equities"]["gappy"]
    gap_pos = dh.timeline.get_loc(dropped_date)
    prev_close = reindexed.iloc[gap_pos - 1]["close"]
    gap_close = reindexed.iloc[gap_pos]["close"]
    assert gap_close == prev_close
    # Volume on gap day is zero.
    assert reindexed.iloc[gap_pos]["volume"] == 0
    # Asset is still active on the gap day (the spec treats it as
    # "no trading happened" but the bar is still tradeable).
    assert dh.active["equities"]["gappy"][gap_pos]


# ---------------------------------------------------------------------------
# B1 — row-index sentinel harness (the canonical look-ahead-leak detector)
# ---------------------------------------------------------------------------
def test_row_index_sentinel_no_lookahead():
    """Every value at row t equals t; assert no value > current_step
    leaks into the state window at any step."""
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    sentinel = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "open": sentinel,
            "high": sentinel,
            "low": sentinel,
            "close": sentinel,
            "volume": sentinel,
        },
        index=dates,
    )
    dh = DataHandler({"equities": {"A": df}}, lookback=5)
    done = False
    while not done:
        T = dh.current_step
        window = dh.get_window()["equities"]["A"]
        assert window.max() <= T, (
            f"look-ahead leak: at step {T}, window max = {window.max()}"
        )
        # Also assert the last row equals exactly T.
        assert window[-1].max() == T
        done = dh.advance_time()


# ---------------------------------------------------------------------------
# Activity / lookup helpers
# ---------------------------------------------------------------------------
def test_is_active_and_get_active_assets():
    a = _make_ohlcv(pd.date_range("2024-01-01", periods=20, freq="B"))
    b = _make_ohlcv(pd.date_range("2024-01-08", periods=10, freq="B"))
    dh = DataHandler({"equities": {"A": a, "B": b}}, lookback=2)
    # At step 2 (early in timeline), only A active.
    assert dh.is_active("A")
    assert not dh.is_active("B")
    assert dh.get_active_assets() == {"A"}
    # Advance until B becomes active.
    while not dh.is_active("B"):
        dh.advance_time()
    assert {"A", "B"}.issubset(dh.get_active_assets())


def test_get_known_assets_includes_all():
    a = _make_ohlcv(pd.date_range("2024-01-01", periods=10, freq="B"))
    b = _make_ohlcv(pd.date_range("2024-01-01", periods=10, freq="B"))
    dh = DataHandler({"equities": {"A": a, "B": b}}, lookback=2)
    assert dh.get_known_assets() == {"A", "B"}


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------
def test_construction_rejects_invalid_gap_policy():
    df = _make_ohlcv(pd.date_range("2024-01-01", periods=10, freq="B"))
    with pytest.raises(ValueError, match="gap_policy"):
        DataHandler({"equities": {"A": df}}, lookback=3, gap_policy="silly")


def test_construction_rejects_lookback_too_large():
    df = _make_ohlcv(pd.date_range("2024-01-01", periods=10, freq="B"))
    with pytest.raises(ValueError, match="timeline length"):
        DataHandler({"equities": {"A": df}}, lookback=10)


def test_construction_rejects_lookback_below_one():
    df = _make_ohlcv(pd.date_range("2024-01-01", periods=10, freq="B"))
    with pytest.raises(ValueError, match="lookback"):
        DataHandler({"equities": {"A": df}}, lookback=0)


def test_construction_rejects_empty_feed():
    with pytest.raises(ValueError, match="no assets"):
        DataHandler({"equities": {}}, lookback=3)


def test_construction_rejects_no_feeds():
    with pytest.raises(ValueError, match="at least one feed"):
        DataHandler({}, lookback=3)
