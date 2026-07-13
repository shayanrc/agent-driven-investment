"""V1.4.1 — explicit-boundary date_aligned split unit tests.

The date_aligned split can define its four segments either by trading-day
durations (train_rows/val_rows/eval_rows/test_rows walked from train_start) or
by explicit boundary dates (val_start / eval_start / test_start / test_end,
with train_start as the anchor). These verify the explicit form:

1. Produces the exact calendar windows, with start dates snapping to the first
   trading day >= the date and test_end to the last trading day <= it.
2. Is byte-equivalent to the duration form when both describe the same windows.
3. Validates: all-four-or-none, strict ordering, date_aligned-only.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gbdt.spec import _validate_spec
from gbdt.train import (
    SplitSpec,
    carve_universe_aligned,
    segment_bound_indices,
)


def _cal(start: str = "2015-01-01", periods: int = 2000) -> pd.DatetimeIndex:
    # Business-day calendar — no holiday subtleties needed for boundary logic.
    return pd.DatetimeIndex(pd.date_range(start, periods=periods, freq="B"))


def _panel(cal: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(cal))))
    df = pd.DataFrame({
        "date": cal, "ticker": "DEEP",
        "open": c, "high": c * 1.005, "low": c * 0.995,
        "close": c, "adj_close": c, "volume": np.ones(len(cal), dtype=int),
    })
    return df.set_index(["date", "ticker"]).sort_index()


# --------------------------------------------------------------------------
# core behaviour
# --------------------------------------------------------------------------


def test_explicit_boundaries_produce_exact_windows():
    cal = _cal(periods=2000)
    split = SplitSpec(
        mode="date_aligned",
        train_start=cal[0].date(),
        val_start=cal[600].date(),
        eval_start=cal[900].date(),
        test_start=cal[1050].date(),
        test_end=cal[1099].date(),
    )
    assert split.has_explicit_boundaries
    b = segment_bound_indices(split, cal)
    assert b["train"] == (0, 599)
    assert b["val"] == (600, 899)
    assert b["eval"] == (900, 1049)
    assert b["test"] == (1050, 1099)


def test_explicit_equals_duration_when_equivalent():
    """A duration split and an explicit split describing the SAME windows
    yield identical segment_dates AND identical per-ticker fold indices."""
    cal = _cal(periods=2000)
    panel = _panel(cal)
    dur = SplitSpec(
        mode="date_aligned", train_start=cal[0].date(),
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
    )
    expl = SplitSpec(
        mode="date_aligned", train_start=cal[0].date(),
        val_start=cal[600].date(), eval_start=cal[900].date(),
        test_start=cal[1050].date(), test_end=cal[1099].date(),
    )
    assert segment_bound_indices(dur, cal) == segment_bound_indices(expl, cal)
    f_dur = carve_universe_aligned(panel, dur, cal)
    f_expl = carve_universe_aligned(panel, expl, cal)
    assert f_dur.segment_dates == f_expl.segment_dates
    for seg in ("train_idx", "val_idx", "eval_idx", "test_idx"):
        a, b = getattr(f_dur, seg), getattr(f_expl, seg)
        assert a.keys() == b.keys()
        for k in a:
            assert np.array_equal(a[k], b[k])


def test_start_snaps_forward_test_end_snaps_back():
    """Start dates that fall on non-trading days advance to the next trading
    day; test_end falls back to the last trading day <= it."""
    cal = _cal(start="2015-01-01", periods=2000)
    # Ordered boundaries; val_start on a Saturday, test_end on a Sunday.
    sat = pd.Timestamp("2018-01-06")   # Saturday
    sun = pd.Timestamp("2020-06-28")   # Sunday
    assert sat.dayofweek == 5 and sun.dayofweek == 6
    split = SplitSpec(
        mode="date_aligned", train_start=cal[0].date(),
        val_start=sat.date(), eval_start=date(2019, 1, 7),
        test_start=date(2020, 1, 6), test_end=sun.date(),
    )
    b = segment_bound_indices(split, cal)
    # val starts on the first trading day AFTER the Saturday (the Monday).
    assert cal[b["val"][0]] > sat and cal[b["val"][0] - 1] < sat
    assert cal[b["val"][0]].dayofweek == 0
    # test ends on the last trading day BEFORE the Sunday (the Friday).
    assert cal[b["test"][1]] < sun and cal[b["test"][1]].dayofweek == 4


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_unordered_boundaries_raise():
    cal = _cal(periods=2000)
    split = SplitSpec(
        mode="date_aligned", train_start=cal[0].date(),
        val_start=cal[900].date(),        # after eval_start — out of order
        eval_start=cal[600].date(),
        test_start=cal[1050].date(), test_end=cal[1099].date(),
    )
    with pytest.raises(ValueError, match="boundaries"):
        segment_bound_indices(split, cal)


def test_partial_boundaries_not_explicit():
    split = SplitSpec(
        mode="date_aligned", train_start=date(2015, 1, 1),
        val_start=date(2022, 3, 30),   # only one of four set
    )
    assert not split.has_explicit_boundaries


_TARGET = {"universe": "sp500", "direction": "up",
           "threshold_pct": 50, "horizon_days": 50, "max_drawdown": 0.25}


@pytest.mark.parametrize("bad", [
    # 3 of 4 boundary dates → incomplete
    {"mode": "date_aligned", "train_start": "2015-01-01",
     "val_start": "2022-03-30", "eval_start": "2023-07-01",
     "test_start": "2024-07-01"},
    # explicit boundaries but trailing mode
    {"mode": "trailing", "train_start": "2015-01-01",
     "val_start": "2022-03-30", "eval_start": "2023-07-01",
     "test_start": "2024-07-01", "test_end": "2025-06-30"},
    # out-of-order (test_start before eval_start)
    {"mode": "date_aligned", "train_start": "2015-01-01",
     "val_start": "2022-03-30", "eval_start": "2024-07-01",
     "test_start": "2023-07-01", "test_end": "2025-06-30"},
])
def test_validate_spec_rejects_bad_explicit_boundaries(bad):
    with pytest.raises(ValueError, match="split|boundar|increasing|date_aligned"):
        _validate_spec({"target": _TARGET, "split": bad})


def test_validate_spec_accepts_full_ordered_explicit_boundaries():
    _validate_spec({"target": _TARGET, "split": {
        "mode": "date_aligned", "train_start": "2015-01-01",
        "val_start": "2022-03-30", "eval_start": "2023-07-01",
        "test_start": "2024-07-01", "test_end": "2025-06-30",
    }})  # must not raise
