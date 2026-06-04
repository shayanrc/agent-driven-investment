"""V1.4 — date-aligned carve unit tests.

Synthetic 3-ticker panel with mixed histories:

- ``DEEP``:   full history covering the entire calendar window.
- ``MID``:    history starts mid-train; should fail the train gate
              (< ``min_train_rows_per_ticker``) but pass val/eval/test.
- ``IPO``:    very-late IPO; should fail train AND val gates and pass
              only eval/test (or just test, depending on dates).

These verify:

1. Per-segment membership respects the gates: train ≥ 200 valid rows,
   val/eval/test ≥ 1 row.
2. Each kept ticker's positional indices are contiguous, within-window,
   and ordered (train < val < eval < test in calendar time).
3. Segment_dates is populated as ISO strings and matches the
   universe-calendar boundaries.
4. The trailing-mode fold still works (back-compat smoke).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gbdt.train import (
    Fold,
    SplitSpec,
    carve_single_fold,
    carve_universe_aligned,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _synthetic_calendar(start: str = "2020-01-01", periods: int = 1400) -> pd.DatetimeIndex:
    # "B" = business-day frequency; close enough to a real exchange calendar
    # for these tests (no holiday subtleties needed).
    return pd.DatetimeIndex(pd.date_range(start, periods=periods, freq="B"))


def _ticker_frame(ticker: str, dates: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": c, "high": c * 1.005, "low": c * 0.995,
            "close": c, "adj_close": c,
            "volume": np.ones(len(dates), dtype=int),
        }
    )


def _mixed_panel(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """3-ticker panel with mixed start dates anchored to ``cal``.

    - DEEP starts at cal[0], ends at cal[-1].
    - MID starts halfway through the train window.
    - IPO starts inside the eval window.
    """
    deep = _ticker_frame("DEEP", cal, seed=1)
    # MID: start at index 400 (halfway through a 800-day train window
    # anchored at cal[0]).
    mid_dates = cal[400:]
    mid = _ticker_frame("MID", mid_dates, seed=2)
    # IPO: start at index 1250 (inside the eval window — train_rows=800 +
    # val_rows=400 = 1200; eval_rows=200 ⇒ eval = [1200, 1400)).
    ipo_dates = cal[1250:]
    ipo = _ticker_frame("IPO", ipo_dates, seed=3)
    frames = [deep, mid, ipo]
    return pd.concat(frames).set_index(["date", "ticker"]).sort_index()


# ---------------------------------------------------------------------------
# Date-aligned carve tests
# ---------------------------------------------------------------------------


def test_carve_universe_aligned_segment_dates_match_calendar():
    cal = _synthetic_calendar(periods=1400)
    panel = _mixed_panel(cal)
    # Anchor at calendar day 0, durations 800/400/200/100 — total 1500 days.
    # That overflows 1400 → we use a smaller split that fits.
    # Use 600/300/150/50 = 1100 days.
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
        train_start=cal[0].date(),
        min_train_rows_per_ticker=200,
    )
    fold = carve_universe_aligned(panel, split, cal)
    sd = fold.segment_dates
    assert sd is not None
    # Boundaries: train [0, 599], val [600, 899], eval [900, 1049], test [1050, 1099].
    assert sd["train"]["start"] == cal[0].date().isoformat()
    assert sd["train"]["end"] == cal[599].date().isoformat()
    assert sd["val"]["start"] == cal[600].date().isoformat()
    assert sd["val"]["end"] == cal[899].date().isoformat()
    assert sd["eval"]["start"] == cal[900].date().isoformat()
    assert sd["eval"]["end"] == cal[1049].date().isoformat()
    assert sd["test"]["start"] == cal[1050].date().isoformat()
    assert sd["test"]["end"] == cal[1099].date().isoformat()


def test_carve_universe_aligned_membership_deep_in_all_segments():
    cal = _synthetic_calendar(periods=1400)
    panel = _mixed_panel(cal)
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
        train_start=cal[0].date(),
        min_train_rows_per_ticker=200,
    )
    fold = carve_universe_aligned(panel, split, cal)
    for seg, idx_map in (
        ("train", fold.train_idx),
        ("val",   fold.val_idx),
        ("eval",  fold.eval_idx),
        ("test",  fold.test_idx),
    ):
        assert "DEEP" in idx_map, f"DEEP missing from {seg}"


def test_carve_universe_aligned_membership_mid_skips_train():
    cal = _synthetic_calendar(periods=1400)
    panel = _mixed_panel(cal)
    # MID starts at cal[400]; gate=200, train window = cal[0..599] ⇒ MID has
    # 200 rows in train (cal[400..599]) ⇒ JUST meets the gate. Bump gate to
    # 300 so MID is excluded from train but still in val/eval/test.
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
        train_start=cal[0].date(),
        min_train_rows_per_ticker=300,
    )
    fold = carve_universe_aligned(panel, split, cal)
    assert "MID" not in fold.train_idx, "MID has only 200 train rows < gate=300"
    assert "MID" in fold.val_idx, "MID should be in val"
    assert "MID" in fold.eval_idx, "MID should be in eval"
    assert "MID" in fold.test_idx, "MID should be in test"


def test_carve_universe_aligned_membership_ipo_only_in_eval_test():
    cal = _synthetic_calendar(periods=1400)
    panel = _mixed_panel(cal)
    # IPO starts at cal[1250]. train [0..599], val [600..899],
    # eval [900..1049], test [1050..1099] — IPO has 0 rows in train/val/eval.
    # Bump test_rows so IPO falls into test.
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=120,
        train_start=cal[0].date(),
        min_train_rows_per_ticker=200,
    )
    # test = [1050..1169]; IPO @ 1250 → still 0. Push test later by shrinking eval.
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=300, test_rows=120,
        train_start=cal[0].date(),
        min_train_rows_per_ticker=200,
    )
    # train [0..599], val [600..899], eval [900..1199], test [1200..1319].
    # IPO @ 1250 ⇒ test has rows 1250..1319 = 70 rows ⇒ in test only.
    fold = carve_universe_aligned(panel, split, cal)
    assert "IPO" not in fold.train_idx, "IPO should NOT be in train"
    assert "IPO" not in fold.val_idx, "IPO should NOT be in val"
    assert "IPO" not in fold.eval_idx, "IPO should NOT be in eval (starts after eval_end)"
    assert "IPO" in fold.test_idx, "IPO should be in test"


def test_carve_universe_aligned_positions_ordered_and_in_window():
    cal = _synthetic_calendar(periods=1400)
    panel = _mixed_panel(cal)
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
        train_start=cal[0].date(),
        min_train_rows_per_ticker=200,
    )
    fold = carve_universe_aligned(panel, split, cal)
    sd = fold.segment_dates
    for ticker in ("DEEP", "MID"):
        sub = panel.xs(ticker, level="ticker").sort_index()
        sub_dates = sub.index
        for seg, idx_map in (
            ("train", fold.train_idx),
            ("val",   fold.val_idx),
            ("eval",  fold.eval_idx),
            ("test",  fold.test_idx),
        ):
            if ticker not in idx_map:
                continue
            positions = idx_map[ticker]
            # Strictly increasing.
            assert np.all(np.diff(positions) >= 1), (
                f"{ticker}/{seg}: positions not strictly increasing"
            )
            # All within segment window.
            seg_dates = sub_dates[positions]
            seg_start = pd.Timestamp(sd[seg]["start"])
            seg_end = pd.Timestamp(sd[seg]["end"])
            assert seg_dates.min() >= seg_start, f"{ticker}/{seg}: leakage before window"
            assert seg_dates.max() <= seg_end, f"{ticker}/{seg}: leakage after window"


def test_carve_universe_aligned_train_start_on_holiday_advances():
    """``side='left'`` on searchsorted ⇒ a non-trading train_start advances
    to the next trading day. Verify by anchoring at a Saturday."""
    cal = _synthetic_calendar(start="2020-01-06", periods=1400)  # Monday
    panel = _mixed_panel(cal)
    saturday = date(2020, 1, 4)  # before cal[0] = Monday 2020-01-06
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
        train_start=saturday,
        min_train_rows_per_ticker=200,
    )
    fold = carve_universe_aligned(panel, split, cal)
    # train_start should be advanced to the first trading day in cal.
    assert fold.segment_dates["train"]["start"] == cal[0].date().isoformat()


def test_carve_universe_aligned_window_overflow_raises():
    cal = _synthetic_calendar(periods=500)
    panel = _mixed_panel(_synthetic_calendar(periods=1400))
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
        train_start=cal[0].date(),
        min_train_rows_per_ticker=200,
    )
    with pytest.raises(ValueError, match="runs past the end"):
        carve_universe_aligned(panel, split, cal)


def test_carve_universe_aligned_missing_train_start_raises():
    cal = _synthetic_calendar(periods=1400)
    panel = _mixed_panel(cal)
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
        train_start=None,
        min_train_rows_per_ticker=200,
    )
    with pytest.raises(ValueError, match="train_start"):
        carve_universe_aligned(panel, split, cal)


# ---------------------------------------------------------------------------
# carve_single_fold dispatch
# ---------------------------------------------------------------------------


def test_carve_single_fold_dispatches_to_date_aligned():
    cal = _synthetic_calendar(periods=1400)
    panel = _mixed_panel(cal)
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
        train_start=cal[0].date(),
        min_train_rows_per_ticker=200,
    )
    fold = carve_single_fold(panel, split, universe_calendar=cal)
    assert isinstance(fold, Fold)
    assert fold.segment_dates is not None
    assert "DEEP" in fold.train_idx


def test_carve_single_fold_date_aligned_without_calendar_raises():
    panel = _mixed_panel(_synthetic_calendar(periods=1400))
    split = SplitSpec(
        mode="date_aligned",
        train_rows=600, val_rows=300, eval_rows=150, test_rows=50,
        train_start=date(2020, 1, 1),
        min_train_rows_per_ticker=200,
    )
    with pytest.raises(ValueError, match="universe_calendar"):
        carve_single_fold(panel, split)


def test_carve_single_fold_trailing_back_compat():
    """Trailing-mode behaviour must be byte-identical to pre-V1.4."""
    cal = _synthetic_calendar(periods=1600)
    deep = _ticker_frame("DEEP", cal, seed=1)
    panel = deep.set_index(["date", "ticker"]).sort_index()
    split = SplitSpec(train_rows=800, val_rows=400, eval_rows=200, test_rows=100)
    fold = carve_single_fold(panel, split)
    assert fold.segment_dates is None
    assert len(fold.train_idx["DEEP"]) == 800
    assert len(fold.val_idx["DEEP"]) == 400
    assert len(fold.eval_idx["DEEP"]) == 200
    assert len(fold.test_idx["DEEP"]) == 100
