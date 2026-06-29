"""Tests for `_dense_date_bounds` — the sparse-stray-row guard on the registry's
per-segment date columns (the nasdaq_40_50 `test_start=2025-06-05` artifact fix).

A trailing (row-based) split gives each ticker its own last-N-rows window, so a
DELISTED ticker whose history ends months before the cohort lands segment rows dated
well before the cohort's block. A naive MIN(date) is then dragged back to that stray
prefix; the dense-bound guard drops dates carrying < DENSE_FRAC × the median daily
row-count, recovering the true cohort window. The twin in
regenerate_r_precision_at_k_csv.py shares identical logic.
"""
import pandas as pd

from scripts.gbdt.backfill_csv_segment_dates import _dense_date_bounds


def _rows(dates, per_day):
    """A per-(date,ticker) date Series: each date repeated `per_day` times."""
    return pd.to_datetime(pd.Series([d for d in dates for _ in range(per_day)]))


def test_drops_sparse_delisted_prefix():
    # Cohort test block: 10 business days, ~90 tickers/day.
    cohort = pd.bdate_range("2025-12-30", periods=10)
    dense = _rows(cohort, 90)
    # A single delisted ticker leaves 1 row/day across a months-earlier prefix.
    stray = pd.to_datetime(pd.Series(list(pd.bdate_range("2025-06-05", periods=30))))
    dts = pd.concat([stray, dense], ignore_index=True)
    start, end = _dense_date_bounds(dts)
    # The stray prefix is dropped; the cohort block defines start AND end.
    assert start == cohort.min().date().isoformat() == "2025-12-30"
    assert end == cohort.max().date().isoformat()


def test_max_unaffected_by_early_strays():
    cohort = pd.bdate_range("2026-01-05", periods=8)
    dts = pd.concat([
        pd.to_datetime(pd.Series(list(pd.bdate_range("2025-07-01", periods=20)))),  # strays
        _rows(cohort, 60),
    ], ignore_index=True)
    _, end = _dense_date_bounds(dts)
    assert end == cohort.max().date().isoformat()  # MAX never dragged by an early prefix


def test_noop_when_uniform():
    # No strays: every date carries the full cohort → bounds == raw min/max.
    dates = pd.bdate_range("2025-01-01", periods=40)
    dts = _rows(dates, 50)
    start, end = _dense_date_bounds(dts)
    assert start == dates.min().date().isoformat()
    assert end == dates.max().date().isoformat()


def test_empty_series():
    assert _dense_date_bounds(pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))) == (None, None)
