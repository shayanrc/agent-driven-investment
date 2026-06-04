"""V1.4 P5 — Phase 0 cache-currency pre-flight tests.

Covers all three sub-cases per plan §6 + the retry-classification logic:

- Sub-case A: universe-level shortfall → ``CacheCurrencyError`` raised.
- Sub-case B: per-ticker right-edge staleness → 3-retry exponential
  backoff on transient errors (1s/4s/16s), definitive failures skip
  retries, deficient table reflects last_error_type.
- The currency-OK happy path returns ``[]``.
- ``format_deficient_table`` produces a parseable worked-example list.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from gbdt.preflight import (
    CacheCurrencyError,
    DeficientTicker,
    cache_currency_check,
    format_deficient_table,
)


def _calendar(periods: int = 2000) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range("2019-01-02", periods=periods, freq="B"))


def test_sub_case_a_universe_shortfall_raises():
    cal = _calendar(periods=100)
    with pytest.raises(CacheCurrencyError, match="sub-case A"):
        cache_currency_check(
            universe_tickers=["AAA"],
            universe_calendar=cal,
            days_test_end=99,
            horizon_days=25,
            today=date.today(),
            cache_latest_date=lambda t: "2025-01-01",
        )


def test_sub_case_a_test_end_past_today_raises():
    cal = _calendar(periods=2000)  # extends well past today.
    days_test_end = 1500
    horizon = 25
    needed_end = cal[days_test_end + horizon].date()
    # Set "today" before needed_end to force the sub-case A guard.
    fake_today = (needed_end - pd.Timedelta(days=30)).to_pydatetime().date() \
        if hasattr(needed_end, "to_pydatetime") else date.fromordinal(needed_end.toordinal() - 30)
    with pytest.raises(CacheCurrencyError, match="past today"):
        cache_currency_check(
            universe_tickers=["AAA"],
            universe_calendar=cal,
            days_test_end=days_test_end,
            horizon_days=horizon,
            today=fake_today,
            cache_latest_date=lambda t: "2099-01-01",
        )


def test_currency_ok_returns_empty():
    cal = _calendar(periods=2000)
    days_test_end = 100
    horizon = 25
    needed = cal[days_test_end + horizon].date().isoformat()
    deficient = cache_currency_check(
        universe_tickers=["AAA", "BBB"],
        universe_calendar=cal,
        days_test_end=days_test_end,
        horizon_days=horizon,
        today=date(2099, 1, 1),
        cache_latest_date=lambda t: needed,  # exactly meets needed_end.
    )
    assert deficient == []


def test_sub_case_b_no_cache_returns_deficient_no_fetcher():
    cal = _calendar(periods=2000)
    deficient = cache_currency_check(
        universe_tickers=["AAA"],
        universe_calendar=cal,
        days_test_end=100,
        horizon_days=25,
        today=date(2099, 1, 1),
        cache_latest_date=lambda t: None,
        fetcher=None,
    )
    assert len(deficient) == 1
    assert deficient[0].ticker == "AAA"
    assert deficient[0].latest_cached is None
    assert deficient[0].last_error_type == "no_cache"


def test_sub_case_b_stale_no_fetcher_returns_deficient():
    cal = _calendar(periods=2000)
    deficient = cache_currency_check(
        universe_tickers=["AAA"],
        universe_calendar=cal,
        days_test_end=100,
        horizon_days=25,
        today=date(2099, 1, 1),
        cache_latest_date=lambda t: "2019-06-01",  # well before needed_end.
        fetcher=None,
    )
    assert len(deficient) == 1
    assert deficient[0].ticker == "AAA"
    assert deficient[0].latest_cached == "2019-06-01"
    assert deficient[0].last_error_type == "no_fetcher"
    assert deficient[0].days_short > 0


def test_sub_case_b_retry_loop_transient_then_success():
    cal = _calendar(periods=2000)
    needed = cal[125].date().isoformat()
    sleeps: list[float] = []
    # Start stale (before needed_end at cal[125] ≈ June 2019).
    cache_state = {"AAA": "2019-01-02"}

    attempts = {"AAA": 0}

    def fake_fetcher(ticker: str, end: date, back_extend: bool):
        attempts[ticker] += 1
        if attempts[ticker] < 2:
            raise TimeoutError("connection timed out")
        # Success: bump the cache forward.
        cache_state[ticker] = needed
        return pd.DataFrame()

    def lookup(t):
        return cache_state.get(t)

    deficient = cache_currency_check(
        universe_tickers=["AAA"],
        universe_calendar=cal,
        days_test_end=100,
        horizon_days=25,
        today=date(2099, 1, 1),
        cache_latest_date=lookup,
        fetcher=fake_fetcher,
        sleep_fn=sleeps.append,
        backoff_seconds=(1, 4),
    )
    assert deficient == []
    assert attempts["AAA"] == 2          # second attempt succeeded.
    assert sleeps == [1]                  # slept once before the second attempt.


def test_sub_case_b_retry_loop_all_transient_failures():
    cal = _calendar(periods=2000)
    sleeps: list[float] = []

    def fake_fetcher(ticker: str, end: date, back_extend: bool):
        raise TimeoutError("connection timed out")

    deficient = cache_currency_check(
        universe_tickers=["AAA"],
        universe_calendar=cal,
        days_test_end=100,
        horizon_days=25,
        today=date(2099, 1, 1),
        cache_latest_date=lambda t: "2019-06-01",
        fetcher=fake_fetcher,
        sleep_fn=sleeps.append,
        backoff_seconds=(1, 4),
    )
    assert len(deficient) == 1
    assert deficient[0].last_error_type == "transient_max_retries"
    # 3 attempts: sleeps after attempt 0 (1s) + attempt 1 (4s); attempt 2 has no sleep.
    assert sleeps == [1, 4]


def test_sub_case_b_retry_loop_skips_on_definitive_404():
    cal = _calendar(periods=2000)
    sleeps: list[float] = []

    def fake_fetcher(ticker: str, end: date, back_extend: bool):
        raise RuntimeError("provider returned 404 not found")

    deficient = cache_currency_check(
        universe_tickers=["AAA"],
        universe_calendar=cal,
        days_test_end=100,
        horizon_days=25,
        today=date(2099, 1, 1),
        cache_latest_date=lambda t: "2019-06-01",
        fetcher=fake_fetcher,
        sleep_fn=sleeps.append,
        backoff_seconds=(1, 4),
    )
    assert len(deficient) == 1
    assert deficient[0].last_error_type == "definitive:404"
    assert sleeps == []  # never slept; bailed on first attempt.


def test_format_deficient_table_shape():
    rows = [
        DeficientTicker(
            ticker="NSE:RELIANCE",
            latest_cached="2025-09-01",
            needed_through="2025-10-15",
            days_short=44,
            last_error_type="transient_max_retries",
        ),
        DeficientTicker(
            ticker="NASDAQ:XYZ",
            latest_cached=None,
            needed_through="2025-10-15",
            days_short=99999,
            last_error_type="no_cache",
        ),
    ]
    s = format_deficient_table(rows)
    assert "ticker" in s
    assert "latest_cached" in s
    assert "needed_through" in s
    assert "days_short" in s
    assert "last_error_type" in s
    assert "NSE:RELIANCE" in s
    assert "transient_max_retries" in s
    assert "NASDAQ:XYZ" in s
    assert "<none>" in s  # latest_cached=None placeholder rendered.


def test_format_deficient_table_empty():
    assert format_deficient_table([]) == "(none)"
