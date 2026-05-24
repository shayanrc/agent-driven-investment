"""NSE calendar — weekend exclusion + holidays-lib NSE set."""

from __future__ import annotations

from datetime import date

from data_pipelines.domains.nse_equities.calendar import (
    NSECalendar,
    is_trading_day,
)


def test_weekend_is_not_trading():
    assert not is_trading_day(date(2025, 4, 5))   # Saturday
    assert not is_trading_day(date(2025, 4, 6))   # Sunday


def test_weekday_no_holiday_is_trading():
    assert is_trading_day(date(2025, 4, 7))       # Monday, no NSE holiday


def test_known_nse_holidays_are_excluded():
    # Spot-check from the 2025/2026 NSE holiday list surfaced in smoke test.
    assert not is_trading_day(date(2025, 3, 14))  # Holi
    assert not is_trading_day(date(2025, 3, 31))  # Eid al-Fitr
    assert not is_trading_day(date(2025, 4, 14))  # Ambedkar Jayanti
    assert not is_trading_day(date(2026, 1, 26))  # Republic Day
    assert not is_trading_day(date(2026, 4, 3))   # Good Friday 2026


def test_calendar_trading_days_filters_correctly():
    cal = NSECalendar()
    days = cal.trading_days(date(2025, 4, 1), date(2025, 4, 30))
    # April 2025: 30 days, ~8 weekend days, 1 holiday (Ambedkar 4/14, Mahavir
    # Jayanti 4/10) → roughly 19-20 trading days. The exact count depends on
    # what the holidays lib has for that year, so just sanity-check shape.
    assert 18 <= len(days) <= 22
    assert all(d.weekday() < 5 for d in days)
    assert date(2025, 4, 5) not in days   # Sat
    assert date(2025, 4, 14) not in days  # Ambedkar Jayanti


def test_calendar_handles_reverse_range():
    cal = NSECalendar()
    assert cal.trading_days(date(2025, 4, 30), date(2025, 4, 1)) == []


def test_calendar_single_day_range():
    cal = NSECalendar()
    days = cal.trading_days(date(2025, 4, 7), date(2025, 4, 7))
    assert days == [date(2025, 4, 7)]
