"""Per-cadence calendars converge against FRED's period-start dating."""

from __future__ import annotations

from datetime import date

from data_pipelines.domains.fred_macro.calendar import (
    BusinessDayCalendar,
    MonthlyCalendar,
    QuarterlyCalendar,
)


def test_businessday_excludes_weekends_not_holidays():
    days = BusinessDayCalendar().trading_days(date(2020, 1, 1), date(2020, 1, 12))
    assert date(2020, 1, 4) not in days   # Saturday
    assert date(2020, 1, 5) not in days   # Sunday
    assert all(d.weekday() < 5 for d in days)
    # Holidays are deliberately NOT excluded (the adapter densifies instead):
    # Jan 1 2020 (a Wednesday, New Year's) is present.
    assert date(2020, 1, 1) in days


def test_businessday_count_jan_2020():
    assert len(BusinessDayCalendar().trading_days(date(2020, 1, 1), date(2020, 1, 31))) == 23


def test_monthly_firsts_only():
    days = MonthlyCalendar().trading_days(date(2020, 1, 15), date(2020, 5, 1))
    # start mid-January → January 1 excluded; through May 1 inclusive.
    assert days == [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1), date(2020, 5, 1)]


def test_monthly_includes_start_when_first():
    days = MonthlyCalendar().trading_days(date(2020, 1, 1), date(2020, 2, 1))
    assert days == [date(2020, 1, 1), date(2020, 2, 1)]


def test_quarterly_jan_apr_jul_oct():
    days = QuarterlyCalendar().trading_days(date(2020, 1, 1), date(2021, 1, 1))
    assert days == [
        date(2020, 1, 1), date(2020, 4, 1), date(2020, 7, 1),
        date(2020, 10, 1), date(2021, 1, 1),
    ]


def test_quarterly_midquarter_start():
    days = QuarterlyCalendar().trading_days(date(2020, 2, 15), date(2020, 8, 1))
    assert days == [date(2020, 4, 1), date(2020, 7, 1)]


def test_empty_when_start_after_end():
    for cal in (BusinessDayCalendar(), MonthlyCalendar(), QuarterlyCalendar()):
        assert cal.trading_days(date(2020, 5, 1), date(2020, 1, 1)) == []
