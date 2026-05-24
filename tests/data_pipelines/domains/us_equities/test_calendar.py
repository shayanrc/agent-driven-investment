"""Stage 5 tests: NYSE calendar — weekends, holidays, observed-day rule."""

from __future__ import annotations

from datetime import date

import pytest

from data_pipelines.domains.us_equities.calendar import (
    NYSECalendar,
    _holidays_for_year,
    is_trading_day,
)


@pytest.fixture
def cal() -> NYSECalendar:
    return NYSECalendar()


class TestKnownHolidays2025:
    """Sanity check: NYSE-published 2025 holiday list."""

    @pytest.mark.parametrize("d", [
        date(2025, 1, 1),    # New Year's
        date(2025, 1, 20),   # MLK Day (3rd Mon Jan)
        date(2025, 2, 17),   # Presidents' Day (3rd Mon Feb)
        date(2025, 4, 18),   # Good Friday (2 days before Easter Apr 20)
        date(2025, 5, 26),   # Memorial Day (last Mon May)
        date(2025, 6, 19),   # Juneteenth
        date(2025, 7, 4),    # Independence Day
        date(2025, 9, 1),    # Labor Day (1st Mon Sep)
        date(2025, 11, 27),  # Thanksgiving (4th Thu Nov)
        date(2025, 12, 25),  # Christmas
        date(2025, 1, 9),    # Carter funeral (unscheduled closure)
    ])
    def test_known_2025_closure(self, d, cal):
        assert not is_trading_day(d), f"{d} should be a closure"
        # also: not in trading_days range result
        days = cal.trading_days(d, d)
        assert days == []


class TestWeekends:
    def test_saturday_not_trading(self, cal):
        assert not is_trading_day(date(2026, 5, 23))  # Sat
        assert cal.trading_days(date(2026, 5, 23), date(2026, 5, 23)) == []

    def test_sunday_not_trading(self, cal):
        assert not is_trading_day(date(2026, 5, 24))


class TestObservedDayRule:
    def test_jul4_on_saturday_observed_friday(self, cal):
        # 2026: Jul 4 is Sat → NYSE observes Fri Jul 3.
        assert not is_trading_day(date(2026, 7, 3))
        # Sat is also closed (weekend).
        assert not is_trading_day(date(2026, 7, 4))

    def test_christmas_on_friday_observed_friday(self, cal):
        # 2026: Dec 25 is Fri → observed Fri.
        assert not is_trading_day(date(2026, 12, 25))


class TestRange:
    def test_trading_days_count_typical_month(self, cal):
        # 2025-03 has no NYSE holidays. 21 weekdays, no holidays → 21 days.
        days = cal.trading_days(date(2025, 3, 1), date(2025, 3, 31))
        assert len(days) == 21

    def test_range_with_holiday(self, cal):
        # 2025-01-01 through 2025-01-09: Jan 1 (NYD) + Jan 9 (Carter funeral)
        # are closures; Jan 4,5 are weekend.
        days = cal.trading_days(date(2025, 1, 1), date(2025, 1, 9))
        expected = [date(2025, 1, 2), date(2025, 1, 3),
                    date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)]
        assert days == expected

    def test_empty_range(self, cal):
        assert cal.trading_days(date(2026, 1, 10), date(2026, 1, 9)) == []

    def test_single_trading_day(self, cal):
        assert cal.trading_days(date(2026, 5, 22), date(2026, 5, 22)) == [date(2026, 5, 22)]


class TestHolidayCache:
    def test_holidays_for_year_cached(self):
        h1 = _holidays_for_year(2025)
        h2 = _holidays_for_year(2025)
        assert h1 is h2
        assert date(2025, 12, 25) in h1
        assert date(2025, 7, 4) in h1

    def test_juneteenth_only_from_2022(self):
        assert date(2021, 6, 18) not in _holidays_for_year(2021)
        assert date(2022, 6, 20) in _holidays_for_year(2022)  # Jun 19 2022 is Sun → observed Mon


class TestUnscheduledClosures:
    """Day-of-Mourning closures discovered during S&P 500 seed audit."""

    @pytest.mark.parametrize("d", [
        date(1994, 4, 27),   # Nixon
        date(2004, 6, 11),   # Reagan
        date(2007, 1, 2),    # Ford
    ])
    def test_presidential_funeral_closure(self, d):
        assert not is_trading_day(d), (
            f"{d} was a NYSE Day of Mourning closure — must not be a trading day"
        )
