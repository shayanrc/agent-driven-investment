"""NYSE trading calendar (D7 enforcement).

Approximates the NYSE schedule with weekend exclusion plus the standard
US-federal-style holiday set NYSE observes. Good enough for daily-bar gap
detection in v1; an external calendar library (exchange-calendars) is a
hardening upgrade we can take if hand-maintenance friction shows up.

Holidays observed (rule-based, computed per year):
  - New Year's Day                       (Jan 1; observed Mon if Sun)
  - Martin Luther King Jr. Day           (3rd Mon of Jan)
  - Washington's Birthday / Presidents'  (3rd Mon of Feb)
  - Good Friday                          (Fri before Easter Sunday)
  - Memorial Day                         (last Mon of May)
  - Juneteenth                           (Jun 19; observed Mon if Sun, Fri if Sat) [from 2022]
  - Independence Day                     (Jul 4; observed Mon if Sun, Fri if Sat)
  - Labor Day                            (1st Mon of Sep)
  - Thanksgiving                         (4th Thu of Nov)
  - Christmas Day                        (Dec 25; observed Mon if Sun, Fri if Sat)

Early-close days (1pm) are NOT modeled — they're full trading days for the
purposes of daily-bar availability, which is all gap detection cares about.
Unscheduled closures (9/11, Hurricane Sandy 2012-10-29/30, Carter funeral
2025-01-09, etc.) are not modeled rule-by-rule; explicit overrides can be
added if a fetch reveals them.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

# One-off NYSE closures the rule-based generator can't compute.
# Source: https://www.nyse.com/markets/hours-calendars (historical), audited
# against gaps observed when re-fetching cached tickers (S&P 500 seed
# 2026-05-23 surfaced 1994 / 2004 / 2007 entries that were missing).
_UNSCHEDULED_CLOSURES: frozenset[date] = frozenset({
    date(1994, 4, 27),   # Nixon funeral / Day of Mourning
    date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2004, 6, 11),   # Reagan funeral / Day of Mourning
    date(2007, 1, 2),    # Ford funeral / Day of Mourning
    date(2012, 10, 29), date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),   # G.H.W. Bush funeral / Day of Mourning
    date(2025, 1, 9),    # J. Carter funeral / Day of Mourning
})


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """nth occurrence of weekday (Mon=0 .. Sun=6) in month/year."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last `weekday` in month/year (e.g., last Monday of May)."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last = next_month - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _observed_weekday(d: date) -> date:
    """NYSE observed-day rule: Sat → Fri, Sun → Mon."""
    if d.weekday() == 5:  # Sat
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sun
        return d + timedelta(days=1)
    return d


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian computus."""
    a = year % 19
    b = year // 100
    c = year % 100
    d_ = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d_ - g + 15) % 30
    i = c // 4
    k = c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=200)
def _holidays_for_year(year: int) -> frozenset[date]:
    out: set[date] = set()
    out.add(_observed_weekday(date(year, 1, 1)))            # New Year's
    out.add(_nth_weekday(year, 1, 0, 3))                     # MLK Day (3rd Mon Jan)
    out.add(_nth_weekday(year, 2, 0, 3))                     # Presidents' (3rd Mon Feb)
    out.add(_easter_sunday(year) - timedelta(days=2))        # Good Friday
    out.add(_last_weekday(year, 5, 0))                       # Memorial Day (last Mon May)
    if year >= 2022:
        out.add(_observed_weekday(date(year, 6, 19)))        # Juneteenth
    out.add(_observed_weekday(date(year, 7, 4)))             # Independence Day
    out.add(_nth_weekday(year, 9, 0, 1))                     # Labor Day (1st Mon Sep)
    out.add(_nth_weekday(year, 11, 3, 4))                    # Thanksgiving (4th Thu Nov)
    out.add(_observed_weekday(date(year, 12, 25)))           # Christmas
    return frozenset(out)


def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    if d in _holidays_for_year(d.year):
        return False
    if d in _UNSCHEDULED_CLOSURES:
        return False
    return True


class NYSECalendar:
    """Concrete Calendar implementation for the us_equities domain."""

    def trading_days(self, start: date, end: date) -> list[date]:
        if start > end:
            return []
        out: list[date] = []
        d = start
        while d <= end:
            if is_trading_day(d):
                out.append(d)
            d += timedelta(days=1)
        return out
