"""Per-cadence calendars for the fred_macro domain.

FRED dates each observation at the *start* of its period: daily series on the
business day, monthly series on the 1st of the month, quarterly series on the
1st of the quarter (Jan/Apr/Jul/Oct 1). Gap detection enumerates the expected
observation dates for a series and compares against the cache; these three
calendars produce grids that converge against FRED's own dating so a fetched
series doesn't show perpetual phantom gaps.

``FREDMacroDomain.calendar_for(identifier)`` picks the right one per series
(keyed on the frequency declared in ``series_macro.yaml``).

Daily note: FRED daily series follow the bond-market business-day schedule
(SIFMA holidays differ from NYSE). We deliberately use a plain weekday calendar
here and let the adapter densify the covered span to weekdays with ``NaN`` on
no-data days. That makes the exact holiday set irrelevant to gap detection —
every weekday in a series' covered span has a row (value or ``NaN``), so the
daily grid always converges without maintaining a SIFMA holiday table.
"""

from __future__ import annotations

from datetime import date, timedelta


class BusinessDayCalendar:
    """Every weekday (Mon-Fri) in ``[start, end]``.

    Holidays are NOT excluded — see the module docstring (the adapter densifies
    daily series to weekdays, so the holiday set never affects convergence).
    """

    def trading_days(self, start: date, end: date) -> list[date]:
        if start > end:
            return []
        out: list[date] = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                out.append(d)
            d += timedelta(days=1)
        return out


class MonthlyCalendar:
    """The 1st of each month in ``[start, end]`` (FRED's monthly dating)."""

    def trading_days(self, start: date, end: date) -> list[date]:
        if start > end:
            return []
        out: list[date] = []
        d = date(start.year, start.month, 1)
        while d <= end:
            if d >= start:
                out.append(d)
            d = _add_month(d)
        return out


class QuarterlyCalendar:
    """The 1st of Jan/Apr/Jul/Oct in ``[start, end]`` (FRED's quarterly dating)."""

    def trading_days(self, start: date, end: date) -> list[date]:
        if start > end:
            return []
        out: list[date] = []
        q_month = ((start.month - 1) // 3) * 3 + 1  # quarter-start month <= start.month
        d = date(start.year, q_month, 1)
        while d <= end:
            if d >= start:
                out.append(d)
            d = _add_month(d, 3)
        return out


def _add_month(d: date, n: int = 1) -> date:
    """First-of-month ``n`` months after ``d`` (``d`` is assumed day==1)."""
    m0 = d.month - 1 + n
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    return date(y, m, 1)
