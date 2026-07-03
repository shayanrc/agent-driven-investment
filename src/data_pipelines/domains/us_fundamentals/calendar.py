"""Reporting-lag-aware quarterly calendar for the us_fundamentals domain.

Gap detection asks the calendar "which observations should exist in
[start, end]?". For fundamentals the honest answer is time-dependent: the
quarter ending Jun 30 *exists* on Jul 1, but no company has filed yet — a grid
that emits it immediately would re-hit the provider chain every day for weeks
(soft-failing each time) until 10-Qs land. So the grid emits a quarter-end
only once ``today >= grid_date + lag_days``.

``lag_days`` defaults to 45: the large/accelerated-filer 10-Q deadline is
40–45 days after fiscal period end, so by day 45 most of the universe has
filed and one seed pass fills most tickers. Late filers simply soft-fail and
retry on the next seed (dispatch preserves cache on failure). The trade-off is
deliberate staleness for early filers — an off-cycle company (WMT, fiscal end
Jan 31, files ~3 weeks later) has its numbers public well before its Mar-31
grid date is even reached; those rows still arrive early in practice because
any triggered fetch stores the full parsed history, but the *grid* never
demands them early.

``today_fn`` is injectable for tests; production uses the wall clock — the
grid is *meant* to move with time (that is the reporting-lag semantics), while
everything downstream of gap detection stays deterministic per D8.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

from data_pipelines.domains.us_fundamentals.schema import quarter_ends

DEFAULT_LAG_DAYS = 45


class QuarterEndCalendar:
    """Calendar quarter-ends (Mar 31 / Jun 30 / Sep 30 / Dec 31) in
    ``[start, end]``, excluding grid dates younger than ``lag_days``."""

    def __init__(
        self,
        lag_days: int = DEFAULT_LAG_DAYS,
        today_fn: Callable[[], date] = date.today,
    ):
        if lag_days < 0:
            raise ValueError("lag_days must be >= 0")
        self._lag = timedelta(days=lag_days)
        self._today_fn = today_fn

    def trading_days(self, start: date, end: date) -> list[date]:
        cutoff = self._today_fn() - self._lag
        return quarter_ends(start, min(end, cutoff))
