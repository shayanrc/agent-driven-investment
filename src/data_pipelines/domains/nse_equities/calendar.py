"""NSE trading calendar (D7 enforcement).

Backed by `holidays.financial_holidays('NSE')` from the maintained `holidays`
library, which encodes NSE's published holiday list (lunar/computed dates for
Holi, Ram Navami, Eid, etc., plus fixed-date civil holidays). Coverage is
12–16 holidays/year going back decades; library is maintained per release.

NSE trading hours are 9:15–15:30 IST. Per D7 we store the date column as a
naive midnight UTC value (open question 12 in V1 plan); the actual trade-day
in IST and the UTC midnight differ by ~5.5h, but downstream consumers
(analog_mc) treat date as opaque, and the cache layer uses the date column as
a primary key — never re-projected through tz logic. Adapters that receive
tz-aware timestamps from upstream MUST convert to the IST trade date BEFORE
flooring to midnight; otherwise an IST trade-day at 00:00 IST = 18:30 UTC the
prior day would land in the wrong row.

Weekend rule: closed Sat + Sun. Plus the `holidays` lib NSE set.

If the library's NSE coverage falls behind (e.g., NSE announces an unscheduled
closure mid-year), surface that as a hand-pinned override here — but only
after confirming the lib doesn't already have it on the next release.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import holidays

# Hand-pinned NSE closures not present in the holidays library. Empty for now
# — populate ONLY when a gap-detection failure surfaces a real omission.
# Pattern mirrors us_equities/calendar.py:_UNSCHEDULED_CLOSURES.
_UNSCHEDULED_CLOSURES: frozenset[date] = frozenset()


@lru_cache(maxsize=1)
def _nse_holidays():
    # `holidays.financial_holidays('NSE')` returns a dict-like instance that
    # lazily materializes years as `.get(date)` / membership tests are made.
    # Cached at module level so all calls share state (and the lib's internal
    # year-cache stays warm).
    return holidays.financial_holidays("NSE")


def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:  # Sat=5, Sun=6
        return False
    if d in _nse_holidays():
        return False
    if d in _UNSCHEDULED_CLOSURES:
        return False
    return True


class NSECalendar:
    """Concrete Calendar implementation for the nse_equities domain."""

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
