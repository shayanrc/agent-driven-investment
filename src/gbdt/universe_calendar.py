"""Universe → trading-calendar resolver (V1.4 date-aligned splits).

The V1.4 date-aligned carve (``gbdt.train.carve_universe_aligned``) needs a
``pandas.DatetimeIndex`` of trading days for the universe's exchange. This
module owns the universe → calendar dispatch:

- Each ``configs/gbdt/default.yaml::universes::<name>`` block may declare a
  ``calendar`` key (e.g. ``"NYSE"`` or ``"NSE"``). When present it wins.
- When absent we infer from the universe name prefix per V1.4 §4.4:
    - ``nifty*``, ``midcap*`` → ``NSE``
    - ``sp500``, ``nasdaq100``, ``russell1000`` → ``NYSE``
- Anything else → raise.

The returned ``DatetimeIndex`` is the calendar's ``schedule()`` index
(market-open timestamps, tz-stripped to naive dates) over a window that
generously covers ``train_start`` through "today + a year" so the
``searchsorted`` boundaries in ``carve_universe_aligned`` never run off
either end.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

# Universe-name → MIC mapping. Keep the explicit map small + declarative so
# adding a new universe is one-line.
_NSE_NAME_PREFIXES: tuple[str, ...] = (
    "nifty",
    "midcap",
)
_NYSE_NAMES: frozenset[str] = frozenset({"sp500", "nasdaq100", "russell1000"})


def resolve_calendar_name(
    universe: str,
    universe_block: dict | None = None,
) -> str:
    """Pick the MIC for ``universe``. Explicit ``calendar`` in the block wins."""
    if universe_block is not None:
        explicit = universe_block.get("calendar")
        if explicit:
            return str(explicit)
    name = universe.lower()
    if name in _NYSE_NAMES:
        return "NYSE"
    for pref in _NSE_NAME_PREFIXES:
        if name.startswith(pref):
            return "NSE"
    raise KeyError(
        f"resolve_calendar_name: cannot infer trading calendar for universe "
        f"{universe!r}. Add a 'calendar: NYSE' / 'calendar: NSE' key to "
        f"configs/gbdt/default.yaml::universes::{universe} or extend the "
        f"prefix/name maps in gbdt.universe_calendar."
    )


def get_calendar(
    universe: str,
    universe_block: dict | None = None,
    *,
    start: date | str = "2010-01-01",
    end: date | str | None = None,
) -> pd.DatetimeIndex:
    """Return the universe's trading-day index over ``[start, end]``.

    ``end`` defaults to today + 1 year, which is generous enough for any
    plausible ``test_end`` plus the longest configured horizon (V1.4 plan
    §6). The returned index is tz-naive (date-only granularity), matching
    the panel's index.
    """
    import pandas_market_calendars as mcal  # local import — heavy module

    mic = resolve_calendar_name(universe, universe_block)
    cal = mcal.get_calendar(mic)
    end_d = end if end is not None else (date.today() + timedelta(days=365))
    sched = cal.schedule(start_date=start, end_date=end_d)
    # ``schedule`` returns a DatetimeIndex of market-open timestamps; downstream
    # we only care about calendar dates, so normalize away the time-of-day.
    idx = pd.DatetimeIndex(sched.index).normalize()
    # Strip timezone so it matches the panel's tz-naive date index.
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


__all__ = ["resolve_calendar_name", "get_calendar"]
