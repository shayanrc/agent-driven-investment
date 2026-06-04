"""V1.4 P5 — Phase 0 cache-currency pre-flight for date-aligned cells.

Per V1.4 plan §6 / D6:

- Sub-case A (universe-level shortfall): ``test_end + horizon`` must not
  fall past today; otherwise we REFUSE the run with a clear message.
- Sub-case B (per-ticker right-edge staleness): for each ticker in the
  universe, the cache's right-edge date must be ≥ ``test_end + horizon``.
  When short, retry ``data_pipelines.fetch(ticker, end=today,
  back_extend=False)`` up to 3 times on transient errors (timeout,
  ConnectionError, HTTP 429, HTTP 5xx) with 1s/4s/16s exponential
  backoff. Definitive failures (HTTP 404, malformed response, empty
  response from a healthy provider) skip retries.
- Sub-case C (per-ticker history depth on the LEFT): auto-handled by the
  membership gate in ``carve_universe_aligned`` — late-IPO tickers
  contribute only to segments they have valid features for. No REFUSE
  here.

When deficient tickers remain after retries, raise
:class:`CacheCurrencyError` with the worked-example list (table per
plan §6).

Pre-flight is opt-in: trailing-mode runs (the default) skip this entire
module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


# Backoff schedule (seconds) per D6: 1s, 4s, 16s — three attempts total.
_BACKOFF_SECONDS = (1, 4, 16)


class CacheCurrencyError(RuntimeError):
    """Pre-flight REFUSE — cache cannot satisfy the date-aligned window."""


@dataclass
class DeficientTicker:
    ticker: str
    latest_cached: str | None  # ISO date or None when ticker has no cache row.
    needed_through: str        # ISO date.
    days_short: int            # calendar-days between latest_cached and needed.
    last_error_type: str       # 'no_cache' | 'still_stale' | 'transient_max_retries' | 'definitive:<code>'


def _is_transient_error(exc: BaseException) -> bool:
    """Per D6: timeout, ConnectionError, HTTP 429, HTTP 5xx → retry."""
    name = type(exc).__name__.lower()
    if "timeout" in name or "connectionerror" in name or "connectionrefused" in name:
        return True
    # HTTP-style errors — inspect message for status code.
    msg = str(exc).lower()
    if "429" in msg or "503" in msg or "502" in msg or "500" in msg or "504" in msg:
        return True
    if " timed out" in msg or "timeout" in msg:
        return True
    if "connection" in msg and ("reset" in msg or "refused" in msg):
        return True
    return False


def _classify_definitive(exc: BaseException) -> str | None:
    """Returns 'definitive:<short_tag>' for known-definitive failures.

    Returned tag drives the deficient-table's ``last_error_type``. Returns
    None when the error is genuinely transient (caller should retry).
    """
    msg = str(exc).lower()
    if "404" in msg:
        return "definitive:404"
    if "malformed" in msg or "schema" in msg:
        return "definitive:malformed"
    return None


def _try_fetch_with_retries(
    ticker: str,
    today: date,
    *,
    fetcher: Callable[..., pd.DataFrame],
    sleep_fn: Callable[[float], None] = time.sleep,
    backoff_seconds: tuple[int, ...] = _BACKOFF_SECONDS,
) -> str | None:
    """Try ``fetcher(ticker, end=today, back_extend=False)`` up to 3 times
    on transient errors. Returns the final classification tag:

    - ``None`` on success.
    - ``'transient_max_retries'`` when every attempt raised a transient
      error.
    - ``'definitive:<tag>'`` on first definitive failure (no retry).
    """
    last_tag: str | None = "transient_max_retries"
    for attempt, sleep_s in enumerate(backoff_seconds):
        try:
            fetcher(ticker, end=today, back_extend=False)
            return None
        except BaseException as exc:  # broad catch — any provider can throw anything.
            tag = _classify_definitive(exc)
            if tag is not None:
                return tag
            if not _is_transient_error(exc):
                # Unknown class — treat as definitive to avoid pointless retries.
                return f"definitive:{type(exc).__name__}"
            last_tag = "transient_max_retries"
            if attempt < len(backoff_seconds) - 1:
                sleep_fn(sleep_s)
    return last_tag


def cache_currency_check(
    *,
    universe_tickers: Iterable[str],
    universe_calendar: pd.DatetimeIndex,
    days_test_end: int,           # positional index of test_end in universe_calendar
    horizon_days: int,
    today: date,
    cache_latest_date: Callable[[str], str | None],
    fetcher: Callable[..., pd.DataFrame] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    backoff_seconds: tuple[int, ...] = _BACKOFF_SECONDS,
) -> list[DeficientTicker]:
    """Phase 0 cache-currency check per V1.4 plan §6.

    Returns a list of deficient tickers (one entry per ticker that
    couldn't be brought up to ``needed_end`` even after retries). An empty
    list means the cache is current and the run may proceed. Sub-case A
    is raised inline (``CacheCurrencyError``) since it's universe-level.

    Parameters
    ----------
    universe_tickers : Iterable[str]
        Every ticker the spec's universe expects to load.
    universe_calendar : pd.DatetimeIndex
        Trading-day index for the universe (NYSE/NSE).
    days_test_end : int
        Positional index of ``test_end`` in ``universe_calendar`` (the
        same value carve_universe_aligned uses).
    horizon_days : int
        ``target.horizon_days`` — needed_end = cal[days_test_end + horizon_days].
    today : date
        Reference for sub-case A and the auto-fetch ``end`` parameter.
    cache_latest_date : Callable
        Lookup function — typically ``gbdt.data._cache_last_date``. Must
        return an ISO date string or None if the ticker is uncached.
    fetcher : Callable | None
        Auto-fetch function — typically ``data_pipelines.fetch``. When
        ``None``, the deficient-ticker probe skips the retry loop and
        returns deficient entries with ``last_error_type='no_fetcher'``
        (used by tests).
    sleep_fn, backoff_seconds : kwargs for test injection.
    """
    if days_test_end + horizon_days >= len(universe_calendar):
        raise CacheCurrencyError(
            "[preflight] sub-case A: the universe calendar does not extend "
            f"{horizon_days} trading days past test_end (calendar idx "
            f"{days_test_end} + horizon={horizon_days} ≥ len={len(universe_calendar)}). "
            "Either extend the calendar (get_calendar(end=...)) or pick a "
            "shorter horizon."
        )
    needed_end_ts = universe_calendar[days_test_end + horizon_days]
    needed_end = needed_end_ts.date()
    if needed_end > today:
        raise CacheCurrencyError(
            f"[preflight] sub-case A: test_end + horizon = {needed_end.isoformat()} "
            f"is past today {today.isoformat()}. The cell cannot complete a "
            f"holdout test segment yet — either roll train_start back to a "
            f"window that completes before today, shrink horizon_days, or "
            f"shorten the per-segment durations."
        )

    deficient: list[DeficientTicker] = []
    for ticker in universe_tickers:
        latest_iso = cache_latest_date(ticker)
        if latest_iso is None:
            deficient.append(DeficientTicker(
                ticker=ticker,
                latest_cached=None,
                needed_through=needed_end.isoformat(),
                days_short=(needed_end - date(1900, 1, 1)).days,  # placeholder large value
                last_error_type="no_cache",
            ))
            continue
        latest_d = date.fromisoformat(latest_iso[:10])
        if latest_d >= needed_end:
            continue
        # Sub-case B — auto-fetch with retries.
        if fetcher is None:
            deficient.append(DeficientTicker(
                ticker=ticker,
                latest_cached=latest_iso,
                needed_through=needed_end.isoformat(),
                days_short=(needed_end - latest_d).days,
                last_error_type="no_fetcher",
            ))
            continue
        tag = _try_fetch_with_retries(
            ticker, today,
            fetcher=fetcher,
            sleep_fn=sleep_fn,
            backoff_seconds=backoff_seconds,
        )
        # Re-check cache after the (possibly successful) retry loop.
        latest_iso2 = cache_latest_date(ticker)
        latest_d2 = (
            date.fromisoformat(latest_iso2[:10]) if latest_iso2 is not None
            else None
        )
        if latest_d2 is not None and latest_d2 >= needed_end:
            continue
        deficient.append(DeficientTicker(
            ticker=ticker,
            latest_cached=latest_iso2,
            needed_through=needed_end.isoformat(),
            days_short=(needed_end - (latest_d2 or latest_d)).days,
            last_error_type=tag or "still_stale",
        ))
    return deficient


def format_deficient_table(deficient: list[DeficientTicker]) -> str:
    """Worked-example table per plan §6 — for the REFUSE message."""
    if not deficient:
        return "(none)"
    header = (
        f"{'ticker':<24}  "
        f"{'latest_cached':<14}  "
        f"{'needed_through':<14}  "
        f"{'days_short':>11}  "
        f"last_error_type"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for d in deficient:
        rows.append(
            f"{d.ticker:<24}  "
            f"{(d.latest_cached or '<none>'):<14}  "
            f"{d.needed_through:<14}  "
            f"{d.days_short:>11}  "
            f"{d.last_error_type}"
        )
    return "\n".join(rows)


__all__ = [
    "CacheCurrencyError",
    "DeficientTicker",
    "cache_currency_check",
    "format_deficient_table",
]
