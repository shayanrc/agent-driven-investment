"""fred_macro Domain wire-up — instantiates and registers FREDMacroDomain.

Importing this module is what makes ``fetch("FRED:DGS10", ...)`` work. Domain
#3 — and the first non-equity, non-OHLCV domain: a single ``(date, value)``
series across a curated macro panel (rates, curve, credit, inflation, labor,
growth).

Two firsts vs the equity domains:
  - **Per-series-cadence calendars** (daily / monthly / quarterly) via
    ``calendar_for()`` — the framework extension this domain motivated
    (predicted in ``adding_a_domain.md``).
  - **Single-source adapter with auto keyless/keyed transport** (no fallback
    chain): ``chain_for_gap`` returns just the FRED adapter.

Wiring is explicit (not auto-discovery), matching ``us_equities`` /
``nse_equities`` — chain, calendars, and the frequency map are visible here.
"""

from __future__ import annotations

import logging

from data_pipelines.adapter import Adapter
from data_pipelines.domain import Calendar, Domain, DomainRegistry
from data_pipelines.domains.fred_macro.adapters.fred import FredAdapter
from data_pipelines.domains.fred_macro.calendar import (
    BusinessDayCalendar,
    MonthlyCalendar,
    QuarterlyCalendar,
)
from data_pipelines.domains.fred_macro.config import FredMacroConfig
from data_pipelines.domains.fred_macro.registry import VALID_PREFIXES, parse_identifier
from data_pipelines.domains.fred_macro.schema import FRED_SCHEMA
from data_pipelines.domains.fred_macro.universe import load_frequency_map
from data_pipelines.schema import Schema

_log = logging.getLogger(__name__)


class FREDMacroDomain(Domain):
    """Domain #3 — FRED macro series, (date, value), per-series cadence."""

    def __init__(self, config: FredMacroConfig | None = None):
        self._config = config or FredMacroConfig()
        # Curated series → frequency, loaded once. Drives per-series calendar
        # selection (calendar_for) and the adapter's daily-densify decision.
        # If the config is absent (unusual — e.g., a stripped test tree), fall
        # back to an empty map: every series then defaults to business-day.
        try:
            self._frequency_map = load_frequency_map(self._config.default_universe)
        except (FileNotFoundError, ValueError):
            self._frequency_map = {}
        self._calendars: dict[str, Calendar] = {
            "daily": BusinessDayCalendar(),
            "monthly": MonthlyCalendar(),
            "quarterly": QuarterlyCalendar(),
        }
        self._adapters: dict[str, Adapter] = {
            "fred": FredAdapter(self._config, frequency_map=self._frequency_map),
        }

    @property
    def name(self) -> str: return "fred_macro"

    @property
    def identifier_prefixes(self) -> tuple[str, ...]: return VALID_PREFIXES

    @property
    def schema(self) -> Schema: return FRED_SCHEMA

    @property
    def calendar(self) -> Calendar:
        # Required single-calendar slot on the ABC; also the fallback for
        # unknown-frequency (out-of-universe) series. Business-day is the
        # most permissive convergent grid (see calendar.py docstring).
        return self._calendars["daily"]

    @property
    def config(self) -> FredMacroConfig: return self._config

    @property
    def adapters(self) -> dict[str, Adapter]: return dict(self._adapters)

    def parse_identifier(self, identifier: str) -> tuple[str, str]:
        return parse_identifier(identifier)

    def calendar_for(self, identifier: str) -> Calendar:
        _, series_id = parse_identifier(identifier)
        freq = self._frequency_map.get(series_id)
        if freq is None:
            _log.warning(
                "fred_macro: series %s is not in curated universe %r; using the "
                "business-day calendar for gap detection (best-effort — a "
                "monthly/quarterly out-of-universe series may show benign "
                "phantom gaps).",
                series_id, self._config.default_universe,
            )
            return self._calendars["daily"]
        return self._calendars[freq]

    def chain_for_gap(
        self, identifier: str, gap_size_trading_days: int, has_cache: bool,
    ) -> list[Adapter]:
        # Single-source: FRED is the only provider. No seed/update/fallback
        # tiering, no per-identifier routing — one adapter for every series.
        return [self._adapters["fred"]]


# Side effect: register on import. Importing data_pipelines.domains.fred_macro
# is what makes the domain available to fetch().
_DOMAIN_INSTANCE = FREDMacroDomain()
DomainRegistry.register(_DOMAIN_INSTANCE)


def get_domain() -> FREDMacroDomain:
    """Test/CLI hook for the singleton instance."""
    return _DOMAIN_INSTANCE
