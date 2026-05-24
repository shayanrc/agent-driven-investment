"""nse_equities Domain wire-up — instantiates and registers NSEDomain.

Importing this module is what makes `fetch("NSE:RELIANCE", ...)` and
`fetch("NIFTY:50", ...)` work. The framework triggers this via
`data_pipelines.__init__`, which imports each shipped domain at startup.

Chain composition (post-Refactor-A, per-identifier):
  - NSE: equity        → [jugaad, nselib, yfinance]  (jugaad fastest +
        most authoritative for NSE bhav data; nselib + yfinance as
        progressive fallbacks)
  - NIFTY: index       → [nselib, yfinance]          (nselib has true
        TRADED_QTY for the ~3-year window upstream provides; yfinance
        backfills older OHLC via dispatch's partial-fill continuation.
        jugaad excluded — its index payload has no VOLUME field.)
  - BSE: anything      → [yfinance]                  (jugaad + nselib
        both NSE-only.)

All three NSE providers are range-aware (jugaad's `from_date`/`to_date`,
nselib's `from_date`/`to_date`, yfinance's `start`/`end`), so chain
composition is purely identifier-driven, not gap-size-driven.
"""

from __future__ import annotations

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domain import Calendar, Domain, DomainRegistry
from data_pipelines.domains.nse_equities.adapters.jugaad import JugaadAdapter
from data_pipelines.domains.nse_equities.adapters.nselib import NSElibAdapter
from data_pipelines.domains.nse_equities.adapters.yfinance import YFinanceNSEAdapter
from data_pipelines.domains.nse_equities.calendar import NSECalendar
from data_pipelines.domains.nse_equities.config import NSEEquitiesConfig
from data_pipelines.domains.nse_equities.registry import (
    VALID_PREFIXES,
    parse_identifier,
)
from data_pipelines.domains.nse_equities.schema import (
    OHLCV_SCHEMA,
    merge_overlap_nse_equities,
)
from data_pipelines.schema import Schema


class NSEDomain(Domain):
    """Domain #2 — NSE (India) daily equities + NIFTY indices.

    Chain logic: per-identifier (see module docstring). Per-adapter
    EmptyPayload short-circuits remain as a defense in depth for BSE: /
    NIFTY: routed through the wrong adapter.

    Adjustment precedence: yfinance "full" beats jugaad/nselib "none" via
    merge_overlap_nse_equities, so a later jugaad refresh never regresses
    adj_close.
    """

    def __init__(self, config: NSEEquitiesConfig | None = None):
        self._config = config or NSEEquitiesConfig()
        self._calendar = NSECalendar()
        self._adapters: dict[str, Adapter] = {
            "jugaad":   JugaadAdapter(self._config),
            "nselib":   NSElibAdapter(self._config),
            "yfinance": YFinanceNSEAdapter(self._config),
        }

    @property
    def name(self) -> str: return "nse_equities"

    @property
    def identifier_prefixes(self) -> tuple[str, ...]: return VALID_PREFIXES

    @property
    def schema(self) -> Schema: return OHLCV_SCHEMA

    @property
    def calendar(self) -> Calendar: return self._calendar

    @property
    def config(self) -> NSEEquitiesConfig: return self._config

    @property
    def adapters(self) -> dict[str, Adapter]: return dict(self._adapters)

    def parse_identifier(self, identifier: str) -> tuple[str, str]:
        return parse_identifier(identifier)

    def chain_for_gap(
        self, identifier: str, gap_size_trading_days: int, has_cache: bool,
    ) -> list[Adapter]:
        prefix, _ = parse_identifier(identifier)
        if prefix == "NIFTY":
            chain = []
            for name in ("nselib", "yfinance"):
                if name in self._adapters:
                    chain.append(self._adapters[name])
            return chain
        if prefix == "BSE":
            return ([self._adapters["yfinance"]]
                    if "yfinance" in self._adapters else [])
        # NSE equities — default chain order from config.
        return [self._adapters[name] for name in self._config.chain_order
                if name in self._adapters]

    def merge_overlap(
        self,
        existing: pd.DataFrame,
        new: pd.DataFrame,
        existing_sources: list[dict],
        new_source: dict,
    ) -> pd.DataFrame:
        return merge_overlap_nse_equities(existing, new, existing_sources, new_source)


# Side-effect import: registers on import. data_pipelines/__init__.py imports
# this module so DomainRegistry sees both us_equities and nse_equities.
_DOMAIN_INSTANCE = NSEDomain()
DomainRegistry.register(_DOMAIN_INSTANCE)


def get_domain() -> NSEDomain:
    """Test/CLI hook for the singleton instance."""
    return _DOMAIN_INSTANCE
