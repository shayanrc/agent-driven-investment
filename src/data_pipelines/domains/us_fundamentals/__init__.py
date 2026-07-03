"""us_fundamentals Domain wire-up — instantiates and registers USFundamentalsDomain.

Domain #4 — quarterly company fundamentals (revenue, net income, operating
cash flow, capex, FCF, weighted shares, EPS) for the US equity universes,
keyed ``FUND:<TICKER>`` on a calendar quarter-end grid.

Like the price domains, fundamentals get a tiered provider chain:

  1. **macrotrends** (primary) — scraped statement pages; deep pre-normalized
     quarterly history (2011→now), $M units, both seed and update (every page
     carries full history).
  2. **edgar** (secondary) — SEC XBRL companyfacts; official, ~2008→now, and
     the only source of ``filed_date`` (point-in-time discipline).
  3. **yfinance** (tertiary) — last ~5 quarters only; newest-quarter last
     resort, mirroring its role in the us_equities chain.

Unlike the equity chain there is no seed/update tier split by gap size —
every provider returns its full available history per request, so one chain
serves any gap. Adapters register themselves into ``_adapters`` as their
phases land (Phases 2–4 of ``docs/data_pipelines/V3_US_FUNDAMENTALS_PLAN.md``);
``chain_for_gap`` serves whatever subset exists, in chain order.
"""

from __future__ import annotations

from data_pipelines.adapter import Adapter
from data_pipelines.domain import Calendar, Domain, DomainRegistry
from data_pipelines.domains.us_fundamentals.calendar import QuarterEndCalendar
from data_pipelines.domains.us_fundamentals.config import USFundamentalsConfig
from data_pipelines.domains.us_fundamentals.registry import (
    VALID_PREFIXES,
    parse_identifier,
)
from data_pipelines.domains.us_fundamentals.schema import US_FUNDAMENTALS_SCHEMA
from data_pipelines.schema import Schema

# Provider precedence — primary → secondary → tertiary.
CHAIN_ORDER: tuple[str, ...] = ("macrotrends", "edgar", "yfinance")


class USFundamentalsDomain(Domain):
    """Domain #4 — quarterly US company fundamentals, tiered provider chain."""

    def __init__(self, config: USFundamentalsConfig | None = None):
        self._config = config or USFundamentalsConfig()
        self._calendar = QuarterEndCalendar(
            lag_days=self._config.reporting_lag_days
        )
        self._adapters: dict[str, Adapter] = {}
        self._register_adapters()

    def _register_adapters(self) -> None:
        """Instantiate the provider adapters that have landed (Phases 2-4)."""
        # Populated as adapter phases land; empty dict = unfillable gaps
        # (dispatch raises AllProvidersFailed), which is correct until then.

    @property
    def name(self) -> str: return "us_fundamentals"

    @property
    def identifier_prefixes(self) -> tuple[str, ...]: return VALID_PREFIXES

    @property
    def schema(self) -> Schema: return US_FUNDAMENTALS_SCHEMA

    @property
    def calendar(self) -> Calendar: return self._calendar

    @property
    def config(self) -> USFundamentalsConfig: return self._config

    @property
    def adapters(self) -> dict[str, Adapter]: return dict(self._adapters)

    def parse_identifier(self, identifier: str) -> tuple[str, str]:
        return parse_identifier(identifier)

    def chain_for_gap(
        self, identifier: str, gap_size_trading_days: int, has_cache: bool,
    ) -> list[Adapter]:
        # One chain for every ticker and gap size: each provider returns its
        # full available history per request, so there is no seed/update
        # threshold semantics to encode (see module docstring).
        return [
            self._adapters[k] for k in CHAIN_ORDER if k in self._adapters
        ]


# Side effect: register on import — importing data_pipelines.domains.
# us_fundamentals is what makes fetch("FUND:AAPL", ...) work.
_DOMAIN_INSTANCE = USFundamentalsDomain()
DomainRegistry.register(_DOMAIN_INSTANCE)


def get_domain() -> USFundamentalsDomain:
    """Test/CLI hook for the singleton instance."""
    return _DOMAIN_INSTANCE
