"""in_fundamentals Domain wire-up — instantiates and registers
InFundamentalsDomain.

Domain #5 — quarterly Indian company fundamentals (revenue, net income, EPS,
derived weighted shares; quarterly cash flow does not exist in India) for the
NSE equity universes, keyed ``INFUND:<TICKER>`` on the calendar quarter-end
grid shared with us_fundamentals.

Single-provider chain (the fred_macro precedent):

  1. **nse_xbrl** — NSE corporate-filings metadata (native, exchange-
     timestamped ``filingDate`` — point-in-time truth without an enrichment
     pass) + Ind-AS XBRL instances from ``nsearchives.nseindia.com``.

BSE and yfinance-IN fallbacks are parked in ``V4_TBD.md``. Per-filing XBRL
files make the fetches gap-bounded, so seed and update use the same chain.
"""

from __future__ import annotations

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domain import Calendar, Domain, DomainRegistry
from data_pipelines.domains.in_fundamentals.config import InFundamentalsConfig
from data_pipelines.domains.in_fundamentals.registry import (
    VALID_PREFIXES,
    parse_identifier,
)
from data_pipelines.domains.in_fundamentals.schema import IN_FUNDAMENTALS_SCHEMA
from data_pipelines.domains.us_fundamentals.calendar import QuarterEndCalendar
from data_pipelines.schema import Schema

CHAIN_ORDER: tuple[str, ...] = ("nse_xbrl",)


class InFundamentalsDomain(Domain):
    """Domain #5 — quarterly Indian company fundamentals, NSE XBRL chain."""

    def __init__(self, config: InFundamentalsConfig | None = None):
        self._config = config or InFundamentalsConfig()
        self._calendar = QuarterEndCalendar(
            lag_days=self._config.reporting_lag_days
        )
        self._adapters: dict[str, Adapter] = {}
        self._register_adapters()

    def _register_adapters(self) -> None:
        from data_pipelines.domains.in_fundamentals.adapters.nse_xbrl import (
            NSEXbrlAdapter,
        )

        self._adapters["nse_xbrl"] = NSEXbrlAdapter(self._config)

    @property
    def name(self) -> str: return "in_fundamentals"

    @property
    def identifier_prefixes(self) -> tuple[str, ...]: return VALID_PREFIXES

    @property
    def schema(self) -> Schema: return IN_FUNDAMENTALS_SCHEMA

    @property
    def calendar(self) -> Calendar: return self._calendar

    @property
    def config(self) -> InFundamentalsConfig: return self._config

    @property
    def adapters(self) -> dict[str, Adapter]: return dict(self._adapters)

    def parse_identifier(self, identifier: str) -> tuple[str, str]:
        return parse_identifier(identifier)

    def chain_for_gap(
        self, identifier: str, gap_size_trading_days: int, has_cache: bool,
    ) -> list[Adapter]:
        # One chain for every ticker and gap size — the adapter's fetches are
        # gap-bounded internally (per-filing XBRL downloads).
        return [
            self._adapters[k] for k in CHAIN_ORDER if k in self._adapters
        ]

    def merge_overlap(
        self,
        existing: pd.DataFrame,
        new: pd.DataFrame,
        existing_sources: list[dict],
        new_source: dict,
    ) -> pd.DataFrame:
        """Per-cell first-written-wins, new fills holes.

        Identical policy to us_fundamentals (same point-in-time posture:
        history already in the cache is never silently rewritten by a later
        refresh — NSE re-filings and audited revisions lose to the
        as-first-published row); NaN/NaT cells are filled from the new frame.
        ``fcf`` is recomputed after the cell-merge for internal consistency —
        NaN-propagating, so it stays honestly NaN while India files no
        quarterly cash flow.
        """
        merged = (
            existing.set_index("date")
            .combine_first(new.set_index("date"))
            .reset_index()
        )
        merged["fcf"] = merged["ocf"] - merged["capex"]
        return (
            merged[self.schema.column_names]
            .sort_values("date")
            .reset_index(drop=True)
        )


# Side effect: register on import — importing data_pipelines.domains.
# in_fundamentals is what makes fetch("INFUND:RELIANCE", ...) work.
_DOMAIN_INSTANCE = InFundamentalsDomain()
DomainRegistry.register(_DOMAIN_INSTANCE)


def get_domain() -> InFundamentalsDomain:
    """Test/CLI hook for the singleton instance."""
    return _DOMAIN_INSTANCE
