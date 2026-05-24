"""us_equities Domain wire-up — instantiates and registers with DomainRegistry.

Importing this module is what makes `fetch("NYSE:AAPL", ...)` work.

Wiring is intentionally explicit (not auto-discovery) so the chain order,
adapter set, and thresholds are visible in one place. Other domains follow
the same shape under domains/<name>/__init__.py.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domain import Calendar, Domain, DomainRegistry
from data_pipelines.domains.us_equities.adapters.stooq import StooqAdapter
from data_pipelines.domains.us_equities.adapters.tiingo import TiingoAdapter
from data_pipelines.domains.us_equities.adapters.yfinance import YFinanceAdapter
from data_pipelines.domains.us_equities.calendar import NYSECalendar
from data_pipelines.domains.us_equities.config import USEquitiesConfig
from data_pipelines.domains.us_equities.registry import (
    VALID_PREFIXES,
    parse_identifier,
)
from data_pipelines.domains.us_equities.schema import (
    OHLCV_SCHEMA,
    merge_overlap_us_equities,
)
from data_pipelines.schema import Schema


class USEquitiesDomain(Domain):
    """The v1 reference domain: NYSE + NASDAQ + INDEX, daily OHLCV.

    Chain logic (per V1_IMPLEMENTATION_PLAN.md §dispatch):
      - Cold cache OR gap > big_gap_threshold_days → [stooq]
      - Smaller gap with cache → [tiingo, yfinance] in order

    Adjustment-quality precedence is handled in merge_overlap, delegating to
    merge_overlap_us_equities (full beats split_only on adj_close).
    """

    def __init__(self, config: USEquitiesConfig | None = None):
        self._config = config or USEquitiesConfig()
        self._calendar = NYSECalendar()
        self._adapters: dict[str, Adapter] = {
            "stooq": StooqAdapter(self._config),
            "tiingo": TiingoAdapter(self._config),
            "yfinance": YFinanceAdapter(self._config),
        }

    @property
    def name(self) -> str: return "us_equities"

    @property
    def identifier_prefixes(self) -> tuple[str, ...]: return VALID_PREFIXES

    @property
    def schema(self) -> Schema: return OHLCV_SCHEMA

    @property
    def calendar(self) -> Calendar: return self._calendar

    @property
    def config(self) -> USEquitiesConfig: return self._config

    @property
    def adapters(self) -> dict[str, Adapter]: return dict(self._adapters)

    def parse_identifier(self, identifier: str) -> tuple[str, str]:
        return parse_identifier(identifier)

    def chain_for_gap(
        self, identifier: str, gap_size_trading_days: int, has_cache: bool,
    ) -> list[Adapter]:
        # INDEX:* gets its own chain. Tiingo's free-tier daily endpoint
        # short-circuits indices with ProviderError ("does not support
        # INDEX symbols"); Stooq's bulk CSV is IP-gated for us but may
        # work elsewhere and (when it works) reports a true index-level
        # volume. yfinance always succeeds for indices and reports the
        # constituent-aggregate as Volume (~2.6B/day for ^SPX). Order:
        # yfinance first (always-on baseline + full coverage), Stooq
        # opportunistic backup that, IF accessible, might bring a richer
        # signal. Skip Tiingo entirely for indices to avoid the dead leg.
        prefix, _ = parse_identifier(identifier)
        if prefix == "INDEX":
            chain = []
            if "yfinance" in self._adapters:
                chain.append(self._adapters["yfinance"])
            if "stooq" in self._adapters:
                chain.append(self._adapters["stooq"])
            return chain

        # Equity path (NYSE/NASDAQ) — unchanged from the v1 design.
        update_chain = [self._adapters[name] for name in self._config.chain_update
                        if name in self._adapters]
        if not has_cache or gap_size_trading_days > self._config.big_gap_threshold_days:
            # Seed first; fall through to update tiers if seed fails (e.g.,
            # Stooq's IP-gated apikey response, network errors). Tiingo and
            # yfinance both accept range params and can backfill full history
            # — slower per call but reliable. Discovered during v1 smoke that
            # the IP-gating made [stooq]-only chains brittle.
            return [self._adapters[self._config.chain_seed], *update_chain]
        return update_chain

    def merge_overlap(
        self,
        existing: pd.DataFrame,
        new: pd.DataFrame,
        existing_sources: list[dict],
        new_source: dict,
    ) -> pd.DataFrame:
        return merge_overlap_us_equities(existing, new, existing_sources, new_source)


# Side effect: register on import. Importing data_pipelines.domains.us_equities
# is what makes the domain available to fetch().
_DOMAIN_INSTANCE = USEquitiesDomain()
DomainRegistry.register(_DOMAIN_INSTANCE)


def get_domain() -> USEquitiesDomain:
    """Test/CLI hook for the singleton instance."""
    return _DOMAIN_INSTANCE
