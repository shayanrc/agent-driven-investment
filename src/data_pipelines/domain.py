"""Domain ABC + DomainRegistry.

A Domain bundles everything specific to one kind of time-series data:
schema, identifier parser, universe, calendar, adapter chain, and per-domain
config. The framework dispatches by looking up the domain whose registered
identifier prefix matches the incoming `fetch(identifier, ...)` call.

v1 ships exactly one domain — us_equities — but the abstraction exists so
domain #2 (NSE, FRED, commodities, ...) plugs in without touching the
framework. Per the plan: do not over-crystallize; this ABC declares only what
us_equities actually needs.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable, Protocol

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.errors import UnknownDomain
from data_pipelines.schema import Schema

# Domain.name, time_column, and schema column names are interpolated into
# DDL and SQL via f-strings in cache.py (sqlite3 cannot parameterize
# identifiers). Today every value comes from trusted in-repo code, but a
# typo or a future domain registering a name like 'x; DROP TABLE' would
# produce valid injection. Reject anything that isn't a plain SQL
# identifier at registration time so the unsafe value never reaches a
# query builder.
_VALID_SQL_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _validate_sql_ident(value: str, what: str) -> None:
    if not isinstance(value, str) or not _VALID_SQL_IDENT.match(value):
        raise ValueError(
            f"invalid SQL identifier for {what}: {value!r} "
            f"(must match [A-Za-z_][A-Za-z0-9_]{{0,63}}; this value is "
            f"interpolated into cache DDL/SQL and cannot be parameterized)"
        )


class Calendar(Protocol):
    """Minimal calendar interface used by cache.detect_gaps.

    A domain supplies a Calendar (NYSE for us_equities, ECB for euro rates,
    etc.). The framework asks only one thing of it: "list the valid time
    points in [start, end]". For us_equities those are trading days; for
    FRED-style series they might be every weekday or every month-end.
    """

    def trading_days(self, start: date, end: date) -> list[date]:
        ...


class Domain(ABC):
    """Plug-in for one kind of time-series data (US equities, FRED macro, ...).

    Concrete domains live under domains/<name>/__init__.py and call
    DomainRegistry.register(...) on import. The framework reads only what's
    declared by the ABC; everything else is the domain's private concern.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short canonical name, e.g., "us_equities". Used in cache paths."""

    @property
    @abstractmethod
    def identifier_prefixes(self) -> tuple[str, ...]:
        """Prefixes this domain claims, e.g., ("NYSE", "NASDAQ", "INDEX")
        for us_equities. The registry indexes domains by these.
        """

    @property
    @abstractmethod
    def schema(self) -> Schema:
        """Canonical processed-layer schema for this domain."""

    @abstractmethod
    def parse_identifier(self, identifier: str) -> tuple[str, str]:
        """Split `<PREFIX>:<SYMBOL>` → (exchange_or_namespace, symbol).

        The prefix is used as the path segment under
        data/processed/<domain>/<prefix>/<symbol>/. For domains without an
        exchange concept (e.g., FRED), the prefix segment can be a placeholder
        like '-'.
        """

    @property
    @abstractmethod
    def calendar(self) -> Calendar:
        """Calendar this domain's gap detection uses."""

    @abstractmethod
    def chain_for_gap(
        self,
        identifier: str,
        gap_size_trading_days: int,
        has_cache: bool,
    ) -> list[Adapter]:
        """Ordered list of adapters dispatch should try for this gap.

        Domains may route different identifiers through different chains —
        e.g., us_equities prefers yfinance for INDEX:* (Stooq is gated,
        Tiingo doesn't cover indices) but [stooq, tiingo, yfinance] for
        equities. The identifier parameter exists for that case; ignore it
        if your domain uses one chain for all symbols.

        Each adapter is tried in order on transient/typed failures (D5).
        Per dispatch's partial-fill semantics: after a provider returns
        rows, the dispatcher re-detects remaining sub-gaps and continues
        the chain to fill them. So putting "best quality but partial coverage"
        first and "weaker quality but full coverage" later is a legitimate
        and useful pattern (used for NSE NIFTY: → [nselib, yfinance]).

        Returning an empty list means "this gap is unfillable" — dispatch
        will raise AllProvidersFailed.
        """

    @property
    def time_column(self) -> str:
        """Name of the time-axis column in the canonical schema. v1 domains
        all use 'date'; future intraday domains might use 'timestamp'.
        """
        return "date"

    def merge_overlap(
        self,
        existing: pd.DataFrame,
        new: pd.DataFrame,
        existing_sources: list[dict],
        new_source: dict,
    ) -> pd.DataFrame:
        """Resolve rows present on both sides of a merge.

        Default: new wins entirely on overlapping dates. Domains that need
        per-column precedence (us_equities preserving full-quality adj_close
        even when split-only Stooq rows arrive later) override this.

        Inputs are restricted to overlapping rows. Output is the resolved
        rows in canonical schema, indexed by time_column ascending.
        """
        return new


class DomainRegistry:
    """Process-global singleton mapping identifier prefixes → Domain instances.

    Concrete domains call register(...) at import time from their package
    __init__.py. resolve(...) parses the prefix off an identifier and returns
    the corresponding domain, raising UnknownDomain on miss.

    Duplicate registration of the same prefix is a programmer error and
    raises immediately — per open question 7 in V1_IMPLEMENTATION_PLAN.md.
    """

    _by_prefix: dict[str, Domain] = {}

    @classmethod
    def register(cls, domain: Domain) -> None:
        _validate_sql_ident(domain.name, f"{type(domain).__name__}.name")
        _validate_sql_ident(
            domain.time_column, f"{type(domain).__name__}.time_column"
        )
        for col in domain.schema.columns:
            _validate_sql_ident(
                col.name,
                f"{type(domain).__name__}.schema column {col.name!r}",
            )
        for prefix in domain.identifier_prefixes:
            if prefix in cls._by_prefix and cls._by_prefix[prefix] is not domain:
                raise ValueError(
                    f"duplicate domain prefix {prefix!r}: "
                    f"{cls._by_prefix[prefix].name!r} already registered, "
                    f"cannot also register {domain.name!r}"
                )
            cls._by_prefix[prefix] = domain

    @classmethod
    def resolve(cls, identifier: str) -> Domain:
        if ":" not in identifier:
            raise UnknownDomain(identifier, sorted(cls._by_prefix.keys()))
        prefix = identifier.split(":", 1)[0]
        if prefix not in cls._by_prefix:
            raise UnknownDomain(identifier, sorted(cls._by_prefix.keys()))
        return cls._by_prefix[prefix]

    @classmethod
    def registered_prefixes(cls) -> list[str]:
        return sorted(cls._by_prefix.keys())

    @classmethod
    def registered_domains(cls) -> Iterable[Domain]:
        # de-duplicate: one Domain may claim multiple prefixes.
        seen: dict[int, Domain] = {}
        for d in cls._by_prefix.values():
            seen[id(d)] = d
        return seen.values()

    @classmethod
    def _reset(cls) -> None:
        """Test-only: clear all registrations."""
        cls._by_prefix.clear()
