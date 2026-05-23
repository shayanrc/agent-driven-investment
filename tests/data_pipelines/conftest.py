"""Shared fixtures: synthetic domain with fake calendar, sample DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import pytest

from data_pipelines.domain import Calendar, Domain, DomainRegistry
from data_pipelines.schema import ColumnSpec, Schema


@dataclass
class FakeCalendar:
    """Calendar that treats every day in a pinned set as a 'trading day'.

    Default: every weekday in the requested range. Override `holidays` to
    skip specific dates.
    """

    holidays: frozenset[date] = frozenset()

    def trading_days(self, start: date, end: date) -> list[date]:
        out: list[date] = []
        d = start
        while d <= end:
            if d.weekday() < 5 and d not in self.holidays:
                out.append(d)
            d += timedelta(days=1)
        return out


class FakeDomain(Domain):
    """Minimal Domain for framework-level tests. (date, value) schema."""

    def __init__(
        self,
        name: str = "fake",
        prefixes: tuple[str, ...] = ("FAKE",),
        calendar: Calendar | None = None,
        overlap_policy: str = "new_wins",  # or "existing_wins"
        adapters: list | None = None,
        big_gap_threshold: int = 10,
    ):
        self._name = name
        self._prefixes = prefixes
        self._calendar = calendar or FakeCalendar()
        self._schema = Schema(columns=(
            ColumnSpec("date", "datetime64[ns]"),
            ColumnSpec("value", "float64"),
        ))
        self._overlap_policy = overlap_policy
        self._adapters = adapters or []
        self._big_gap_threshold = big_gap_threshold

    @property
    def name(self): return self._name
    @property
    def identifier_prefixes(self): return self._prefixes
    @property
    def schema(self): return self._schema
    @property
    def calendar(self): return self._calendar

    def parse_identifier(self, identifier):
        prefix, sym = identifier.split(":", 1)
        return prefix, sym

    def merge_overlap(self, existing, new, existing_sources, new_source):
        if self._overlap_policy == "existing_wins":
            return existing
        return new

    def chain_for_gap(self, gap_size_trading_days: int, has_cache: bool):
        """First adapter is treated as seed, rest as update chain.

        Triggers seed when: no cache OR gap exceeds threshold.
        """
        if not self._adapters:
            return []
        if not has_cache or gap_size_trading_days > self._big_gap_threshold:
            return [self._adapters[0]]
        return self._adapters[1:] if len(self._adapters) > 1 else [self._adapters[0]]


@pytest.fixture(autouse=True)
def _isolate_registry():
    DomainRegistry._reset()
    yield
    DomainRegistry._reset()


@pytest.fixture
def fake_domain() -> FakeDomain:
    return FakeDomain()


def make_df(dates: Iterable[date], values: Iterable[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)).astype("datetime64[ns]"),
        "value": [float(v) for v in values],
    })
