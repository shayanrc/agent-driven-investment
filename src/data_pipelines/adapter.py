"""Adapter ABC — provider-specific data source plug-in for a domain.

Every concrete adapter (Stooq, Tiingo, yfinance, eventual FRED, etc.) implements
this contract. The fetch/parse split is deliberate: fetch writes raw bytes
atomically to data/raw/ and returns the path; parse reads that file and returns
a DataFrame still in source-native shape. Schema normalization is the next step
(at the cache.py boundary), NOT the adapter's job — keeps the raw layer honest
and the reprocess-from-raw path deterministic (D8).

Adapters raise typed errors from data_pipelines.errors on failure; the dispatch
layer handles chain fallthrough.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import pandas as pd


class Adapter(ABC):
    """Provider-specific data source for one domain.

    Concrete subclasses live under domains/<domain>/adapters/. They are
    instantiated by the domain's __init__.py and registered in the domain's
    chain config (seed / update / fallback).
    """

    name: str  # provider name, e.g., "stooq", "tiingo". Used in raw paths and meta.

    # Optional: map of source-native column names → canonical schema names.
    # Dispatch passes this to domain.schema.normalize() after parse().
    # Default None means parse() already returns canonical names.
    source_column_map: dict[str, str] | None = None

    # Optional: extra fields the framework merges into the sources[] entry in
    # _meta.json. us_equities adapters use this for adjustment_quality.
    extra_meta: dict = {}

    @abstractmethod
    def fetch(
        self,
        identifier: str,
        start: date | None = None,
        end: date | None = None,
        *,
        data_root: Path,
    ) -> Path:
        """Download source data, write raw bytes atomically, return raw path.

        Implementations call raw_store.write_raw_atomic(data_root, ...) to
        land bytes. Source-native format (CSV for Stooq, JSON for Tiingo,
        parquet for yfinance). Range arguments are optional — some providers
        (Stooq's simple CSV endpoint) don't accept ranges and return full
        history; the adapter should still record the *requested* range in the
        raw filename (the audit-of-the-request); authoritative coverage is
        derived from the parsed DataFrame's date column at meta build time.

        `data_root` is passed in by dispatch — adapters do not bind to a
        global path at construction time, so the same adapter instance works
        for tests (tmp_path), CLI runs (./data), and notebook sessions.

        Raises ProviderError / EmptyPayload / MissingAPIKey on failure. Never
        raises SchemaMismatch (that's a parse-stage concern).
        """

    @abstractmethod
    def parse(self, raw_path: Path) -> pd.DataFrame:
        """Read a raw file produced by self.fetch and return a DataFrame.

        Returned shape is source-native (columns/dtypes as the provider gave
        them) — normalization to the canonical domain schema happens later via
        Schema.normalize(). Pure function of the file contents; no network.

        Must be deterministic — same raw bytes → same DataFrame, every time
        (D8). No timestamps, no random tie-breaking.
        """

    def health_check(self) -> bool:
        """Best-effort liveness check for the provider.

        Default implementation returns True; concrete adapters override to
        ping the API or verify credentials are present. Used by the CLI
        `health` subcommand. Never raises; on failure returns False.
        """
        return True
