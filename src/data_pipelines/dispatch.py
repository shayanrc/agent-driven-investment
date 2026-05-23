"""Public fetch() dispatch: identifier → domain → cache → adapter chain → return.

The orchestration is intentionally linear:

  resolve domain → read cache → detect gaps → for each gap: try the domain's
  adapter chain in order, parse + normalize + validate, merge into running
  cache, atomic-write → return the requested slice.

Each adapter call lands raw bytes via raw_store BEFORE parse runs (so a parse
failure leaves the raw on disk for inspection and reprocess). Schema
normalization + validation happen at this boundary — adapters never call them.
Provider failures fall through the chain silently per D5; exhaustion of the
chain raises AllProvidersFailed and leaves the cache untouched for that gap.

Returns:
    fetch(...) → DataFrame  (canonical schema, requested-range slice)
    fetch_with_meta(...) → (DataFrame, FetchMeta)  for the agent-tool path
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipelines.cache import (
    detect_gaps,
    merge_cache,
    read_processed,
    write_processed_atomic,
)
from data_pipelines.domain import Domain, DomainRegistry
from data_pipelines.errors import (
    AllProvidersFailed,
    EmptyPayload,
    ProviderError,
    SchemaMismatch,
)


@dataclass
class FetchMeta:
    """JSON-serializable summary of a fetch call. The agent-tool wrapper
    consumes this directly.
    """

    identifier: str
    domain: str
    range: dict[str, str]                              # {"start": "...", "end": "..."}
    row_count: int
    cache_was_cold: bool
    gaps_filled: list[dict[str, Any]] = field(default_factory=list)
    providers_failed: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "domain": self.domain,
            "range": self.range,
            "row_count": self.row_count,
            "cache_was_cold": self.cache_was_cold,
            "gaps_filled": self.gaps_filled,
            "providers_failed": self.providers_failed,
        }


def fetch(
    identifier: str,
    start: str | date,
    end: str | date,
    frequency: str = "daily",
    data_root: str | Path = "data",
) -> pd.DataFrame:
    """Public surface. Returns the canonical DataFrame for the requested
    range, populating the cache as needed.
    """
    df, _ = fetch_with_meta(identifier, start, end, frequency, data_root)
    return df


def fetch_with_meta(
    identifier: str,
    start: str | date,
    end: str | date,
    frequency: str = "daily",
    data_root: str | Path = "data",
) -> tuple[pd.DataFrame, FetchMeta]:
    """fetch() + JSON-serializable FetchMeta. Use this from the CLI and the
    eventual agent-tool wrapper.
    """
    if frequency != "daily":
        raise NotImplementedError(f"frequency={frequency!r}; v1 supports 'daily' only")

    start_d = _as_date(start)
    end_d = _as_date(end)
    if start_d > end_d:
        raise ValueError(f"start ({start_d}) is after end ({end_d})")

    data_root_p = Path(data_root)

    domain = DomainRegistry.resolve(identifier)
    cached_df, cached_meta = read_processed(data_root_p, domain, identifier)
    cache_was_cold = cached_df is None

    # Cap gap detection at the cache's earliest known date. The cache was
    # built by asking providers for the broadest possible range, so its first
    # row is the asset's earliest available date. Requesting earlier than
    # that hits providers needlessly (network round trip → empty payload →
    # soft fail) on every re-seed of a post-IPO ticker. If a new vintage
    # source ever exposes pre-cache data, purge that ticker to re-discover.
    effective_start = start_d
    if cached_df is not None and len(cached_df) > 0:
        cache_first = _as_date(cached_df[domain.time_column].iloc[0])
        if effective_start < cache_first:
            effective_start = cache_first
    gaps = detect_gaps(
        cached_df, effective_start, end_d, domain.calendar, domain.time_column
    )

    gaps_filled: list[dict[str, Any]] = []
    providers_failed: list[dict[str, str]] = []

    for gap_start, gap_end in gaps:
        gap_days = len(domain.calendar.trading_days(gap_start, gap_end))
        has_cache = cached_df is not None and len(cached_df) > 0
        chain = domain.chain_for_gap(gap_days, has_cache)

        if not chain:
            raise AllProvidersFailed(
                identifier,
                [ProviderError("<none>", identifier, "domain returned empty chain")],
            )

        gap_failures: list[ProviderError] = []
        success_df: pd.DataFrame | None = None
        success_source: dict[str, Any] = {}

        for adapter in chain:
            try:
                exchange, ticker = domain.parse_identifier(identifier)
                raw_path = adapter.fetch(
                    identifier, start=gap_start, end=gap_end,
                    data_root=data_root_p,
                )
                src_df = adapter.parse(raw_path)
                norm_df = domain.schema.normalize(
                    src_df,
                    source_column_map=getattr(adapter, "source_column_map", None),
                    provider=adapter.name,
                    identifier=identifier,
                )
                domain.schema.validate(
                    norm_df, provider=adapter.name, identifier=identifier
                )
                success_df = norm_df
                success_source = _build_source_meta(
                    adapter=adapter,
                    raw_path=raw_path,
                    df=norm_df,
                    time_column=domain.time_column,
                )
                break
            except EmptyPayload as e:
                gap_failures.append(e)
                providers_failed.append({"provider": e.provider, "reason": e.reason})
                continue
            except ProviderError as e:
                gap_failures.append(e)
                providers_failed.append({"provider": e.provider, "reason": e.reason})
                continue
            except SchemaMismatch as e:
                gap_failures.append(
                    ProviderError(e.provider, identifier, f"schema mismatch: {e.details}")
                )
                providers_failed.append({"provider": e.provider, "reason": f"schema mismatch: {e.details}"})
                continue

        if success_df is None:
            # Soft-fail: when the cache already has data AND at least one
            # provider gave an authoritative EmptyPayload (= "no data exists
            # for this range") response, treat the gap as legitimately
            # unfillable. Other failures in the same chain (Stooq IP-gate,
            # Tiingo 429, network errors) are environmental noise and don't
            # contradict the EmptyPayload signal. Common case: pre-IPO date
            # ranges where the asset didn't yet exist.
            # Hard-fail when no provider returned an authoritative answer
            # (all errors are environmental) — we genuinely don't know.
            any_empty = any(isinstance(f, EmptyPayload) for f in gap_failures)
            has_cache = cached_df is not None and len(cached_df) > 0
            if any_empty and has_cache:
                providers_failed.append({
                    "provider": "<chain>",
                    "reason": (f"gap {gap_start}..{gap_end} unfillable "
                               f"(provider(s) returned empty; asset may not "
                               f"have existed); existing cache preserved"),
                })
                continue
            raise AllProvidersFailed(identifier, gap_failures)

        cached_df, cached_meta = merge_cache(
            cached_df, success_df, cached_meta, success_source, domain
        )
        write_processed_atomic(
            data_root_p, domain, identifier, cached_df, cached_meta
        )
        gaps_filled.append({
            "gap": {"start": gap_start.isoformat(), "end": gap_end.isoformat()},
            "provider": success_source["provider"],
            "rows": int(len(success_df)),
        })

    sliced = _slice(cached_df, start_d, end_d, domain.time_column)

    meta = FetchMeta(
        identifier=identifier,
        domain=domain.name,
        range={"start": start_d.isoformat(), "end": end_d.isoformat()},
        row_count=int(len(sliced)),
        cache_was_cold=cache_was_cold,
        gaps_filled=gaps_filled,
        providers_failed=providers_failed,
    )
    return sliced, meta


def _slice(df: pd.DataFrame | None, start: date, end: date, tcol: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    mask = (df[tcol] >= pd.Timestamp(start)) & (df[tcol] <= pd.Timestamp(end))
    return df[mask].reset_index(drop=True)


def _build_source_meta(
    *, adapter, raw_path: Path, df: pd.DataFrame, time_column: str,
) -> dict[str, Any]:
    extra = dict(getattr(adapter, "extra_meta", {}) or {})
    return {
        "provider": adapter.name,
        "raw_file": raw_path.name,
        "covers": {
            "start": _as_date(df[time_column].iloc[0]).isoformat(),
            "end": _as_date(df[time_column].iloc[-1]).isoformat(),
        },
        **extra,
    }


def _as_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    ts = pd.Timestamp(x)
    return ts.date()
