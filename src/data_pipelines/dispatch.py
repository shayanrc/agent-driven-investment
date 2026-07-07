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
    _as_date,
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
    back_extend: bool = False,
) -> pd.DataFrame:
    """Public surface. Returns the canonical DataFrame for the requested
    range, populating the cache as needed.

    ``back_extend`` (default False) bypasses the cache-first cap so providers
    are asked for pre-cache history. See :func:`fetch_with_meta` for the full
    rationale.
    """
    df, _ = fetch_with_meta(
        identifier, start, end, frequency, data_root, back_extend=back_extend,
    )
    return df


def fetch_with_meta(
    identifier: str,
    start: str | date,
    end: str | date,
    frequency: str = "daily",
    data_root: str | Path = "data",
    back_extend: bool = False,
) -> tuple[pd.DataFrame, FetchMeta]:
    """fetch() + JSON-serializable FetchMeta. Use this from the CLI and the
    eventual agent-tool wrapper.

    ``back_extend`` (default False): when True, skip the cache-first cap on
    ``effective_start`` so the dispatcher asks providers for the full
    requested range, *including dates earlier than the cache's current first
    row*. Use this for user-initiated deep-history extension after the cache
    was originally seeded with a shallower start date. The default is False
    because the cap exists for a reason — re-asking providers for pre-cache
    dates on every routine refresh is wasteful (the asset's true earliest
    date is typically already on disk) and a new-vintage answer is suspicious
    by default. Pass ``back_extend=True`` only when you are deliberately
    going further back than the existing seed. When the requested ``start``
    is not earlier than the cache's first row, the flag is a no-op (same
    behaviour as the default path).
    """
    if frequency != "daily":
        raise NotImplementedError(f"frequency={frequency!r}; v1 supports 'daily' only")

    start_d = _as_date(start)
    end_d = _as_date(end)
    if start_d > end_d:
        raise ValueError(f"start ({start_d}) is after end ({end_d})")

    data_root_p = Path(data_root)

    domain = DomainRegistry.resolve(identifier)
    # Resolve the calendar once per call. Most domains return their single
    # trading calendar; per-series-cadence domains (fred_macro: daily vs
    # monthly vs quarterly) return the calendar matching this identifier so
    # gap detection enumerates the dates the series actually publishes.
    cal = domain.calendar_for(identifier)
    cached_df, cached_meta = read_processed(data_root_p, domain, identifier)
    cache_was_cold = cached_df is None

    # Cap gap detection at the cache's earliest known date. The cache was
    # built by asking providers for the broadest possible range, so its first
    # row is the asset's earliest available date. Requesting earlier than
    # that hits providers needlessly (network round trip → empty payload →
    # soft fail) on every re-seed of a post-IPO ticker. If a new vintage
    # source ever exposes pre-cache data, purge that ticker to re-discover.
    #
    # back_extend=True is the explicit opt-out: skip the cap so the
    # dispatcher asks providers for the full requested range, including
    # dates earlier than cache_first. Use this when deliberately extending
    # an existing ticker's history further back than initially seeded.
    # When start_d is already >= cache_first the flag has no effect.
    effective_start = start_d
    if cached_df is not None and len(cached_df) > 0:
        cache_first = _as_date(cached_df[domain.time_column].iloc[0])
        if effective_start < cache_first and not back_extend:
            effective_start = cache_first
    gaps = detect_gaps(
        cached_df, effective_start, end_d, cal, domain.time_column
    )

    gaps_filled: list[dict[str, Any]] = []
    providers_failed: list[dict[str, str]] = []

    for gap_start, gap_end in gaps:
        gap_days = len(cal.trading_days(gap_start, gap_end))
        has_cache = cached_df is not None and len(cached_df) > 0
        chain = domain.chain_for_gap(identifier, gap_days, has_cache)

        if not chain:
            raise AllProvidersFailed(
                identifier,
                [ProviderError("<none>", identifier, "domain returned empty chain")],
            )

        # Partial-fill continuation: a provider may legitimately return rows
        # for only PART of the requested gap (NSE NIFTY: from nselib is
        # capped at ~3 fiscal years; future cases may include FRED's
        # per-series start dates). After each provider's successful write,
        # re-detect the remaining sub-gaps inside [gap_start, gap_end] and
        # continue the chain to fill them. This is bounded: each provider
        # is tried at most once per top-level gap, and re-detection uses
        # the just-updated cache so it converges on each iteration.
        gap_failures: list[ProviderError] = []
        any_success = False

        for adapter in chain:
            sub_gaps = detect_gaps(
                cached_df, gap_start, gap_end,
                cal, domain.time_column,
            )
            if not sub_gaps:
                # Cache now covers the entire top-level gap; chain done.
                break

            for sub_start, sub_end in sub_gaps:
                try:
                    raw_path = adapter.fetch(
                        identifier, start=sub_start, end=sub_end,
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
                    success_source = _build_source_meta(
                        adapter=adapter,
                        raw_path=raw_path,
                        df=norm_df,
                        time_column=domain.time_column,
                        identifier=identifier,
                    )
                    cached_df, cached_meta = merge_cache(
                        cached_df, norm_df, cached_meta, success_source, domain
                    )
                    write_processed_atomic(
                        data_root_p, domain, identifier, cached_df, cached_meta
                    )
                    gaps_filled.append({
                        "gap": {"start": sub_start.isoformat(),
                                "end": sub_end.isoformat()},
                        "provider": success_source["provider"],
                        "rows": int(len(norm_df)),
                    })
                    any_success = True
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

        # After the chain has been exhausted, see if anything in the top-level
        # gap is still missing.
        leftover = detect_gaps(
            cached_df, gap_start, gap_end,
            cal, domain.time_column,
        )
        if leftover:
            any_empty = any(isinstance(f, EmptyPayload) for f in gap_failures)
            has_existing_cache = (
                cached_df is not None
                and len(cached_df) > 0
                # If we filled some sub-gaps this round, the cache now has
                # data even if it was cold at fetch start.
            )
            if any_success or (any_empty and has_existing_cache):
                # Soft-fail: chain captured what was available; record the
                # uncovered remainder for the caller to see in providers_failed.
                # "unfillable" wording covers both the "all empty, nothing
                # filled" case (pre-IPO ranges, internal calendar gaps) and
                # the "partial fill, residual uncovered" case (NSE NIFTY:
                # nselib coverage stops at ~3 yrs and yfinance lacks the rest).
                if any_success:
                    reason = (f"gap {gap_start}..{gap_end} only partially "
                              f"filled; {len(leftover)} sub-range(s) remain "
                              f"unfillable after exhausting chain; existing "
                              f"cache preserved")
                else:
                    reason = (f"gap {gap_start}..{gap_end} unfillable "
                              f"(provider(s) returned empty; asset may not "
                              f"have existed); existing cache preserved")
                providers_failed.append({
                    "provider": "<chain>",
                    "reason": reason,
                })
                continue
            # No provider returned data AND no prior cache to fall back on —
            # genuinely unknown. Hard fail.
            raise AllProvidersFailed(identifier, gap_failures)

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
    identifier: str = "<unknown>",
) -> dict[str, Any]:
    # A provider that returned 200 OK with zero post-normalize rows reaches
    # here as an empty DataFrame (schema.validate accepts empty frames whose
    # columns/dtypes match). Without this guard, ``df[time_column].iloc[0]``
    # raises a bare IndexError that escapes the dispatch chain-fallthrough
    # (which only catches typed ProviderError subclasses). Convert to
    # EmptyPayload so the existing soft-fail / hard-fail logic handles it.
    if len(df) == 0:
        raise EmptyPayload(adapter.name, identifier)
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


