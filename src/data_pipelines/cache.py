"""Processed-layer cache: read, atomic write, gap detection, merge.

Layout: data/processed/<domain>/<exchange>/<ticker>/{daily.parquet, _meta.json}.
Both files are written atomically (D2). The parquet lands first; _meta.json
last — so a half-written cache is detectable (parquet without matching meta).

Gap detection is calendar-aware: requested range is expanded to the domain's
valid time points, then compared against what's already in the cache. The
output is a list of (start, end) inclusive ranges, where consecutive missing
trading days are collapsed into one gap.

Merge is two-stage: (1) framework concatenates non-overlapping rows in time
order; (2) for overlapping rows, the domain's merge_overlap() decides
precedence. Default = new wins. us_equities will override to preserve
full-quality adj_close.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.domain import Calendar, Domain

META_SCHEMA_VERSION = 1
PROCESSED_PARQUET_NAME = "daily.parquet"
PROCESSED_META_NAME = "_meta.json"


def processed_dir(
    data_root: Path,
    domain_name: str,
    exchange: str,
    ticker: str,
    processed_subdir: str = "processed",
) -> Path:
    return Path(data_root) / processed_subdir / domain_name / exchange / ticker


def read_processed(
    data_root: Path,
    domain: Domain,
    identifier: str,
    processed_subdir: str = "processed",
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """Read the cached processed DataFrame and meta for one identifier.

    Returns (None, None) if either file is missing. A parquet without a
    matching _meta.json is treated as missing (the meta is the commit
    marker — D2).
    """
    exchange, ticker = domain.parse_identifier(identifier)
    d = processed_dir(data_root, domain.name, exchange, ticker, processed_subdir)
    parquet = d / PROCESSED_PARQUET_NAME
    meta = d / PROCESSED_META_NAME
    if not parquet.is_file() or not meta.is_file():
        return None, None
    df = pd.read_parquet(parquet)
    meta_obj = json.loads(meta.read_text())
    return df, meta_obj


def write_processed_atomic(
    data_root: Path,
    domain: Domain,
    identifier: str,
    df: pd.DataFrame,
    meta: dict,
    processed_subdir: str = "processed",
) -> Path:
    """Write the canonical parquet and meta JSON atomically.

    Order is load-bearing: parquet temp+fsync+rename FIRST, then meta last.
    A crash between the two leaves a parquet without meta, which read_processed
    treats as no cache — re-fetchable, never silently consumed as truth.

    The DataFrame must already conform to domain.schema (validate at call
    site, not here — keeps write a simple I/O primitive).
    """
    exchange, ticker = domain.parse_identifier(identifier)
    d = processed_dir(data_root, domain.name, exchange, ticker, processed_subdir)
    d.mkdir(parents=True, exist_ok=True)

    parquet_final = d / PROCESSED_PARQUET_NAME
    meta_final = d / PROCESSED_META_NAME

    _atomic_write_bytes(
        parquet_final,
        _df_to_parquet_bytes(df),
    )
    _atomic_write_bytes(
        meta_final,
        json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"),
    )
    return parquet_final


def detect_gaps(
    cached_df: pd.DataFrame | None,
    requested_start: date,
    requested_end: date,
    calendar: Calendar,
    time_column: str = "date",
) -> list[tuple[date, date]]:
    """Return inclusive (start, end) ranges of calendar time points missing
    from the cache for the requested window.

    Cold cache → one gap covering the whole requested range.
    Cache covers requested exactly → empty list.
    Internal hole → one gap per contiguous run of missing trading days.
    """
    valid_days = calendar.trading_days(requested_start, requested_end)
    if not valid_days:
        return []

    cached_set: set[date] = set()
    if cached_df is not None and len(cached_df) > 0:
        cached_set = {
            _as_date(d) for d in cached_df[time_column].tolist()
        }

    runs: list[list[date]] = []
    current: list[date] | None = None
    for d in valid_days:
        if d in cached_set:
            if current is not None:
                runs.append(current)
                current = None
        else:
            if current is None:
                current = [d, d]
            else:
                current[1] = d
    if current is not None:
        runs.append(current)

    return [(r[0], r[1]) for r in runs]


def merge_cache(
    existing_df: pd.DataFrame | None,
    new_df: pd.DataFrame,
    existing_meta: dict | None,
    new_source: dict,
    domain: Domain,
) -> tuple[pd.DataFrame, dict]:
    """Merge new_df into existing_df, dedupe by time_column with domain
    precedence on overlap, and update meta.

    new_source is the dict appended to meta["sources"] — caller (dispatch)
    supplies provider name, raw_file basename, covers={start,end},
    adjustment_quality, etc.

    Returns (merged_df, updated_meta). The DataFrame is sorted ascending by
    time_column and re-indexed 0..N-1.
    """
    tcol = domain.time_column

    if existing_df is None or len(existing_df) == 0:
        merged = new_df.sort_values(tcol).reset_index(drop=True)
        meta = _build_meta(
            existing_meta=None,
            domain=domain,
            df=merged,
            new_source=new_source,
        )
        return merged, meta

    existing_dates = set(existing_df[tcol])
    new_dates = set(new_df[tcol])
    overlap = existing_dates & new_dates

    non_overlap_existing = existing_df[~existing_df[tcol].isin(overlap)]
    non_overlap_new = new_df[~new_df[tcol].isin(overlap)]

    if overlap:
        overlap_existing = existing_df[existing_df[tcol].isin(overlap)].sort_values(tcol).reset_index(drop=True)
        overlap_new = new_df[new_df[tcol].isin(overlap)].sort_values(tcol).reset_index(drop=True)
        existing_sources = (existing_meta or {}).get("sources", [])
        resolved = domain.merge_overlap(
            overlap_existing, overlap_new, existing_sources, new_source
        )
    else:
        resolved = new_df.iloc[0:0]

    merged = pd.concat(
        [non_overlap_existing, resolved, non_overlap_new],
        ignore_index=True,
    ).sort_values(tcol).reset_index(drop=True)

    meta = _build_meta(
        existing_meta=existing_meta,
        domain=domain,
        df=merged,
        new_source=new_source,
    )
    return merged, meta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_write_bytes(final_path: Path, payload: bytes) -> None:
    """Temp file in same directory → fsync → os.replace. D2 atomicity.

    Cleans up the temp file if the replace step raises.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, final_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """In-memory parquet serialization; bytes go through the atomic writer."""
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    return buf.getvalue()


def _as_date(x) -> date:
    """Coerce a timestamp / datetime / date scalar to a plain `date`."""
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    ts = pd.Timestamp(x)
    return ts.date()


def _build_meta(
    *,
    existing_meta: dict | None,
    domain: Domain,
    df: pd.DataFrame,
    new_source: dict,
) -> dict:
    tcol = domain.time_column
    sources = list((existing_meta or {}).get("sources", []))
    sources.append(new_source)
    return {
        "schema_version": META_SCHEMA_VERSION,
        "domain": domain.name,
        "row_count": int(len(df)),
        "range": {
            "start": _as_date(df[tcol].iloc[0]).isoformat(),
            "end": _as_date(df[tcol].iloc[-1]).isoformat(),
        },
        "last_fetch_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": sources,
    }
