"""Processed-layer cache: SQLite-backed.

Layout: single global DB at ``data/processed.db``.
For each registered domain, two tables exist (created on first write):

    <domain>_data : columns mirror domain.schema, plus a leading `ticker`
                    column. Primary key (ticker, <time_column>).
    <domain>_meta : (ticker, schema_version, row_count, range_start,
                     range_end, last_fetch_utc, sources_json) keyed by ticker.

The meta `sources_json` is a TEXT column holding the JSON-serialized
per-source provenance list (provider, raw_file, covers, adjustment_quality,
etc.) — same shape as the v1 parquet-era `_meta.json` `sources` array.

D2 atomicity is provided by SQLite transactions: each write_processed_atomic
opens a single BEGIN..COMMIT, deletes existing rows for the ticker, inserts
the new rows, and upserts the meta row. Partial writes are impossible — a
crash mid-commit rolls back.

D3 determinism: reads always ORDER BY <time_column> and return DataFrames
with canonical schema dtypes (datetime64[ns], float64, int64).

D8 (raw immutability) is unchanged — the raw store still lives in
data/raw/<provider>/... as immutable per-fetch files.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

_log = logging.getLogger(__name__)

from data_pipelines.domain import Calendar, Domain
from data_pipelines.schema import ColumnSpec

META_SCHEMA_VERSION = 1
PROCESSED_DB_NAME = "processed.db"

# One lock per DB path; SQLite handles concurrency internally but our
# multi-statement write transactions benefit from coarse-grained guards in
# the same process. Threaded usage is uncommon for this module today but
# trivial to support.
_DB_LOCKS: dict[Path, threading.Lock] = {}


# ---------------------------------------------------------------------------
# Path / connection helpers
# ---------------------------------------------------------------------------

def processed_db_path(
    data_root: Path,
    processed_subdir: str = "processed.db",
) -> Path:
    """Resolved path to the global processed.db SQLite file."""
    p = Path(data_root) / processed_subdir
    return p


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL + sensible defaults. Caller is responsible
    for close()."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None means Python's sqlite3 issues no implicit BEGIN —
    # the connection runs in SQLite's autocommit mode and *we* drive
    # transactions with explicit BEGIN / COMMIT / ROLLBACK calls. This is
    # how we get D2 atomicity: a crash mid-write rolls back cleanly. Do
    # not remove the explicit BEGIN blocks elsewhere in this module.
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _get_lock(db_path: Path) -> threading.Lock:
    # setdefault is atomic in CPython (single bytecode op under the GIL);
    # check-then-set on a plain dict would allow two threads to create two
    # different Lock objects for the same path, silently defeating the lock.
    return _DB_LOCKS.setdefault(db_path.resolve(), threading.Lock())


# ---------------------------------------------------------------------------
# DDL — derive table schemas from Domain.schema
# ---------------------------------------------------------------------------

def _sql_type_for(spec: ColumnSpec) -> str:
    """Map a ColumnSpec dtype to a SQLite affinity. SQLite is dynamically
    typed — the affinity is advisory but useful for tools that introspect
    the schema.
    """
    dt = spec.dtype
    if dt.startswith("int") or dt.startswith("uint"):
        return "INTEGER"
    if dt.startswith("float"):
        return "REAL"
    if dt.startswith("datetime"):
        return "TIMESTAMP"
    return "TEXT"


def _data_table_name(domain: Domain) -> str:
    return f"{domain.name}_data"


def _meta_table_name(domain: Domain) -> str:
    return f"{domain.name}_meta"


def _ensure_tables(conn: sqlite3.Connection, domain: Domain) -> None:
    """Create the domain's data + meta tables if absent. Idempotent."""
    tcol = domain.time_column
    col_defs = ",\n            ".join(
        f"{c.name} {_sql_type_for(c)}" for c in domain.schema.columns
    )
    data_tbl = _data_table_name(domain)
    meta_tbl = _meta_table_name(domain)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {data_tbl} (
            ticker TEXT NOT NULL,
            {col_defs},
            PRIMARY KEY (ticker, {tcol})
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {meta_tbl} (
            ticker TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            range_start TEXT NOT NULL,
            range_end TEXT NOT NULL,
            last_fetch_utc TEXT NOT NULL,
            sources_json TEXT NOT NULL
        )
    """)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_processed(
    data_root: Path,
    domain: Domain,
    identifier: str,
    processed_subdir: str = "processed.db",
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """Read the cached DataFrame and meta for one identifier.

    Returns (None, None) if the DB file or the per-ticker meta row is missing
    — the meta row is the commit marker, mirroring the parquet-era semantics
    where the .json file was the marker.
    """
    db_path = processed_db_path(data_root, processed_subdir)
    if not db_path.is_file():
        return None, None

    conn = _connect(db_path)
    try:
        _ensure_tables(conn, domain)
        meta_row = conn.execute(
            f"SELECT * FROM {_meta_table_name(domain)} WHERE ticker = ?",
            (identifier,),
        ).fetchone()
        if meta_row is None:
            return None, None

        df = _read_data(conn, domain, identifier)
        return df, _meta_row_to_dict(meta_row, domain.name)
    finally:
        conn.close()


def write_processed_atomic(
    data_root: Path,
    domain: Domain,
    identifier: str,
    df: pd.DataFrame,
    meta: dict,
    processed_subdir: str = "processed.db",
) -> Path:
    """Atomic write of df + meta into the SQLite cache for `identifier`.

    Wraps DELETE-then-INSERT for the data table and REPLACE for the meta
    table in a single transaction. Partial writes are impossible per SQLite
    semantics: if the COMMIT doesn't run, the rollback restores the prior
    state.

    Returns the DB file path (for symmetry with the parquet API).
    """
    db_path = processed_db_path(data_root, processed_subdir)
    lock = _get_lock(db_path)
    conn = _connect(db_path)
    try:
        with lock:
            _ensure_tables(conn, domain)
            conn.execute("BEGIN")
            try:
                _replace_data(conn, domain, identifier, df)
                _upsert_meta(conn, domain, identifier, meta)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
    finally:
        conn.close()
    return db_path


def list_cached_identifiers(
    data_root: Path,
    domain: Domain,
    processed_subdir: str = "processed.db",
) -> list[str]:
    """All identifiers with a meta row for the given domain."""
    db_path = processed_db_path(data_root, processed_subdir)
    if not db_path.is_file():
        return []
    conn = _connect(db_path)
    try:
        _ensure_tables(conn, domain)
        rows = conn.execute(
            f"SELECT ticker FROM {_meta_table_name(domain)} ORDER BY ticker"
        ).fetchall()
        return [r["ticker"] for r in rows]
    finally:
        conn.close()


def purge_identifier(
    data_root: Path,
    domain: Domain,
    identifier: str,
    processed_subdir: str = "processed.db",
) -> bool:
    """Delete all data + meta for one identifier in this domain.
    Returns True if the meta row existed before purge.
    """
    db_path = processed_db_path(data_root, processed_subdir)
    if not db_path.is_file():
        return False
    lock = _get_lock(db_path)
    conn = _connect(db_path)
    try:
        with lock:
            _ensure_tables(conn, domain)
            conn.execute("BEGIN")
            try:
                conn.execute(
                    f"DELETE FROM {_data_table_name(domain)} WHERE ticker = ?",
                    (identifier,),
                )
                cur = conn.execute(
                    f"DELETE FROM {_meta_table_name(domain)} WHERE ticker = ?",
                    (identifier,),
                )
                existed = cur.rowcount > 0
                conn.execute("COMMIT")
                return existed
            except BaseException:
                conn.execute("ROLLBACK")
                raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Gap detection + merge — unchanged from parquet version (pure-DataFrame ops)
# ---------------------------------------------------------------------------

def detect_gaps(
    cached_df: pd.DataFrame | None,
    requested_start: date,
    requested_end: date,
    calendar: Calendar,
    time_column: str = "date",
) -> list[tuple[date, date]]:
    valid_days = calendar.trading_days(requested_start, requested_end)
    if not valid_days:
        return []

    cached_set: set[date] = set()
    if cached_df is not None and len(cached_df) > 0:
        cached_set = {_as_date(d) for d in cached_df[time_column].tolist()}

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
    tcol = domain.time_column

    if existing_df is None or len(existing_df) == 0:
        merged = new_df.sort_values(tcol).reset_index(drop=True)
        merged = _drop_duplicate_timestamps(
            merged, tcol, source=new_source.get("provider", "<unknown>"),
        )
        meta = _build_meta(existing_meta=None, domain=domain, df=merged,
                           new_source=new_source)
        return merged, meta

    existing_dates = set(existing_df[tcol])
    new_dates = set(new_df[tcol])
    overlap = existing_dates & new_dates

    non_overlap_existing = existing_df[~existing_df[tcol].isin(overlap)]
    non_overlap_new = new_df[~new_df[tcol].isin(overlap)]

    if overlap:
        overlap_existing = (existing_df[existing_df[tcol].isin(overlap)]
                            .sort_values(tcol).reset_index(drop=True))
        overlap_new = (new_df[new_df[tcol].isin(overlap)]
                       .sort_values(tcol).reset_index(drop=True))
        existing_sources = (existing_meta or {}).get("sources", [])
        resolved = domain.merge_overlap(
            overlap_existing, overlap_new, existing_sources, new_source,
        )
    else:
        resolved = new_df.iloc[0:0]

    merged = pd.concat(
        [non_overlap_existing, resolved, non_overlap_new], ignore_index=True,
    ).sort_values(tcol).reset_index(drop=True)

    # Defensive de-dup on the time column. The cache's PRIMARY KEY is
    # (ticker, <time_column>), so any duplicates here would crash the SQLite
    # write with an opaque IntegrityError (see issue #36). Duplicates can leak
    # in via two paths:
    #   (1) An adapter returning multiple rows for the same trading day
    #       (observed with jugaad-data on a handful of pre-2015 NSE bhav
    #       rows; the upstream NSE archive sometimes carries duplicate
    #       records for re-issued / corrected entries).
    #   (2) An overlap-resolution policy that yields rows beyond what the
    #       set-diff above accounted for (theoretical; current
    #       merge_overlap implementations are well-behaved, but a future
    #       domain could break this invariant).
    # Keeping `last` matches the existing "new wins" precedence for ordinary
    # overlap (later non-overlap rows in `non_overlap_new` follow earlier
    # rows in `non_overlap_existing` after the concat → sort_values is a
    # stable sort in pandas, so the last duplicate is the most-recently-
    # sourced row).
    merged = _drop_duplicate_timestamps(
        merged, tcol, source=new_source.get("provider", "<unknown>"),
    )

    meta = _build_meta(existing_meta=existing_meta, domain=domain, df=merged,
                       new_source=new_source)
    return merged, meta


def _drop_duplicate_timestamps(
    df: pd.DataFrame, time_column: str, *, source: str,
) -> pd.DataFrame:
    """Drop rows with duplicate values in ``time_column``, keeping the last.

    Returns the de-duped DataFrame, re-indexed. Emits a WARNING-level log
    line per call that dropped rows, naming the source so provider
    misbehavior surfaces in the run log rather than silently propagating
    into the cache.

    No-op when the input has no duplicates — common case, zero overhead
    beyond a single ``duplicated().any()`` scan.
    """
    if len(df) == 0:
        return df
    dup_mask = df[time_column].duplicated(keep="last")
    n_dropped = int(dup_mask.sum())
    if n_dropped == 0:
        return df
    dup_dates = sorted({_as_date(d) for d in df.loc[dup_mask, time_column]})
    sample = ", ".join(d.isoformat() for d in dup_dates[:5])
    suffix = f" (+{len(dup_dates) - 5} more)" if len(dup_dates) > 5 else ""
    _log.warning(
        "merge_cache: dropped %d duplicate row(s) on '%s' from source=%r; "
        "kept last per timestamp. Sample dup-dates: %s%s",
        n_dropped, time_column, source, sample, suffix,
    )
    return df.loc[~dup_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal: data table read / write
# ---------------------------------------------------------------------------

def _read_data(
    conn: sqlite3.Connection, domain: Domain, identifier: str,
) -> pd.DataFrame:
    cols = [c.name for c in domain.schema.columns]
    col_list = ", ".join(cols)
    tcol = domain.time_column
    query = (
        f"SELECT {col_list} FROM {_data_table_name(domain)} "
        f"WHERE ticker = ? ORDER BY {tcol}"
    )
    df = pd.read_sql(query, conn, params=(identifier,))
    # SQLite returns dates as strings (ISO TEXT); coerce columns back to
    # canonical dtypes so the round-trip is invariant.
    for c in domain.schema.columns:
        if df[c.name].dtype.name == c.dtype:
            continue
        if c.dtype.startswith("datetime"):
            df[c.name] = pd.to_datetime(df[c.name]).astype(c.dtype)
        else:
            df[c.name] = df[c.name].astype(c.dtype)
    return df.reset_index(drop=True)


def _replace_data(
    conn: sqlite3.Connection, domain: Domain, identifier: str,
    df: pd.DataFrame,
) -> None:
    conn.execute(
        f"DELETE FROM {_data_table_name(domain)} WHERE ticker = ?",
        (identifier,),
    )
    if len(df) == 0:
        return
    cols = [c.name for c in domain.schema.columns]
    placeholders = ", ".join(["?"] * (len(cols) + 1))  # +1 for ticker
    col_names = ", ".join(["ticker", *cols])
    sql = (
        f"INSERT INTO {_data_table_name(domain)} ({col_names}) "
        f"VALUES ({placeholders})"
    )
    # Extract column-wise, NOT via df.iterrows(): iterrows coerces every cell in
    # a row to one common dtype, so a NaN in a float column of a row that also
    # carries a datetime column comes back as NaT and fails to bind ("type
    # 'NaTType' is not supported"). Iterating each column independently keeps
    # the cell's own dtype and lets us map any missing value (NaN/NaT/None) to
    # SQL NULL. Equity domains have no nullable columns; fred_macro's nullable
    # `value` is the first to exercise this path.
    col_values: dict[str, list] = {}
    for c in cols:
        out: list = []
        for v in df[c]:
            if pd.isna(v):
                out.append(None)
            elif isinstance(v, pd.Timestamp):
                out.append(v.isoformat(sep=" "))  # datetime → ISO TEXT
            elif hasattr(v, "item"):
                out.append(v.item())               # numpy scalar → python scalar
            else:
                out.append(v)
        col_values[c] = out
    rows = [
        (identifier, *(col_values[c][i] for c in cols))
        for i in range(len(df))
    ]
    conn.executemany(sql, rows)


def _upsert_meta(
    conn: sqlite3.Connection, domain: Domain, identifier: str, meta: dict,
) -> None:
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {_meta_table_name(domain)}
            (ticker, schema_version, row_count, range_start, range_end,
             last_fetch_utc, sources_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            identifier,
            int(meta.get("schema_version", META_SCHEMA_VERSION)),
            int(meta.get("row_count", 0)),
            meta.get("range", {}).get("start", ""),
            meta.get("range", {}).get("end", ""),
            meta.get("last_fetch_utc", ""),
            json.dumps(meta.get("sources", []), sort_keys=True),
        ),
    )


def _meta_row_to_dict(row: sqlite3.Row, domain_name: str) -> dict:
    return {
        "schema_version": int(row["schema_version"]),
        "domain": domain_name,
        "row_count": int(row["row_count"]),
        "range": {"start": row["range_start"], "end": row["range_end"]},
        "last_fetch_utc": row["last_fetch_utc"],
        "sources": json.loads(row["sources_json"]),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _as_date(x) -> date:
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
