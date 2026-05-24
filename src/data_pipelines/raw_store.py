"""Immutable raw-payload store (D2 atomic, D8 immutable).

Adapters call write_raw_atomic(...) the moment they receive provider bytes,
BEFORE any parsing. If parse/normalize blows up later, the raw bytes are
preserved on disk so the failure can be reproduced and the fix verified
without re-hitting the API.

Filename pattern (immutable contract — open question 4 in V1_IMPLEMENTATION_PLAN.md):
    data/raw/<provider>/<domain>/<exchange>/<ticker>/<UTC_ts>_<start>_<end>.<ext>

Examples:
    data/raw/stooq/us_equities/NYSE/AAPL/20260523T143022Z_1986-01-02_2026-05-22.csv
    data/raw/tiingo/us_equities/NYSE/AAPL/20260523T143022Z_2026-05-20_2026-05-22.json
    data/raw/fred/fred_macro/-/DGS10/20260523T143022Z_2010-01-01_2026-05-22.json

UTC timestamp resolution is whole seconds; a collision means two writes
fired in the same second for the same (provider, domain, exchange, ticker,
start, end). That's a programmer error and write_raw_atomic raises.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

# Filename format constants — pinned for D8 / open question 4.
_TS_FMT = "%Y%m%dT%H%M%SZ"
_DATE_FMT = "%Y-%m-%d"
_FILENAME_RE = re.compile(
    r"^(?P<ts>\d{8}T\d{6}Z)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.(?P<ext>[A-Za-z0-9]+)$"
)


@dataclass(frozen=True)
class RawFilename:
    """Parsed raw-file filename components."""

    timestamp: datetime  # UTC, tz-aware
    range_start: date
    range_end: date
    ext: str


def encode_filename(
    timestamp: datetime,
    range_start: date,
    range_end: date,
    ext: str,
) -> str:
    """Build the canonical raw-file filename. Inverse of parse_filename."""
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be tz-aware (UTC)")
    ts_utc = timestamp.astimezone(timezone.utc)
    ext_clean = ext.lstrip(".")
    return (
        f"{ts_utc.strftime(_TS_FMT)}_"
        f"{range_start.strftime(_DATE_FMT)}_"
        f"{range_end.strftime(_DATE_FMT)}."
        f"{ext_clean}"
    )


def parse_filename(name: str) -> RawFilename:
    """Round-trip-deterministic parser. Raises ValueError on bad input."""
    m = _FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"not a valid raw filename: {name!r}")
    ts = datetime.strptime(m["ts"], _TS_FMT).replace(tzinfo=timezone.utc)
    return RawFilename(
        timestamp=ts,
        range_start=datetime.strptime(m["start"], _DATE_FMT).date(),
        range_end=datetime.strptime(m["end"], _DATE_FMT).date(),
        ext=m["ext"],
    )


def raw_dir(
    data_root: Path,
    provider: str,
    domain: str,
    exchange: str,
    ticker: str,
    raw_subdir: str = "raw",
) -> Path:
    """Compute the directory holding all raw files for one identifier+provider."""
    return Path(data_root) / raw_subdir / provider / domain / exchange / ticker


def write_raw_atomic(
    data_root: Path,
    provider: str,
    domain: str,
    exchange: str,
    ticker: str,
    payload: bytes,
    range_start: date,
    range_end: date,
    ext: str,
    *,
    timestamp: datetime | None = None,
    raw_subdir: str = "raw",
) -> Path:
    """Write `payload` to data/raw/<provider>/<domain>/<exchange>/<ticker>/
    <UTC_ts>_<start>_<end>.<ext> atomically.

    Atomicity (D2): write to a temp file in the same directory, fsync,
    os.replace to final name. Never leaves a partial file at the final path.

    Immutability (D8): if the final path already exists, raise FileExistsError.
    Adapters should provide a unique timestamp; collisions are programmer
    error.

    Returns the final Path.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    target_dir = raw_dir(data_root, provider, domain, exchange, ticker, raw_subdir)
    target_dir.mkdir(parents=True, exist_ok=True)

    fname = encode_filename(timestamp, range_start, range_end, ext)
    final_path = target_dir / fname
    if final_path.exists():
        raise FileExistsError(
            f"raw file already exists (D8 immutability): {final_path}"
        )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{fname}.", suffix=".tmp", dir=target_dir
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, final_path)
    except BaseException:
        # Best-effort cleanup; swallow if temp is already gone.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return final_path


def list_raw(
    data_root: Path,
    provider: str,
    domain: str,
    exchange: str,
    ticker: str,
    raw_subdir: str = "raw",
) -> list[Path]:
    """List all raw files for one (provider, domain, exchange, ticker), sorted
    by filename (which sorts by UTC timestamp lexicographically because
    %Y%m%dT%H%M%SZ is collation-friendly).

    Returns an empty list if the directory does not exist. Skips files that
    don't match the canonical pattern (stray .tmp leftovers, junk).
    """
    d = raw_dir(data_root, provider, domain, exchange, ticker, raw_subdir)
    if not d.is_dir():
        return []
    out = []
    for entry in d.iterdir():
        if not entry.is_file():
            continue
        try:
            parse_filename(entry.name)
        except ValueError:
            continue
        out.append(entry)
    return sorted(out, key=lambda p: p.name)
