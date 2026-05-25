"""Data composition for the forecasters framework.

Resolves the call's data source (``--identifier`` via ``data_pipelines.fetch``
or ``--data-path`` via direct CSV/parquet read) and returns a canonical-schema
DataFrame. The framework hands this DataFrame to the backend; no module under
``src/forecasters/`` performs any time-series modeling itself.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd


def prepare_data(
    *,
    identifier: str | None = None,
    data_path: str | Path | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    date_col: str = "date",
    close_col: str = "adj_close",
    data_root: str | Path = "data",
) -> pd.DataFrame:
    """Return a DataFrame to feed the backend.

    Exactly one of ``identifier`` or ``data_path`` must be supplied.

    - ``identifier``: calls ``data_pipelines.fetch(identifier, start, end)``.
      Requires ``start`` and ``end``.
    - ``data_path``: reads a CSV or parquet file directly. ``start`` and
      ``end`` (if supplied) slice the resulting DataFrame on ``date_col``.

    The returned DataFrame is the raw DataFrame from the source — the backend
    is responsible for column-name handling (see
    ``analog_mc.forecaster._forecast`` for a fallback for FRED-style columns).
    """
    if (identifier is None) == (data_path is None):
        raise ValueError(
            "prepare_data: pass exactly one of identifier or data_path"
        )

    if identifier is not None:
        if start is None or end is None:
            raise ValueError(
                "prepare_data(identifier=...) requires start and end dates"
            )
        # Lazy import: data_pipelines pulls in a domain registry whose import
        # cost is non-trivial. Defer until we actually need it.
        # Domains are registered as side effects of importing their packages —
        # `data_pipelines.__init__` does not import them eagerly (per the
        # data_pipelines design), so we must import each one explicitly here.
        from data_pipelines import fetch
        import data_pipelines.domains.us_equities  # noqa: F401  (registers NYSE/NASDAQ/INDEX)
        import data_pipelines.domains.nse_equities  # noqa: F401  (registers NSE/BSE/NIFTY)
        return fetch(identifier, start=start, end=end, data_root=data_root)

    # ---- data_path branch -------------------------------------------------
    path = Path(data_path)  # type: ignore[arg-type]
    if not path.is_file():
        raise FileNotFoundError(f"data_path {path} does not exist")
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # Slice by date if requested (and the date column is present).
    if date_col in df.columns and (start is not None or end is not None):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        if start is not None:
            df = df[df[date_col] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df[date_col] <= pd.Timestamp(end)]
        df = df.reset_index(drop=True)
    elif "observation_date" in df.columns and (start is not None or end is not None):
        # FRED-style fallback (the NASDAQ100 CSV).
        df = df.copy()
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        if start is not None:
            df = df[df["observation_date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["observation_date"] <= pd.Timestamp(end)]
        df = df.reset_index(drop=True)
    return df


def data_hash(df: pd.DataFrame, date_col: str | None = None, close_col: str | None = None) -> str:
    """Stable content hash of (date, close) pairs in the DataFrame.

    Auto-detects the date / close columns if the canonical-schema names are
    present; otherwise falls back to ``observation_date`` / ``NASDAQ100``
    (the project's NASDAQ100 FRED-style CSV).
    """
    dc = date_col or ("date" if "date" in df.columns else "observation_date")
    cc = close_col or ("adj_close" if "adj_close" in df.columns else "NASDAQ100")
    if dc not in df.columns or cc not in df.columns:
        raise ValueError(
            f"data_hash: cannot find date/close columns; tried ({dc!r}, {cc!r}); "
            f"have {list(df.columns)}"
        )
    h = hashlib.sha256()
    s = df[[dc, cc]].dropna().sort_values(dc)
    for ts, val in zip(s[dc].tolist(), s[cc].tolist(), strict=True):
        h.update(pd.Timestamp(ts).strftime("%Y-%m-%d").encode())
        h.update(b":")
        h.update(f"{float(val):.10g}".encode())
        h.update(b"\n")
    return "sha256:" + h.hexdigest()
