"""nse_equities canonical processed-layer schema.

Daily OHLCV + adjusted close. Shape is identical to us_equities by coincidence
(both are daily equity OHLCV), but the Schema instance is intentionally
separate — per V1_IMPLEMENTATION_PLAN.md §"What not to do (v1.7)": don't
crystallize a shared OHLCV abstraction until a third domain demands it.

`adj_close` semantics (D4):
    - "full"  → split-and-dividend adjusted (yfinance .NS Adj Close)
    - "none"  → unadjusted, equal to close (jugaad-data, nselib — both pull
               raw NSE bhav data; corporate-action adjustments are not applied
               upstream)

The overlap merge policy preserves "full" rows when a "none" update would
otherwise overwrite them. Important when the chain runs jugaad → nselib first
(raw OHLC, no adj) and yfinance later (adj_close="full"): once yfinance has
populated adj_close, a later jugaad fetch must not regress it back to close.

All NSE rows are in INR; currency is recorded per-source in
_meta.sources[].currency for audit (open question 11 in V1 plan) rather than
adding a per-row column.

**Source semantics:**
- jugaad-data, nselib: raw OHLC + raw volume from NSE bhav. No splits applied.
- yfinance (.NS): same split-adjusted-OHLC-disguised-as-raw caveat as
  us_equities — yfinance always back-applies splits regardless of auto_adjust.
  Documented in detail in domains/us_equities/schema.py.
"""

from __future__ import annotations

import pandas as pd

from data_pipelines.schema import ColumnSpec, Schema

OHLCV_SCHEMA = Schema(columns=(
    ColumnSpec("date", "datetime64[ns]"),
    ColumnSpec("open", "float64"),
    ColumnSpec("high", "float64"),
    ColumnSpec("low", "float64"),
    ColumnSpec("close", "float64"),
    ColumnSpec("adj_close", "float64"),
    ColumnSpec("volume", "int64"),
))

# Adjustment-quality strings — pinned for D4 enforcement and meta auditing.
QUALITY_FULL = "full"
QUALITY_NONE = "none"


def merge_overlap_nse_equities(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    existing_sources: list[dict],
    new_source: dict,
) -> pd.DataFrame:
    """Per-column precedence for the NSE-equities OHLCV schema.

    For overlapping dates:
      - Raw OHLC + volume: new wins (later fetch is more authoritative for
        any provider-side corrections).
      - adj_close: "full" beats "none". If new is "none" and the existing
        row's adj_close source was "full", keep the existing adj_close.

    The existing-quality lookup uses the most-recent source in
    existing_sources; this is the same heuristic as us_equities and is
    documented to be approximate (row-level provenance is v2).
    """
    new_quality = new_source.get("adjustment_quality", QUALITY_FULL)
    existing_quality = (
        existing_sources[-1].get("adjustment_quality", QUALITY_FULL)
        if existing_sources else QUALITY_FULL
    )

    if new_quality == QUALITY_NONE and existing_quality == QUALITY_FULL:
        resolved = new.copy().reset_index(drop=True)
        existing_indexed = existing.set_index("date")["adj_close"]
        resolved["adj_close"] = (
            resolved["date"].map(existing_indexed).astype("float64")
        )
        return resolved

    return new
