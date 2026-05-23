"""us_equities canonical processed-layer schema.

Daily OHLCV + split-and-dividend-adjusted close. Schema invariance across
all three adapters (Stooq, Tiingo, yfinance) is non-negotiable (D1).

`adj_close` semantics (D4):
    - "full"        → split-and-dividend adjusted (Tiingo, yfinance)
    - "split_only"  → split-only (Stooq); flagged in _meta.sources for audit

The overlap merge policy below preserves "full" rows when a "split_only"
update would otherwise overwrite them — important when Stooq is re-run as
a seed top-up after Tiingo has already populated newer dates.

**OHLC source-semantic caveat (discovered via v1 parity audit 2026-05-23):**
The open/high/low/close columns are NOT guaranteed to be true-raw historical
prices. Tiingo returns true raw OHLC; yfinance only exposes split-adjusted
OHLC (its `auto_adjust=False` flag controls dividends only — splits are
always back-applied internally). For tickers served by yfinance the OHLC
columns are split-adjusted-disguised-as-raw. The source provenance is
recorded per row-range in _meta.sources[].provider so consumers can tell.

The recommended downstream practice is to use adj_close for any quantitative
calculation (returns, volatility, log-returns) — it agrees across all three
sources to within <0.02% in cross-source audits. OHLC is best treated as
informational / display; if you need true-raw OHLC pin the source to Tiingo.
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
QUALITY_SPLIT_ONLY = "split_only"


def merge_overlap_us_equities(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    existing_sources: list[dict],
    new_source: dict,
) -> pd.DataFrame:
    """Per-column precedence for the US-equities OHLCV schema.

    For overlapping dates:
      - Raw OHLC + volume: new wins (later fetch is more authoritative for
        any provider-side corrections).
      - adj_close: "full" beats "split_only". If new is split_only and the
        existing row's adj_close source was full, keep the existing adj_close.

    The "existing row was full" check is approximate: we look at the most
    recent source in existing_sources and assume rows trace to it. This is
    acceptable for v1 because (a) Stooq seeds happen before Tiingo updates
    in normal flow, and (b) the audit trail in meta.sources lets a human
    reconcile later. A row-level provenance column is a v2 concern.
    """
    new_quality = new_source.get("adjustment_quality", QUALITY_FULL)
    # Inspect last existing source for its quality flag.
    existing_quality = (
        existing_sources[-1].get("adjustment_quality", QUALITY_FULL)
        if existing_sources else QUALITY_FULL
    )

    if new_quality == QUALITY_SPLIT_ONLY and existing_quality == QUALITY_FULL:
        # Take new OHLCV but preserve existing adj_close.
        resolved = new.copy().reset_index(drop=True)
        existing_indexed = existing.set_index("date")["adj_close"]
        resolved["adj_close"] = resolved["date"].map(existing_indexed).astype("float64")
        return resolved

    return new
