"""nse_equities schema + merge precedence tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipelines.domains.nse_equities.schema import (
    OHLCV_SCHEMA,
    QUALITY_FULL,
    QUALITY_NONE,
    merge_overlap_nse_equities,
)
from data_pipelines.errors import SchemaMismatch


def _make_row(d, *, open_=100.0, close=101.0, adj_close=101.0, vol=1000):
    return {
        "date": pd.Timestamp(d).asm8.astype("datetime64[ns]"),
        "open": open_, "high": close + 1, "low": open_ - 1,
        "close": close, "adj_close": adj_close, "volume": np.int64(vol),
    }


def _df(rows):
    df = pd.DataFrame(rows)
    return df.astype({c.name: c.dtype for c in OHLCV_SCHEMA.columns})


def test_schema_columns_match_us_equities_shape():
    # Same column set/order — but the Schema *instance* must be separate.
    from data_pipelines.domains.us_equities.schema import OHLCV_SCHEMA as US
    assert OHLCV_SCHEMA.column_names == US.column_names
    assert OHLCV_SCHEMA is not US


def test_schema_validates_clean_frame():
    df = _df([_make_row("2025-04-01"), _make_row("2025-04-02")])
    OHLCV_SCHEMA.validate(df, provider="t", identifier="NSE:RELIANCE")


def test_schema_rejects_missing_column():
    df = _df([_make_row("2025-04-01")]).drop(columns=["adj_close"])
    with pytest.raises(SchemaMismatch, match="column mismatch"):
        OHLCV_SCHEMA.validate(df, provider="t", identifier="NSE:RELIANCE")


def test_schema_rejects_nan_in_non_nullable():
    df = _df([_make_row("2025-04-01")])
    df.loc[0, "close"] = np.nan
    with pytest.raises(SchemaMismatch, match="non-nullable"):
        OHLCV_SCHEMA.validate(df, provider="t", identifier="NSE:RELIANCE")


def test_normalize_renames_and_casts():
    raw = pd.DataFrame([{
        "DATE": "2025-04-01",
        "OPEN": "100.0", "HIGH": "102.0", "LOW": "99.0",
        "CLOSE": "101.0", "ADJ_CLOSE": "101.0", "VOLUME": "12345",
    }])
    out = OHLCV_SCHEMA.normalize(raw, source_column_map={
        "DATE": "date", "OPEN": "open", "HIGH": "high", "LOW": "low",
        "CLOSE": "close", "ADJ_CLOSE": "adj_close", "VOLUME": "volume",
    })
    OHLCV_SCHEMA.validate(out)
    assert out["date"].dtype == "datetime64[ns]"
    assert out["volume"].dtype == "int64"


# ---- merge precedence ----


def test_merge_new_full_replaces_old_none():
    # Old was none; new is full → new wins entirely (default branch).
    existing = _df([_make_row("2025-04-01", close=100, adj_close=100)])
    new = _df([_make_row("2025-04-01", close=100, adj_close=95)])
    result = merge_overlap_nse_equities(
        existing, new,
        existing_sources=[{"provider": "jugaad", "adjustment_quality": QUALITY_NONE}],
        new_source={"provider": "yfinance", "adjustment_quality": QUALITY_FULL},
    )
    assert result["adj_close"].iloc[0] == 95


def test_merge_new_none_preserves_old_full_adj_close():
    # Old was full (yfinance); new is none (jugaad re-fetch) → keep old adj_close.
    existing = _df([_make_row("2025-04-01", close=100, adj_close=92)])
    new = _df([_make_row("2025-04-01", close=101, adj_close=101)])
    result = merge_overlap_nse_equities(
        existing, new,
        existing_sources=[{"provider": "yfinance", "adjustment_quality": QUALITY_FULL}],
        new_source={"provider": "jugaad", "adjustment_quality": QUALITY_NONE},
    )
    # OHLCV from new, adj_close from existing.
    assert result["close"].iloc[0] == 101
    assert result["adj_close"].iloc[0] == 92


def test_merge_both_none_uses_new():
    existing = _df([_make_row("2025-04-01", close=100, adj_close=100)])
    new = _df([_make_row("2025-04-01", close=101, adj_close=101)])
    result = merge_overlap_nse_equities(
        existing, new,
        existing_sources=[{"provider": "jugaad", "adjustment_quality": QUALITY_NONE}],
        new_source={"provider": "nselib", "adjustment_quality": QUALITY_NONE},
    )
    assert result["adj_close"].iloc[0] == 101


def test_merge_no_existing_sources_defaults_to_full():
    # Edge: empty existing_sources — code defaults existing quality to full.
    existing = _df([_make_row("2025-04-01", adj_close=88)])
    new = _df([_make_row("2025-04-01", adj_close=99)])
    result = merge_overlap_nse_equities(
        existing, new,
        existing_sources=[],
        new_source={"provider": "jugaad", "adjustment_quality": QUALITY_NONE},
    )
    # existing assumed full → preserve its adj_close.
    assert result["adj_close"].iloc[0] == 88
