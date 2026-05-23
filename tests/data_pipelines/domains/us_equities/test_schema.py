"""Stage 5 tests: us_equities schema invariants + adjustment-quality merge."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data_pipelines.domains.us_equities.schema import (
    OHLCV_SCHEMA,
    QUALITY_FULL,
    QUALITY_SPLIT_ONLY,
    merge_overlap_us_equities,
)
from data_pipelines.errors import SchemaMismatch


def _ohlcv(dates, opens, highs, lows, closes, adj_closes, vols):
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)).astype("datetime64[ns]"),
        "open": [float(x) for x in opens],
        "high": [float(x) for x in highs],
        "low": [float(x) for x in lows],
        "close": [float(x) for x in closes],
        "adj_close": [float(x) for x in adj_closes],
        "volume": [int(x) for x in vols],
    })


class TestSchemaShape:
    def test_columns_in_canonical_order(self):
        assert OHLCV_SCHEMA.column_names == [
            "date", "open", "high", "low", "close", "adj_close", "volume"
        ]

    def test_valid_df_passes(self):
        df = _ohlcv([date(2026, 1, 5)], [100], [101], [99], [100.5], [100.5], [1_000_000])
        OHLCV_SCHEMA.validate(df)

    def test_rejects_missing_adj_close(self):
        df = _ohlcv([date(2026, 1, 5)], [100], [101], [99], [100.5], [100.5], [1])
        df = df.drop(columns=["adj_close"])
        with pytest.raises(SchemaMismatch):
            OHLCV_SCHEMA.validate(df)

    def test_rejects_wrong_volume_dtype(self):
        df = _ohlcv([date(2026, 1, 5)], [100], [101], [99], [100.5], [100.5], [1])
        df["volume"] = df["volume"].astype("float64")
        with pytest.raises(SchemaMismatch):
            OHLCV_SCHEMA.validate(df)


class TestMergeOverlapPolicy:
    def test_default_new_wins_when_qualities_match(self):
        existing = _ohlcv([date(2026, 1, 5)], [100], [101], [99], [100.5], [100.5], [1_000])
        new = _ohlcv([date(2026, 1, 5)], [110], [112], [109], [111.5], [111.5], [2_000])
        new_source = {"adjustment_quality": QUALITY_FULL}
        out = merge_overlap_us_equities(existing, new, [{"adjustment_quality": QUALITY_FULL}], new_source)
        # New values everywhere.
        assert out["adj_close"].iloc[0] == 111.5
        assert out["close"].iloc[0] == 111.5

    def test_split_only_preserves_existing_full_adj_close(self):
        # Tiingo (full) wrote first; Stooq (split_only) re-fetched same date.
        existing = _ohlcv([date(2026, 1, 5)], [100], [101], [99], [100.5], [99.0], [1_000])
        new = _ohlcv([date(2026, 1, 5)], [100], [101], [99], [100.5], [100.5], [1_000])
        existing_sources = [{"adjustment_quality": QUALITY_FULL}]
        new_source = {"adjustment_quality": QUALITY_SPLIT_ONLY}
        out = merge_overlap_us_equities(existing, new, existing_sources, new_source)
        # adj_close held to existing 99.0 (the dividend-adjusted value).
        assert out["adj_close"].iloc[0] == 99.0
        # Other columns from new (which may differ in raw OHLC after corrections).
        assert out["close"].iloc[0] == 100.5

    def test_full_overrides_split_only(self):
        existing = _ohlcv([date(2026, 1, 5)], [100], [101], [99], [100.5], [100.5], [1])
        new = _ohlcv([date(2026, 1, 5)], [100], [101], [99], [100.5], [99.0], [1])
        existing_sources = [{"adjustment_quality": QUALITY_SPLIT_ONLY}]
        new_source = {"adjustment_quality": QUALITY_FULL}
        out = merge_overlap_us_equities(existing, new, existing_sources, new_source)
        assert out["adj_close"].iloc[0] == 99.0
