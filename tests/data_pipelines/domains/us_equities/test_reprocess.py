"""Stage 9 D8 enforcement: reprocessing a fixed set of raw files yields a
deterministic processed DataFrame across runs.

Strategy:
  1. Pre-stage raw fixtures from tests/data_pipelines/fixtures/us_equities/
     into a tmp data_root under the canonical raw layout.
  2. Run parse + schema.normalize + merge_cache for each provider in a
     defined order (stooq → tiingo → yfinance).
  3. Capture the processed DataFrame.
  4. Repeat from scratch.
  5. Assert the two processed DataFrames are bit-identical (same dtypes,
     same rows, same values, same column order).

We compare DataFrame content rather than raw parquet bytes — parquet writer
metadata (creation timestamps, kvmetadata) can vary across writes even when
the data is identical, and what consumers actually care about is the
deterministic data, not file-level byte equality.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data_pipelines.cache import merge_cache
from data_pipelines.domains.us_equities import USEquitiesDomain
from data_pipelines.domains.us_equities.adapters.stooq import StooqAdapter
from data_pipelines.domains.us_equities.adapters.tiingo import TiingoAdapter
from data_pipelines.domains.us_equities.adapters.yfinance import YFinanceAdapter
from data_pipelines.domains.us_equities.schema import (
    OHLCV_SCHEMA,
    QUALITY_FULL,
    QUALITY_SPLIT_ONLY,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "us_equities"


def _yf_parquet_fixture(tmp_path: Path) -> Path:
    """Build a yfinance parquet fixture in tmp_path (kept off-disk to avoid
    parquet-byte instability across pandas versions)."""
    df = pd.DataFrame(
        {
            "Open": [187.10, 188.30],
            "High": [188.70, 189.95],
            "Low": [186.20, 187.50],
            "Close": [188.45, 189.65],
            "Adj Close": [188.45, 189.65],
            "Volume": [50100000, 51200000],
        },
        index=pd.DatetimeIndex(["2026-01-12", "2026-01-13"], name="Date"),
    ).reset_index()
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    p = tmp_path / "yf_fixture.parquet"
    p.write_bytes(buf.getvalue())
    return p


def _reprocess_pipeline(stooq_raw: Path, tiingo_raw: Path, yf_raw: Path,
                         domain: USEquitiesDomain) -> pd.DataFrame:
    """Deterministic re-derivation: parse → normalize → merge in fixed order."""
    stooq = StooqAdapter()
    tiingo = TiingoAdapter()
    yf = YFinanceAdapter()

    cached_df: pd.DataFrame | None = None
    cached_meta: dict | None = None

    for adapter, raw in [(stooq, stooq_raw), (tiingo, tiingo_raw), (yf, yf_raw)]:
        src_df = adapter.parse(raw)
        norm_df = domain.schema.normalize(
            src_df,
            source_column_map=getattr(adapter, "source_column_map", None),
            provider=adapter.name, identifier="NYSE:AAPL",
        )
        domain.schema.validate(norm_df, provider=adapter.name, identifier="NYSE:AAPL")
        new_source = {
            "provider": adapter.name,
            "raw_file": raw.name,
            "covers": {
                "start": norm_df["date"].iloc[0].date().isoformat(),
                "end": norm_df["date"].iloc[-1].date().isoformat(),
            },
            **getattr(adapter, "extra_meta", {}),
        }
        cached_df, cached_meta = merge_cache(
            cached_df, norm_df, cached_meta, new_source, domain,
        )
    return cached_df


@pytest.fixture
def staged_raws(tmp_path: Path) -> tuple[Path, Path, Path]:
    stooq_src = FIXTURES / "stooq" / "aapl_sample.csv"
    tiingo_src = FIXTURES / "tiingo" / "aapl_sample.json"
    stooq_dst = tmp_path / "stooq.csv"
    tiingo_dst = tmp_path / "tiingo.json"
    stooq_dst.write_bytes(stooq_src.read_bytes())
    tiingo_dst.write_bytes(tiingo_src.read_bytes())
    yf_dst = _yf_parquet_fixture(tmp_path)
    return stooq_dst, tiingo_dst, yf_dst


@pytest.fixture
def domain() -> USEquitiesDomain:
    return USEquitiesDomain()


class TestD8Determinism:
    def test_two_runs_yield_identical_dataframe(self, staged_raws, domain):
        s, t, y = staged_raws
        df1 = _reprocess_pipeline(s, t, y, domain)
        df2 = _reprocess_pipeline(s, t, y, domain)
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)

    def test_dtypes_canonical(self, staged_raws, domain):
        s, t, y = staged_raws
        df = _reprocess_pipeline(s, t, y, domain)
        # All canonical columns present in canonical order with canonical dtypes.
        assert list(df.columns) == [c.name for c in OHLCV_SCHEMA.columns]
        for col in OHLCV_SCHEMA.columns:
            assert str(df[col.name].dtype) == col.dtype, (
                f"{col.name}: expected {col.dtype}, got {df[col.name].dtype}"
            )

    def test_sorted_ascending(self, staged_raws, domain):
        s, t, y = staged_raws
        df = _reprocess_pipeline(s, t, y, domain)
        assert df["date"].is_monotonic_increasing

    def test_overlap_resolved_per_us_equities_policy(self, staged_raws, domain):
        s, t, y = staged_raws
        df = _reprocess_pipeline(s, t, y, domain)
        # Tiingo (full) and Stooq (split_only) overlap on 2026-01-02, 01-05, 01-06.
        # Per merge_overlap_us_equities, when new=split_only would overwrite
        # existing=full, the full adj_close is preserved. Stooq lands first
        # (split_only); then Tiingo lands (full → wins entirely). So all
        # overlap rows end up with Tiingo's adj_close.
        for d in [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]:
            row = df[df["date"] == pd.Timestamp(d)].iloc[0]
            # Tiingo's adj_close in the fixture equals its close.
            assert row["adj_close"] == row["close"]


class TestAdjustmentMetadata:
    def test_quality_constants_match_extra_meta(self):
        assert StooqAdapter().extra_meta["adjustment_quality"] == QUALITY_SPLIT_ONLY
        assert TiingoAdapter().extra_meta["adjustment_quality"] == QUALITY_FULL
        assert YFinanceAdapter().extra_meta["adjustment_quality"] == QUALITY_FULL


class TestWireUp:
    def test_domain_exposes_full_chain(self, domain):
        adapters = domain.adapters
        assert set(adapters.keys()) == {"stooq", "tiingo", "yfinance"}
        assert isinstance(adapters["stooq"], StooqAdapter)
        assert isinstance(adapters["tiingo"], TiingoAdapter)
        assert isinstance(adapters["yfinance"], YFinanceAdapter)

    def test_chain_for_gap_cold(self, domain):
        # Cold cache → seed first, then update tiers as fallback (IP-gate hardening).
        chain = domain.chain_for_gap(gap_size_trading_days=5, has_cache=False)
        assert [a.name for a in chain] == ["stooq", "tiingo", "yfinance"]

    def test_chain_for_gap_small_with_cache(self, domain):
        chain = domain.chain_for_gap(gap_size_trading_days=5, has_cache=True)
        assert [a.name for a in chain] == ["tiingo", "yfinance"]

    def test_chain_for_gap_big_with_cache(self, domain):
        # Big gap with cache also tries seed first, falls through to update tiers.
        chain = domain.chain_for_gap(gap_size_trading_days=100, has_cache=True)
        assert [a.name for a in chain] == ["stooq", "tiingo", "yfinance"]

    def test_identifier_prefixes_registered(self, domain):
        assert "NYSE" in domain.identifier_prefixes
        assert "NASDAQ" in domain.identifier_prefixes
        assert "INDEX" in domain.identifier_prefixes
