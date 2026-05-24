"""D8 enforcement (nse_equities): reprocessing fixed raw bytes yields a
deterministic processed DataFrame across runs.

Mirrors tests/data_pipelines/domains/us_equities/test_reprocess.py — same
strategy: pre-stage the committed adapter fixtures, run
parse + schema.normalize + merge_cache in a fixed order (jugaad → nselib →
yfinance), capture, repeat, assert frame equality.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipelines.cache import merge_cache
from data_pipelines.domains.nse_equities import NSEDomain
from data_pipelines.domains.nse_equities.adapters.jugaad import JugaadAdapter
from data_pipelines.domains.nse_equities.adapters.nselib import NSElibAdapter
from data_pipelines.domains.nse_equities.adapters.yfinance import YFinanceNSEAdapter
from data_pipelines.domains.nse_equities.schema import (
    OHLCV_SCHEMA,
    QUALITY_FULL,
    QUALITY_NONE,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "nse_equities"


def _reprocess_pipeline(
    jugaad_raw: Path, nselib_raw: Path, yf_raw: Path, domain: NSEDomain,
) -> pd.DataFrame:
    jugaad = JugaadAdapter()
    nselib_a = NSElibAdapter()
    yf = YFinanceNSEAdapter()

    cached_df: pd.DataFrame | None = None
    cached_meta: dict | None = None

    for adapter, raw in [(jugaad, jugaad_raw), (nselib_a, nselib_raw), (yf, yf_raw)]:
        src_df = adapter.parse(raw)
        norm_df = domain.schema.normalize(
            src_df,
            source_column_map=getattr(adapter, "source_column_map", None),
            provider=adapter.name, identifier="NSE:RELIANCE",
        )
        domain.schema.validate(norm_df, provider=adapter.name, identifier="NSE:RELIANCE")
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
    jugaad_src = FIXTURES / "jugaad" / "RELIANCE_2025-04.json"
    nselib_src = FIXTURES / "nselib" / "RELIANCE_2025-04.csv"
    yf_src = FIXTURES / "yfinance" / "RELIANCE_NS_2025-04.parquet"
    jugaad_dst = tmp_path / "jugaad.json"
    nselib_dst = tmp_path / "nselib.csv"
    yf_dst = tmp_path / "yfinance.parquet"
    jugaad_dst.write_bytes(jugaad_src.read_bytes())
    nselib_dst.write_bytes(nselib_src.read_bytes())
    yf_dst.write_bytes(yf_src.read_bytes())
    return jugaad_dst, nselib_dst, yf_dst


@pytest.fixture
def domain() -> NSEDomain:
    return NSEDomain()


class TestD8Determinism:
    def test_two_runs_yield_identical_dataframe(self, staged_raws, domain):
        j, n, y = staged_raws
        df1 = _reprocess_pipeline(j, n, y, domain)
        df2 = _reprocess_pipeline(j, n, y, domain)
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)

    def test_dtypes_canonical(self, staged_raws, domain):
        j, n, y = staged_raws
        df = _reprocess_pipeline(j, n, y, domain)
        assert list(df.columns) == [c.name for c in OHLCV_SCHEMA.columns]
        for col in OHLCV_SCHEMA.columns:
            assert str(df[col.name].dtype) == col.dtype

    def test_sorted_ascending(self, staged_raws, domain):
        j, n, y = staged_raws
        df = _reprocess_pipeline(j, n, y, domain)
        assert df["date"].is_monotonic_increasing

    def test_adj_close_full_quality_preserved(self, staged_raws, domain):
        """yfinance lands last, providing QUALITY_FULL adj_close — the result
        should reflect yfinance's adjusted values where they differ from
        close (RELIANCE has historical dividends; April 2025 sees the small
        dividend wedge between close and adj_close)."""
        j, n, y = staged_raws
        df = _reprocess_pipeline(j, n, y, domain)
        # At least one row where adj_close < close (yfinance dividend adj).
        assert (df["adj_close"] < df["close"]).any()


class TestAdjustmentMetadata:
    def test_quality_constants_match_extra_meta(self):
        assert JugaadAdapter().extra_meta["adjustment_quality"] == QUALITY_NONE
        assert NSElibAdapter().extra_meta["adjustment_quality"] == QUALITY_NONE
        assert YFinanceNSEAdapter().extra_meta["adjustment_quality"] == QUALITY_FULL

    def test_currency_inr_on_all_three(self):
        assert JugaadAdapter().extra_meta["currency"] == "INR"
        assert NSElibAdapter().extra_meta["currency"] == "INR"
        assert YFinanceNSEAdapter().extra_meta["currency"] == "INR"
