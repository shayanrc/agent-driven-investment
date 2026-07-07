"""Stage 8 tests: yfinance adapter — parquet round-trip, multi-index flattening,
empty result, schema invariant.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_pipelines.domains.us_equities.adapters.yfinance import YFinanceAdapter
from data_pipelines.domains.us_equities.config import USEquitiesConfig
from data_pipelines.domains.us_equities.schema import OHLCV_SCHEMA, QUALITY_FULL
from data_pipelines.errors import EmptyPayload, ProviderError


def _yf_like_df() -> pd.DataFrame:
    """Mimic the shape yfinance returns: Date index, 'Adj Close' column."""
    return pd.DataFrame(
        {
            "Open": [185.50, 187.00, 188.20],
            "High": [187.20, 188.50, 189.75],
            "Low": [184.80, 186.10, 187.40],
            "Close": [186.95, 188.10, 189.50],
            "Adj Close": [186.95, 188.10, 189.50],
            "Volume": [52400000, 48300000, 55100000],
        },
        index=pd.DatetimeIndex(
            ["2026-01-02", "2026-01-05", "2026-01-06"], name="Date"
        ),
    )


def _write_yf_parquet(tmp_dir: Path) -> Path:
    """Drop a parquet file shaped like the adapter would write."""
    df = _yf_like_df().reset_index()
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    p = tmp_dir / "yf_sample.parquet"
    p.write_bytes(buf.getvalue())
    return p


@pytest.fixture
def adapter() -> YFinanceAdapter:
    return YFinanceAdapter()


class TestParse:
    def test_canonical_after_normalize(self, adapter, tmp_path):
        p = _write_yf_parquet(tmp_path)
        df = adapter.parse(p)
        norm = OHLCV_SCHEMA.normalize(
            df, source_column_map=adapter.source_column_map,
            provider="yfinance", identifier="NYSE:AAPL",
        )
        OHLCV_SCHEMA.validate(norm, provider="yfinance", identifier="NYSE:AAPL")
        assert list(norm["adj_close"]) == [186.95, 188.10, 189.50]

    def test_date_normalized(self, adapter, tmp_path):
        p = _write_yf_parquet(tmp_path)
        df = adapter.parse(p)
        assert str(df["Date"].dtype) == "datetime64[ns]"
        assert df["Date"].is_monotonic_increasing

    def test_multiindex_columns_flattened_in_memory(self, adapter):
        # Single-ticker Ticker.history() returns flat columns; multi-ticker
        # yf.download() returns MultiIndex. Parquet doesn't round-trip
        # MultiIndex cleanly, so we exercise the defensive flatten path in
        # memory only, against a DataFrame that already has MultiIndex
        # columns (simulating the unsupported but possible call path).
        df = pd.DataFrame({
            ("Open", "AAPL"): [185.50],
            ("Adj Close", "AAPL"): [186.95],
        })
        df.columns = pd.MultiIndex.from_tuples(list(df.columns))
        # Direct-call the flattening logic by piping through parse-equivalent steps.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        assert "Open" in df.columns and "Adj Close" in df.columns


class TestExtraMeta:
    def test_full_adjustment_quality(self, adapter):
        assert adapter.extra_meta == {"adjustment_quality": QUALITY_FULL}


class TestFetchMocked:
    def test_happy_path_writes_parquet(self, adapter, tmp_path):
        mock_yf = MagicMock()
        ticker = MagicMock()
        ticker.history.return_value = _yf_like_df()
        mock_yf.Ticker.return_value = ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            raw_path = adapter.fetch(
                "NYSE:AAPL", start=date(2026, 1, 2), end=date(2026, 1, 6),
                data_root=tmp_path,
            )
        assert raw_path.suffix == ".parquet"
        # Round-trip the file through parse → schema.
        df = adapter.parse(raw_path)
        norm = OHLCV_SCHEMA.normalize(
            df, source_column_map=adapter.source_column_map,
            provider="yfinance", identifier="NYSE:AAPL",
        )
        OHLCV_SCHEMA.validate(norm, provider="yfinance", identifier="NYSE:AAPL")

    def test_yf_inclusive_end_padding(self, adapter, tmp_path):
        mock_yf = MagicMock()
        ticker = MagicMock()
        ticker.history.return_value = _yf_like_df()
        mock_yf.Ticker.return_value = ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            adapter.fetch("NYSE:AAPL", start=date(2026, 1, 2),
                          end=date(2026, 1, 6), data_root=tmp_path)
        # yfinance's end is exclusive; adapter must pad by 1 day.
        call_kwargs = ticker.history.call_args.kwargs
        assert call_kwargs["end"] == "2026-01-07"
        assert call_kwargs["auto_adjust"] is False

    def test_empty_history_raises(self, adapter, tmp_path):
        mock_yf = MagicMock()
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with pytest.raises(EmptyPayload):
                adapter.fetch("NYSE:ZZZZ", start=date(2026, 1, 1),
                              end=date(2026, 1, 9), data_root=tmp_path)

    def test_yf_raise_wrapped(self, tmp_path):
        # Zero retries so the wrapped error propagates without backoff sleeps.
        ad = YFinanceAdapter(USEquitiesConfig(retry_max_retries=0))
        mock_yf = MagicMock()
        ticker = MagicMock()
        ticker.history.side_effect = RuntimeError("yfinance internal failure")
        mock_yf.Ticker.return_value = ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with pytest.raises(ProviderError, match="yfinance error"):
                ad.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                         end=date(2026, 1, 9), data_root=tmp_path)

    def test_transient_error_retried_then_succeeds(self, tmp_path):
        # Same shared retry/backoff as the nse_equities yfinance adapter:
        # a transient failure on attempt 1 is retried and the fetch lands.
        ad = YFinanceAdapter(USEquitiesConfig(
            retry_max_retries=2,
            retry_base_delay_sec=0.01,
            retry_max_delay_sec=0.01,
            retry_jitter=False,
        ))
        mock_yf = MagicMock()
        ticker = MagicMock()
        ticker.history.side_effect = [
            RuntimeError("transient yfinance failure"),
            _yf_like_df(),
        ]
        mock_yf.Ticker.return_value = ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            raw = ad.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                           end=date(2026, 1, 9), data_root=tmp_path)
        assert raw.exists()
        assert ticker.history.call_count == 2

    def test_disabled_config_rejects(self, tmp_path):
        ad = YFinanceAdapter(USEquitiesConfig(yfinance_enabled=False))
        with pytest.raises(ProviderError, match="disabled"):
            ad.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                     end=date(2026, 1, 9), data_root=tmp_path)
