"""Stage 6 tests: Stooq adapter — URL build, parse, empty-body handling,
end-to-end via mocked HTTP, schema invariant.

Online smoke test gated on PYTEST_ONLINE=1.
"""

from __future__ import annotations

import os
from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_pipelines.domains.us_equities.adapters.stooq import (
    StooqAdapter,
    _is_apikey_required,
    _looks_like_data,
)
from data_pipelines.domains.us_equities.schema import OHLCV_SCHEMA, QUALITY_SPLIT_ONLY
from data_pipelines.errors import EmptyPayload, ProviderError

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "us_equities" / "stooq"

# Captured at import time so the autouse env-scrub fixture below doesn't hide
# it from the online smoke test.
_REAL_STOOQ_KEY = os.environ.get("STOOQ_API_KEY")


@pytest.fixture
def adapter() -> StooqAdapter:
    return StooqAdapter()


@pytest.fixture(autouse=True)
def _scrub_env():
    prev = os.environ.pop("STOOQ_API_KEY", None)
    yield
    if prev is not None:
        os.environ["STOOQ_API_KEY"] = prev


class TestURLBuild:
    def test_nyse_url_no_key(self, adapter):
        url = adapter._build_url("NYSE:AAPL", None)
        assert "stooq.com" in url
        assert "s=aapl.us" in url
        assert "i=d" in url
        assert "apikey" not in url  # absent when key is None

    def test_nyse_url_with_key(self, adapter):
        url = adapter._build_url("NYSE:AAPL", "TESTKEY")
        assert "apikey=TESTKEY" in url

    def test_nasdaq_url(self, adapter):
        url = adapter._build_url("NASDAQ:MSFT", "TESTKEY")
        assert "s=msft.us" in url
        assert "apikey=TESTKEY" in url

    def test_index_url_uses_slug(self, adapter):
        url = adapter._build_url("INDEX:^SPX", None)
        assert "s=%5Espx" in url  # URL-encoded ^ = %5E

    def test_index_unknown_falls_back_to_lowercase(self, adapter):
        url = adapter._build_url("INDEX:^VIX", None)
        assert "s=%5Evix" in url


class TestKeyResolution:
    def test_resolve_returns_none_when_unset(self, adapter):
        assert adapter._resolve_key() is None

    def test_resolve_returns_value_when_set(self, adapter):
        os.environ["STOOQ_API_KEY"] = "fake-key"
        assert adapter._resolve_key() == "fake-key"

    def test_health_check_true_regardless_of_key(self, adapter):
        # Stooq doesn't require a key universally; health is module-level.
        assert adapter.health_check() is True
        os.environ["STOOQ_API_KEY"] = "fake-key"
        assert adapter.health_check() is True


class TestLooksLikeData:
    def test_empty(self):
        assert not _looks_like_data(b"")

    def test_whitespace_only(self):
        assert not _looks_like_data(b"   \n\n  ")

    def test_no_data_response(self):
        assert not _looks_like_data(b"No data\n")
        assert not _looks_like_data(b"NO DATA")

    def test_header_only(self):
        assert not _looks_like_data(b"Date,Open,High,Low,Close,Volume\n")

    def test_real_data(self):
        assert _looks_like_data(b"Date,Open,High,Low,Close,Volume\n2026-01-02,1,2,3,4,5\n")


class TestApikeyRequiredDetection:
    def test_help_page_detected(self):
        text = (FIXTURES / "apikey_required.html").read_bytes()
        assert _is_apikey_required(text)

    def test_real_csv_not_flagged(self):
        text = (FIXTURES / "aapl_sample.csv").read_bytes()
        assert not _is_apikey_required(text)

    def test_case_insensitive(self):
        assert _is_apikey_required(b"GET YOUR APIKEY: register here")

    def test_empty_not_flagged(self):
        assert not _is_apikey_required(b"")


class TestParse:
    def test_canonical_columns(self, adapter):
        df = adapter.parse(FIXTURES / "aapl_sample.csv")
        # Stooq has no adj_close; adapter fabricates from close.
        assert {"date", "open", "high", "low", "close", "adj_close", "volume"}.issubset(df.columns)
        assert (df["adj_close"] == df["close"]).all()

    def test_normalize_passes_schema(self, adapter):
        df = adapter.parse(FIXTURES / "aapl_sample.csv")
        norm = OHLCV_SCHEMA.normalize(df, provider="stooq", identifier="NYSE:AAPL")
        OHLCV_SCHEMA.validate(norm, provider="stooq", identifier="NYSE:AAPL")

    def test_sorted_ascending(self, adapter):
        df = adapter.parse(FIXTURES / "aapl_sample.csv")
        assert df["date"].is_monotonic_increasing

    def test_extra_meta_split_only(self, adapter):
        assert adapter.extra_meta == {"adjustment_quality": QUALITY_SPLIT_ONLY}


class TestFetchMocked:
    def _mock_response(self, payload: bytes, status: int = 200):
        m = MagicMock()
        m.status = status
        m.read.return_value = payload
        m.__enter__ = lambda self_: self_
        m.__exit__ = lambda *a, **kw: False
        return m

    def test_happy_path_writes_raw_no_key(self, adapter, tmp_path):
        # No STOOQ_API_KEY in env; adapter should still attempt the fetch.
        payload = (FIXTURES / "aapl_sample.csv").read_bytes()
        with patch("data_pipelines.domains.us_equities.adapters.stooq."
                   "urllib.request.urlopen",
                   return_value=self._mock_response(payload)):
            raw_path = adapter.fetch(
                "NYSE:AAPL", start=date(2026, 1, 2), end=date(2026, 1, 9),
                data_root=tmp_path,
            )
        assert raw_path.exists()
        assert raw_path.read_bytes() == payload
        rel = raw_path.relative_to(tmp_path)
        parts = rel.parts
        assert parts[:5] == ("raw", "stooq", "us_equities", "NYSE", "AAPL")
        assert parts[5].endswith(".csv")

    def test_apikey_required_raises_no_key_in_env(self, adapter, tmp_path):
        # IP-gated case, no key in env → error message hints at registration/IP.
        payload = (FIXTURES / "apikey_required.html").read_bytes()
        with patch("data_pipelines.domains.us_equities.adapters.stooq."
                   "urllib.request.urlopen",
                   return_value=self._mock_response(payload)):
            with pytest.raises(ProviderError, match="API-key-required") as exc:
                adapter.fetch("NYSE:AAPL", data_root=tmp_path)
        assert "gating this IP" in str(exc.value)
        raw_dir = tmp_path / "raw" / "stooq" / "us_equities" / "NYSE" / "AAPL"
        assert not raw_dir.exists() or not list(raw_dir.iterdir())

    def test_apikey_required_raises_invalid_key(self, adapter, tmp_path):
        # Key set but Stooq still rejects → error hints at invalid key.
        os.environ["STOOQ_API_KEY"] = "fake-invalid-key"
        payload = (FIXTURES / "apikey_required.html").read_bytes()
        with patch("data_pipelines.domains.us_equities.adapters.stooq."
                   "urllib.request.urlopen",
                   return_value=self._mock_response(payload)):
            with pytest.raises(ProviderError, match="invalid STOOQ_API_KEY"):
                adapter.fetch("NYSE:AAPL", data_root=tmp_path)

    def test_empty_body_raises_empty_payload(self, adapter, tmp_path):
        with patch("data_pipelines.domains.us_equities.adapters.stooq."
                   "urllib.request.urlopen",
                   return_value=self._mock_response(b"No data\n")):
            with pytest.raises(EmptyPayload):
                adapter.fetch("NYSE:ZZZZ", data_root=tmp_path)
        raw_dir = tmp_path / "raw" / "stooq" / "us_equities" / "NYSE" / "ZZZZ"
        assert not raw_dir.exists() or not list(raw_dir.iterdir())

    def test_http_error_raises_provider_error(self, adapter, tmp_path):
        import urllib.error
        with patch("data_pipelines.domains.us_equities.adapters.stooq."
                   "urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       "url", 503, "Service Unavailable", {}, BytesIO())):
            with pytest.raises(ProviderError, match="503"):
                adapter.fetch("NYSE:AAPL", data_root=tmp_path)

    def test_timeout_raises_provider_error(self, adapter, tmp_path):
        with patch("data_pipelines.domains.us_equities.adapters.stooq."
                   "urllib.request.urlopen",
                   side_effect=TimeoutError()):
            with pytest.raises(ProviderError, match="timeout"):
                adapter.fetch("NYSE:AAPL", data_root=tmp_path)

    def test_key_not_in_error_message(self, adapter, tmp_path):
        secret = "super-secret-stooq-key-do-not-leak"
        os.environ["STOOQ_API_KEY"] = secret
        import urllib.error
        with patch("data_pipelines.domains.us_equities.adapters.stooq."
                   "urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       "url", 503, "Service Unavailable", {}, BytesIO())):
            with pytest.raises(ProviderError) as exc:
                adapter.fetch("NYSE:AAPL", data_root=tmp_path)
        assert secret not in str(exc.value)


@pytest.mark.skipif(
    os.environ.get("PYTEST_ONLINE") != "1",
    reason="online smoke test gated on PYTEST_ONLINE=1 (STOOQ_API_KEY optional)",
)
class TestOnline:
    def test_real_aapl_fetch_and_parse(self, tmp_path):
        if _REAL_STOOQ_KEY:
            os.environ["STOOQ_API_KEY"] = _REAL_STOOQ_KEY
        ad = StooqAdapter()
        try:
            raw_path = ad.fetch(
                "NYSE:AAPL", start=date(2020, 1, 1), end=date.today(),
                data_root=tmp_path,
            )
        except ProviderError as e:
            if "API-key-required" in str(e) and not _REAL_STOOQ_KEY:
                # Expected on IP-gated networks without a key. Validates that
                # the failure surfaces as a typed error (not silent corruption)
                # but we can't actually fetch — skip rather than red.
                pytest.skip(
                    "Stooq is gating this IP and no STOOQ_API_KEY is set; "
                    "typed error path verified — register a key to enable "
                    "actual fetch"
                )
            raise
        assert raw_path.exists() and raw_path.stat().st_size > 1000
        df = ad.parse(raw_path)
        norm = OHLCV_SCHEMA.normalize(df, provider="stooq", identifier="NYSE:AAPL")
        OHLCV_SCHEMA.validate(norm, provider="stooq", identifier="NYSE:AAPL")
        assert len(norm) > 100
