"""Stage 7 tests: Tiingo adapter — URL build, key safety, retry, parse, schema."""

from __future__ import annotations

import json
import os
import urllib.error
from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_pipelines.domains.us_equities.adapters.tiingo import (
    TiingoAdapter,
    _has_rows,
)
from data_pipelines.domains.us_equities.config import USEquitiesConfig
from data_pipelines.domains.us_equities.schema import OHLCV_SCHEMA, QUALITY_FULL
from data_pipelines.errors import (
    EmptyPayload,
    MissingAPIKey,
    ProviderError,
)

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "us_equities" / "tiingo"

# Captured at import (before the autouse env-scrub fixture wipes it) so the
# online smoke test can restore the real key inside the test body.
_REAL_TIINGO_KEY = os.environ.get("TIINGO_API_KEY")


@pytest.fixture
def adapter() -> TiingoAdapter:
    return TiingoAdapter(USEquitiesConfig(tiingo_max_retries=2))


@pytest.fixture(autouse=True)
def _scrub_env():
    """Remove TIINGO_API_KEY around every test; restore after."""
    prev = os.environ.pop("TIINGO_API_KEY", None)
    yield
    if prev is not None:
        os.environ["TIINGO_API_KEY"] = prev


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Class-level circuit-breaker state must not bleed across tests."""
    from data_pipelines.domains.us_equities.adapters.tiingo import TiingoAdapter
    TiingoAdapter._consecutive_429s = 0
    TiingoAdapter._circuit_open_until = 0.0
    yield
    TiingoAdapter._consecutive_429s = 0
    TiingoAdapter._circuit_open_until = 0.0


class TestKeyResolution:
    def test_missing_key_raises_before_network(self, adapter, tmp_path):
        with pytest.raises(MissingAPIKey) as exc:
            adapter.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                          end=date(2026, 1, 9), data_root=tmp_path)
        assert exc.value.env_var == "TIINGO_API_KEY"

    def test_health_check_false_when_no_key(self, adapter):
        assert adapter.health_check() is False

    def test_health_check_true_with_key(self, adapter):
        os.environ["TIINGO_API_KEY"] = "fake-key"
        assert adapter.health_check() is True

    def test_key_not_in_error_message(self, adapter, tmp_path):
        os.environ["TIINGO_API_KEY"] = "super-secret-token-do-not-leak"
        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       "url", 401, "Unauthorized", {}, BytesIO())):
            with pytest.raises(ProviderError) as exc:
                adapter.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                              end=date(2026, 1, 2), data_root=tmp_path)
        assert "super-secret-token" not in str(exc.value)
        assert "super-secret-token" not in repr(exc.value)


class TestURLBuild:
    def test_nyse_url(self, adapter):
        url = adapter._build_url("NYSE:AAPL", date(2026, 1, 1), date(2026, 1, 9))
        assert "tiingo.com" in url
        assert "/tiingo/daily/aapl/prices" in url
        assert "startDate=2026-01-01" in url
        assert "endDate=2026-01-09" in url
        # Key must NOT be in URL (D6 — token in Authorization header only).
        assert "token=" not in url.lower()

    def test_index_rejected(self, adapter):
        with pytest.raises(ProviderError, match="INDEX"):
            adapter._build_url("INDEX:^SPX", date(2026, 1, 1), date(2026, 1, 9))


class TestHasRows:
    def test_empty_bytes(self): assert not _has_rows(b"")
    def test_empty_array(self): assert not _has_rows(b"[]")
    def test_invalid_json(self): assert not _has_rows(b"not json")
    def test_dict_payload(self): assert not _has_rows(b'{"detail": "error"}')
    def test_real_array(self):
        assert _has_rows(b'[{"date": "2026-01-01"}]')


class TestParse:
    def test_renames_adj_close(self, adapter):
        df = adapter.parse(FIXTURES / "aapl_sample.json")
        # Adapter does NOT rename — that's source_column_map's job.
        # But the column must be present after schema.normalize.
        norm = OHLCV_SCHEMA.normalize(
            df, source_column_map=adapter.source_column_map,
            provider="tiingo", identifier="NYSE:AAPL",
        )
        OHLCV_SCHEMA.validate(norm, provider="tiingo", identifier="NYSE:AAPL")
        assert list(norm["adj_close"]) == [186.95, 188.10, 189.50]

    def test_date_pinned_to_ns_midnight(self, adapter):
        df = adapter.parse(FIXTURES / "aapl_sample.json")
        assert str(df["date"].dtype) == "datetime64[ns]"
        # All times should be midnight.
        assert (df["date"].dt.normalize() == df["date"]).all()

    def test_sorted_ascending(self, adapter):
        df = adapter.parse(FIXTURES / "aapl_sample.json")
        assert df["date"].is_monotonic_increasing

    def test_empty_array_yields_empty_df(self, adapter):
        df = adapter.parse(FIXTURES / "empty.json")
        assert len(df) == 0


def _mock_resp(payload: bytes, status: int = 200):
    m = MagicMock()
    m.status = status
    m.read.return_value = payload
    m.__enter__ = lambda self_: self_
    m.__exit__ = lambda *a, **kw: False
    return m


class TestFetchMocked:
    def test_happy_path_writes_raw_json(self, adapter, tmp_path):
        os.environ["TIINGO_API_KEY"] = "fake-key"
        payload = (FIXTURES / "aapl_sample.json").read_bytes()
        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen",
                   return_value=_mock_resp(payload)):
            raw_path = adapter.fetch(
                "NYSE:AAPL", start=date(2026, 1, 2), end=date(2026, 1, 6),
                data_root=tmp_path,
            )
        assert raw_path.suffix == ".json"
        assert raw_path.read_bytes() == payload

    def test_empty_array_raises_empty_payload(self, adapter, tmp_path):
        os.environ["TIINGO_API_KEY"] = "fake-key"
        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen",
                   return_value=_mock_resp(b"[]")):
            with pytest.raises(EmptyPayload):
                adapter.fetch("NYSE:ZZZZ", start=date(2026, 1, 1),
                              end=date(2026, 1, 2), data_root=tmp_path)

    def test_401_raises_provider_error_no_key_leak(self, adapter, tmp_path):
        os.environ["TIINGO_API_KEY"] = "x" * 64
        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       "url", 401, "Unauthorized", {}, BytesIO())):
            with pytest.raises(ProviderError, match="401"):
                adapter.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                              end=date(2026, 1, 9), data_root=tmp_path)

    def test_429_retries_then_succeeds(self, adapter, tmp_path):
        os.environ["TIINGO_API_KEY"] = "fake-key"
        payload = (FIXTURES / "aapl_sample.json").read_bytes()
        call_count = {"n": 0}

        def side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise urllib.error.HTTPError("url", 429, "Too Many Requests", {}, BytesIO())
            return _mock_resp(payload)

        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen", side_effect=side_effect), \
             patch("data_pipelines.domains.us_equities.adapters.tiingo.time.sleep"):
            raw_path = adapter.fetch(
                "NYSE:AAPL", start=date(2026, 1, 1), end=date(2026, 1, 9),
                data_root=tmp_path,
            )
        assert raw_path.exists()
        assert call_count["n"] == 2

    def test_circuit_breaker_trips_after_threshold(self, adapter, tmp_path):
        from data_pipelines.domains.us_equities.adapters.tiingo import TiingoAdapter
        # Reset class-level state.
        TiingoAdapter._consecutive_429s = 0
        TiingoAdapter._circuit_open_until = 0.0

        os.environ["TIINGO_API_KEY"] = "fake-key"
        # Simulate sustained 429s. Threshold is 3; retries inside one fetch
        # accrue per attempt, so the breaker can trip mid-fetch on a long
        # retry chain, or across consecutive fetches.
        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       "url", 429, "Too Many Requests", {}, BytesIO())), \
             patch("data_pipelines.domains.us_equities.adapters.tiingo.time.sleep"):
            for i in range(3):
                with pytest.raises(ProviderError):
                    adapter.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                                   end=date(2026, 1, 9), data_root=tmp_path)
        # After the breaker trips, subsequent calls fail fast WITHOUT making
        # a network request.
        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen") as urlopen_mock, \
             patch("data_pipelines.domains.us_equities.adapters.tiingo.time.sleep"):
            with pytest.raises(ProviderError, match="circuit breaker"):
                adapter.fetch("NYSE:MSFT", start=date(2026, 1, 1),
                               end=date(2026, 1, 9), data_root=tmp_path)
            assert urlopen_mock.call_count == 0  # zero network calls
        # Cleanup.
        TiingoAdapter._consecutive_429s = 0
        TiingoAdapter._circuit_open_until = 0.0

    def test_circuit_breaker_resets_on_success(self, adapter, tmp_path):
        from data_pipelines.domains.us_equities.adapters.tiingo import TiingoAdapter
        TiingoAdapter._consecutive_429s = 2  # one below threshold
        TiingoAdapter._circuit_open_until = 0.0

        os.environ["TIINGO_API_KEY"] = "fake-key"
        payload = (FIXTURES / "aapl_sample.json").read_bytes()
        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen",
                   return_value=_mock_resp(payload)):
            adapter.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                           end=date(2026, 1, 9), data_root=tmp_path)
        # 429 streak reset to 0 after success.
        assert TiingoAdapter._consecutive_429s == 0
        TiingoAdapter._consecutive_429s = 0
        TiingoAdapter._circuit_open_until = 0.0

    def test_429_exhausts_retries(self, adapter, tmp_path):
        os.environ["TIINGO_API_KEY"] = "fake-key"
        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       "url", 429, "Too Many Requests", {}, BytesIO())), \
             patch("data_pipelines.domains.us_equities.adapters.tiingo.time.sleep"):
            with pytest.raises(ProviderError, match="429"):
                adapter.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                              end=date(2026, 1, 9), data_root=tmp_path)

    def test_token_in_auth_header_not_url(self, adapter, tmp_path):
        os.environ["TIINGO_API_KEY"] = "the-secret-token"
        captured: dict = {}

        def capturing_urlopen(req, *args, **kwargs):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            return _mock_resp((FIXTURES / "aapl_sample.json").read_bytes())

        with patch("data_pipelines.domains.us_equities.adapters.tiingo."
                   "urllib.request.urlopen", side_effect=capturing_urlopen):
            adapter.fetch("NYSE:AAPL", start=date(2026, 1, 1),
                          end=date(2026, 1, 9), data_root=tmp_path)

        assert "the-secret-token" not in captured["url"]
        # Authorization header carries the token.
        assert any("the-secret-token" in v for v in captured["headers"].values())


@pytest.mark.skipif(
    os.environ.get("PYTEST_ONLINE") != "1" or not _REAL_TIINGO_KEY,
    reason="online smoke test gated on PYTEST_ONLINE=1 and TIINGO_API_KEY env var",
)
class TestOnline:
    def test_real_aapl_fetch_and_parse(self, tmp_path):
        os.environ["TIINGO_API_KEY"] = _REAL_TIINGO_KEY  # restore after autouse scrub
        adapter = TiingoAdapter(USEquitiesConfig(tiingo_max_retries=2))
        raw_path = adapter.fetch(
            "NYSE:AAPL", start=date(2025, 1, 1), end=date(2025, 1, 31),
            data_root=tmp_path,
        )
        assert raw_path.exists()
        df = adapter.parse(raw_path)
        norm = OHLCV_SCHEMA.normalize(
            df, source_column_map=adapter.source_column_map,
            provider="tiingo", identifier="NYSE:AAPL",
        )
        OHLCV_SCHEMA.validate(norm, provider="tiingo", identifier="NYSE:AAPL")
        assert len(norm) >= 10
        assert adapter.extra_meta == {"adjustment_quality": QUALITY_FULL}
