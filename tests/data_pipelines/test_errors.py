"""Stage 1 tests: typed exception construction and message shape."""

from __future__ import annotations

import pytest

from data_pipelines.errors import (
    AllProvidersFailed,
    DataPipelinesError,
    EmptyPayload,
    MissingAPIKey,
    ProviderError,
    SchemaMismatch,
    UnknownDomain,
)


def test_provider_error_carries_context():
    e = ProviderError("tiingo", "NYSE:AAPL", "HTTP 503")
    assert e.provider == "tiingo"
    assert e.identifier == "NYSE:AAPL"
    assert e.reason == "HTTP 503"
    assert "tiingo" in str(e) and "NYSE:AAPL" in str(e) and "HTTP 503" in str(e)


def test_empty_payload_is_provider_error_subclass():
    e = EmptyPayload("stooq", "NYSE:ZZZZ")
    assert isinstance(e, ProviderError)
    assert "empty payload" in str(e)


def test_schema_mismatch_carries_details():
    e = SchemaMismatch("stooq", "NYSE:AAPL", "missing column 'adj_close'")
    assert e.provider == "stooq"
    assert "missing column" in str(e)


def test_missing_api_key_does_not_leak_key():
    e = MissingAPIKey("tiingo", "TIINGO_API_KEY")
    # The env var name appears, but no value should be in the message.
    assert "TIINGO_API_KEY" in str(e)
    assert e.env_var == "TIINGO_API_KEY"


def test_all_providers_failed_summarizes_chain():
    failures = [
        ProviderError("tiingo", "NYSE:AAPL", "HTTP 503"),
        ProviderError("yfinance", "NYSE:AAPL", "rate limited"),
    ]
    e = AllProvidersFailed("NYSE:AAPL", failures)
    assert e.failures == failures
    msg = str(e)
    assert "tiingo" in msg and "yfinance" in msg


def test_unknown_domain_lists_known_prefixes():
    e = UnknownDomain("FOO:BAR", ["NYSE", "NASDAQ", "INDEX"])
    assert "FOO:BAR" in str(e)
    assert "NYSE" in str(e)


def test_all_inherit_from_root():
    for cls in (ProviderError, EmptyPayload, SchemaMismatch,
                MissingAPIKey, AllProvidersFailed, UnknownDomain):
        with pytest.raises(DataPipelinesError):
            if cls is ProviderError:
                raise cls("p", "i", "r")
            elif cls is EmptyPayload:
                raise cls("p", "i")
            elif cls is SchemaMismatch:
                raise cls("p", "i", "d")
            elif cls is MissingAPIKey:
                raise cls("p", "ENV")
            elif cls is AllProvidersFailed:
                raise cls("i", [])
            elif cls is UnknownDomain:
                raise cls("i", [])
