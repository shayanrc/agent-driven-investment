"""Stage 5 tests: us_equities identifier parser."""

from __future__ import annotations

import pytest

from data_pipelines.domains.us_equities.registry import (
    SUPPORTED_INDICES,
    VALID_PREFIXES,
    parse_identifier,
)


class TestHappyPath:
    def test_nyse(self):
        assert parse_identifier("NYSE:AAPL") == ("NYSE", "AAPL")

    def test_nasdaq(self):
        assert parse_identifier("NASDAQ:MSFT") == ("NASDAQ", "MSFT")

    def test_index_spx(self):
        assert parse_identifier("INDEX:^SPX") == ("INDEX", "^SPX")


class TestNormalization:
    def test_lowercase_prefix_uppercased(self):
        assert parse_identifier("nyse:aapl") == ("NYSE", "AAPL")

    def test_lowercase_symbol_uppercased(self):
        assert parse_identifier("NYSE:aapl") == ("NYSE", "AAPL")

    def test_index_alpha_uppercased_keeps_caret(self):
        assert parse_identifier("INDEX:^spx") == ("INDEX", "^SPX")


class TestRejections:
    def test_no_prefix(self):
        with pytest.raises(ValueError, match="missing domain prefix"):
            parse_identifier("AAPL")

    def test_unknown_prefix(self):
        with pytest.raises(ValueError, match="unknown us_equities prefix"):
            parse_identifier("NSE:RELIANCE")

    def test_empty_symbol(self):
        with pytest.raises(ValueError, match="empty symbol"):
            parse_identifier("NYSE:")

    def test_index_without_caret_rejected(self):
        with pytest.raises(ValueError, match="must start with"):
            parse_identifier("INDEX:SPX")


class TestSupportedIndicesConstant:
    def test_canonical_set(self):
        assert SUPPORTED_INDICES == {"^SPX", "^NDX", "^DJI", "^RUT"}

    def test_out_of_set_index_still_parses(self):
        # Parser is permissive on well-formed identifiers; universe membership
        # is enforced elsewhere.
        assert parse_identifier("INDEX:^VIX") == ("INDEX", "^VIX")


def test_valid_prefixes_constant():
    assert VALID_PREFIXES == ("NYSE", "NASDAQ", "INDEX")
