"""Identifier parser + universe loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from data_pipelines.domains.nse_equities.registry import (
    NIFTY_INDEX_SLUGS,
    VALID_PREFIXES,
    parse_identifier,
    resolve_nifty_slug,
)
from data_pipelines.domains.nse_equities.universe import (
    is_in_universe,
    load_universe,
)


# ---- parser ----


@pytest.mark.parametrize("identifier,expected", [
    ("NSE:RELIANCE", ("NSE", "RELIANCE")),
    ("nse:reliance", ("NSE", "RELIANCE")),
    ("NSE:M&M", ("NSE", "M&M")),
    ("NSE:BAJAJ-AUTO", ("NSE", "BAJAJ-AUTO")),
    ("BSE:RELIANCE", ("BSE", "RELIANCE")),
    ("NIFTY:50", ("NIFTY", "50")),
    ("NIFTY:bank", ("NIFTY", "BANK")),
])
def test_parse_valid(identifier, expected):
    assert parse_identifier(identifier) == expected


def test_parse_missing_prefix():
    with pytest.raises(ValueError, match="missing domain prefix"):
        parse_identifier("RELIANCE")


def test_parse_unknown_prefix():
    with pytest.raises(ValueError, match="unknown nse_equities prefix"):
        parse_identifier("NASDAQ:AAPL")


def test_parse_empty_symbol():
    with pytest.raises(ValueError, match="empty symbol"):
        parse_identifier("NSE:")


def test_valid_prefixes_documented():
    # Spec lock — adding a prefix is a deliberate change.
    assert VALID_PREFIXES == ("NSE", "BSE", "NIFTY")


# ---- index slug resolution ----


@pytest.mark.parametrize("alias,upstream", [
    ("50", "NIFTY 50"),
    ("BANK", "NIFTY BANK"),
    ("NEXT50", "NIFTY NEXT 50"),
])
def test_nifty_slugs(alias, upstream):
    assert resolve_nifty_slug(alias) == upstream


def test_nifty_slug_case_insensitive():
    assert resolve_nifty_slug("50") == "NIFTY 50"
    assert resolve_nifty_slug("bank") == "NIFTY BANK"


def test_unknown_nifty_slug_returns_none():
    assert resolve_nifty_slug("MEGACAP9000") is None


def test_nifty_50_is_present():
    # Spec lock — "50" is the canonical short alias.
    assert "50" in NIFTY_INDEX_SLUGS


# ---- universe loader ----


def test_load_universe_nifty50_size_and_shape():
    universe = load_universe("nifty50")
    # 50 constituents + 1 index = 51.
    assert len(universe) == 51
    # Spot-check well-known constituents.
    assert "NSE:RELIANCE" in universe
    assert "NSE:TCS" in universe
    assert "NSE:INFY" in universe
    assert "NIFTY:50" in universe


def test_load_universe_all_identifiers_parse():
    """Every entry must round-trip through the parser."""
    for ident in load_universe("nifty50"):
        prefix, symbol = parse_identifier(ident)
        assert prefix in VALID_PREFIXES
        assert symbol


def test_is_in_universe():
    assert is_in_universe("NSE:RELIANCE", "nifty50")
    assert not is_in_universe("NSE:NOTREAL", "nifty50")


def test_load_universe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_universe("does_not_exist", config_root=tmp_path)


def test_load_universe_custom_root(tmp_path):
    p = tmp_path / "universe_tiny.yaml"
    p.write_text(yaml.safe_dump({
        "tickers": ["NSE:FOO", "NSE:BAR"],
        "indices": ["NIFTY:50"],
    }))
    out = load_universe("tiny", config_root=tmp_path)
    assert out == ["NSE:FOO", "NSE:BAR", "NIFTY:50"]
