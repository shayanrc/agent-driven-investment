"""NSEDomain wire-up — chain composition + registry interplay."""

from __future__ import annotations

from data_pipelines.domain import DomainRegistry
from data_pipelines.domains.nse_equities import NSEDomain, get_domain
from data_pipelines.domains.nse_equities.adapters.jugaad import JugaadAdapter
from data_pipelines.domains.nse_equities.adapters.nselib import NSElibAdapter
from data_pipelines.domains.nse_equities.adapters.yfinance import YFinanceNSEAdapter
from data_pipelines.domains.nse_equities.schema import OHLCV_SCHEMA


def test_domain_metadata():
    d = NSEDomain()
    assert d.name == "nse_equities"
    assert set(d.identifier_prefixes) == {"NSE", "BSE", "NIFTY"}
    assert d.schema is OHLCV_SCHEMA
    assert d.time_column == "date"


def test_get_domain_returns_singleton():
    a = get_domain()
    b = get_domain()
    assert a is b


def test_chain_for_nse_equity_three_adapters():
    """NSE: equities → [jugaad, nselib, yfinance] (gap-size/cache agnostic)."""
    d = NSEDomain()
    cold = d.chain_for_gap("NSE:RELIANCE", gap_size_trading_days=10, has_cache=False)
    warm = d.chain_for_gap("NSE:RELIANCE", gap_size_trading_days=1, has_cache=True)
    big = d.chain_for_gap("NSE:RELIANCE", gap_size_trading_days=10000, has_cache=True)
    for chain in (cold, warm, big):
        assert [type(a).__name__ for a in chain] == [
            "JugaadAdapter", "NSElibAdapter", "YFinanceNSEAdapter",
        ]


def test_chain_for_nifty_index_nselib_first_then_yfinance():
    """NIFTY: → [nselib, yfinance]. nselib has true TRADED_QTY for its
    ~3-year window; yfinance backfills older OHLC via partial-fill
    continuation. jugaad excluded (no volume in index payload)."""
    d = NSEDomain()
    chain = d.chain_for_gap("NIFTY:50", gap_size_trading_days=100, has_cache=False)
    assert [type(a).__name__ for a in chain] == [
        "NSElibAdapter", "YFinanceNSEAdapter",
    ]


def test_chain_for_bse_yfinance_only():
    """BSE: → [yfinance] only (jugaad + nselib are NSE-only)."""
    d = NSEDomain()
    chain = d.chain_for_gap("BSE:RELIANCE", gap_size_trading_days=10, has_cache=False)
    assert [type(a).__name__ for a in chain] == ["YFinanceNSEAdapter"]


def test_parse_identifier_round_trip():
    d = NSEDomain()
    assert d.parse_identifier("NSE:RELIANCE") == ("NSE", "RELIANCE")
    assert d.parse_identifier("NIFTY:50") == ("NIFTY", "50")


def test_registry_registration():
    DomainRegistry._reset()
    DomainRegistry.register(NSEDomain())
    resolved = DomainRegistry.resolve("NSE:TCS")
    assert resolved.name == "nse_equities"
    resolved = DomainRegistry.resolve("NIFTY:50")
    assert resolved.name == "nse_equities"


def test_merge_overlap_delegates_to_schema_module():
    """Smoke — verify the domain hooks the right merge fn (vs default new-wins)."""
    import pandas as pd
    from data_pipelines.domains.nse_equities.schema import QUALITY_FULL, QUALITY_NONE
    d = NSEDomain()
    existing = pd.DataFrame({
        "date": pd.to_datetime(["2025-04-01"]).astype("datetime64[ns]"),
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.5], "adj_close": [88.0], "volume": [1000],
    }).astype({"volume": "int64"})
    new = pd.DataFrame({
        "date": pd.to_datetime(["2025-04-01"]).astype("datetime64[ns]"),
        "open": [101.0], "high": [102.0], "low": [100.0],
        "close": [101.5], "adj_close": [101.5], "volume": [2000],
    }).astype({"volume": "int64"})
    out = d.merge_overlap(
        existing, new,
        existing_sources=[{"provider": "yfinance", "adjustment_quality": QUALITY_FULL}],
        new_source={"provider": "jugaad", "adjustment_quality": QUALITY_NONE},
    )
    # close was overwritten by jugaad; adj_close preserved from yfinance.
    assert out["close"].iloc[0] == 101.5
    assert out["adj_close"].iloc[0] == 88.0
