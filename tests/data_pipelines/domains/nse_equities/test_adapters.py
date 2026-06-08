"""nse_equities adapter tests — parse() against committed fixtures + selected
fetch-path branches that don't need the network (BSE → EmptyPayload, etc.).

Fixtures are real downloads from each provider for RELIANCE 2025-04 (equity)
and NIFTY 50 (index where supported). Captured under
tests/data_pipelines/fixtures/nse_equities/{jugaad,nselib,yfinance}/.

These tests deliberately do NOT hit the network. The online round-trip is
exercised by the V1.7 seed/parity scripts; CI runs the fast offline path.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data_pipelines.domains.nse_equities.adapters.jugaad import JugaadAdapter
from data_pipelines.domains.nse_equities.adapters.nselib import NSElibAdapter
from data_pipelines.domains.nse_equities.adapters.yfinance import YFinanceNSEAdapter
from data_pipelines.domains.nse_equities.schema import (
    OHLCV_SCHEMA,
    QUALITY_FULL,
    QUALITY_NONE,
)
from data_pipelines.errors import EmptyPayload

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "nse_equities"


def _normalize_and_validate(df, adapter, identifier):
    out = OHLCV_SCHEMA.normalize(
        df, source_column_map=adapter.source_column_map,
        provider=adapter.name, identifier=identifier,
    )
    OHLCV_SCHEMA.validate(out, provider=adapter.name, identifier=identifier)
    return out


# ---- jugaad ----


def test_jugaad_parse_reliance_fixture():
    df = JugaadAdapter().parse(FIXTURES / "jugaad" / "RELIANCE_2025-04.json")
    assert set(["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "ADJ_CLOSE", "VOLUME"]).issubset(df.columns)
    # 19 trading days in April 2025
    assert len(df) == 19
    # Dates monotonic ascending; IST trade dates (no UTC-shift residue)
    assert df["DATE"].is_monotonic_increasing
    assert df["DATE"].iloc[0] == pd.Timestamp("2025-04-01")
    assert df["DATE"].iloc[-1] == pd.Timestamp("2025-04-30")
    # adj_close mirrors close per QUALITY_NONE
    assert (df["ADJ_CLOSE"] == df["CLOSE"]).all()


def test_jugaad_parse_then_normalize_passes_schema():
    df = JugaadAdapter().parse(FIXTURES / "jugaad" / "RELIANCE_2025-04.json")
    out = _normalize_and_validate(df, JugaadAdapter(), "NSE:RELIANCE")
    assert out["date"].dtype == "datetime64[ns]"
    assert out["volume"].dtype == "int64"
    # Smoke on actual values from the upstream payload
    assert (out["volume"] > 0).all()
    assert (out["close"] > 1000).all()  # RELIANCE was around 1200-1400 INR in Apr-2025
    # No duplicate dates — would otherwise blow up the (ticker, date) PK in
    # cache writes. Regression for the "jugaad returns ALL series" trap that
    # surfaced in the v1.7 NIFTY 50 seed run.
    assert out["date"].is_unique


def test_jugaad_parse_drops_non_eq_series(tmp_path):
    """If the raw payload contains both EQ + BL rows for the same date, parse()
    must keep only EQ — block-deal series rows are out of scope for daily
    OHLCV and would collide on the (ticker, date) primary key."""
    import json
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps([
        # EQ row
        {"CH_TIMESTAMP": "2025-04-01T18:30:00.000Z", "CH_SERIES": "EQ",
         "CH_OPENING_PRICE": 100.0, "CH_TRADE_HIGH_PRICE": 102.0,
         "CH_TRADE_LOW_PRICE": 99.0, "CH_CLOSING_PRICE": 101.0,
         "CH_TOT_TRADED_QTY": 1000},
        # BL row, same date — must be dropped
        {"CH_TIMESTAMP": "2025-04-01T18:30:00.000Z", "CH_SERIES": "BL",
         "CH_OPENING_PRICE": 99.5, "CH_TRADE_HIGH_PRICE": 99.5,
         "CH_TRADE_LOW_PRICE": 99.5, "CH_CLOSING_PRICE": 99.5,
         "CH_TOT_TRADED_QTY": 50},
    ]))
    df = JugaadAdapter().parse(p)
    assert len(df) == 1
    assert df["CLOSE"].iloc[0] == 101.0   # EQ row, not BL


def test_jugaad_adapter_meta_is_quality_none_inr():
    a = JugaadAdapter()
    assert a.extra_meta == {"adjustment_quality": QUALITY_NONE, "currency": "INR"}


def test_jugaad_raises_empty_for_nifty_identifier(tmp_path):
    """The jugaad index endpoint is broken (KeyError 'd'); adapter short-circuits."""
    with pytest.raises(EmptyPayload):
        JugaadAdapter().fetch("NIFTY:50", date(2025, 4, 1), date(2025, 4, 30), data_root=tmp_path)


def test_jugaad_raises_empty_for_bse_identifier(tmp_path):
    """jugaad-data is NSE-only; BSE: identifier falls through immediately."""
    with pytest.raises(EmptyPayload):
        JugaadAdapter().fetch("BSE:RELIANCE", date(2025, 4, 1), date(2025, 4, 30), data_root=tmp_path)


# ---- nselib ----


def test_nselib_parse_reliance_equity_fixture():
    df = NSElibAdapter().parse(FIXTURES / "nselib" / "RELIANCE_2025-04.csv")
    assert set(["date", "open", "high", "low", "close", "adj_close", "volume"]) <= set(df.columns)
    assert len(df) == 19
    assert df["date"].is_monotonic_increasing
    # Indian-comma cleanup must produce real numbers, not strings
    assert pd.api.types.is_numeric_dtype(df["volume"])
    assert (df["volume"] > 0).all()
    assert (df["close"] > 1000).all()
    # adj_close = close for QUALITY_NONE
    assert (df["adj_close"] == df["close"]).all()


def test_nselib_parse_nifty50_index_fixture():
    df = NSElibAdapter().parse(FIXTURES / "nselib" / "NIFTY50_2025-04.csv")
    assert len(df) == 19
    assert df["date"].is_monotonic_increasing
    # NIFTY 50 was ~23000-24500 in Apr 2025
    assert (df["close"] > 20000).all() and (df["close"] < 30000).all()
    # Index TRADED_QTY → volume
    assert (df["volume"] > 0).all()


def test_nselib_parse_then_normalize_passes_schema_equity():
    df = NSElibAdapter().parse(FIXTURES / "nselib" / "RELIANCE_2025-04.csv")
    _normalize_and_validate(df, NSElibAdapter(), "NSE:RELIANCE")


def test_nselib_parse_then_normalize_passes_schema_index():
    df = NSElibAdapter().parse(FIXTURES / "nselib" / "NIFTY50_2025-04.csv")
    _normalize_and_validate(df, NSElibAdapter(), "NIFTY:50")


def test_nselib_parse_drops_non_eq_series(tmp_path):
    """If the nselib CSV payload contains both EQ + BL rows for the same date,
    parse() must keep only EQ. Mirrors the JugaadAdapter regression — same
    bug class (#244 / GH #133): nselib's price_volume_data also returns the
    multi-series rows, and BL/T0/N1 collide on the (ticker, date) PK."""
    p = tmp_path / "mixed.csv"
    p.write_text(
        "Symbol,Series,Date,PrevClose,OpenPrice,HighPrice,LowPrice,LastPrice,"
        "ClosePrice,AveragePrice,TotalTradedQuantity,Turnover₹,No.ofTrades\n"
        # EQ row
        'RELIANCE,EQ,01-Apr-2025,"99.0","100.0","102.0","99.0","101.0","101.0",'
        '"100.5","1000","1,00,500.00","50"\n'
        # BL row, same date — must be dropped
        'RELIANCE,BL,01-Apr-2025,"99.0","99.5","99.5","99.5","99.5","99.5",'
        '"99.5","50","4,975.00","1"\n'
        # N1 row, same date — must also be dropped
        'RELIANCE,N1,01-Apr-2025,"99.0","99.0","99.0","99.0","99.0","99.0",'
        '"99.0","10","990.00","1"\n'
    )
    df = NSElibAdapter().parse(p)
    assert len(df) == 1
    assert df["close"].iloc[0] == 101.0  # EQ row, not BL/N1


def test_nselib_meta_is_quality_none_inr():
    a = NSElibAdapter()
    assert a.extra_meta == {"adjustment_quality": QUALITY_NONE, "currency": "INR"}


def test_nselib_raises_empty_for_bse(tmp_path):
    with pytest.raises(EmptyPayload):
        NSElibAdapter().fetch("BSE:RELIANCE", date(2025, 4, 1), date(2025, 4, 30), data_root=tmp_path)


def test_nselib_raises_empty_for_unknown_nifty_alias(tmp_path):
    with pytest.raises(EmptyPayload):
        NSElibAdapter().fetch("NIFTY:MEGACAP9000", date(2025, 4, 1), date(2025, 4, 30), data_root=tmp_path)


# ---- yfinance .NS ----


def test_yfinance_parse_reliance_fixture():
    df = YFinanceNSEAdapter().parse(FIXTURES / "yfinance" / "RELIANCE_NS_2025-04.parquet")
    # source-native columns survive
    assert set(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]) <= set(df.columns)
    assert len(df) == 19
    assert df["Date"].is_monotonic_increasing


def test_yfinance_parse_nifty_fixture():
    df = YFinanceNSEAdapter().parse(FIXTURES / "yfinance" / "NSEI_2025-04.parquet")
    assert len(df) == 19
    assert (df["Close"] > 20000).all() and (df["Close"] < 30000).all()


def test_yfinance_parse_then_normalize_passes_schema():
    df = YFinanceNSEAdapter().parse(FIXTURES / "yfinance" / "RELIANCE_NS_2025-04.parquet")
    out = _normalize_and_validate(df, YFinanceNSEAdapter(), "NSE:RELIANCE")
    # adj_close differs from close (full split+div adjustment)
    assert (out["adj_close"] != out["close"]).any()


def test_yfinance_meta_is_quality_full_inr():
    a = YFinanceNSEAdapter()
    assert a.extra_meta == {"adjustment_quality": QUALITY_FULL, "currency": "INR"}


def test_yfinance_symbol_mapping():
    a = YFinanceNSEAdapter()
    assert a._yf_symbol("NSE:RELIANCE") == "RELIANCE.NS"
    assert a._yf_symbol("BSE:RELIANCE") == "RELIANCE.BO"
    assert a._yf_symbol("NIFTY:50") == "^NSEI"
    # Unknown NIFTY alias → None (caller raises EmptyPayload)
    assert a._yf_symbol("NIFTY:DOESNOTEXIST") is None


def test_yfinance_unknown_nifty_raises_empty(tmp_path):
    with pytest.raises(EmptyPayload):
        YFinanceNSEAdapter().fetch(
            "NIFTY:DOESNOTEXIST", date(2025, 4, 1), date(2025, 4, 30),
            data_root=tmp_path,
        )


# ---- cross-adapter consistency on the same fixture day ----


def test_close_agreement_across_sources_on_reliance():
    """On 2025-04-01 (no corporate action), close should agree to <0.5% across
    all three sources. Detects accidental column swaps or scale errors."""
    j = JugaadAdapter().parse(FIXTURES / "jugaad" / "RELIANCE_2025-04.json")
    n = NSElibAdapter().parse(FIXTURES / "nselib" / "RELIANCE_2025-04.csv")
    y = YFinanceNSEAdapter().parse(FIXTURES / "yfinance" / "RELIANCE_NS_2025-04.parquet")

    j_n = _normalize_and_validate(j, JugaadAdapter(), "NSE:RELIANCE")
    n_n = _normalize_and_validate(n, NSElibAdapter(), "NSE:RELIANCE")
    y_n = _normalize_and_validate(y, YFinanceNSEAdapter(), "NSE:RELIANCE")

    target_date = pd.Timestamp("2025-04-01")
    j_close = float(j_n.loc[j_n["date"] == target_date, "close"].iloc[0])
    n_close = float(n_n.loc[n_n["date"] == target_date, "close"].iloc[0])
    y_close = float(y_n.loc[y_n["date"] == target_date, "close"].iloc[0])

    # Round to handle float epsilon — the three should be effectively equal
    # since RELIANCE had no corporate action in April 2025.
    base = max(j_close, n_close, y_close)
    assert abs(j_close - n_close) / base < 0.005
    assert abs(j_close - y_close) / base < 0.005
