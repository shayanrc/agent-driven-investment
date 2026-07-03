"""panel — pure per-ticker assembly (fundamentals + prices + splits → ratios)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valuation.panel import PANEL_COLUMNS, build_ticker_panel, latest_snapshot


def _quarterly():
    # 5 consecutive dated quarters, revenue ramping, ~constant shares
    return pd.DataFrame({
        "fiscal_period_end": pd.to_datetime([
            "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"]),
        "filed_date": pd.to_datetime([
            "2024-05-01", "2024-08-01", "2024-11-01", "2025-02-15", "2025-05-01"]),
        "revenue": [9000.0, 9500.0, 10000.0, 12000.0, 10500.0],
        "net_income": [1800.0, 1900.0, 2000.0, 2400.0, 2100.0],
        "fcf": [1500.0, 1600.0, 1700.0, 2000.0, 1750.0],
        "shares_diluted": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    })


def _prices(dates, px):
    return pd.DataFrame({"date": pd.to_datetime(dates), "adj_close": px})


class TestBuildTickerPanel:
    def test_point_in_time_ratio_flips_on_filing(self):
        # TTM through 2024-12-31 (rev 9000+9500+10000+12000=40500) becomes
        # effective 2025-02-15. Two price days straddle it at the same price.
        q = _quarterly()
        prices = _prices(["2025-02-14", "2025-02-15"], [100.0, 100.0])
        out = build_ticker_panel("FUND:TEST", q, prices, pd.Series(dtype="float64"))
        before = out[out["date"] == "2025-02-14"].iloc[0]
        after = out[out["date"] == "2025-02-15"].iloc[0]
        # before the filing: no 4-quarter window is effective yet → NaN
        assert np.isnan(before["pe"])
        # on the filing: TTM live. mcap = 100*1000 = 100000; NI_ttm = 8100
        assert after["revenue_ttm"] == 40500.0
        assert after["net_income_ttm"] == 8100.0
        assert after["pe"] == 100000.0 / 8100.0
        assert after["asof_fiscal_period_end"] == pd.Timestamp("2024-12-31")

    def test_price_moves_daily_fundamentals_step(self):
        q = _quarterly()
        # two days both after the 2025-02-15 filing, different prices
        prices = _prices(["2025-03-01", "2025-03-02"], [100.0, 110.0])
        out = build_ticker_panel("FUND:TEST", q, prices, pd.Series(dtype="float64"))
        d1, d2 = out.iloc[0], out.iloc[1]
        # same TTM (fundamentals constant between filings)
        assert d1["net_income_ttm"] == d2["net_income_ttm"] == 8100.0
        # PE scales with price (10% higher price → 10% higher PE)
        assert d2["pe"] == pytest.approx(d1["pe"] * 1.1)

    def test_split_adjusts_shares(self):
        # a 2:1 split AFTER all fiscal ends → reported shares lifted ×2, so
        # per-share halves and mcap uses double the shares.
        q = _quarterly()
        splits = pd.Series({pd.Timestamp("2025-06-01"): 2.0})
        prices = _prices(["2025-03-01"], [100.0])
        out = build_ticker_panel("FUND:TEST", q, prices, splits)
        row = out.iloc[0]
        assert row["shares"] == 2000.0                 # 1000 × 2
        assert row["eps_ttm"] == 8100.0 / 2000.0        # halved
        assert row["market_cap"] == 100.0 * 2000.0

    def test_columns_canonical(self):
        out = build_ticker_panel(
            "FUND:TEST", _quarterly(), _prices(["2025-03-01"], [100.0]),
            pd.Series(dtype="float64"))
        assert list(out.columns) == list(PANEL_COLUMNS)


class TestLatestSnapshot:
    def test_one_row_per_ticker_most_recent(self):
        panel = pd.DataFrame({
            "ticker": ["FUND:A", "FUND:A", "FUND:B"],
            "date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-15"]),
            "pe": [10.0, 11.0, 20.0],
        })
        snap = latest_snapshot(panel)
        assert len(snap) == 2
        assert snap[snap["ticker"] == "FUND:A"]["pe"].item() == 11.0
