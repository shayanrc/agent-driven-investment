"""Validation gates for the point-in-time panel — the causal invariants that
must hold for the ratios to be usable as model features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from valuation.panel import build_ticker_panel

_NO_SPLITS = pd.Series(dtype="float64")


def _quarterly(revenue):
    return pd.DataFrame({
        "fiscal_period_end": pd.to_datetime([
            "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"]),
        "filed_date": pd.to_datetime([
            "2024-05-01", "2024-08-01", "2024-11-01", "2025-02-15", "2025-05-01"]),
        "revenue": revenue,
        "net_income": [r * 0.2 for r in revenue],
        "fcf": [r * 0.15 for r in revenue],
        "shares_diluted": [1000.0] * 5,
    })


def _prices(dates, px):
    return pd.DataFrame({"date": pd.to_datetime(dates), "adj_close": px})


class TestNoLookAhead:
    def test_future_filing_does_not_change_past_row(self):
        # A day 2025-03-01: after q4 filing (2025-02-15), before q5 (2025-05-01).
        prices = _prices(["2025-03-01"], [100.0])
        base = _quarterly([9000, 9500, 10000, 12000, 10500])
        # perturb ONLY q5 (2025-03-31, filed 2025-05-01 — the future)
        perturbed = _quarterly([9000, 9500, 10000, 12000, 999999])

        r_base = build_ticker_panel("FUND:T", base, prices, _NO_SPLITS).iloc[0]
        r_pert = build_ticker_panel("FUND:T", perturbed, prices, _NO_SPLITS).iloc[0]
        # the 2025-03-01 row reflects TTM through q4; q5 is not yet public
        assert r_base["revenue_ttm"] == r_pert["revenue_ttm"] == 40500.0
        assert r_base["pe"] == r_pert["pe"]

    def test_already_filed_change_does_change_row(self):
        # sanity: perturbing q4 (already filed by 2025-03-01) DOES move the row
        prices = _prices(["2025-03-01"], [100.0])
        base = _quarterly([9000, 9500, 10000, 12000, 10500])
        perturbed = _quarterly([9000, 9500, 10000, 20000, 10500])
        r_base = build_ticker_panel("FUND:T", base, prices, _NO_SPLITS).iloc[0]
        r_pert = build_ticker_panel("FUND:T", perturbed, prices, _NO_SPLITS).iloc[0]
        assert r_pert["revenue_ttm"] == 48500.0
        assert r_pert["revenue_ttm"] != r_base["revenue_ttm"]


class TestTTMStepsOnlyOnFilings:
    def test_fundamentals_constant_between_filings_price_varies(self):
        # 4 days all inside [2025-02-15, 2025-05-01) at different prices
        prices = _prices(
            ["2025-02-20", "2025-03-15", "2025-04-01", "2025-04-30"],
            [100.0, 120.0, 90.0, 110.0])
        out = build_ticker_panel(
            "FUND:T", _quarterly([9000, 9500, 10000, 12000, 10500]),
            prices, _NO_SPLITS)
        # every TTM column is constant across the window
        for col in ("revenue_ttm", "net_income_ttm", "fcf_ttm", "eps_ttm", "shares"):
            assert out[col].nunique() == 1
        # but PE tracks price exactly (pe/price = shares/ni_ttm = const)
        ratio = out["pe"] / out["adj_close"]
        assert np.allclose(ratio, ratio.iloc[0])


class TestSanity:
    def test_ps_positive_where_defined(self):
        prices = _prices(["2025-03-01", "2025-04-01"], [100.0, 110.0])
        out = build_ticker_panel(
            "FUND:T", _quarterly([9000, 9500, 10000, 12000, 10500]),
            prices, _NO_SPLITS)
        ps = out["ps"].dropna()
        assert (ps > 0).all()

    def test_no_row_before_first_full_ttm(self):
        # only 3 quarters of prices' worth of history is available before the
        # first 4-quarter window is effective (2025-02-15)
        prices = _prices(["2024-12-01", "2025-01-15"], [100.0, 100.0])
        out = build_ticker_panel(
            "FUND:T", _quarterly([9000, 9500, 10000, 12000, 10500]),
            prices, _NO_SPLITS)
        # both dates precede the first effective_date → all ratios NaN
        assert out["pe"].isna().all()
        assert out["asof_filed_date"].isna().all()
