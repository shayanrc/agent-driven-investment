"""ratios — per-share metrics, PE/PS/P-FCF, yields, denominator handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from valuation.ratios import compute_ratios


def _df(rows):
    return pd.DataFrame(
        rows,
        columns=["adj_close", "shares", "revenue_ttm", "net_income_ttm", "fcf_ttm"],
    )


class TestRatios:
    def test_basic_math(self):
        # price 100, 1000M shares → mcap 100,000 $M; TTM rev 40,000 / NI 8,000 /
        # FCF 5,000 $M.
        out = compute_ratios(_df([[100.0, 1000.0, 40000.0, 8000.0, 5000.0]])).iloc[0]
        assert out["market_cap"] == 100000.0
        assert out["eps_ttm"] == 8.0          # 8000/1000
        assert out["rev_ps_ttm"] == 40.0
        assert out["pe"] == 12.5              # 100000/8000
        assert out["ps"] == 2.5
        assert out["p_fcf"] == 20.0
        assert out["earnings_yield"] == 0.08  # 8000/100000
        assert out["sales_yield"] == 0.4
        assert out["fcf_yield"] == 0.05
        # pe and earnings_yield are reciprocals
        assert out["pe"] * out["earnings_yield"] == 1.0

    def test_negative_earnings_pe_nan_yield_signed(self):
        out = compute_ratios(_df([[100.0, 1000.0, 40000.0, -8000.0, -5000.0]])).iloc[0]
        assert np.isnan(out["pe"])            # loss → PE not meaningful
        assert np.isnan(out["p_fcf"])         # negative FCF → P/FCF NaN
        assert out["earnings_yield"] == -0.08  # signed, finite
        assert out["fcf_yield"] == -0.05
        assert out["ps"] == 2.5               # sales still positive

    def test_zero_denominator_nan(self):
        out = compute_ratios(_df([[100.0, 1000.0, 0.0, 0.0, 0.0]])).iloc[0]
        assert np.isnan(out["pe"])
        assert np.isnan(out["ps"])
        assert np.isnan(out["p_fcf"])
        assert np.isnan(out["eps_ttm"]) is False or out["eps_ttm"] == 0.0

    def test_missing_inputs_propagate_nan(self):
        out = compute_ratios(_df([[100.0, np.nan, 40000.0, 8000.0, 5000.0]])).iloc[0]
        assert np.isnan(out["market_cap"])
        assert np.isnan(out["pe"])
        assert np.isnan(out["eps_ttm"])

    def test_all_ratio_columns_present(self):
        from valuation.ratios import RATIO_COLUMNS
        out = compute_ratios(_df([[100.0, 1000.0, 40000.0, 8000.0, 5000.0]]))
        for c in RATIO_COLUMNS:
            assert c in out.columns
