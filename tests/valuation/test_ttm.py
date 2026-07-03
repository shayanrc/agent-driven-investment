"""ttm — point-in-time trailing-twelve-month engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from valuation.ttm import asof_daily, build_ttm_timeline


def _q(fe, filed, revenue=None, net_income=None, fcf=None, shares=None):
    n = len(fe)
    def col(v, default):
        return [default] * n if v is None else v
    return pd.DataFrame({
        "fiscal_period_end": pd.to_datetime(fe),
        "filed_date": pd.to_datetime(filed),
        "revenue": col(revenue, 100.0),
        "net_income": col(net_income, 20.0),
        "fcf": col(fcf, 15.0),
        "shares": col(shares, 1000.0),
    })


class TestBuildTTM:
    def test_sums_trailing_four_quarters(self):
        q = _q(
            ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"],
            ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-15", "2025-05-01"],
            revenue=[10, 20, 30, 40, 50],
        )
        tl = build_ttm_timeline(q)
        # first snapshot at the 4th quarter (2024-12-31): 10+20+30+40 = 100
        assert len(tl) == 2
        first = tl.iloc[0]
        assert first["asof_fiscal_period_end"] == pd.Timestamp("2024-12-31")
        assert first["revenue_ttm"] == 100.0
        # effective on the newest filing in the window
        assert first["effective_date"] == pd.Timestamp("2025-02-15")
        # second snapshot rolls: 20+30+40+50 = 140, shares from newest quarter
        assert tl.iloc[1]["revenue_ttm"] == 140.0

    def test_fewer_than_four_quarters_empty(self):
        q = _q(["2024-03-31", "2024-06-30", "2024-09-30"],
               ["2024-05-01", "2024-08-01", "2024-11-01"])
        assert build_ttm_timeline(q).empty

    def test_undated_quarter_blocks_windows_containing_it(self):
        # Q3 filed_date is NaT → any window including it is dropped.
        q = _q(
            ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"],
            ["2024-05-01", "2024-08-01", None, "2025-02-15", "2025-05-01"],
        )
        tl = build_ttm_timeline(q)
        # windows ending 2024-12-31 and 2025-03-31 both contain the NaT Q3 → none
        assert tl.empty

    def test_missing_quarter_gap_rejected(self):
        # a ~6-month gap (missing 2024-06-30) → the window isn't 4 consecutive
        q = _q(
            ["2023-12-31", "2024-03-31", "2024-09-30", "2024-12-31"],
            ["2024-02-15", "2024-05-01", "2024-11-01", "2025-02-15"],
        )
        assert build_ttm_timeline(q).empty

    def test_metric_nan_if_any_component_missing(self):
        q = _q(
            ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"],
            ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-15"],
            revenue=[10, np.nan, 30, 40], net_income=[1, 2, 3, 4],
        )
        tl = build_ttm_timeline(q)
        assert np.isnan(tl.iloc[0]["revenue_ttm"])       # one component NaN
        assert tl.iloc[0]["net_income_ttm"] == 10.0      # all present → summed


class TestAsofDaily:
    def test_forward_fills_and_respects_effective_date(self):
        q = _q(
            ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"],
            ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-15", "2025-05-01"],
            revenue=[10, 20, 30, 40, 50],
        )
        tl = build_ttm_timeline(q)
        days = pd.Series(pd.to_datetime([
            "2025-01-01",  # before first effective (2025-02-15) → NaN
            "2025-02-15",  # exactly on → first snapshot
            "2025-03-20",  # still first snapshot (next effective is 2025-05-01)
            "2025-06-01",  # second snapshot
        ]))
        out = asof_daily(tl, days)
        assert np.isnan(out.iloc[0]["revenue_ttm"])
        assert out.iloc[1]["revenue_ttm"] == 100.0
        assert out.iloc[2]["revenue_ttm"] == 100.0
        assert out.iloc[3]["revenue_ttm"] == 140.0

    def test_empty_timeline_gives_all_nan(self):
        tl = build_ttm_timeline(_q(["2024-03-31"], ["2024-05-01"]))
        out = asof_daily(tl, pd.Series(pd.to_datetime(["2025-01-01"])))
        assert np.isnan(out.iloc[0]["revenue_ttm"])
