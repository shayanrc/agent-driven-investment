"""EdgarAdapter: XBRL normalization against the real (trimmed) AAPL fixture.

The fixture is AAPL's companyfacts JSON captured 2026-07-03, trimmed to the
tags in the priority lists. The strongest checks are CROSS-PROVIDER: EDGAR's
recovered quarters (including YTD-differenced OCF/capex and FY−3QYTD Q4s)
must agree with macrotrends' independently published values for the same
quarters.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data_pipelines.domains.us_fundamentals.adapters.edgar import (
    EdgarAdapter,
    _dedupe_periods,
)
from data_pipelines.domains.us_fundamentals.schema import US_FUNDAMENTALS_SCHEMA

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "us_fundamentals"


@pytest.fixture(scope="module")
def aapl() -> pd.DataFrame:
    return EdgarAdapter().parse(
        FIXTURES / "edgar_aapl_companyfacts_trimmed.json"
    )


def _row(df: pd.DataFrame, grid: str) -> pd.Series:
    hit = df.loc[df["date"] == pd.Timestamp(grid)]
    assert len(hit) == 1, f"expected exactly one row at grid {grid}"
    return hit.iloc[0]


class TestParse:
    def test_canonical_schema(self, aapl):
        US_FUNDAMENTALS_SCHEMA.validate(aapl)

    def test_deep_history_every_quarter_recovered(self, aapl):
        # design analysis: 71-72 quarters 2008 → 2026, 4/4 per fiscal year
        assert len(aapl) >= 68
        assert aapl["date"].min() <= pd.Timestamp("2009-06-30")
        assert aapl["date"].max() >= pd.Timestamp("2026-03-31")
        assert aapl["date"].is_unique and aapl["date"].is_monotonic_increasing

    def test_latest_quarter_direct_values(self, aapl):
        row = _row(aapl, "2026-03-31")
        # fiscal end is the TRUE period end; grid is snapped
        assert row["fiscal_period_end"] == pd.Timestamp("2026-03-28")
        assert row["revenue"] == pytest.approx(111184.0)
        assert row["net_income"] == pytest.approx(29578.0)
        # point-in-time: the Q2 FY2026 10-Q hit EDGAR on 2026-05-01
        assert row["filed_date"] == pd.Timestamp("2026-05-01")

    def test_ytd_differenced_ocf_capex_match_macrotrends(self, aapl):
        # AAPL 10-Q cash-flow statements are YTD-only: Q2 FY2026 OCF/capex
        # exist ONLY via H1 − Q1 differencing. Macrotrends publishes the
        # discrete quarter independently: OCF 28,702 / capex 1,971 / Q1
        # 53,925 & 2,373 ($M).
        q2 = _row(aapl, "2026-03-31")
        assert q2["ocf"] == pytest.approx(28702.0)
        assert q2["capex"] == pytest.approx(1971.0)
        assert q2["fcf"] == pytest.approx(28702.0 - 1971.0)
        q1 = _row(aapl, "2025-12-31")
        assert q1["ocf"] == pytest.approx(53925.0)
        assert q1["capex"] == pytest.approx(2373.0)

    def test_q4_from_fy_minus_3q_ytd_matches_macrotrends(self, aapl):
        # NI for fiscal Q4s FY2021+ is never directly reported (SEC dropped
        # the Item 302 quarterly footnote) — recovered as FY − 3Q-YTD.
        # Macrotrends independently publishes Q4 FY2025 NI = 27,466 $M
        # (quarter ending 2025-09-27 → grid 2025-09-30).
        q4 = _row(aapl, "2025-09-30")
        assert q4["net_income"] == pytest.approx(27466.0)
        assert q4["revenue"] == pytest.approx(102466.0)

    def test_q4_shares_derived_and_eps_filled(self, aapl):
        # Q4 weighted shares: 4×FY − ΣQ1..3 (macrotrends just recycles the
        # FY average, 15,004.7 M; the derived value must sit near it AND
        # continue the buyback-driven within-year decline below Q3's
        # 14,948 M). EPS filled from NI/shares ≈ the reported 1.84.
        q4 = _row(aapl, "2025-09-30")
        q3 = _row(aapl, "2025-06-30")
        assert q4["shares_diluted"] == pytest.approx(15004.7, rel=0.02)
        assert q4["shares_diluted"] < q3["shares_diluted"]
        assert q4["eps_diluted"] == pytest.approx(1.84, abs=0.02)

    def test_scaling_is_millions(self, aapl):
        # raw USD → $M: revenue in the 1e4-1e5 range, never 1e10+
        assert aapl["revenue"].max() < 1e6
        assert aapl["revenue"].max() > 1e5  # AAPL holiday quarters >$100B

    def test_filed_date_always_present_and_causal(self, aapl):
        assert aapl["filed_date"].notna().all()
        # a filing always lands AFTER its fiscal period ends
        assert (aapl["filed_date"] > aapl["fiscal_period_end"]).all()


class TestPointInTimeDedup:
    def test_earliest_filed_wins(self):
        # Two filings report the same (start, end): the original 10-Q and a
        # later comparative with a restated value. Earliest filed wins.
        pool = [
            (date(2017, 10, 1), date(2017, 12, 30), 88293.0,
             date(2019, 1, 30), 0, "later-comparative"),
            (date(2017, 10, 1), date(2017, 12, 30), 88293.0,
             date(2018, 2, 2), 2, "original-10q"),
        ]
        out = _dedupe_periods(pool)
        val, filed = out[(date(2017, 10, 1), date(2017, 12, 30))]
        assert filed == date(2018, 2, 2)

    def test_tag_rank_breaks_same_filing_ties(self):
        # Same filing carries two tags for the period → lower rank wins.
        pool = [
            (date(2018, 1, 1), date(2018, 3, 31), 100.0,
             date(2018, 5, 1), 4, "accn-1"),   # Revenues (rank 4)
            (date(2018, 1, 1), date(2018, 3, 31), 90.0,
             date(2018, 5, 1), 0, "accn-1"),   # RevenueFromContract (rank 0)
        ]
        out = _dedupe_periods(pool)
        val, _ = out[(date(2018, 1, 1), date(2018, 3, 31))]
        assert val == 90.0

    def test_asc606_transition_keeps_original_filing(self, aapl):
        # The quarter ending 2017-12-30 appears in FOUR filings in the
        # fixture: the original 10-Q (SalesRevenueNet, filed 2018-02-02), a
        # same-tag comparative (2018-11-05), and two post-ASC-606
        # comparatives under the higher-priority RevenueFromContract tag
        # (2019-01-30, 2019-10-31). Naive prefer-first-tag would pick the
        # 2019 comparative and destroy the point-in-time filed_date;
        # earliest-filed keeps the original 10-Q.
        row = aapl.loc[
            aapl["fiscal_period_end"] == pd.Timestamp("2017-12-30")
        ].iloc[0]
        assert row["revenue"] == pytest.approx(88293.0)
        assert row["filed_date"] == pd.Timestamp("2018-02-02")


class TestEmptyAndMissing:
    def test_empty_facts_yield_empty_frame(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text('{"cik": 1, "entityName": "X", "facts": {}}')
        df = EdgarAdapter().parse(p)
        assert df.empty
        assert list(df.columns) == US_FUNDAMENTALS_SCHEMA.column_names
