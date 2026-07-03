"""resolve_filed_dates — the pure matching core of the filed_date enrichment.

Guards the two invariants that off-calendar fiscal filers (CAVA's 13-week
restaurant quarters, FERG's July fiscal year) exposed: a match needs both
reportDate↔fiscal-end proximity AND the causal rule (filing after period end),
and anything unconfirmed is cleared to NaT (never a stale companyfacts guess).
"""

from __future__ import annotations

import pandas as pd

from scripts.data_pipelines.enrich_fundamentals_filed_date import (
    resolve_filed_dates,
)


def _cached(rows):
    df = pd.DataFrame(rows, columns=["date", "fiscal_period_end", "filed_date"])
    for c in df.columns:
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    return df


class TestResolveFiledDates:
    def test_clean_calendar_filer_matches(self):
        cached = _cached([
            ("2025-12-31", "2025-12-31", None),
            ("2026-03-31", "2026-03-31", None),
        ])
        subs = {"2025-12-31": "2026-02-05", "2026-03-31": "2026-04-30"}
        out = resolve_filed_dates(cached, subs, 20)
        assert out.iloc[0] == pd.Timestamp("2026-02-05")
        assert out.iloc[1] == pd.Timestamp("2026-04-30")

    def test_off_calendar_true_fiscal_end_matches(self):
        # FERG-style: macrotrends stored the TRUE fiscal end (Jul 31), the
        # 10-Q reportDate matches it → confirmed.
        cached = _cached([("2025-09-30", "2025-07-31", None)])
        subs = {"2025-07-31": "2025-09-26"}
        out = resolve_filed_dates(cached, subs, 20)
        assert out.iloc[0] == pd.Timestamp("2025-09-26")

    def test_calendar_normalized_fiscal_end_clears(self):
        # CAVA-style: macrotrends mislabeled a ~Apr-20 quarter as 2025-06-30.
        # The only filing snapping to that grid (reportDate 2025-04-20) is 71
        # days from the labeled fiscal end → no confirmation → NaT, and any
        # stale prior value is cleared.
        cached = _cached([("2025-06-30", "2025-06-30", "2025-05-16")])
        subs = {"2025-04-20": "2025-05-16"}
        out = resolve_filed_dates(cached, subs, 20)
        assert pd.isna(out.iloc[0])

    def test_causal_invariant_rejects_filing_before_period_end(self):
        # Even within proximity, a filing dated on/before the fiscal end is a
        # mis-snap and must be rejected.
        cached = _cached([("2025-06-30", "2025-06-30", None)])
        subs = {"2025-06-25": "2025-06-20"}  # "filed" before period end
        out = resolve_filed_dates(cached, subs, 20)
        assert pd.isna(out.iloc[0])

    def test_unconfirmed_stale_value_is_cleared(self):
        # No submissions entry for this grid → authoritative rewrite clears the
        # existing (companyfacts residual) date.
        cached = _cached([("2013-03-31", "2013-03-31", "2014-05-01")])
        out = resolve_filed_dates(cached, {}, 20)
        assert pd.isna(out.iloc[0])

    def test_fiscal_year_collision_picks_closest(self):
        # Two reportDates snap to the same grid (fiscal-year-change stub);
        # the one closest to the row's fiscal end wins.
        cached = _cached([("2025-12-31", "2025-12-31", None)])
        subs = {"2025-12-28": "2026-02-25", "2025-11-15": "2025-12-20"}
        out = resolve_filed_dates(cached, subs, 20)
        assert out.iloc[0] == pd.Timestamp("2026-02-25")
