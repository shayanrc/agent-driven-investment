"""us_fundamentals schema: wide nullable quarterly row + the grid-snap util."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data_pipelines.domains.us_fundamentals.schema import (
    METRIC_COLUMNS,
    US_FUNDAMENTALS_SCHEMA,
    dedupe_grid_collisions,
    quarter_ends,
    snap_to_quarter_end,
)
from data_pipelines.errors import SchemaMismatch


def _row(grid, fiscal_end, filed=None, **metrics):
    base = {
        "date": pd.Timestamp(grid),
        "fiscal_period_end": pd.Timestamp(fiscal_end),
        "filed_date": pd.Timestamp(filed) if filed else pd.NaT,
    }
    for m in METRIC_COLUMNS:
        base[m] = float(metrics.get(m, np.nan))
    return base


def _df(rows):
    df = pd.DataFrame(rows)
    for c in ("date", "fiscal_period_end", "filed_date"):
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    for m in METRIC_COLUMNS:
        df[m] = df[m].astype("float64")
    return df[[c.name for c in US_FUNDAMENTALS_SCHEMA.columns]]


class TestSchema:
    def test_column_names_and_order(self):
        assert US_FUNDAMENTALS_SCHEMA.column_names == [
            "date", "fiscal_period_end", "filed_date",
            "revenue", "net_income", "ocf", "capex", "fcf",
            "shares_basic", "shares_diluted", "eps_basic", "eps_diluted",
        ]

    def test_metric_columns_are_the_float_columns(self):
        assert METRIC_COLUMNS == (
            "revenue", "net_income", "ocf", "capex", "fcf",
            "shares_basic", "shares_diluted", "eps_basic", "eps_diluted",
        )

    def test_accepts_nan_metrics_and_nat_filed_date(self):
        df = _df([_row("2026-03-31", "2026-03-28", revenue=111184.0)])
        US_FUNDAMENTALS_SCHEMA.validate(df)  # no raise

    def test_rejects_nat_grid_date(self):
        df = _df([_row("2026-03-31", "2026-03-28", revenue=1.0)])
        df.loc[0, "date"] = pd.NaT
        with pytest.raises(SchemaMismatch):
            US_FUNDAMENTALS_SCHEMA.validate(df)

    def test_rejects_nat_fiscal_period_end(self):
        df = _df([_row("2026-03-31", "2026-03-28", revenue=1.0)])
        df.loc[0, "fiscal_period_end"] = pd.NaT
        with pytest.raises(SchemaMismatch):
            US_FUNDAMENTALS_SCHEMA.validate(df)


class TestSnap:
    @pytest.mark.parametrize("fiscal_end,expected", [
        # exact grid dates map to themselves
        (date(2026, 3, 31), date(2026, 3, 31)),
        (date(2025, 12, 31), date(2025, 12, 31)),
        # 52/53-week wobble: within 7 days after a grid date → snap back
        (date(2017, 7, 1), date(2017, 6, 30)),   # AAPL fiscal Q3 FY2017
        (date(2023, 7, 1), date(2023, 6, 30)),   # AAPL fiscal Q3 FY2023
        (date(2024, 1, 5), date(2023, 12, 31)),  # 5 days ≤ tolerance
        # beyond tolerance → snap forward
        (date(2024, 1, 8), date(2024, 3, 31)),   # 8 days > tolerance
        (date(2025, 1, 31), date(2025, 3, 31)),  # WMT fiscal Q4
        (date(2025, 4, 30), date(2025, 6, 30)),  # WMT fiscal Q1
        (date(2026, 3, 28), date(2026, 3, 31)),  # AAPL: a few days short
        (date(2025, 11, 30), date(2025, 12, 31)),  # Feb/May/Aug/Nov cycle
    ])
    def test_snap_cases(self, fiscal_end, expected):
        assert snap_to_quarter_end(fiscal_end) == expected

    def test_off_cycle_company_never_collides(self):
        # A Jan/Apr/Jul/Oct filer's four quarters land on four distinct
        # grid dates.
        ends = [date(2025, 1, 31), date(2025, 4, 30),
                date(2025, 7, 31), date(2025, 10, 31)]
        snapped = [snap_to_quarter_end(e) for e in ends]
        assert len(set(snapped)) == 4

    def test_quarter_ends_enumeration(self):
        assert quarter_ends(date(2025, 1, 1), date(2025, 12, 31)) == [
            date(2025, 3, 31), date(2025, 6, 30),
            date(2025, 9, 30), date(2025, 12, 31),
        ]
        # inclusive bounds, clipped start
        assert quarter_ends(date(2025, 6, 30), date(2025, 9, 30)) == [
            date(2025, 6, 30), date(2025, 9, 30),
        ]
        assert quarter_ends(date(2025, 7, 1), date(2025, 6, 30)) == []


class TestDedupeGridCollisions:
    def test_keeps_row_closest_to_grid_date(self):
        # A fiscal-year-change stub and a regular quarter both snapped to
        # 2025-09-30; the regular quarter (closer fiscal end) wins.
        df = _df([
            _row("2025-09-30", "2025-08-02", revenue=1.0),   # 59 days away
            _row("2025-09-30", "2025-09-27", revenue=2.0),   # 3 days away
            _row("2025-12-31", "2025-12-27", revenue=3.0),
        ])
        out = dedupe_grid_collisions(df)
        assert len(out) == 2
        assert out.loc[out["date"] == pd.Timestamp("2025-09-30"),
                       "revenue"].item() == 2.0

    def test_empty_frame_passthrough(self):
        df = _df([_row("2025-12-31", "2025-12-27", revenue=1.0)]).iloc[:0]
        out = dedupe_grid_collisions(df)
        assert out.empty


class TestCacheRoundTrip:
    def test_nan_and_nat_survive_sqlite(self, tmp_path):
        # The fred_macro lesson (adding_a_domain.md): nullable columns must
        # round-trip NaN↔NULL through the cache. This schema adds a nullable
        # *datetime* (filed_date) — NaT must survive too.
        from data_pipelines.cache import read_processed, write_processed_atomic
        from data_pipelines.domains.us_fundamentals import USFundamentalsDomain

        domain = USFundamentalsDomain()
        df = _df([
            _row("2026-03-31", "2026-03-28", filed="2026-05-01",
                 revenue=111184.0, net_income=29578.0, ocf=24000.0,
                 capex=4344.0, fcf=19656.0),
            _row("2025-12-31", "2025-12-27",
                 revenue=143756.0),  # all else NaN, filed_date NaT
        ])
        US_FUNDAMENTALS_SCHEMA.validate(df)
        meta = {
            "schema_version": 1, "row_count": len(df),
            "range": {"start": "2025-12-31", "end": "2026-03-31"},
            "last_fetch_utc": "2026-07-03T00:00:00Z", "sources": [],
        }
        write_processed_atomic(tmp_path, domain, "FUND:AAPL", df, meta)
        back, meta_back = read_processed(tmp_path, domain, "FUND:AAPL")

        assert meta_back is not None
        # SQLite orders by date ascending; align before comparing.
        expected = df.sort_values("date").reset_index(drop=True)
        pd.testing.assert_frame_equal(back, expected)
        # NaT / NaN explicitly preserved
        assert back["filed_date"].isna().sum() == 1
        assert back["ocf"].isna().sum() == 1
