"""MacrotrendsAdapter: parse() against real captured page data (AAPL + WMT).

Fixtures are raw-envelope JSONs whose pages hold the REAL ``originalData``
blobs captured from macrotrends on 2026-07-03, wrapped in minimal HTML (page
chrome trimmed). Spot-check values below are as published on that date.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from data_pipelines.domains.us_fundamentals.adapters.macrotrends import (
    MacrotrendsAdapter,
)
from data_pipelines.domains.us_fundamentals.schema import US_FUNDAMENTALS_SCHEMA
from data_pipelines.errors import EmptyPayload

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "us_fundamentals"


@pytest.fixture(scope="module")
def aapl() -> pd.DataFrame:
    return MacrotrendsAdapter().parse(FIXTURES / "macrotrends_aapl_envelope.json")


@pytest.fixture(scope="module")
def wmt() -> pd.DataFrame:
    return MacrotrendsAdapter().parse(FIXTURES / "macrotrends_wmt_envelope.json")


class TestParseAAPL:
    def test_canonical_schema(self, aapl):
        US_FUNDAMENTALS_SCHEMA.validate(aapl)

    def test_deep_history(self, aapl):
        # 59 quarters 2011-09-30 → 2026-03-31 as captured
        assert len(aapl) >= 55
        assert aapl["date"].min() == pd.Timestamp("2011-09-30")
        assert aapl["date"].max() >= pd.Timestamp("2026-03-31")

    def test_known_values_latest_quarter(self, aapl):
        row = aapl.loc[aapl["date"] == pd.Timestamp("2026-03-31")].iloc[0]
        assert row["revenue"] == pytest.approx(111184.0)
        assert row["ocf"] == pytest.approx(28702.0)
        # PP&E change -1971 → capex +1971, fcf = 28702 - 1971
        assert row["capex"] == pytest.approx(1971.0)
        assert row["fcf"] == pytest.approx(26731.0)
        # fiscal period end is the macrotrends month-end date, grid snapped
        assert row["fiscal_period_end"] == pd.Timestamp("2026-03-31")

    def test_shares_in_millions(self, aapl):
        # AAPL diluted share count ~14-17B over the window → 14,000-17,500 M
        recent = aapl.loc[aapl["date"] >= pd.Timestamp("2020-01-01")]
        assert recent["shares_diluted"].between(14000, 18000).all()

    def test_no_filed_date(self, aapl):
        assert aapl["filed_date"].isna().all()

    def test_grid_dates_unique_and_sorted(self, aapl):
        assert aapl["date"].is_unique
        assert aapl["date"].is_monotonic_increasing


class TestParseWMTOffCycleFiscal:
    def test_fiscal_month_ends_snap_forward(self, wmt):
        US_FUNDAMENTALS_SCHEMA.validate(wmt)
        # WMT dates quarters at Jan/Apr/Jul/Oct month-ends; every grid date
        # must land on the calendar quarter-end grid with no collisions.
        months = set(wmt["date"].dt.month)
        assert months <= {3, 6, 9, 12}
        assert wmt["date"].is_unique
        # the true fiscal end survives: Jan-31-ending quarters → Mar-31 grid
        jan_rows = wmt.loc[wmt["fiscal_period_end"].dt.month == 1]
        assert not jan_rows.empty
        assert (jan_rows["date"].dt.month == 3).all()

    def test_quarter_ending_2026_01_31(self, wmt):
        row = wmt.loc[
            wmt["fiscal_period_end"] == pd.Timestamp("2026-01-31")
        ].iloc[0]
        assert row["date"] == pd.Timestamp("2026-03-31")
        assert row["revenue"] > 100000  # WMT holiday quarter, ~$190B


class TestFetchPlumbing:
    def _adapter_with_slug_map(self, slug_map):
        adapter = MacrotrendsAdapter()
        adapter._slug_map_cache = slug_map
        return adapter

    def test_unknown_ticker_raises_empty_payload(self, tmp_path):
        adapter = self._adapter_with_slug_map({"AAPL": "AAPL/apple"})
        with pytest.raises(EmptyPayload):
            adapter.fetch("FUND:ZZZTOP", data_root=tmp_path)

    def test_dash_ticker_resolves_dot_slug(self, tmp_path):
        # Universe spelling BRK-B ↔ macrotrends spelling BRK.B
        adapter = self._adapter_with_slug_map(
            {"BRK.B": "BRK.B/berkshire-hathaway"}
        )
        pages = {
            page: "<script>var originalData = [];</script>"
            for page in ("income-statement", "cash-flow-statement")
        }

        def fake_get(url, identifier):
            page = url.split("/")[-1].split("?")[0]
            # a page WITH data so fetch proceeds
            return (
                b'<script>var originalData = [{"field_name": "Revenue", '
                b'"popup_icon": "", "2025-12-31": "1.0"}];</script>'
            )

        with patch.object(adapter, "_get", side_effect=fake_get):
            raw = adapter.fetch("FUND:BRK-B", data_root=tmp_path)
        doc = json.loads(raw.read_text())
        assert doc["slug"] == "BRK.B/berkshire-hathaway"
        assert set(doc["pages"]) == {"income-statement", "cash-flow-statement"}
        # raw lands under the dash (identifier) spelling
        assert "/BRK-B/" in str(raw)

    def test_page_without_original_data_raises_empty_payload(self, tmp_path):
        adapter = self._adapter_with_slug_map({"AAPL": "AAPL/apple"})
        with patch.object(
            adapter, "_get", return_value=b"<html>no table here</html>"
        ):
            with pytest.raises(EmptyPayload):
                adapter.fetch("FUND:AAPL", data_root=tmp_path)

    def test_parse_roundtrip_through_fetch_envelope(self, tmp_path):
        # fetch writes an envelope parse() can consume (D8 reprocess path).
        adapter = self._adapter_with_slug_map({"AAPL": "AAPL/apple"})
        real = (FIXTURES / "macrotrends_aapl_envelope.json").read_text()
        pages = json.loads(real)["pages"]

        def fake_get(url, identifier):
            page = url.split("/")[-1].split("?")[0]
            return pages[page].encode()

        with patch.object(adapter, "_get", side_effect=fake_get):
            raw = adapter.fetch("FUND:AAPL", data_root=tmp_path)
        df = adapter.parse(raw)
        US_FUNDAMENTALS_SCHEMA.validate(df)
        assert len(df) >= 55
