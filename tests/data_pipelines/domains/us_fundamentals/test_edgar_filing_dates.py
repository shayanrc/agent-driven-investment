"""EdgarAdapter.filing_dates — SEC submissions API → reportDate→filingDate map.

Uses a small synthetic submissions doc (the real shape: parallel form /
filingDate / reportDate arrays under filings.recent, with older filings in
paginated filings.files[]).
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import patch

import pytest

from data_pipelines.domains.us_fundamentals.adapters.edgar import EdgarAdapter


def _submissions(recent, files=None):
    return {
        "cik": 1652044, "name": "TESTCO",
        "filings": {"recent": recent, "files": files or []},
    }


RECENT = {
    "form":       ["10-Q", "10-K",       "10-K/A",     "8-K",        "10-Q"],
    "filingDate": ["2025-04-25", "2025-02-05", "2025-06-30", "2025-01-10", "2024-10-30"],
    "reportDate": ["2025-03-31", "2024-12-31", "2024-12-31", "2025-01-09", "2024-09-30"],
}


class TestFilingDates:
    def _adapter(self):
        a = EdgarAdapter()
        a._cik_map_cache = {"TESTCO": 1652044}
        return a

    def test_maps_report_to_earliest_filing(self, tmp_path):
        adapter = self._adapter()
        payload = json.dumps(_submissions(RECENT)).encode()
        with patch.object(adapter, "_get", return_value=payload):
            out = adapter.filing_dates("FUND:TESTCO", data_root=tmp_path)
        # 10-Q and 10-K periods mapped
        assert out[date(2025, 3, 31)] == date(2025, 4, 25)
        assert out[date(2024, 9, 30)] == date(2024, 10, 30)
        # Q4 comes from the 10-K's own filing date (exact, not derived)
        assert out[date(2024, 12, 31)] == date(2025, 2, 5)

    def test_amendment_loses_to_original_earliest_filed(self, tmp_path):
        # The 10-K/A (filed 2025-06-30) restates the same 2024-12-31 period as
        # the original 10-K (filed 2025-02-05); earliest-filed keeps the 10-K.
        adapter = self._adapter()
        payload = json.dumps(_submissions(RECENT)).encode()
        with patch.object(adapter, "_get", return_value=payload):
            out = adapter.filing_dates("FUND:TESTCO", data_root=tmp_path)
        assert out[date(2024, 12, 31)] == date(2025, 2, 5)

    def test_non_periodic_forms_excluded(self, tmp_path):
        # The 8-K (reportDate 2025-01-09) must not appear.
        adapter = self._adapter()
        payload = json.dumps(_submissions(RECENT)).encode()
        with patch.object(adapter, "_get", return_value=payload):
            out = adapter.filing_dates("FUND:TESTCO", data_root=tmp_path)
        assert date(2025, 1, 9) not in out

    def test_paginated_files_merged(self, tmp_path):
        # Older filings live in a separate file; both blocks contribute.
        adapter = self._adapter()
        main = _submissions(RECENT, files=[{"name": "CIK-submissions-001.json"}])
        older = {
            "form": ["10-Q"], "filingDate": ["2023-07-25"],
            "reportDate": ["2023-06-30"],
        }
        def fake_get(url, identifier):
            if "submissions-001" in url:
                return json.dumps(older).encode()
            return json.dumps(main).encode()
        with patch.object(adapter, "_get", side_effect=fake_get):
            out = adapter.filing_dates("FUND:TESTCO", data_root=tmp_path)
        assert out[date(2023, 6, 30)] == date(2023, 7, 25)
        assert out[date(2024, 12, 31)] == date(2025, 2, 5)

    def test_no_cik_returns_empty(self, tmp_path):
        adapter = EdgarAdapter()
        adapter._cik_map_cache = {}  # ticker not an SEC filer
        out = adapter.filing_dates("FUND:NOPE", data_root=tmp_path)
        assert out == {}

    def test_lands_raw_audit_copy(self, tmp_path):
        adapter = self._adapter()
        payload = json.dumps(_submissions(RECENT)).encode()
        with patch.object(adapter, "_get", return_value=payload):
            adapter.filing_dates("FUND:TESTCO", data_root=tmp_path)
        raws = list(
            (tmp_path / "raw" / "edgar_submissions").rglob("*.json")
        )
        assert len(raws) == 1
