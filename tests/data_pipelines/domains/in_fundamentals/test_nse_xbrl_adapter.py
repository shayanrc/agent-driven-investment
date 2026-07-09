"""nse_xbrl adapter parse(): fixture-driven, no network.

The fixture is a real envelope captured live 2026-07-09: the RELIANCE
quarterly-results metadata (6 records, both bases) + the Q3 FY25
(quarter end 2024-12-31, standalone, filed 2025-01-16 20:20) Ind-AS XBRL
instance. Only that one record has XML — the others exercise the
skip-records-without-attachments path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipelines.domains.in_fundamentals.adapters.nse_xbrl import (
    NSEXbrlAdapter,
)
from data_pipelines.domains.in_fundamentals.schema import (
    IN_FUNDAMENTALS_SCHEMA,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "in_fundamentals" / "nse_envelope_reliance.json"
)


@pytest.fixture(scope="module")
def parsed() -> pd.DataFrame:
    return NSEXbrlAdapter().parse(FIXTURE)


class TestParse:
    def test_one_row_only_records_with_xml(self, parsed):
        # 6 metadata records, 1 XBRL attachment → exactly 1 row.
        assert len(parsed) == 1

    def test_canonical_columns(self, parsed):
        assert list(parsed.columns) == list(IN_FUNDAMENTALS_SCHEMA.column_names)

    def test_grid_date_is_quarter_end(self, parsed):
        assert parsed.loc[0, "date"] == pd.Timestamp("2024-12-31")
        assert parsed.loc[0, "fiscal_period_end"] == pd.Timestamp("2024-12-31")

    def test_quarter_context_not_ytd(self, parsed):
        # RIL Q3 FY25 standalone revenue = 1,282,600 INR-M (the YTD context
        # in the same instance carries 3,966,450 — picking it would mean the
        # context filter failed).
        assert parsed.loc[0, "revenue"] == pytest.approx(1_282_600.0)

    def test_net_income_inr_millions(self, parsed):
        assert parsed.loc[0, "net_income"] == pytest.approx(87_210.0)

    def test_filed_date_native(self, parsed):
        assert parsed.loc[0, "filed_date"] == pd.Timestamp("2025-01-16")

    def test_standalone_flag(self, parsed):
        assert parsed.loc[0, "consolidated"] == 0.0

    def test_eps_present(self, parsed):
        assert parsed.loc[0, "eps_diluted"] > 0

    def test_shares_derived_weighted_consistent(self, parsed):
        row = parsed.loc[0]
        assert row["shares_diluted"] == pytest.approx(
            row["net_income"] / row["eps_diluted"]
        )
        # RIL has ~13.5B shares → ~13,500 M (sanity band)
        assert 10_000 < row["shares_diluted"] < 20_000

    def test_no_quarterly_cashflow_in_india(self, parsed):
        assert parsed[["ocf", "capex", "fcf"]].isna().all().all()

    def test_deterministic_reparse(self, parsed):
        again = NSEXbrlAdapter().parse(FIXTURE)
        pd.testing.assert_frame_equal(parsed, again)


class TestBasisSelection:
    def test_consolidated_preferred_when_both_have_xml(self, tmp_path):
        import json
        doc = json.loads(FIXTURE.read_text())
        xml = doc["xbrl"]["1189823"]
        cons = next(
            r for r in doc["metadata"]
            if r["toDate"] == "31-Dec-2024"
            and r["consolidated"] == "Consolidated"
        )
        # give the consolidated sibling the same XML payload
        doc["xbrl"][str(cons["seqNumber"])] = xml
        p = tmp_path / "env.json"
        p.write_text(json.dumps(doc))
        df = NSEXbrlAdapter().parse(p)
        assert len(df) == 1
        assert df.loc[0, "consolidated"] == 1.0


class TestUndefinedContextConvention:
    """2018-2021-era instances reference contextRef="OneD" without defining
    it (filing-tool quirk). The positional column convention (OneD = current
    quarter) is the fallback — live regression fixture from the pilot."""

    FIXTURE_2018 = FIXTURE.parent / "nse_envelope_reliance_2018.json"

    def test_2018_instance_parses_via_convention(self):
        df = NSEXbrlAdapter().parse(self.FIXTURE_2018)
        assert len(df) == 1
        row = df.loc[0]
        assert row["date"] == pd.Timestamp("2018-06-30")
        assert row["revenue"] == pytest.approx(1_330_690.0)
        assert row["net_income"] == pytest.approx(94_850.0)
        assert row["filed_date"] == pd.Timestamp("2018-08-07")
        assert row["consolidated"] == 1.0
        assert 5_000 < row["shares_diluted"] < 7_000


class TestIntegratedFilingStream:
    """SEBI's Integrated Filing regime replaced the classic quarterly stream
    from Q4 FY25 — records come from a second endpoint with different field
    names and interleaved governance filings. Live TCS Q1 FY26 capture."""

    FIXTURE_INT = FIXTURE.parent / "nse_envelope_tcs_integrated.json"

    def test_integrated_record_parses(self):
        df = NSEXbrlAdapter().parse(self.FIXTURE_INT)
        assert len(df) == 1
        row = df.loc[0]
        assert row["date"] == pd.Timestamp("2025-06-30")
        assert row["revenue"] == pytest.approx(634_370.0)
        assert row["net_income"] == pytest.approx(128_190.0)
        assert row["consolidated"] == 1.0
        assert row["filed_date"] == pd.Timestamp("2025-07-10")

    def test_governance_records_skipped(self):
        import json
        doc = json.loads(self.FIXTURE_INT.read_text())
        types = {r["type"] for r in doc["integrated_metadata"]}
        assert "Integrated Filing- Governance" in types  # present in fixture
        df = NSEXbrlAdapter().parse(self.FIXTURE_INT)
        assert len(df) == 1  # ...but never parsed


class TestInsuranceTaxonomies:
    """Life (LI) and general (GI) insurers file premium income instead of
    RevenueFromOperations and ...AfterTax profit variants; they carry no EPS
    facts and no trustworthy share count, so shares/EPS stay honestly NaN.
    Live Q4 FY26 captures: SBILIFE (LI), ICICIGI (GI)."""

    FIXTURE_LI = FIXTURE.parent / "nse_envelope_sbilife_li.json"
    FIXTURE_GI = FIXTURE.parent / "nse_envelope_icicigi_gi.json"

    def test_life_insurer_parses(self):
        df = NSEXbrlAdapter().parse(self.FIXTURE_LI)
        assert len(df) == 1
        row = df.loc[0]
        assert row["date"] == pd.Timestamp("2026-03-31")
        assert row["revenue"] == pytest.approx(279_388.6)   # gross premium
        assert row["net_income"] == pytest.approx(8_046.4)
        assert row["filed_date"] == pd.Timestamp("2026-04-22")
        assert row["consolidated"] == 0.0                   # standalone

    def test_general_insurer_parses(self):
        df = NSEXbrlAdapter().parse(self.FIXTURE_GI)
        assert len(df) == 1
        row = df.loc[0]
        assert row["date"] == pd.Timestamp("2026-03-31")
        assert row["revenue"] == pytest.approx(80_737.0)    # gross premiums written
        assert row["net_income"] == pytest.approx(5_465.6)
        assert row["filed_date"] == pd.Timestamp("2026-04-15")

    def test_insurer_shares_eps_honestly_nan(self):
        for fx in (self.FIXTURE_LI, self.FIXTURE_GI):
            row = NSEXbrlAdapter().parse(fx).loc[0]
            assert pd.isna(row["eps_basic"]) and pd.isna(row["eps_diluted"])
            assert pd.isna(row["shares_basic"]) and pd.isna(row["shares_diluted"])
