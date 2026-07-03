"""YFinanceFundamentalsAdapter: parse() against a real captured envelope.

The fixture is the adapter's own raw envelope for AAPL captured live on
2026-07-03 (~5 quarters — yfinance's full depth). Cross-provider anchor: the
same quarters as the macrotrends/EDGAR fixtures, so values must agree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipelines.domains.us_fundamentals.adapters.yfinance_fund import (
    YFinanceFundamentalsAdapter,
)
from data_pipelines.domains.us_fundamentals.schema import (
    METRIC_COLUMNS,
    US_FUNDAMENTALS_SCHEMA,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "us_fundamentals"


@pytest.fixture(scope="module")
def aapl() -> pd.DataFrame:
    return YFinanceFundamentalsAdapter().parse(
        FIXTURES / "yfinance_aapl_envelope.json"
    )


class TestParse:
    def test_canonical_schema(self, aapl):
        US_FUNDAMENTALS_SCHEMA.validate(aapl)

    def test_shallow_history_is_expected(self, aapl):
        # tertiary provider: last ~5 quarters only
        assert 3 <= len(aapl) <= 8

    def test_values_agree_with_other_providers(self, aapl):
        row = aapl.loc[aapl["date"] == pd.Timestamp("2026-03-31")].iloc[0]
        assert row["revenue"] == pytest.approx(111184.0)
        assert row["ocf"] == pytest.approx(28702.0)
        # yfinance Capital Expenditure is negative → stored positive
        assert row["capex"] == pytest.approx(1971.0)
        assert row["fcf"] == pytest.approx(26731.0)
        assert row["shares_diluted"] == pytest.approx(14725.9, rel=1e-3)
        assert row["eps_diluted"] == pytest.approx(2.01, abs=0.01)

    def test_all_nan_padding_rows_dropped(self, aapl):
        # yfinance pads a trailing empty quarter; it must not survive to
        # claim grid coverage.
        assert not aapl[list(METRIC_COLUMNS)].isna().all(axis=1).any()

    def test_no_filed_date(self, aapl):
        assert aapl["filed_date"].isna().all()
