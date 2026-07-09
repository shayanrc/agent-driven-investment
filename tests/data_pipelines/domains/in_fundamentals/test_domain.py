"""in_fundamentals domain: registry routing, identifier parsing, universe
derivation, merge policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipelines.domain import DomainRegistry
from data_pipelines.domains.in_fundamentals import get_domain
from data_pipelines.domains.in_fundamentals.registry import parse_identifier
from data_pipelines.domains.in_fundamentals.universe import load_universe


class TestIdentifier:
    def test_parse_basic(self):
        assert parse_identifier("INFUND:RELIANCE") == ("-", "RELIANCE")

    def test_parse_lowercase_prefix_and_symbol(self):
        assert parse_identifier("infund:reliance") == ("-", "RELIANCE")

    def test_missing_prefix_rejected(self):
        with pytest.raises(ValueError, match="missing domain prefix"):
            parse_identifier("RELIANCE")

    def test_wrong_prefix_rejected(self):
        with pytest.raises(ValueError, match="unknown in_fundamentals prefix"):
            parse_identifier("FUND:RELIANCE")

    def test_empty_symbol_rejected(self):
        with pytest.raises(ValueError, match="empty ticker"):
            parse_identifier("INFUND: ")


class TestRegistry:
    def test_registry_resolves_infund(self):
        # The conftest resets the registry between tests; re-register the
        # singleton the way the CLI import path does (us_fundamentals
        # test_domain.py precedent).
        DomainRegistry.register(get_domain())
        assert DomainRegistry.resolve("INFUND:RELIANCE").name == "in_fundamentals"

    def test_us_fund_prefix_untouched(self):
        from data_pipelines.domains.us_fundamentals import (
            get_domain as get_us_domain,
        )
        DomainRegistry.register(get_us_domain())
        DomainRegistry.register(get_domain())
        assert DomainRegistry.resolve("FUND:AAPL").name == "us_fundamentals"
        assert DomainRegistry.resolve("INFUND:RELIANCE").name == "in_fundamentals"

    def test_schema_has_consolidated_appended_last(self):
        cols = get_domain().schema.column_names
        assert cols[-1] == "consolidated"
        # shared positional prefix with us_fundamentals
        from data_pipelines.domains.us_fundamentals.schema import (
            US_FUNDAMENTALS_SCHEMA,
        )
        assert tuple(cols[:-1]) == tuple(US_FUNDAMENTALS_SCHEMA.column_names)


class TestUniverse:
    def test_nifty50_derivation(self):
        idents = load_universe("nifty50")
        assert idents == sorted(set(idents))
        assert all(i.startswith("INFUND:") for i in idents)
        assert "INFUND:RELIANCE" in idents
        assert not any("NIFTY" in i.split(":", 1)[1] for i in idents)
        assert 40 <= len(idents) <= 60

    def test_nifty500_is_superset_of_nifty50(self):
        small = set(load_universe("nifty50"))
        big = set(load_universe("nifty500"))
        assert len(small - big) <= 2  # rebalance drift tolerance
        assert len(big) > 400


class TestMergeOverlap:
    def _frame(self, dates, **overrides):
        n = len(dates)
        base = {
            "date": pd.to_datetime(dates),
            "fiscal_period_end": pd.to_datetime(dates),
            "filed_date": pd.to_datetime([None] * n),
            "revenue": [np.nan] * n,
            "net_income": [np.nan] * n,
            "ocf": [np.nan] * n,
            "capex": [np.nan] * n,
            "fcf": [np.nan] * n,
            "shares_basic": [np.nan] * n,
            "shares_diluted": [np.nan] * n,
            "eps_basic": [np.nan] * n,
            "eps_diluted": [np.nan] * n,
            "consolidated": [np.nan] * n,
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_first_written_wins_and_hole_fill(self):
        existing = self._frame(
            ["2024-12-31"], revenue=[100.0],
            filed_date=pd.to_datetime(["2025-01-16"]),
        )
        new = self._frame(
            ["2024-12-31"], revenue=[999.0], net_income=[50.0],
        )
        merged = get_domain().merge_overlap(existing, new, [], {})
        assert merged.loc[0, "revenue"] == 100.0        # existing kept
        assert merged.loc[0, "net_income"] == 50.0      # hole filled
        assert merged.loc[0, "filed_date"] == pd.Timestamp("2025-01-16")

    def test_fcf_stays_nan_without_cashflow(self):
        existing = self._frame(["2024-12-31"], revenue=[100.0])
        new = self._frame(["2025-03-31"], revenue=[110.0])
        merged = get_domain().merge_overlap(existing, new, [], {})
        assert merged["fcf"].isna().all()
        assert list(merged["date"]) == [
            pd.Timestamp("2024-12-31"), pd.Timestamp("2025-03-31"),
        ]
