"""USFundamentalsDomain wiring: registry, identifier parsing, universe, CLI."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from data_pipelines.domain import DomainRegistry
from data_pipelines.domains.us_fundamentals import (
    CHAIN_ORDER,
    USFundamentalsDomain,
    get_domain,
)
from data_pipelines.domains.us_fundamentals.registry import parse_identifier
from data_pipelines.domains.us_fundamentals.universe import load_universe


class TestParseIdentifier:
    def test_basic(self):
        assert parse_identifier("FUND:AAPL") == ("-", "AAPL")

    def test_dash_ticker_preserved(self):
        assert parse_identifier("FUND:BRK-B") == ("-", "BRK-B")

    def test_lowercase_normalized(self):
        assert parse_identifier("fund:aapl") == ("-", "AAPL")

    @pytest.mark.parametrize("bad", ["AAPL", "NYSE:AAPL", "FUND:", "FUND: "])
    def test_rejects(self, bad):
        with pytest.raises(ValueError):
            parse_identifier(bad)


class TestDomain:
    def test_registration_resolves_fund_prefix(self):
        # The autouse conftest fixture resets the registry per test;
        # re-register the singleton the way the CLI import path does.
        DomainRegistry.register(get_domain())
        assert DomainRegistry.resolve("FUND:AAPL").name == "us_fundamentals"

    def test_chain_follows_declared_order(self):
        domain = USFundamentalsDomain()
        chain = domain.chain_for_gap("FUND:AAPL", 1, has_cache=False)
        names = [a.name for a in chain]
        # Whatever subset of providers has landed, order must match
        # CHAIN_ORDER (primary → secondary → tertiary).
        assert names == [n for n in CHAIN_ORDER if n in names]

    def test_time_column_default(self):
        assert USFundamentalsDomain().time_column == "date"


class TestUniverse:
    def test_all_is_union_without_indices(self):
        all_ids = load_universe("all")
        assert all(i.startswith("FUND:") for i in all_ids)
        assert not any("^" in i for i in all_ids)  # no INDEX:^SPX etc.
        assert len(all_ids) == len(set(all_ids))   # de-duplicated
        # union should be roughly the russell1000 + non-overlap
        assert len(all_ids) > 900

    def test_sp500_subset_of_all(self):
        sp500 = set(load_universe("sp500"))
        assert len(sp500) > 400
        assert sp500 <= set(load_universe("all"))
        assert "FUND:AAPL" in sp500

    def test_sorted_deterministic(self):
        ids = load_universe("all")
        assert ids == sorted(ids)


class TestCLIRouting:
    def test_seed_routes_to_us_fundamentals_loader(self):
        from data_pipelines.__main__ import _load_universe_for_domain
        with patch(
            "data_pipelines.__main__.load_us_fundamentals_universe",
            return_value=["FUND:AAPL"],
        ) as loader:
            out = _load_universe_for_domain("us_fundamentals", "all")
        loader.assert_called_once_with("all")
        assert out == ["FUND:AAPL"]


class TestMergeOverlap:
    def _frame(self, rows):
        import numpy as np
        from data_pipelines.domains.us_fundamentals.schema import (
            METRIC_COLUMNS, US_FUNDAMENTALS_SCHEMA,
        )
        df = pd.DataFrame(rows)
        for c in ("date", "fiscal_period_end", "filed_date"):
            df[c] = pd.to_datetime(df.get(c)).astype("datetime64[ns]")
        for m in METRIC_COLUMNS:
            df[m] = pd.Series(df.get(m, np.nan), dtype="float64")
        return df[US_FUNDAMENTALS_SCHEMA.column_names]

    def test_existing_values_win_new_fills_holes(self):
        domain = USFundamentalsDomain()
        existing = self._frame([{
            "date": "2026-03-31", "fiscal_period_end": "2026-03-31",
            "filed_date": None, "revenue": 100.0, "ocf": 50.0,
        }])
        new = self._frame([{
            "date": "2026-03-31", "fiscal_period_end": "2026-03-28",
            "filed_date": "2026-05-01", "revenue": 999.0, "ocf": 51.0,
            "capex": 10.0,
        }])
        out = domain.merge_overlap(existing, new, [], {})
        row = out.iloc[0]
        assert row["revenue"] == 100.0          # existing value kept
        assert row["ocf"] == 50.0               # existing value kept
        assert row["capex"] == 10.0             # hole filled from new
        assert row["filed_date"] == pd.Timestamp("2026-05-01")  # enriched
        # fiscal_period_end: existing (first-written) wins too
        assert row["fiscal_period_end"] == pd.Timestamp("2026-03-31")

    def test_fcf_recomputed_consistently_on_mixed_rows(self):
        domain = USFundamentalsDomain()
        existing = self._frame([{
            "date": "2026-03-31", "fiscal_period_end": "2026-03-31",
            "filed_date": None, "ocf": 100.0,   # capex/fcf missing
        }])
        new = self._frame([{
            "date": "2026-03-31", "fiscal_period_end": "2026-03-31",
            "filed_date": None, "ocf": 102.0, "capex": 20.0, "fcf": 82.0,
        }])
        out = domain.merge_overlap(existing, new, [], {})
        row = out.iloc[0]
        # ocf existing (100) + capex new (20) → fcf must be 80, NOT new's 82
        assert row["fcf"] == 80.0
