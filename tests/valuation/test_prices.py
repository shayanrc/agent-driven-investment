"""prices — id mapping + split-basis alignment (pure parts)."""

from __future__ import annotations

import pandas as pd
import pytest

from valuation.prices import (
    adjust_shares_to_latest_basis,
    cumulative_split_factor,
    nse_equities_identifier,
    us_equities_identifier,
)


class TestIdentifierMap:
    def test_maps_known_symbols(self):
        # AAPL is NASDAQ in the nasdaq100/sp500 YAMLs
        assert us_equities_identifier("FUND:AAPL") == "NASDAQ:AAPL"
        # a NYSE name
        assert us_equities_identifier("FUND:JPM") == "NYSE:JPM"

    def test_unknown_symbol_returns_none(self):
        assert us_equities_identifier("FUND:ZZZNOPE") is None

    def test_accepts_bare_symbol(self):
        assert us_equities_identifier("AAPL") == "NASDAQ:AAPL"


class TestNseIdentifierMap:
    def test_maps_known_symbol(self):
        # RELIANCE is in the nifty500 universe as NSE:RELIANCE
        assert nse_equities_identifier("INFUND:RELIANCE") == "NSE:RELIANCE"

    def test_unknown_symbol_returns_none(self):
        # not a nifty500 constituent
        assert nse_equities_identifier("INFUND:ZZZNOPE") is None

    def test_accepts_bare_symbol(self):
        assert nse_equities_identifier("RELIANCE") == "NSE:RELIANCE"

    def test_index_pseudo_ticker_excluded(self):
        # the NIFTY:500 index entry in the universe is not a mappable equity
        assert nse_equities_identifier("INFUND:500") is None


class TestSplitFactor:
    def _splits(self):
        # AAPL-like: 4:1 on 2020-08-31 (and an earlier 7:1 on 2014-06-09)
        return pd.Series(
            {pd.Timestamp("2014-06-09"): 7.0, pd.Timestamp("2020-08-31"): 4.0}
        ).sort_index()

    def test_no_splits_factor_one(self):
        assert cumulative_split_factor(pd.Series(dtype="float64"),
                                       pd.Timestamp("2020-01-01")) == 1.0

    def test_before_all_splits_multiplies_all(self):
        # a share count from 2010 predates both splits → ×28
        assert cumulative_split_factor(self._splits(),
                                       pd.Timestamp("2010-03-31")) == 28.0

    def test_between_splits(self):
        # after the 2014 split, before the 2020 split → only ×4 remains
        assert cumulative_split_factor(self._splits(),
                                       pd.Timestamp("2019-03-31")) == 4.0

    def test_after_all_splits_factor_one(self):
        assert cumulative_split_factor(self._splits(),
                                       pd.Timestamp("2021-03-31")) == 1.0

    def test_on_split_date_is_exclusive(self):
        # "strictly after": a fiscal end exactly ON the split date is already
        # in post-split basis for that split.
        assert cumulative_split_factor(self._splits(),
                                       pd.Timestamp("2020-08-31")) == 1.0


class TestAdjustShares:
    def test_lifts_pre_split_shares(self):
        splits = pd.Series({pd.Timestamp("2020-08-31"): 4.0})
        fe = pd.Series(pd.to_datetime(["2020-03-31", "2021-03-31"]))
        shares = pd.Series([4300.0, 16800.0])  # pre-split ~4.3B, post ~16.8B
        adj = adjust_shares_to_latest_basis(fe, shares, splits)
        # pre-split count ×4 → ~17,200; post-split unchanged
        assert adj.iloc[0] == pytest.approx(17200.0)
        assert adj.iloc[1] == pytest.approx(16800.0)
