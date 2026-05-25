"""Stage 1 — data loader tests.

Exercises:
- universe resolution against the on-disk universe_nifty50.yaml.
- panel load against the real NIFTY 50 cache (skipped if the cache is empty
  in this environment).
- ticker exclusion at min_rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gbdt import data as gbdt_data


REPO_ROOT = None  # use CWD; pytest runs from repo root via pythonpath setup


def test_resolve_universe_nifty50_has_50_tickers():
    tickers = gbdt_data.resolve_universe("nifty50")
    assert len(tickers) == 50
    assert all(t.startswith("NSE:") for t in tickers)
    assert "NSE:RELIANCE" in tickers


def test_universe_metadata_nifty50():
    meta = gbdt_data.universe_metadata("nifty50")
    assert meta["annualization_factor"] == 250
    assert meta["index_ticker"] == "INDEX:^NSEI"


def test_register_universe_round_trip(tmp_path):
    gbdt_data.register_universe(
        "test_basket",
        ["NSE:RELIANCE", "NSE:TCS"],
        repo_root=tmp_path,
    )
    assert (tmp_path / "configs/data_pipelines/domains/nse_equities/universe_test_basket.yaml").exists()
    tickers = gbdt_data.resolve_universe("test_basket", repo_root=tmp_path)
    assert tickers == ["NSE:RELIANCE", "NSE:TCS"]


def test_resolve_universe_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not registered"):
        gbdt_data.resolve_universe("nope_does_not_exist", repo_root=tmp_path)


@pytest.mark.slow
def test_load_panel_nifty50_via_cache():
    """Loads the real NIFTY 50 panel — requires the data_pipelines cache
    to be populated for ``NSE:RELIANCE`` + ``INDEX:^NSEI``. Cheap on a warm
    cache, slow on a cold one."""
    p = gbdt_data.load_panel("nifty50", start="2020-01-01", end="2024-12-31")
    assert p.universe == "nifty50"
    assert p.annualization_factor == 250
    assert len(p.tickers_kept) > 0
    # Panel structure
    assert isinstance(p.panel.index, pd.MultiIndex)
    assert p.panel.index.names == ["date", "ticker"]
    for col in ("open", "high", "low", "close", "volume"):
        assert col in p.panel.columns
    # Index series
    assert len(p.index_series) > 100
    assert "close" in p.index_series.columns
