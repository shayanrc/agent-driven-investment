"""Stage 1 — data loader tests.

Exercises:
- universe resolution against the on-disk universe_nifty50.yaml.
- panel load against the real NIFTY 50 cache (skipped if the cache is empty
  in this environment).
- ticker exclusion at min_rows.
- NaN-row filter + cache freshness telemetry (PR #8 review, Minor 1+4).
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

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
    assert meta["index_ticker"] == "NIFTY:50"


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
    cache, slow on a cold one.

    We pin ``min_rows=500`` (well under the production 1600) so a freshly
    seeded ~5y cache still keeps tickers and the structural assertions
    below can run — see PR #8 review (Low 5). A test of the production
    1600-row gate belongs in a separate cache-state probe, not here.
    """
    p = gbdt_data.load_panel(
        "nifty50", start="2020-01-01", end="2024-12-31", min_rows=500,
    )
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


# ---------------------------------------------------------------------------
# Helpers + fixtures for the stubbed-cache tests below.
# ---------------------------------------------------------------------------


def _seed_stub_cache(
    tmp_path,
    ticker: str,
    rows: list[tuple],
    last_meta_date: str | None = None,
) -> None:
    """Build a minimal ``data/processed.db`` with the schema ``_cache_read``
    expects. Used by the NaN-filter + staleness tests so we don't depend on
    the real NSE cache.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(data_dir / "processed.db"))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS nse_equities_data ("
            "ticker TEXT, date TEXT, open REAL, high REAL, low REAL, "
            "close REAL, adj_close REAL, volume REAL"
            ")"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS nse_equities_meta ("
            "ticker TEXT PRIMARY KEY, range_end TEXT"
            ")"
        )
        con.executemany(
            "INSERT INTO nse_equities_data "
            "(ticker, date, open, high, low, close, adj_close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(ticker, *r) for r in rows],
        )
        if last_meta_date is not None:
            con.execute(
                "INSERT OR REPLACE INTO nse_equities_meta (ticker, range_end) "
                "VALUES (?, ?)",
                (ticker, last_meta_date),
            )
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Minor 4 — NaN-row filter at cache read.
# ---------------------------------------------------------------------------


def test_cache_read_drops_all_nan_ohlcv_rows(tmp_path):
    rows = [
        ("2024-01-02", 100.0, 101.0, 99.5, 100.5, 100.5, 1_000_000),
        # All-OHLCV-NaN row — should be dropped.
        ("2024-01-03", None, None, None, None, None, None),
        ("2024-01-04", 101.0, 102.0, 100.5, 101.5, 101.5, 1_100_000),
    ]
    _seed_stub_cache(tmp_path, "NSE:STUBCO", rows)
    df, n_dropped = gbdt_data._cache_read(
        "NSE:STUBCO", "1990-01-01", None,
        repo_root=tmp_path, return_nan_count=True,
    )
    assert len(df) == 2
    assert n_dropped == 1
    # And the surviving dates are the non-NaN ones.
    assert df["date"].dt.date.tolist() == [
        date(2024, 1, 2), date(2024, 1, 4),
    ]


def test_cache_read_partial_nan_row_kept(tmp_path):
    """A row with any non-null OHLCV value must survive — only fully-blank
    rows are dropped."""
    rows = [
        ("2024-01-02", 100.0, 101.0, 99.5, 100.5, 100.5, 1_000_000),
        # close present, others NaN — must survive.
        ("2024-01-03", None, None, None, 100.7, None, None),
    ]
    _seed_stub_cache(tmp_path, "NSE:STUBPARTIAL", rows)
    df, n_dropped = gbdt_data._cache_read(
        "NSE:STUBPARTIAL", "1990-01-01", None,
        repo_root=tmp_path, return_nan_count=True,
    )
    assert len(df) == 2
    assert n_dropped == 0


# ---------------------------------------------------------------------------
# Regression — end-date inclusivity with a time-suffixed date column.
# Cached dates carry a 'YYYY-MM-DD 00:00:00' component; a bare 'date <= end'
# string-compared '…D 00:00:00' as GREATER than 'D' and silently dropped the
# end day (off-by-one that hid the most recent bar from load_panel / fresh
# inference). The fix is a half-open interval [start_day, end_day + 1).
# ---------------------------------------------------------------------------


def test_cache_read_end_date_inclusive_with_time_component(tmp_path):
    rows = [
        ("2024-01-02 00:00:00", 100.0, 101.0, 99.5, 100.5, 100.5, 1_000_000),
        ("2024-01-03 00:00:00", 101.0, 102.0, 100.5, 101.5, 101.5, 1_100_000),
        ("2024-01-04 00:00:00", 102.0, 103.0, 101.5, 102.5, 102.5, 1_200_000),
    ]
    _seed_stub_cache(tmp_path, "NSE:STUBEND", rows)

    # end == the LAST day → that day MUST be included (the bug dropped it).
    df = gbdt_data._cache_read("NSE:STUBEND", "2024-01-01", "2024-01-04", repo_root=tmp_path)
    assert df["date"].max() == pd.Timestamp("2024-01-04")
    assert len(df) == 3

    # end == a MIDDLE day → inclusive of that day, excludes later days.
    df_mid = gbdt_data._cache_read("NSE:STUBEND", "2024-01-01", "2024-01-03", repo_root=tmp_path)
    assert df_mid["date"].dt.date.tolist() == [date(2024, 1, 2), date(2024, 1, 3)]

    # start == first day → start day included (start-side inclusivity preserved).
    df_start = gbdt_data._cache_read("NSE:STUBEND", "2024-01-03", "2024-01-04", repo_root=tmp_path)
    assert df_start["date"].dt.date.tolist() == [date(2024, 1, 3), date(2024, 1, 4)]

    # end == a date object (not str) → same inclusivity.
    df_obj = gbdt_data._cache_read("NSE:STUBEND", "2024-01-01", date(2024, 1, 4), repo_root=tmp_path)
    assert len(df_obj) == 3

    # end is None → all rows.
    df_none = gbdt_data._cache_read("NSE:STUBEND", "2024-01-01", None, repo_root=tmp_path)
    assert len(df_none) == 3


# ---------------------------------------------------------------------------
# Minor 1 — cache freshness telemetry.
# ---------------------------------------------------------------------------


def test_ensure_universe_cached_flags_stale_ticker(tmp_path, caplog):
    """A cache whose max-date is older than ``staleness_days`` must be
    flagged ``is_stale=True`` with an age in days, and a WARNING log emitted.
    """
    stale_last = (date.today() - timedelta(days=60)).isoformat()
    # Build enough rows to pass min_rows=3 below.
    rows = [
        (f"2024-01-{i:02d}", 100.0, 101.0, 99.5, 100.5, 100.5, 1_000_000)
        for i in range(2, 12)
    ]
    _seed_stub_cache(tmp_path, "NSE:STALE", rows, last_meta_date=stale_last)

    caplog.set_level("WARNING", logger="gbdt.data")
    statuses = gbdt_data.ensure_universe_cached(
        ["NSE:STALE"], start=None, end=None,
        min_rows=3, repo_root=tmp_path, staleness_days=14,
    )
    st = statuses["NSE:STALE"]
    assert st.kept is True
    assert st.is_stale is True
    assert st.cache_last_date == stale_last
    assert st.cache_age_days >= 60
    assert any("stale cache" in rec.message for rec in caplog.records)


def test_ensure_universe_cached_fresh_ticker_not_flagged(tmp_path):
    fresh_last = (date.today() - timedelta(days=2)).isoformat()
    rows = [
        (f"2024-01-{i:02d}", 100.0, 101.0, 99.5, 100.5, 100.5, 1_000_000)
        for i in range(2, 12)
    ]
    _seed_stub_cache(tmp_path, "NSE:FRESH", rows, last_meta_date=fresh_last)
    statuses = gbdt_data.ensure_universe_cached(
        ["NSE:FRESH"], start=None, end=None,
        min_rows=3, repo_root=tmp_path, staleness_days=14,
    )
    st = statuses["NSE:FRESH"]
    assert st.kept is True
    assert st.is_stale is False
    assert st.cache_age_days is not None and st.cache_age_days <= 3


# ---------------------------------------------------------------------------
# Nit 7 — register_universe is deterministic when listed_at is omitted.
# ---------------------------------------------------------------------------


def test_register_universe_omits_listed_at_by_default(tmp_path):
    """Without ``listed_at``, the YAML must NOT carry today's date — otherwise
    every regeneration would dirty the diff."""
    path = gbdt_data.register_universe(
        "stub_basket", ["NSE:A", "NSE:B"], repo_root=tmp_path,
    )
    import yaml as _yaml
    payload = _yaml.safe_load(path.read_text())
    assert "listed_at" not in payload
    assert payload["tickers"] == ["NSE:A", "NSE:B"]


def test_register_universe_records_listed_at_when_provided(tmp_path):
    path = gbdt_data.register_universe(
        "stub_dated", ["NSE:A"], repo_root=tmp_path, listed_at="2024-06-01",
    )
    import yaml as _yaml
    payload = _yaml.safe_load(path.read_text())
    assert payload["listed_at"] == "2024-06-01"
