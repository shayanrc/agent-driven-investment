"""Universe resolution — hard-fail + multi-universe registration (PR #100).

Backstory: a silent ``.get(name, NIFTY_DEFAULTS)`` fallback in
``gbdt.data.universe_metadata`` caused Exp 2 (nifty100 +10%/20d/dd5%) to
silently compute every macro feature (F1/F5/F9/F9b) against NIFTY:50
instead of NIFTY:100, because nifty100 wasn't pre-registered in
``configs/gbdt/default.yaml::universes``. The fix:

1. Unknown universe raises ``KeyError`` (no fallback).
2. ``configs/gbdt/default.yaml`` ships the 6 missing universes — 3 NSE
   (nifty100, nifty_midcap_150, nifty500) + 3 US (sp500, nasdaq100,
   russell1000).
3. ``ticker_prefix`` becomes optional — US universes set it to ``null``
   because constituents span exchanges (``NASDAQ:AAPL``, ``NYSE:JPM``)
   and are stored fully-prefixed in the universe YAML.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import yaml

from gbdt import data as gbdt_data


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Hard-fail on unknown universe (no silent fallback).
# ---------------------------------------------------------------------------


def test_unknown_universe_hard_raises():
    """A spec referencing an unregistered universe must raise KeyError with
    a message that names the missing block and points at default.yaml.
    """
    with pytest.raises(KeyError) as exc:
        gbdt_data.universe_metadata("bogus_universe_xyz")
    msg = str(exc.value)
    assert "bogus_universe_xyz" in msg
    assert "universes::" in msg
    assert "configs/gbdt/default.yaml" in msg


def test_unknown_universe_hard_raises_with_isolated_root(tmp_path):
    """Same, but against an isolated repo root with no default.yaml. The
    fallback is what the bug was; absence of config must surface, not paper
    over."""
    with pytest.raises(KeyError):
        gbdt_data.universe_metadata("anything_at_all", repo_root=tmp_path)


# ---------------------------------------------------------------------------
# 2. Each registered universe resolves with the right schema.
# ---------------------------------------------------------------------------


_NSE_UNIVERSES = ("nifty50", "nifty100", "nifty_midcap_150", "nifty500")
_US_UNIVERSES = ("sp500", "nasdaq100", "russell1000")
_KNOWN_UNIVERSES = _NSE_UNIVERSES + _US_UNIVERSES


@pytest.mark.parametrize("name", _KNOWN_UNIVERSES)
def test_each_known_universe_resolves(name):
    """All 7 universes have a complete metadata block with the right
    annualization factor + ticker_prefix semantics for the asset class.
    """
    meta = gbdt_data.universe_metadata(name)
    assert isinstance(meta, dict), f"{name}: metadata must be a dict"
    assert "source" in meta, f"{name}: missing source"
    assert "index_ticker" in meta, f"{name}: missing index_ticker"
    assert "annualization_factor" in meta, f"{name}: missing annualization_factor"

    if name in _NSE_UNIVERSES:
        assert meta["annualization_factor"] == 250
        assert meta["ticker_prefix"] == "NSE:", (
            f"{name}: NSE universes use 'NSE:' prefix; got {meta.get('ticker_prefix')!r}"
        )
        # All NSE indices are routed via NIFTY: in our cache, never INDEX:.
        assert meta["index_ticker"].startswith("NIFTY:"), (
            f"{name}: NSE index_ticker should use NIFTY: prefix"
        )
    else:  # US
        assert meta["annualization_factor"] == 252
        assert meta.get("ticker_prefix") is None, (
            f"{name}: US universes must have ticker_prefix=null because their "
            f"constituents span NYSE+NASDAQ and are stored pre-prefixed; "
            f"got {meta.get('ticker_prefix')!r}"
        )
        assert meta["index_ticker"].startswith("INDEX:"), (
            f"{name}: US index_ticker should use INDEX: prefix"
        )


# ---------------------------------------------------------------------------
# 3. ticker_prefix=null path: pre-prefixed constituents load cleanly.
# ---------------------------------------------------------------------------


def _seed_us_stub_cache(tmp_path: Path, ticker: str, n_rows: int = 20) -> None:
    """Minimal us_equities_data table with ``n_rows`` for one ticker."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(data_dir / "processed.db"))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS us_equities_data ("
            "ticker TEXT, date TEXT, open REAL, high REAL, low REAL, "
            "close REAL, adj_close REAL, volume REAL)"
        )
        con.executemany(
            "INSERT INTO us_equities_data "
            "(ticker, date, open, high, low, close, adj_close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (ticker, f"2024-01-{i:02d}", 100.0 + i, 101.0 + i, 99.0 + i,
                 100.5 + i, 100.5 + i, 1_000_000)
                for i in range(1, n_rows + 1)
            ],
        )
        con.commit()
    finally:
        con.close()


def test_null_ticker_prefix_handles_preprefixed_tickers(tmp_path, monkeypatch):
    """A US-style universe (ticker_prefix=null) with constituents already
    carrying their exchange prefix (NASDAQ:AAPL, NYSE:JPM) must load cleanly
    via ``_cache_read`` — no double-prefixing, correct table routing.

    Concretely: NYSE:/NASDAQ:/INDEX: tickers must route to the
    ``us_equities_data`` table, not the NSE one. The pre-fix routing
    incorrectly bucketed ``INDEX:`` into NSE because both Indian indices
    and US indices used the same prefix in the dispatcher.
    """
    _seed_us_stub_cache(tmp_path, "NASDAQ:AAPL")
    _seed_us_stub_cache(tmp_path, "NYSE:JPM")
    _seed_us_stub_cache(tmp_path, "INDEX:^SPX")

    for t in ("NASDAQ:AAPL", "NYSE:JPM", "INDEX:^SPX"):
        df = gbdt_data._cache_read(
            t, "1990-01-01", None, repo_root=tmp_path,
        )
        assert len(df) == 20, f"{t}: expected 20 rows, got {len(df)}"
        assert "close" in df.columns
        # Sanity: no exception, no silent empty frame — the table dispatch
        # correctly routed us to us_equities_data.


def test_index_ticker_does_not_route_to_nse_table(tmp_path):
    """Regression: INDEX:^SPX (US benchmark) used to route to nse_equities_data
    because the NSE prefix list incorrectly claimed ``INDEX:``. With only the
    nse table seeded (no us), a US INDEX read must yield 0 rows or raise —
    NOT silently return a NIFTY index row.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(data_dir / "processed.db"))
    try:
        # Seed NSE table with a NIFTY:50 row (would have been the false positive).
        con.execute(
            "CREATE TABLE IF NOT EXISTS nse_equities_data ("
            "ticker TEXT, date TEXT, open REAL, high REAL, low REAL, "
            "close REAL, adj_close REAL, volume REAL)"
        )
        con.execute(
            "INSERT INTO nse_equities_data VALUES "
            "('NIFTY:50', '2024-01-01', 1, 1, 1, 1, 1, 1)"
        )
        # Create empty us_equities_data so the SELECT doesn't error.
        con.execute(
            "CREATE TABLE IF NOT EXISTS us_equities_data ("
            "ticker TEXT, date TEXT, open REAL, high REAL, low REAL, "
            "close REAL, adj_close REAL, volume REAL)"
        )
        con.commit()
    finally:
        con.close()

    df = gbdt_data._cache_read(
        "INDEX:^SPX", "1990-01-01", None, repo_root=tmp_path,
    )
    assert len(df) == 0, (
        "INDEX:^SPX must route to us_equities_data (empty here) not nse_equities_data"
    )


# ---------------------------------------------------------------------------
# 4. resolve_universe honors the source: path (US YAMLs live in us_equities/).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected_dir", [
    ("nifty50", "nse_equities"),
    ("sp500", "us_equities"),
    ("nasdaq100", "us_equities"),
    ("russell1000", "us_equities"),
])
def test_resolve_universe_uses_source_path(name, expected_dir):
    """The default.yaml ``source:`` field determines where the constituent
    YAML lives. NSE universes live under ``nse_equities/``; US universes
    live under ``us_equities/``. Verify the metadata points at the right
    domain directory and (for files that exist on this checkout) that
    resolve_universe succeeds with a non-empty ticker list.

    The 3 NSE deep-universe YAMLs (nifty100/midcap150/nifty500) are
    promotion-pending — see task #93(d)/#96 — so we skip those gracefully.
    """
    meta = gbdt_data.universe_metadata(name)
    assert expected_dir in meta["source"], (
        f"{name}: source should live under {expected_dir}/; got {meta['source']!r}"
    )

    source_path = REPO_ROOT / meta["source"]
    if not source_path.exists():
        pytest.skip(
            f"constituent YAML for {name} not promoted on this branch — "
            f"see task #93(d)/#96"
        )
    tickers = gbdt_data.resolve_universe(name)
    assert len(tickers) > 0, f"{name}: empty ticker list"


# ---------------------------------------------------------------------------
# 5. Self-service flow (existing path) still works for an inline basket
#     once the orchestrator pre-writes the default.yaml block.
# ---------------------------------------------------------------------------


def test_self_service_requires_default_yaml_block(tmp_path):
    """Re-asserts the contract: register_universe writes the YAML, but you
    must *also* append a universes::<name> block to default.yaml before
    universe_metadata() will serve it. The pre-flight orchestrator is
    responsible for that second write; data.py no longer covers for it.
    """
    name = "tmp_basket_with_no_block"
    gbdt_data.register_universe(
        name, ["NSE:RELIANCE", "NSE:TCS"], repo_root=tmp_path,
    )
    # YAML exists, but no block in (nonexistent) default.yaml.
    with pytest.raises(KeyError):
        gbdt_data.universe_metadata(name, repo_root=tmp_path)

    # Once the orchestrator writes the block, it resolves.
    cfg_path = tmp_path / "configs/gbdt/default.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump({
        "universes": {
            name: {
                "source": f"configs/data_pipelines/domains/nse_equities/universe_{name}.yaml",
                "index_ticker": "NIFTY:50",
                "ticker_prefix": "NSE:",
                "annualization_factor": 250,
            }
        }
    }))
    meta = gbdt_data.universe_metadata(name, repo_root=tmp_path)
    assert meta["annualization_factor"] == 250
    assert meta["index_ticker"] == "NIFTY:50"
