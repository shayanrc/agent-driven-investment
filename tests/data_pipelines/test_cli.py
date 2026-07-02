"""Stage 10 CLI tests: argparse routing + integration via mocked adapters.

These tests bypass real network by monkey-patching the adapter chain on the
already-registered us_equities domain.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Importing the CLI module triggers us_equities registration (and reset
# fixture in the parent conftest scrubs it between tests — we re-register
# below).
from data_pipelines.__main__ import build_parser, main
from data_pipelines.adapter import Adapter
from data_pipelines.domain import DomainRegistry
from data_pipelines.domains.us_equities import USEquitiesDomain
from data_pipelines.raw_store import write_raw_atomic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeUSEquitiesAdapter(Adapter):
    """Stand-in for any us_equities adapter — writes raw + parses a canned df."""

    def __init__(self, name: str, df_factory, extra_meta: dict | None = None):
        self.name = name
        self._df_factory = df_factory
        self.extra_meta = extra_meta or {"adjustment_quality": "full"}
        self.source_column_map = None
        self.calls = 0

    def fetch(self, identifier, start=None, end=None, *, data_root):
        from data_pipelines.domains.us_equities.registry import parse_identifier
        self.calls += 1
        prefix, symbol = parse_identifier(identifier)
        from datetime import datetime, timezone
        ts = datetime(2026, 5, 23, 14, 30, self.calls, tzinfo=timezone.utc)
        return write_raw_atomic(
            data_root,
            provider=self.name, domain="us_equities",
            exchange=prefix, ticker=symbol,
            payload=b"raw",
            range_start=start or date(2026, 1, 1),
            range_end=end or date(2026, 1, 31),
            ext="csv",
            timestamp=ts,
        )

    def parse(self, raw_path):
        return self._df_factory()


def _ohlcv_df(dates: list[date], close_base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(dates).astype("datetime64[ns]"),
        "open": [close_base + i for i in range(len(dates))],
        "high": [close_base + 1 + i for i in range(len(dates))],
        "low": [close_base - 1 + i for i in range(len(dates))],
        "close": [close_base + 0.5 + i for i in range(len(dates))],
        "adj_close": [close_base + 0.5 + i for i in range(len(dates))],
        "volume": [1_000_000 + i for i in range(len(dates))],
    })


@pytest.fixture
def us_equities_with_fake_adapters(tmp_path):
    """Build a USEquitiesDomain with fake adapters, registered in the registry."""
    domain = USEquitiesDomain()
    fake_seed = _FakeUSEquitiesAdapter(
        "stooq",
        lambda: _ohlcv_df([date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]),
        extra_meta={"adjustment_quality": "split_only"},
    )
    fake_upd = _FakeUSEquitiesAdapter(
        "tiingo",
        lambda: _ohlcv_df([date(2026, 1, 8), date(2026, 1, 9)]),
    )
    fake_fb = _FakeUSEquitiesAdapter(
        "yfinance",
        lambda: _ohlcv_df([date(2026, 1, 8), date(2026, 1, 9)]),
    )
    domain._adapters = {
        "stooq": fake_seed, "tiingo": fake_upd, "yfinance": fake_fb,
    }
    DomainRegistry.register(domain)
    return domain, tmp_path


# ---------------------------------------------------------------------------
# Argparse routing
# ---------------------------------------------------------------------------

class TestParser:
    def test_fetch_required_args(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["fetch", "NYSE:AAPL"])  # missing --start/--end

    def test_fetch_parses_dates(self):
        parser = build_parser()
        args = parser.parse_args([
            "fetch", "NYSE:AAPL", "--start", "2026-01-01", "--end", "2026-01-31",
        ])
        assert args.cmd == "fetch"
        assert args.identifier == "NYSE:AAPL"
        assert args.start == "2026-01-01"
        # back_extend defaults to False — existing callers' behaviour preserved.
        assert args.back_extend is False

    def test_fetch_back_extend_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "fetch", "NYSE:AAPL",
            "--start", "2015-01-01", "--end", "2026-01-31",
            "--back-extend",
        ])
        assert args.back_extend is True

    def test_seed_defaults_to_sp500(self):
        parser = build_parser()
        args = parser.parse_args([
            "seed", "--start", "2026-01-01", "--end", "2026-01-31",
        ])
        assert args.domain == "us_equities"
        assert args.universe == "sp500"

    def test_reprocess_mutex(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["reprocess", "--identifier", "NYSE:AAPL", "--all"])


# ---------------------------------------------------------------------------
# Integration smoke
# ---------------------------------------------------------------------------

class TestFetchCmd:
    def test_fetch_invokes_seed_and_writes_cache(
        self, us_equities_with_fake_adapters, capsys
    ):
        domain, root = us_equities_with_fake_adapters
        rc = main([
            "--data-root", str(root),
            "fetch", "NYSE:AAPL",
            "--start", "2026-01-05", "--end", "2026-01-07",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        meta = json.loads(out.split("rows:")[0])
        assert meta["identifier"] == "NYSE:AAPL"
        assert meta["cache_was_cold"] is True
        assert meta["gaps_filled"][0]["provider"] == "stooq"
        # SQLite cache: a single DB at data/processed.db with rows for AAPL.
        from data_pipelines.cache import (
            list_cached_identifiers,
            processed_db_path,
        )
        assert processed_db_path(root).is_file()
        assert "NYSE:AAPL" in list_cached_identifiers(root, domain)


class TestListDomainsCmd:
    def test_list_domains(self, us_equities_with_fake_adapters, capsys):
        rc = main(["list-domains"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert any(d["name"] == "us_equities" for d in out)
        ue = next(d for d in out if d["name"] == "us_equities")
        assert "NYSE" in ue["prefixes"]
        assert set(ue["adapters"]) == {"stooq", "tiingo", "yfinance"}


class TestHealthCmd:
    def test_health_us_equities(self, us_equities_with_fake_adapters, capsys):
        rc = main(["health", "--domain", "us_equities"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert any(entry["domain"] == "us_equities" for entry in out)


class TestSeedCmd:
    def test_seed_bulk_against_mini_universe(
        self, us_equities_with_fake_adapters, tmp_path, capsys
    ):
        # Write a 3-ticker universe config to a tmp config root and point to it.
        config_root = tmp_path / "configs" / "data_pipelines" / "domains" / "us_equities"
        config_root.mkdir(parents=True)
        (config_root / "universe_mini.yaml").write_text(
            "universe: mini\n"
            "tickers:\n  - NYSE:JPM\n  - NASDAQ:AAPL\n  - NYSE:WMT\n"
        )

        # Patch the universe loader to look at our tmp config root.
        with patch("data_pipelines.__main__.load_us_equities_universe",
                   side_effect=lambda name: [
                       "NYSE:JPM", "NASDAQ:AAPL", "NYSE:WMT",
                   ] if name == "mini" else []):
            rc = main([
                "--data-root", str(us_equities_with_fake_adapters[1]),
                "seed", "--universe", "mini",
                "--start", "2026-01-05", "--end", "2026-01-07",
            ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "3/3 succeeded" in out

    def test_seed_parallel_writes_all_tickers(
        self, us_equities_with_fake_adapters, tmp_path, capsys
    ):
        # --jobs > 1 fetches concurrently; the per-DB write lock must keep every
        # ticker's rows intact (no clobber), so the cache is identical to the
        # sequential path (goal.md determinism) and all 3 tickers land.
        domain, root = us_equities_with_fake_adapters
        with patch("data_pipelines.__main__.load_us_equities_universe",
                   side_effect=lambda name: ["NYSE:JPM", "NASDAQ:AAPL", "NYSE:WMT"]
                   if name == "mini" else []):
            rc = main([
                "--data-root", str(root),
                "seed", "--universe", "mini", "--jobs", "3",
                "--start", "2026-01-05", "--end", "2026-01-07",
            ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "jobs=3" in out and "3/3 succeeded" in out
        from data_pipelines.cache import list_cached_identifiers
        cached = set(list_cached_identifiers(root, domain))
        assert {"NYSE:JPM", "NASDAQ:AAPL", "NYSE:WMT"} <= cached


class TestListCachedCmd:
    def test_list_cached_after_fetch(
        self, us_equities_with_fake_adapters, capsys
    ):
        domain, root = us_equities_with_fake_adapters
        main([
            "--data-root", str(root),
            "fetch", "NYSE:AAPL",
            "--start", "2026-01-05", "--end", "2026-01-07",
        ])
        capsys.readouterr()  # flush

        rc = main(["--data-root", str(root), "list-cached"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert any(entry["identifier"] == "NYSE:AAPL" for entry in out)


class TestReprocessCmd:
    def test_reprocess_no_raw_no_op(
        self, us_equities_with_fake_adapters, capsys
    ):
        domain, root = us_equities_with_fake_adapters
        rc = main([
            "--data-root", str(root),
            "reprocess", "--identifier", "NYSE:NOPE",
        ])
        assert rc == 0

    def test_reprocess_after_fetch(
        self, us_equities_with_fake_adapters, capsys
    ):
        domain, root = us_equities_with_fake_adapters
        main([
            "--data-root", str(root),
            "fetch", "NYSE:AAPL",
            "--start", "2026-01-05", "--end", "2026-01-07",
        ])
        capsys.readouterr()
        rc = main([
            "--data-root", str(root),
            "reprocess", "--identifier", "NYSE:AAPL",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "NYSE:AAPL" in out


class TestPurgeCmd:
    def test_purge_requires_yes(
        self, us_equities_with_fake_adapters, capsys
    ):
        domain, root = us_equities_with_fake_adapters
        rc = main([
            "--data-root", str(root),
            "purge", "--identifier", "NYSE:AAPL",
        ])
        assert rc == 2

    def test_purge_removes_processed_dir(
        self, us_equities_with_fake_adapters, capsys
    ):
        domain, root = us_equities_with_fake_adapters
        main([
            "--data-root", str(root),
            "fetch", "NYSE:AAPL",
            "--start", "2026-01-05", "--end", "2026-01-07",
        ])
        capsys.readouterr()
        rc = main([
            "--data-root", str(root),
            "purge", "--identifier", "NYSE:AAPL", "--yes",
        ])
        assert rc == 0
        assert not (root / "processed" / "us_equities" / "NYSE" / "AAPL").exists()
