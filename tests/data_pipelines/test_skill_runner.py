"""Smoke tests for scripts/data_pipelines/skill_runner.py.

Avoids real network calls by writing a small synthetic processed.db with the
us_equities schema and pointing the runner at it via --data-root (health) or
by pre-seeding rows the fetcher will return from cache (fetch).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# Side-effect imports — register domains for the test DB writes below.
import data_pipelines.domains.us_equities  # noqa: F401
import data_pipelines.domains.nse_equities  # noqa: F401

from data_pipelines.cache import write_processed_atomic
from data_pipelines.domain import DomainRegistry
from data_pipelines.domains.us_equities import get_domain as get_us_domain
from scripts.data_pipelines.skill_runner import main


# The autouse conftest fixture resets the registry between tests; we
# re-register the production domains in our tests by re-import (the
# decorators/registration calls re-execute on side-effect re-import once the
# registry has been cleared).


@pytest.fixture
def seeded_db(tmp_path: Path):
    """A tiny processed.db at tmp_path/data with one us_equities row set.

    Tests pass --data-root tmp_path to point the runner at this DB.
    """
    DomainRegistry._reset()
    import importlib
    import data_pipelines.domains.us_equities as us_mod
    import data_pipelines.domains.nse_equities as nse_mod
    importlib.reload(us_mod)
    importlib.reload(nse_mod)

    domain = get_us_domain()
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    # Build a small DataFrame matching the us_equities schema.
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.5] * 5,
        "adj_close": [100.5] * 5,
        "volume": [1000] * 5,
    }).astype({
        "open": "float64", "high": "float64", "low": "float64",
        "close": "float64", "adj_close": "float64", "volume": "int64",
    })
    meta = {
        "schema_version": 1,
        "domain": "us_equities",
        "row_count": len(df),
        "range": {
            "start": df["date"].iloc[0].date().isoformat(),
            "end": df["date"].iloc[-1].date().isoformat(),
        },
        "last_fetch_utc": "2026-05-24T12:00:00Z",
        "sources": [{
            "provider": "tiingo",
            "raw_file": "stub.csv",
            "covers": {
                "start": df["date"].iloc[0].date().isoformat(),
                "end": df["date"].iloc[-1].date().isoformat(),
            },
            "adjustment_quality": "full",
        }],
    }
    write_processed_atomic(data_root, domain, "NASDAQ:STUB", df, meta)
    return data_root


# ----------------------------------------------------------------------------
# health subcommand
# ----------------------------------------------------------------------------


def test_health_no_args_table(seeded_db: Path, capsys) -> None:
    rc = main(["health", "--data-root", str(seeded_db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "total identifiers: 1" in out
    assert "us_equities" in out


def test_health_no_args_json(seeded_db: Path, capsys) -> None:
    rc = main(["health", "--data-root", str(seeded_db), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_identifiers"] == 1
    assert payload["total_rows"] == 5
    assert "us_equities" in payload["per_domain"]
    # Seeded DB has exactly one row with last_fetch_utc = 2026-05-24T12:00:00Z,
    # so oldest and newest must both pin to that timestamp.
    assert payload["oldest_last_fetch_utc"] == "2026-05-24T12:00:00Z"
    assert payload["newest_last_fetch_utc"] == "2026-05-24T12:00:00Z"


def test_health_identifier_cached(seeded_db: Path, capsys) -> None:
    rc = main(["health", "--data-root", str(seeded_db),
               "--identifier", "NASDAQ:STUB", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cached"] is True
    assert payload["rows"] == 5
    assert payload["range"]["start"] == "2024-01-02"
    assert payload["sources"][0]["provider"] == "tiingo"


def test_health_identifier_not_cached(seeded_db: Path, capsys) -> None:
    rc = main(["health", "--data-root", str(seeded_db),
               "--identifier", "NASDAQ:DOESNOTEXIST", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cached"] is False


def test_health_identifier_unknown_domain_exits_nonzero(seeded_db: Path, capsys) -> None:
    rc = main(["health", "--data-root", str(seeded_db),
               "--identifier", "FOO:BAR"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "UnknownDomain" in err


def test_health_domain_filter(seeded_db: Path, capsys) -> None:
    rc = main(["health", "--data-root", str(seeded_db),
               "--domain", "us_equities", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload["per_domain"].keys()) == ["us_equities"]


# ----------------------------------------------------------------------------
# fetch subcommand
# ----------------------------------------------------------------------------


def test_fetch_unknown_domain_exits_nonzero(seeded_db: Path, capsys, monkeypatch) -> None:
    # Point CWD's "data" path away from the seeded one (data/ doesn't exist
    # in tmp_path for fetch — it looks at relative ./data by default).
    monkeypatch.chdir(seeded_db.parent)
    rc = main(["fetch", "--identifier", "FOO:BAR",
               "--start", "2020-01-01", "--end", "2020-12-31"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "UnknownDomain" in err


def test_fetch_cached_identifier_serves_from_cache(seeded_db: Path, capsys, monkeypatch) -> None:
    """Pre-seeded cache covers the requested range, so fetch returns 5 rows."""
    monkeypatch.chdir(seeded_db.parent)
    rc = main(["fetch", "--identifier", "NASDAQ:STUB",
               "--start", "2024-01-02", "--end", "2024-01-08", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["identifier"] == "NASDAQ:STUB"
    assert payload["rows"] == 5
    assert payload["cache_was_cold"] is False
    assert payload["providers_failed"] == []
