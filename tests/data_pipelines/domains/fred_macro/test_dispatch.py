"""End-to-end fetch() through dispatch + SQLite cache, with the network
transport stubbed. Covers the two FRED-specific risk paths:

  * the per-series-cadence calendar actually drives gap detection in dispatch
    (a monthly series stores monthly rows and converges, not a daily grid), and
  * NaN (FRED "." / no-data) survives the SQLite NULL round-trip.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from data_pipelines.cache import read_processed
from data_pipelines.dispatch import fetch_with_meta
from data_pipelines.domain import DomainRegistry
from data_pipelines.domains.fred_macro import FREDMacroDomain
from data_pipelines.domains.fred_macro.adapters.fred import FredAdapter, _csv_has_values
from data_pipelines.domains.fred_macro.registry import parse_identifier
from data_pipelines.errors import EmptyPayload
from data_pipelines.raw_store import write_raw_atomic

# Canned "series → {iso_date: value}". DGS10 daily (with a "." gap day),
# CPIAUCSL monthly. The stub emits only rows inside the requested range, as
# FRED's range-filtered endpoints do.
_CANNED = {
    "DGS10": {
        "2020-01-02": "1.88", "2020-01-03": "1.79", "2020-01-06": ".",
        "2020-01-07": "1.83", "2020-01-08": "1.87",
    },
    "CPIAUCSL": {
        "2020-01-01": "257.971", "2020-02-01": "258.678", "2020-03-01": "258.115",
    },
}


class _StubFredAdapter(FredAdapter):
    """Real parse()/densify, stubbed network: synthesizes a fredgraph-style CSV
    for the requested [start, end] from a canned series dict."""

    def fetch(self, identifier, start=None, end=None, *, data_root: Path):
        _, sid = parse_identifier(identifier)
        rows = [f"observation_date,{sid}"]
        for iso, val in sorted(_CANNED.get(sid, {}).items()):
            if start <= date.fromisoformat(iso) <= end:
                rows.append(f"{iso},{val}")
        payload = ("\n".join(rows) + "\n").encode()
        if not _csv_has_values(payload):
            raise EmptyPayload(self.name, identifier)
        return write_raw_atomic(
            data_root, provider=self.name, domain="fred_macro", exchange="-",
            ticker=sid, payload=payload, range_start=start, range_end=end,
            ext="csv", timestamp=datetime.now(timezone.utc),
        )


@pytest.fixture
def fred_domain():
    """A FRED domain with the network stubbed, registered for this test
    (the autouse registry-reset in conftest clears it between tests)."""
    dom = FREDMacroDomain()
    dom._adapters["fred"] = _StubFredAdapter(frequency_map=dom._frequency_map)
    DomainRegistry.register(dom)
    return dom


def test_daily_fetch_and_convergence(tmp_path, fred_domain):
    df, m1 = fetch_with_meta(
        "FRED:DGS10", date(2020, 1, 2), date(2020, 1, 8), data_root=tmp_path,
    )
    # 5 weekdays in range, all present (incl. the densified "." day).
    assert len(df) == 5
    assert m1.gaps_filled  # cold cache → something was filled
    # Second identical fetch: the daily grid converged — no new gaps.
    _, m2 = fetch_with_meta(
        "FRED:DGS10", date(2020, 1, 2), date(2020, 1, 8), data_root=tmp_path,
    )
    assert m2.gaps_filled == []


def test_nan_survives_sqlite_roundtrip(tmp_path, fred_domain):
    fetch_with_meta(
        "FRED:DGS10", date(2020, 1, 2), date(2020, 1, 8), data_root=tmp_path,
    )
    # Re-read from disk (not the in-memory frame) to exercise the SQLite
    # NaN→NULL→NaN path that the equity domains never hit.
    cached, _ = read_processed(tmp_path, fred_domain, "FRED:DGS10")
    assert str(cached["value"].dtype) == "float64"
    hole = cached.loc[cached["date"] == np.datetime64("2020-01-06"), "value"].iloc[0]
    assert np.isnan(hole)
    assert cached["value"].notna().sum() == 4  # the other 4 weekdays have values


def test_monthly_uses_monthly_calendar(tmp_path, fred_domain):
    df, _ = fetch_with_meta(
        "FRED:CPIAUCSL", date(2020, 1, 1), date(2020, 3, 31), data_root=tmp_path,
    )
    # Monthly cadence → 3 rows (Jan/Feb/Mar firsts), NOT a ~65-weekday grid.
    assert len(df) == 3
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == [
        "2020-01-01", "2020-02-01", "2020-03-01",
    ]
    # Convergence: a re-fetch fills no new gaps (proves the monthly grid, not
    # the default daily one, is what dispatch reconciles against).
    _, m2 = fetch_with_meta(
        "FRED:CPIAUCSL", date(2020, 1, 1), date(2020, 3, 31), data_root=tmp_path,
    )
    assert m2.gaps_filled == []


def test_single_source_chain():
    dom = FREDMacroDomain()
    chain = dom.chain_for_gap("FRED:DGS10", gap_size_trading_days=100, has_cache=True)
    assert len(chain) == 1
    assert chain[0].name == "fred"
