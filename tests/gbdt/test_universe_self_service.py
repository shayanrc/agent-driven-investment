"""Stage 1 — universe self-service.

The agent surface registers a fresh universe YAML before falling through
to the cache check. This test confirms the end-to-end loop works against
a tmp_path repo root.
"""

from __future__ import annotations

import pytest

from gbdt import data as gbdt_data


def test_register_then_resolve(tmp_path):
    name = "tmp_2_ticker_basket"
    tickers = ["NSE:RELIANCE", "NSE:TCS"]
    gbdt_data.register_universe(name, tickers, repo_root=tmp_path)
    out = gbdt_data.resolve_universe(name, repo_root=tmp_path)
    assert out == tickers


def test_self_service_universe_defaults_metadata(tmp_path):
    """A universe registered on first use that has no default.yaml entry
    must still resolve a sensible metadata block (NSE defaults).
    """
    gbdt_data.register_universe(
        "no_default_entry", ["NSE:RELIANCE"], repo_root=tmp_path,
    )
    meta = gbdt_data.universe_metadata("no_default_entry", repo_root=tmp_path)
    assert meta["annualization_factor"] == 250
    assert meta["index_ticker"] == "INDEX:^NSEI"


def test_self_service_overwrites_idempotently(tmp_path):
    name = "tmp_basket"
    gbdt_data.register_universe(name, ["NSE:A"], repo_root=tmp_path)
    gbdt_data.register_universe(name, ["NSE:A", "NSE:B"], repo_root=tmp_path)
    assert gbdt_data.resolve_universe(name, repo_root=tmp_path) == ["NSE:A", "NSE:B"]


def test_resolve_universe_empty_raises(tmp_path):
    import yaml
    p = tmp_path / "configs/data_pipelines/domains/nse_equities/universe_empty.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({"universe": "empty", "tickers": []}))
    with pytest.raises(ValueError, match="no tickers"):
        gbdt_data.resolve_universe("empty", repo_root=tmp_path)
