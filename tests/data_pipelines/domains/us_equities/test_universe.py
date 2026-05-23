"""Stage 5 tests: us_equities universe loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_pipelines.domains.us_equities import universe as uni


@pytest.fixture
def custom_config_root(tmp_path: Path) -> Path:
    (tmp_path / "universe_test.yaml").write_text(
        "universe: test\n"
        "tickers:\n  - NYSE:AAA\n  - NASDAQ:BBB\n"
        "indices:\n  - INDEX:^TST\n"
    )
    uni._load_yaml.cache_clear()
    return tmp_path


def test_load_default_sp500_universe_is_nonempty():
    out = uni.load_universe("sp500")
    assert len(out) > 0
    # Every entry has a prefix.
    assert all(":" in e for e in out)


def test_load_universe_with_custom_root(custom_config_root):
    out = uni.load_universe("test", custom_config_root)
    assert out == ["NYSE:AAA", "NASDAQ:BBB", "INDEX:^TST"]


def test_is_in_universe(custom_config_root):
    assert uni.is_in_universe("NYSE:AAA", "test", custom_config_root)
    assert not uni.is_in_universe("NYSE:ZZZ", "test", custom_config_root)


def test_missing_universe_raises():
    uni._load_yaml.cache_clear()
    with pytest.raises(FileNotFoundError):
        uni.load_universe("nope_does_not_exist")


def test_default_sp500_includes_indices():
    out = uni.load_universe("sp500")
    indices = [x for x in out if x.startswith("INDEX:")]
    assert "INDEX:^SPX" in indices
