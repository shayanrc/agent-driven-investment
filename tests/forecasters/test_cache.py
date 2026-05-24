"""Tests for forecasters.cache — content-addressed forecast result cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forecasters.cache import cache_key, read_cached, write_cached


def _good_result(horizon: int = 4, n_paths: int = 5) -> dict[str, Any]:
    return {
        "paths": np.arange(n_paths * horizon, dtype=np.float64).reshape(n_paths, horizon),
        "anchors": {
            "origin_date": "2020-01-02",
            "horizon_dates": [f"2020-01-{3 + i:02d}" for i in range(horizon)],
        },
        "summary": {
            "median": [0.0] * horizon,
            "p05": [-1.0] * horizon,
            "p25": [-0.5] * horizon,
            "p75": [0.5] * horizon,
            "p95": [1.0] * horizon,
            "crps": 0.04,
        },
        "metadata": {
            "backend_name": "fake",
            "preset_name": "p",
            "preset_hash": "sha256:h",
            "config_hash": "sha256:c",
            "n_paths": n_paths,
            "seed_used": 7,
        },
        "warnings": ["hello"],
    }


def _k(**overrides) -> str:
    base = dict(
        preset_name="v24-default",
        preset_content_hash="sha256:abc",
        identifier="NASDAQ100",
        data_path=None,
        start="2010-01-01",
        end="2024-12-31",
        origin="2024-01-02",
        horizon=60,
        seed=42,
    )
    base.update(overrides)
    return cache_key(**base)


# ----------------------------------------------------------------------------
# cache_key — stability + variance
# ----------------------------------------------------------------------------


def test_cache_key_stable() -> None:
    assert _k() == _k()


def test_cache_key_changes_with_each_arg() -> None:
    base = _k()
    assert _k(preset_name="other") != base
    assert _k(preset_content_hash="sha256:def") != base
    assert _k(identifier="NYSE:AAPL") != base
    assert _k(start="2010-01-02") != base
    assert _k(end="2024-12-30") != base
    assert _k(origin="2024-01-03") != base
    assert _k(horizon=61) != base
    assert _k(seed=43) != base
    # data_path vs identifier — different keys.
    assert _k(identifier=None, data_path="data/NASDAQ100.csv") != base


def test_cache_key_seed_none_distinct_from_zero() -> None:
    assert _k(seed=None) != _k(seed=0)


def test_cache_key_is_short_hex() -> None:
    k = _k()
    assert len(k) == 16
    int(k, 16)  # parseable


# ----------------------------------------------------------------------------
# write_cached / read_cached round-trip
# ----------------------------------------------------------------------------


def test_round_trip_round_trips_paths_and_summary(tmp_path: Path) -> None:
    key = _k()
    r = _good_result(horizon=4, n_paths=5)
    out_dir = write_cached(key, r, cache_root=tmp_path)
    assert out_dir.is_dir()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "paths.npz").is_file()
    assert (out_dir / "warnings.json").is_file()

    loaded = read_cached(key, cache_root=tmp_path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded["paths"], r["paths"])
    assert loaded["anchors"] == r["anchors"]
    assert loaded["summary"] == r["summary"]
    assert loaded["metadata"] == r["metadata"]
    assert loaded["warnings"] == r["warnings"]


def test_read_cached_returns_none_for_missing(tmp_path: Path) -> None:
    assert read_cached(_k(), cache_root=tmp_path) is None


def test_different_keys_isolated(tmp_path: Path) -> None:
    k1 = _k()
    k2 = _k(origin="2024-02-01")
    r1 = _good_result()
    r2 = _good_result()
    r2["paths"] = r2["paths"] + 100
    write_cached(k1, r1, cache_root=tmp_path)
    write_cached(k2, r2, cache_root=tmp_path)
    out1 = read_cached(k1, cache_root=tmp_path)
    out2 = read_cached(k2, cache_root=tmp_path)
    assert out1 is not None and out2 is not None
    np.testing.assert_array_equal(out1["paths"], r1["paths"])
    np.testing.assert_array_equal(out2["paths"], r2["paths"])


def test_preset_edit_invalidates(tmp_path: Path) -> None:
    """Changing the preset content hash → new cache key → cache miss."""
    k_before = _k(preset_content_hash="sha256:111")
    k_after = _k(preset_content_hash="sha256:222")
    write_cached(k_before, _good_result(), cache_root=tmp_path)
    assert read_cached(k_before, cache_root=tmp_path) is not None
    assert read_cached(k_after, cache_root=tmp_path) is None


def test_write_to_default_root_overridden_by_arg(tmp_path: Path) -> None:
    """Even if DEFAULT_CACHE_ROOT exists, an explicit cache_root wins."""
    key = _k()
    out_dir = write_cached(key, _good_result(), cache_root=tmp_path)
    assert out_dir.is_relative_to(tmp_path)


def test_write_idempotent_if_target_exists(tmp_path: Path) -> None:
    """Concurrent writers: a second write_cached call doesn't crash."""
    key = _k()
    r1 = _good_result()
    write_cached(key, r1, cache_root=tmp_path)
    # Simulate a second writer arriving with different data — the existing
    # cache should win (atomic-rename pattern means we discard the new work).
    r2 = _good_result()
    r2["paths"] = r2["paths"] + 999
    write_cached(key, r2, cache_root=tmp_path)
    loaded = read_cached(key, cache_root=tmp_path)
    np.testing.assert_array_equal(loaded["paths"], r1["paths"])  # first write preserved
