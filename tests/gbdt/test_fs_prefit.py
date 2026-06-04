"""Unit tests for the fs_prefit module (V1.3 Option B P2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbdt.fs_prefit import (
    FSPrefitResult,
    cliff_cut,
    fs_prefit_cache_key,
    hp_sha256,
    load_fs_prefit_cache,
    run_fs_prefit,
    save_fs_prefit_cache,
)


# ---------------------------------------------------------------------------
# cliff_cut
# ---------------------------------------------------------------------------


def test_cliff_cut_at_1pct_of_top():
    """Plan example: importances [100, 50, 5, 2, 0.5, 0.1] → kept = [100, 50, 5, 2]."""
    imp = pd.Series(
        {"a": 100.0, "b": 50.0, "c": 5.0, "d": 2.0, "e": 0.5, "f": 0.1},
    )
    kept, dropped, top, threshold = cliff_cut(imp, cliff_pct=0.01)
    # top = 100; threshold = 1.0 → keep all >= 1.0 = {a, b, c, d}
    assert set(kept) == {"a", "b", "c", "d"}
    assert set(dropped) == {"e", "f"}
    assert top == 100.0
    assert threshold == 1.0


def test_cliff_cut_keeps_sort_order_desc():
    imp = pd.Series({"a": 5.0, "b": 100.0, "c": 50.0, "d": 2.0})
    kept, dropped, top, threshold = cliff_cut(imp, cliff_pct=0.01)
    # Sorted desc: b, c, a, d
    assert kept == ["b", "c", "a", "d"]
    # all > 1.0 (top * 0.01)


def test_cliff_cut_handles_all_zero():
    imp = pd.Series({"a": 0.0, "b": 0.0})
    kept, dropped, top, threshold = cliff_cut(imp, cliff_pct=0.01)
    # Keep everything when there's no signal to cliff on.
    assert set(kept) == {"a", "b"}
    assert dropped == []
    assert top == 0.0


def test_cliff_cut_handles_empty_series():
    imp = pd.Series([], dtype=float)
    kept, dropped, top, threshold = cliff_cut(imp, cliff_pct=0.01)
    assert kept == []
    assert dropped == []


def test_cliff_cut_uses_custom_pct():
    imp = pd.Series({"a": 100.0, "b": 50.0, "c": 30.0, "d": 5.0})
    # 50% cliff → kept = {a, b} (>= 50)
    kept, dropped, top, threshold = cliff_cut(imp, cliff_pct=0.50)
    assert set(kept) == {"a", "b"}
    assert set(dropped) == {"c", "d"}
    assert threshold == 50.0


# ---------------------------------------------------------------------------
# hp_sha256 — deterministic
# ---------------------------------------------------------------------------


def test_hp_sha256_deterministic():
    h1 = hp_sha256({"max_depth": 6, "eta": 0.3})
    h2 = hp_sha256({"eta": 0.3, "max_depth": 6})    # different key order
    assert h1 == h2


def test_hp_sha256_distinguishes_values():
    assert hp_sha256({"max_depth": 6}) != hp_sha256({"max_depth": 8})


# ---------------------------------------------------------------------------
# fs_prefit_cache_key
# ---------------------------------------------------------------------------


def test_fs_prefit_cache_key_deterministic():
    args = dict(
        universe="nasdaq100",
        features_source_sha256="abc123",
        snapshot_end="2026-01-01",
        default_hp_sha256="hpsha",
    )
    k1 = fs_prefit_cache_key(**args)
    k2 = fs_prefit_cache_key(**args)
    assert k1 == k2


def test_fs_prefit_cache_key_distinct_per_universe():
    base = dict(
        features_source_sha256="abc123",
        snapshot_end="2026-01-01",
        default_hp_sha256="hpsha",
    )
    a = fs_prefit_cache_key(universe="nasdaq100", **base)
    b = fs_prefit_cache_key(universe="sp500", **base)
    assert a != b


def test_fs_prefit_cache_key_distinct_per_default_hp():
    base = dict(
        universe="nasdaq100",
        features_source_sha256="abc",
        snapshot_end="2026-01-01",
    )
    a = fs_prefit_cache_key(default_hp_sha256="aaa", **base)
    b = fs_prefit_cache_key(default_hp_sha256="bbb", **base)
    assert a != b


# ---------------------------------------------------------------------------
# Cache load/save round-trip
# ---------------------------------------------------------------------------


def test_cache_roundtrip(tmp_path):
    result = FSPrefitResult(
        kept_features=["a", "b"], dropped_features=["c"],
        top_importance=10.0, cliff_threshold=0.1,
        backend="xgboost", default_hp_sha256="abc",
        cliff_pct=0.01, fit_seconds=1.5,
        importance_kept={"a": 10.0, "b": 5.0},
    )
    key = "test_key_123"
    save_fs_prefit_cache(tmp_path, key, result)
    loaded = load_fs_prefit_cache(tmp_path, key)
    assert loaded is not None
    assert loaded.kept_features == ["a", "b"]
    assert loaded.dropped_features == ["c"]
    assert loaded.backend == "xgboost"
    assert loaded.cliff_pct == 0.01
    assert loaded.importance_kept == {"a": 10.0, "b": 5.0}


def test_cache_miss_returns_none(tmp_path):
    assert load_fs_prefit_cache(tmp_path, "nonexistent_key") is None


def test_cache_corrupt_returns_none(tmp_path):
    """A non-JSON cache file should yield None (treated as a miss)."""
    cache_dir = tmp_path / "fs_prefit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "bad_key.json").write_text("{not json")
    assert load_fs_prefit_cache(tmp_path, "bad_key") is None


# ---------------------------------------------------------------------------
# run_fs_prefit
# ---------------------------------------------------------------------------


def _fake_fit_returning_importance(importance_series):
    """Build a fake fit_one closure that returns the given importance Series."""
    def fit_one(*, hp, **kw):
        return importance_series
    return fit_one


def test_run_fs_prefit_cliff_cut_default():
    """Default cliff_pct=0.01: importances [100, 50, 5, 2, 0.5] → kept top 4."""
    imp = pd.Series({"a": 100.0, "b": 50.0, "c": 5.0, "d": 2.0, "e": 0.5})
    fit_one = _fake_fit_returning_importance(imp)
    result = run_fs_prefit(
        X_train=pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "e": [5]}),
        y_train=np.array([1]),
        w_train=None,
        fit_one=fit_one,
        backend="xgboost",
        default_hp={"max_depth": 6, "eta": 0.3},
    )
    assert set(result.kept_features) == {"a", "b", "c", "d"}
    assert result.dropped_features == ["e"]
    assert result.cliff_pct == 0.01
    assert result.backend == "xgboost"
    assert result.default_hp_sha256 == hp_sha256({"max_depth": 6, "eta": 0.3})
    assert result.fit_seconds >= 0.0


def test_run_fs_prefit_custom_cliff_pct():
    imp = pd.Series({"a": 100.0, "b": 50.0, "c": 5.0, "d": 0.5})
    fit_one = _fake_fit_returning_importance(imp)
    result = run_fs_prefit(
        X_train=pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]}),
        y_train=np.array([1]),
        w_train=None,
        fit_one=fit_one,
        backend="catboost",
        default_hp={"depth": 6},
        cliff_pct=0.10,
    )
    # 10% cliff → threshold = 10; kept = {a, b} (>= 10)
    assert set(result.kept_features) == {"a", "b"}


def test_run_fs_prefit_accepts_dict_importance():
    """The fit_one callable might return a dict instead of a Series — accepted."""
    def fit_one(*, hp, **kw):
        return {"a": 10.0, "b": 1.0, "c": 0.05}
    result = run_fs_prefit(
        X_train=pd.DataFrame({"a": [1], "b": [2], "c": [3]}),
        y_train=np.array([1]),
        w_train=None,
        fit_one=fit_one,
        backend="xgboost",
        default_hp={"max_depth": 6},
    )
    assert "a" in result.kept_features
    assert "b" in result.kept_features
    # c: 0.05 < threshold (10 * 0.01 = 0.1) → dropped.
    assert "c" in result.dropped_features
