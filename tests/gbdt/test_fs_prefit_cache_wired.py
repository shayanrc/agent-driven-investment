"""Wire-up tests for the FS-prefit cache (V1.3 Option B D6.2.A).

The fs_prefit.py module ships cache-key + load/save helpers; this file
verifies those helpers are actually called from ``_maybe_run_scout_and_prefit``
(default mode) so sibling cells sharing a universe + snapshot + default-HP
don't re-pay the prefit fit cost.

Cache-MISS path: pre-existing FS-prefit run produces a fresh fit AND writes
a cache entry on disk.

Cache-HIT path: the cache exists at the expected key BEFORE the call →
``run_fs_prefit`` is not invoked (we monkeypatch it to raise) and the run
still succeeds with the cached kept-features list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt import fs_prefit as fs_prefit_mod
from gbdt.fs_prefit import (
    FSPrefitResult,
    fs_prefit_cache_key,
    hp_sha256,
    save_fs_prefit_cache,
)
from gbdt.train import walk_forward_train


def _toy_panel(n_per_ticker: int = 1600, n_tickers: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_per_ticker, freq="B")
    frames = []
    for i in range(n_tickers):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_per_ticker)))
        df = pd.DataFrame({
            "date": dates,
            "ticker": f"T{i}",
            "open": c, "high": c * 1.005, "low": c * 0.995,
            "close": c, "adj_close": c,
            "volume": np.ones(n_per_ticker, dtype=int),
        })
        frames.append(df)
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    n_total = len(panel)
    X = pd.DataFrame(
        rng.normal(0, 1, (n_total, 6)),
        index=panel.index,
        columns=["sig", "n1", "n2", "n3", "n4", "n5"],
    )
    y = ((X["sig"] + rng.normal(0, 0.3, n_total)) > 0).astype(int)
    return panel, X, y


def _make_hp() -> dict:
    return {"iterations": 20, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05}


# ---------------------------------------------------------------------------
# Cache MISS — fit runs AND a cache entry is written.
# ---------------------------------------------------------------------------


def test_fs_prefit_cache_miss_runs_fit_and_writes_cache(tmp_path):
    panel, X, y = _toy_panel(1600, 3, seed=21)
    hp = _make_hp()
    cliff_pct = 0.0    # keep all features
    backend = "catboost"
    universe = "toyverse"
    snapshot_end = "2026-06-01"
    features_sha = "feature_src_v1"

    # Sanity: cache root is empty.
    assert not (tmp_path / "fs_prefit").exists()

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp=hp, max_iterations=2,
        scout_spec={"enabled": False},
        fs_prefit_spec={"enabled": True, "cliff_pct": cliff_pct},
        callback_mode="default",
        fs_prefit_universe=universe,
        fs_prefit_cache_root=str(tmp_path),
        fs_prefit_features_source_sha256=features_sha,
        fs_prefit_snapshot_end_iso=snapshot_end,
    )

    # Report shape: cache_hit=False, cache_key populated.
    pf = result.scout_report["fs_prefit"]
    assert pf["enabled"] is True
    assert pf["cache_hit"] is False
    assert pf["cache_key"] is not None
    # The cache file was written on disk under tmp_path/fs_prefit/<key>.json.
    cache_dir = tmp_path / "fs_prefit"
    assert cache_dir.exists()
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    # And its name matches the reported key.
    assert cache_files[0].name == f"{pf['cache_key']}.json"


# ---------------------------------------------------------------------------
# Cache HIT — run_fs_prefit is NOT called; cached kept-feature list is used.
# ---------------------------------------------------------------------------


def test_fs_prefit_cache_hit_skips_fit(monkeypatch, tmp_path):
    panel, X, y = _toy_panel(1600, 3, seed=22)
    hp = _make_hp()
    cliff_pct = 0.0
    backend = "catboost"
    universe = "toyverse"
    snapshot_end = "2026-06-01"
    features_sha = "feature_src_v1"

    # Pre-populate the cache with the key the runner will compute.
    key = fs_prefit_cache_key(
        universe=universe,
        features_source_sha256=features_sha,
        snapshot_end=snapshot_end,
        default_hp_sha256=hp_sha256(
            {"hp": dict(hp), "cliff_pct": float(cliff_pct),
             "backend": str(backend)},
        ),
    )
    # Pretend the cached prefit kept only the signal column "sig".
    cached_result = FSPrefitResult(
        kept_features=["sig"],
        dropped_features=["n1", "n2", "n3", "n4", "n5"],
        top_importance=100.0, cliff_threshold=1.0,
        backend=backend, default_hp_sha256="precomputed",
        cliff_pct=cliff_pct, fit_seconds=0.001,
        importance_kept={"sig": 100.0},
    )
    save_fs_prefit_cache(tmp_path, key, cached_result)

    # Mock run_fs_prefit so the cache HIT is proven by failure-on-call.
    call_counter = {"n": 0}

    def _explode(*args, **kwargs):
        call_counter["n"] += 1
        raise AssertionError(
            "run_fs_prefit was called on a cache hit — cache wiring is broken."
        )

    monkeypatch.setattr(fs_prefit_mod, "run_fs_prefit", _explode)

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp=hp, max_iterations=2,
        scout_spec={"enabled": False},
        fs_prefit_spec={"enabled": True, "cliff_pct": cliff_pct},
        callback_mode="default",
        fs_prefit_universe=universe,
        fs_prefit_cache_root=str(tmp_path),
        fs_prefit_features_source_sha256=features_sha,
        fs_prefit_snapshot_end_iso=snapshot_end,
    )

    # run_fs_prefit was never invoked.
    assert call_counter["n"] == 0
    pf = result.scout_report["fs_prefit"]
    assert pf["enabled"] is True
    assert pf["cache_hit"] is True
    assert pf["cache_key"] == key
    # The kept-feature list came from the cached entry.
    assert pf["n_kept"] == 1
    assert pf["n_dropped"] == 5


# ---------------------------------------------------------------------------
# Caching off when any key component is missing (back-compat).
# ---------------------------------------------------------------------------


def test_fs_prefit_no_cache_when_inputs_missing(tmp_path):
    """If the runner can't supply universe / cache_root / source_sha /
    snapshot_end, the prefit still runs (no exception) — the cache is just
    bypassed. ``cache_key`` in the report is None."""
    panel, X, y = _toy_panel(1600, 3, seed=23)
    hp = _make_hp()
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp=hp, max_iterations=2,
        scout_spec={"enabled": False},
        fs_prefit_spec={"enabled": True, "cliff_pct": 0.0},
        callback_mode="default",
        # Intentionally omit fs_prefit_universe / cache_root / ... — back-compat.
    )
    pf = result.scout_report["fs_prefit"]
    assert pf["enabled"] is True
    assert pf["cache_hit"] is False
    assert pf["cache_key"] is None
    # And no cache file was created.
    assert not (tmp_path / "fs_prefit").exists()
