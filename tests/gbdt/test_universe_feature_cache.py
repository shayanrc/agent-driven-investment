"""task #183 — universe-level (cross-cell) feature-matrix cache.

The per-run cache (task #181, :mod:`gbdt.feature_cache`) is keyed on the full
target tuple, so sibling cells in a same-universe sweep do NOT share it and
each pays the full ~5 h (russell1000) feature build. The universe-level cache
keys EVERYTHING that determines :func:`gbdt.features.build_feature_matrix`'s
output EXCEPT the target tuple — every cell in a same-universe sweep hashes to
the same key and shares the build.

These tests cover the three correctness pillars:

  1. **Key composition.** The key EXCLUDES the target tuple and INCLUDES every
     non-target input (universe, split, feature config, seed, code commit,
     data snapshot).
  2. **Round-trip fidelity.** ``build → write → load`` returns a matrix
     IDENTICAL (values + dtypes + index + column order) to the one built.
  3. **Miss-on-mismatch.** Any change to the non-target inputs yields a
     different key ⇒ ``load_cache`` misses ⇒ caller rebuilds. Absent files,
     corrupt parquet, sidecar shape disagreements all return ``None``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gbdt import feature_cache as per_cell_cache
from gbdt import features as gbdt_features
from gbdt import universe_feature_cache as ufc
from gbdt.leakage_harness import make_synthetic_panel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synth_panel_with_index(n_rows: int = 250, n_tickers: int = 4, seed: int = 0):
    panel = make_synthetic_panel(n_rows, n_tickers, seed=seed)
    rng = np.random.default_rng(seed + 99)
    dates = panel.index.get_level_values("date").unique()
    rets = rng.normal(0, 0.008, size=len(dates))
    close = 1000.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0.0, 0.005, len(dates)))
    low = close * (1.0 - rng.uniform(0.0, 0.005, len(dates)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, len(dates)))
    index_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": np.ones(len(dates))},
        index=pd.Index(dates, name="date"),
    )
    return panel, index_df


def _build_matrix(panel, index_df):
    """Mirror the runner's Phase 2: build then drop all-NaN columns."""
    X = gbdt_features.build_feature_matrix(panel, index_df)
    return X.dropna(axis=1, how="all")


# Base kwargs for compute_key — EXCLUDES the target tuple (that's the whole
# point of #183). Compare to ``_BASE_KW`` in ``test_feature_cache.py``, which
# DOES include ``target``.
_BASE_KW = dict(
    universe="nifty50",
    split={"train_rows": 800, "val_rows": 400, "eval_rows": 200,
           "test_rows": 100, "min_rows_per_ticker": 1500},
    lookbacks=(5, 10, 20, 50, 100, 200),
    families="all",
    exclude=[],
    random_seed=42,
)


@pytest.fixture
def panel_and_key():
    panel, index_df = _synth_panel_with_index(seed=0)
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    key = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    return panel, index_df, key


# ---------------------------------------------------------------------------
# (1) Key composition — drops target, keeps everything else
# ---------------------------------------------------------------------------


def test_universe_key_signature_excludes_target_tuple():
    """The function signature itself must not accept a ``target`` kwarg —
    accidentally re-introducing one would defeat the purpose of the layer.

    Also (task #190): the signature must NOT accept ``code_commit`` /
    ``code_dirty`` — those were dropped to fix the over-strict per-commit
    invalidation; the feature-code signature now carries a SHA-256 of the
    ``gbdt.features`` source instead."""
    import inspect

    sig = inspect.signature(ufc.compute_key)
    params = set(sig.parameters)
    assert "target" not in params, (
        "compute_key must NOT take a 'target' kwarg — that's the entire "
        "point of the universe-level cache (drop the target tuple from the "
        "key so sibling cells share)."
    )
    # Task #190 drop: code_commit + code_dirty must be gone (hard break).
    assert "code_commit" not in params, (
        "compute_key must not accept code_commit (task #190 dropped it — "
        "see PRs #86/#87 cold-rebuild incident)."
    )
    assert "code_dirty" not in params, (
        "compute_key must not accept code_dirty (task #190 dropped it)."
    )
    # And it MUST take every non-target input that determines the build.
    for required in (
        "universe", "split", "lookbacks", "families", "exclude",
        "random_seed", "panel_sig",
    ):
        assert required in params, f"compute_key missing required kwarg: {required!r}"


def test_universe_key_invariant_under_target_swap(panel_and_key):
    """Two cells differing ONLY in their target tuple must hash to the same
    universe key — that's what lets them share the cached matrix."""
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    # Same universe/split/features/seed/etc., regardless of target.
    again = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert again == base_key


def test_universe_key_differs_per_cell_key_does_not(panel_and_key):
    """Sanity check on the contrast: the PER-CELL key DOES change with target,
    while the UNIVERSE key DOES NOT. Documents the design intent."""
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)

    target_a = {"direction": "up", "threshold_pct": 10, "horizon_days": 25,
                "max_drawdown": 0.05, "uniqueness_weighting": True}
    target_b = {"direction": "up", "threshold_pct": 20, "horizon_days": 50,
                "max_drawdown": 0.10, "uniqueness_weighting": True}

    cell_key_a = per_cell_cache.compute_key(
        target=target_a, panel_sig=panel_sig, **_BASE_KW,
    )
    cell_key_b = per_cell_cache.compute_key(
        target=target_b, panel_sig=panel_sig, **_BASE_KW,
    )
    assert cell_key_a != cell_key_b, "per-cell key MUST differ when target differs"

    universe_key_a = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    universe_key_b = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert universe_key_a == universe_key_b == base_key, (
        "universe key MUST be invariant under target changes — that's the cache"
        " hit shared cells get."
    )


def test_universe_key_differs_on_changed_universe(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    other = ufc.compute_key(panel_sig=panel_sig, **{**_BASE_KW, "universe": "nasdaq100"})
    assert other != base_key


def test_universe_key_differs_on_changed_split(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    kw = {**_BASE_KW, "split": {**_BASE_KW["split"], "train_rows": 1200}}
    other = ufc.compute_key(panel_sig=panel_sig, **kw)
    assert other != base_key


def test_universe_key_differs_on_changed_features(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    kw = {**_BASE_KW, "families": ["F1", "F2"]}
    other = ufc.compute_key(panel_sig=panel_sig, **kw)
    assert other != base_key


def test_universe_key_differs_on_changed_lookbacks(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    kw = {**_BASE_KW, "lookbacks": (5, 10, 20)}
    other = ufc.compute_key(panel_sig=panel_sig, **kw)
    assert other != base_key


def test_universe_key_differs_on_changed_exclude(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    kw = {**_BASE_KW, "exclude": ["volume_ratio_*"]}
    other = ufc.compute_key(panel_sig=panel_sig, **kw)
    assert other != base_key


def test_universe_key_differs_on_changed_seed(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    kw = {**_BASE_KW, "random_seed": 7}
    other = ufc.compute_key(panel_sig=panel_sig, **kw)
    assert other != base_key


def test_universe_key_stable_across_what_used_to_be_commit_invalidation(panel_and_key):
    """Task #190 regression guard: pre-#190, ANY commit on main (even one
    that didn't touch features) flipped the universe-cache key and forced
    a ~5 h cold rebuild on russell1000 (per PRs #86/#87 incident). Post-#190
    the key is determined solely by actual feature-build inputs, so
    re-computing with identical inputs gives the same key."""
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    again = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert again == base_key, (
        "universe compute_key MUST be stable across what used to be "
        "commit-level invalidation — task #190 fix."
    )


def test_universe_key_differs_on_changed_features_source(panel_and_key, monkeypatch):
    """The TARGETED invalidator (task #190): a change to ``gbdt.features``
    source flows through ``feature_code_signature``'s ``source_sha256`` and
    flips the cache key. This is what makes "edit features.py → next run
    rebuilds" work without false positives from unrelated commits."""
    import inspect as _inspect
    from gbdt import features as gbdt_features

    panel, index_df, base_key = panel_and_key

    real_getsource = _inspect.getsource
    def fake_getsource(obj):
        if obj is gbdt_features:
            return real_getsource(obj) + "\n# perturbation\n"
        return real_getsource(obj)

    # The signature helper lives in gbdt.feature_cache and is re-used by
    # universe_feature_cache (single source of truth), so patching that one
    # site invalidates both cache keys at once.
    monkeypatch.setattr("gbdt.feature_cache.inspect.getsource", fake_getsource)

    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    other = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert other != base_key


def test_universe_key_differs_on_changed_data_snapshot(panel_and_key):
    panel, index_df, base_key = panel_and_key
    # A different snapshot (one extra ticker / different rows) ⇒ new sig.
    panel2, index2 = _synth_panel_with_index(n_tickers=5, seed=0)
    sig2 = per_cell_cache.panel_signature(panel2, index2)
    other = ufc.compute_key(panel_sig=sig2, **_BASE_KW)
    assert other != base_key


def test_universe_key_differs_on_changed_code_signature(panel_and_key, monkeypatch):
    """A bump to ``EXPECTED_TOTAL_COLS`` (proxy for a feature-engineering
    change) must flow through ``feature_code_signature`` and invalidate the
    key — even with everything else identical."""
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    monkeypatch.setattr(
        gbdt_features, "EXPECTED_TOTAL_COLS",
        gbdt_features.EXPECTED_TOTAL_COLS + 1,
    )
    other = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert other != base_key


# ---------------------------------------------------------------------------
# (2) Round-trip fidelity
# ---------------------------------------------------------------------------


def test_build_cache_reload_returns_identical_matrix(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    ufc.write_cache(tmp_path, X, key, subdir="cache")
    loaded = ufc.load_cache(tmp_path, key, subdir="cache")
    assert loaded is not None, "key match must return the cached matrix"
    # Identical values, dtypes, index, and column order.
    pd.testing.assert_frame_equal(loaded, X, check_exact=True)


def test_write_creates_parent_dirs(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    ufc.write_cache(tmp_path / "nested" / "root", X, key, subdir="cache")
    assert ufc.matrix_path(tmp_path / "nested" / "root", key, subdir="cache").exists()


def test_cache_files_live_under_data_root_subdir(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    ufc.write_cache(tmp_path, X, key, subdir="cache")
    expected_dir = tmp_path / "cache"
    assert expected_dir.is_dir()
    # Two artifacts per key: the parquet + the sidecar.
    assert (expected_dir / f"{key}.parquet").exists()
    assert (expected_dir / f"{key}.key.json").exists()


# ---------------------------------------------------------------------------
# (3) Misses (absent / corrupt / mismatched key)
# ---------------------------------------------------------------------------


def test_load_absent_cache_returns_none(tmp_path, panel_and_key):
    _, _, key = panel_and_key
    assert ufc.load_cache(tmp_path, key, subdir="cache") is None


def test_corrupt_parquet_returns_none(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    ufc.write_cache(tmp_path, X, key, subdir="cache")
    ufc.matrix_path(tmp_path, key, subdir="cache").write_bytes(b"not a parquet")
    assert ufc.load_cache(tmp_path, key, subdir="cache") is None


def test_sidecar_shape_mismatch_returns_none(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    ufc.write_cache(tmp_path, X, key, subdir="cache")
    kpath = ufc.key_path(tmp_path, key, subdir="cache")
    sidecar = json.loads(kpath.read_text())
    sidecar["n_cols"] = sidecar["n_cols"] + 1
    kpath.write_text(json.dumps(sidecar))
    assert ufc.load_cache(tmp_path, key, subdir="cache") is None


def test_schema_version_mismatch_returns_none(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    ufc.write_cache(tmp_path, X, key, subdir="cache")
    kpath = ufc.key_path(tmp_path, key, subdir="cache")
    sidecar = json.loads(kpath.read_text())
    sidecar["schema_version"] = "v-FROM-THE-FUTURE"
    kpath.write_text(json.dumps(sidecar))
    assert ufc.load_cache(tmp_path, key, subdir="cache") is None


def test_changed_seed_misses_cache(tmp_path, panel_and_key):
    """End-to-end miss: a cache written under one key is not served under
    a different key (the rebuild path the runner takes)."""
    panel, index_df, base_key = panel_and_key
    X = _build_matrix(panel, index_df)
    ufc.write_cache(tmp_path, X, base_key, subdir="cache")
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    new_key = ufc.compute_key(panel_sig=panel_sig, **{**_BASE_KW, "random_seed": 7})
    assert ufc.load_cache(tmp_path, new_key, subdir="cache") is None
    # Original key still hits (cache not destroyed).
    assert ufc.load_cache(tmp_path, base_key, subdir="cache") is not None


# ---------------------------------------------------------------------------
# (4) Determinism + cross-target sharing
# ---------------------------------------------------------------------------


def test_key_is_deterministic(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)
    again = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert again == base_key


def test_two_target_cells_share_same_universe_cache(tmp_path, panel_and_key):
    """The integration story end-to-end: cell A builds → writes universe cache;
    cell B (same universe, DIFFERENT target) computes the SAME universe key and
    loads cell A's parquet. This is the russell1000 sweep speedup."""
    panel, index_df, _ = panel_and_key
    panel_sig = per_cell_cache.panel_signature(panel, index_df)

    # Cell A — target tuple #1.
    target_a = {"direction": "up", "threshold_pct": 10, "horizon_days": 25,
                "max_drawdown": 0.05, "uniqueness_weighting": True}
    universe_key_a = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    X = _build_matrix(panel, index_df)
    ufc.write_cache(tmp_path, X, universe_key_a, subdir="cache")

    # Cell B — DIFFERENT target tuple, same universe + everything else.
    target_b = {"direction": "down", "threshold_pct": 20, "horizon_days": 50,
                "max_drawdown": 0.10, "uniqueness_weighting": True}
    # Sanity: per-cell keys DO differ (no spurious per-cell sharing).
    cell_key_a = per_cell_cache.compute_key(
        target=target_a, panel_sig=panel_sig, **_BASE_KW,
    )
    cell_key_b = per_cell_cache.compute_key(
        target=target_b, panel_sig=panel_sig, **_BASE_KW,
    )
    assert cell_key_a != cell_key_b
    # But the universe keys MATCH and cell B's lookup HITS.
    universe_key_b = ufc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert universe_key_b == universe_key_a
    loaded = ufc.load_cache(tmp_path, universe_key_b, subdir="cache")
    assert loaded is not None, "cell B must hit cell A's universe cache"
    # And the loaded matrix is byte-identical to what cell B would have built
    # on its own (the golden snapshot — proves we don't pollute results).
    pd.testing.assert_frame_equal(loaded, X, check_exact=True)


# ---------------------------------------------------------------------------
# (5) Task #190 — SCHEMA_VERSION bump for the code_commit → source_sha256 swap
# ---------------------------------------------------------------------------


def test_schema_version_is_v2():
    """Task #190 bumped SCHEMA_VERSION v1 → v2 so any v1-keyed parquet on
    disk (notably the 6.2 G russell1000 cache from PR #85's build) misses
    cleanly and gets rebuilt under the new key shape. Correctness over
    reuse — we'd rather rebuild once than risk reusing under an
    inconsistent key shape."""
    assert ufc.SCHEMA_VERSION == "v2"
