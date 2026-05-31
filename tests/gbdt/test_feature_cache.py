"""task #181 — per-run feature-matrix cache round-trip + key invalidation.

The agent_file_protocol loop is exit-and-resume: every ``--resume`` is a fresh
process that would otherwise rebuild the full candidate feature matrix from
scratch (~3 h on sp500). The cache persists the built matrix and lets a key
match skip the rebuild. These tests cover the two correctness pillars:

  1. **Round-trip fidelity** — build → ``write_cache`` → ``load_cache`` returns
     a matrix IDENTICAL (values + dtypes + index + column order) to the one
     built, so reuse never changes results.
  2. **Key invalidation** — any change to the determining inputs (seed,
     threshold, data snapshot, feature config, code commit) yields a different
     key, so ``load_cache`` misses and the caller rebuilds.

The build uses the same synthetic-panel helper as ``test_features.py`` so the
cached object is a genuine ``build_feature_matrix`` output, not a toy frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt import feature_cache as fc
from gbdt import features as gbdt_features
from gbdt.leakage_harness import make_synthetic_panel


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


_BASE_KW = dict(
    universe="nifty50",
    target={"direction": "up", "threshold_pct": 10, "horizon_days": 20,
            "max_drawdown": None, "uniqueness_weighting": True},
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
    panel_sig = fc.panel_signature(panel, index_df)
    key = fc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    return panel, index_df, key


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


def test_build_cache_reload_returns_identical_matrix(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)

    run_dir = tmp_path / "cell"
    fc.write_cache(run_dir, X, key)

    loaded = fc.load_cache(run_dir, key)
    assert loaded is not None, "key match must return the cached matrix"
    # Identical values, dtypes, index, and column order.
    pd.testing.assert_frame_equal(loaded, X, check_exact=True)


def test_cache_files_are_co_located_in_run_dir(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    run_dir = tmp_path / "cell"
    fc.write_cache(run_dir, X, key)
    assert fc.matrix_path(run_dir).exists()
    assert fc.key_path(run_dir).exists()
    assert fc.matrix_path(run_dir) == run_dir / "_feature_matrix_cache.parquet"


def test_write_creates_parent_dirs(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    run_dir = tmp_path / "deeply" / "nested" / "cell"
    fc.write_cache(run_dir, X, key)
    assert fc.matrix_path(run_dir).exists()


# ---------------------------------------------------------------------------
# Misses (absent / corrupt) → rebuild
# ---------------------------------------------------------------------------


def test_load_absent_cache_returns_none(tmp_path, panel_and_key):
    _, _, key = panel_and_key
    assert fc.load_cache(tmp_path / "no_such_run", key) is None


def test_corrupt_parquet_returns_none(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    run_dir = tmp_path / "cell"
    fc.write_cache(run_dir, X, key)
    # Truncate the parquet to simulate a crash mid-write / FS corruption.
    fc.matrix_path(run_dir).write_bytes(b"not a parquet")
    assert fc.load_cache(run_dir, key) is None


def test_sidecar_shape_mismatch_returns_none(tmp_path, panel_and_key):
    panel, index_df, key = panel_and_key
    X = _build_matrix(panel, index_df)
    run_dir = tmp_path / "cell"
    fc.write_cache(run_dir, X, key)
    # Tamper the sidecar's recorded shape — must be treated as a miss.
    import json
    sidecar = json.loads(fc.key_path(run_dir).read_text())
    sidecar["n_cols"] = sidecar["n_cols"] + 1
    fc.key_path(run_dir).write_text(json.dumps(sidecar))
    assert fc.load_cache(run_dir, key) is None


# ---------------------------------------------------------------------------
# Key invalidation — changed determining inputs force a rebuild
# ---------------------------------------------------------------------------


def test_key_differs_on_changed_seed(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = fc.panel_signature(panel, index_df)
    kw = {**_BASE_KW, "random_seed": 7}
    other = fc.compute_key(panel_sig=panel_sig, **kw)
    assert other != base_key


def test_key_differs_on_changed_threshold(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = fc.panel_signature(panel, index_df)
    kw = {**_BASE_KW, "target": {**_BASE_KW["target"], "threshold_pct": 20}}
    other = fc.compute_key(panel_sig=panel_sig, **kw)
    assert other != base_key


def test_key_differs_on_changed_horizon(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = fc.panel_signature(panel, index_df)
    kw = {**_BASE_KW, "target": {**_BASE_KW["target"], "horizon_days": 50}}
    other = fc.compute_key(panel_sig=panel_sig, **kw)
    assert other != base_key


def test_compute_key_signature_drops_code_commit_and_code_dirty():
    """Task #190: ``code_commit`` + ``code_dirty`` are HARD-DROPPED from the
    parameter list (not silently ignored). Passing them must TypeError —
    that's how callers find out their invocation needs updating, not by
    silently inheriting the pre-#190 over-strict behavior."""
    import inspect

    sig = inspect.signature(fc.compute_key)
    params = set(sig.parameters)
    assert "code_commit" not in params, (
        "compute_key must not accept code_commit (task #190 dropped it — "
        "see PRs #86/#87 cold-rebuild incident)."
    )
    assert "code_dirty" not in params, (
        "compute_key must not accept code_dirty (task #190 dropped it)."
    )


def test_key_stable_across_what_used_to_be_commit_invalidation(panel_and_key):
    """Task #190 regression guard: pre-#190, two cells differing only by
    ``git rev-parse HEAD`` hashed to different keys, forcing a ~5 h cold
    rebuild on every unrelated commit. Post-#190 the cache key is
    DETERMINED ONLY by the actual feature-build inputs, so re-computing
    the key (no inputs changed) yields the same hash. This is the
    behavioral inverse of the old ``test_key_differs_on_changed_code_commit``.
    """
    panel, index_df, base_key = panel_and_key
    panel_sig = fc.panel_signature(panel, index_df)
    # Re-compute with identical inputs (no commit field at all anymore).
    again = fc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert again == base_key, (
        "compute_key MUST be stable across what used to be commit-level "
        "invalidation — that's the whole point of #190."
    )


def test_key_differs_on_changed_data_snapshot(panel_and_key):
    panel, index_df, base_key = panel_and_key
    # A different data snapshot (one extra ticker / different rows) → new sig.
    panel2, index2 = _synth_panel_with_index(n_tickers=5, seed=0)
    sig2 = fc.panel_signature(panel2, index2)
    other = fc.compute_key(panel_sig=sig2, **_BASE_KW)
    assert other != base_key


def test_changed_seed_misses_cache_and_triggers_rebuild(tmp_path, panel_and_key):
    """End-to-end: a cache written under one key is NOT served under a
    different key (the rebuild path the runner takes)."""
    panel, index_df, base_key = panel_and_key
    X = _build_matrix(panel, index_df)
    run_dir = tmp_path / "cell"
    fc.write_cache(run_dir, X, base_key)

    panel_sig = fc.panel_signature(panel, index_df)
    new_key = fc.compute_key(panel_sig=panel_sig, **{**_BASE_KW, "random_seed": 7})
    assert fc.load_cache(run_dir, new_key) is None  # miss → caller rebuilds
    # But the original key still hits (cache not destroyed).
    assert fc.load_cache(run_dir, base_key) is not None


# ---------------------------------------------------------------------------
# Determinism + non-mutation
# ---------------------------------------------------------------------------


def test_key_is_deterministic(panel_and_key):
    panel, index_df, base_key = panel_and_key
    panel_sig = fc.panel_signature(panel, index_df)
    again = fc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert again == base_key


def test_panel_signature_stable_across_calls(panel_and_key):
    panel, index_df, _ = panel_and_key
    s1 = fc.panel_signature(panel, index_df)
    s2 = fc.panel_signature(panel, index_df)
    assert s1 == s2
    assert s1["panel_rows"] == len(panel)
    assert s1["panel_n_tickers"] == panel.index.get_level_values("ticker").nunique()


# ---------------------------------------------------------------------------
# Task #190 — feature_code_signature carries source_sha256 of gbdt.features
# ---------------------------------------------------------------------------


def test_schema_version_is_v2():
    """Task #190 bumped SCHEMA_VERSION v1 → v2 to invalidate any cache
    written under the old key shape (which keyed on code_commit). Old
    parquets on disk MUST miss cleanly and get rebuilt — never reused
    under an inconsistent schema."""
    assert fc.SCHEMA_VERSION == "v2"


def test_feature_code_signature_includes_source_sha256():
    """The TARGETED invalidator (task #190): a SHA-256 of the
    ``gbdt.features`` module's source text. Without this field, the
    signature would only catch macro-shape changes (family list /
    expected col count), missing in-function bug fixes."""
    sig = fc.feature_code_signature()
    assert "source_sha256" in sig, (
        "feature_code_signature must include source_sha256 of gbdt.features."
    )
    # 64-char hex (SHA-256 hexdigest).
    import re
    assert re.fullmatch(r"[0-9a-f]{64}", sig["source_sha256"]), (
        f"source_sha256 not a SHA-256 hexdigest: {sig['source_sha256']!r}"
    )
    # Coarse shape fields still present for debugging.
    assert "all_families" in sig
    assert "default_lookbacks" in sig
    assert "expected_total_cols" in sig


def test_feature_code_signature_reproducible_no_code_change():
    """Two calls with no code change produce identical signatures —
    in particular, identical ``source_sha256``. Reproducibility is what
    makes the cache stable across processes / agent restarts."""
    sig_a = fc.feature_code_signature()
    sig_b = fc.feature_code_signature()
    assert sig_a == sig_b
    assert sig_a["source_sha256"] == sig_b["source_sha256"]


def test_feature_code_signature_changes_when_features_source_changes(monkeypatch):
    """Monkeypatch the ``gbdt.features`` module to a different source text,
    then re-compute the signature: the source_sha256 must flip. This is
    the "edit features.py → cache invalidates" guarantee."""
    import inspect as _inspect
    from gbdt import features as gbdt_features

    base_sig = fc.feature_code_signature()
    base_hash = base_sig["source_sha256"]

    # Pretend gbdt.features has different source text by patching
    # inspect.getsource at the call site (gbdt.feature_cache imports it
    # locally as ``inspect.getsource``).
    real_getsource = _inspect.getsource
    def fake_getsource(obj):
        if obj is gbdt_features:
            return real_getsource(obj) + "\n# extra line that perturbs the hash\n"
        return real_getsource(obj)

    monkeypatch.setattr("gbdt.feature_cache.inspect.getsource", fake_getsource)

    perturbed_sig = fc.feature_code_signature()
    assert perturbed_sig["source_sha256"] != base_hash, (
        "source_sha256 must flip when gbdt.features source text changes."
    )


def test_compute_key_differs_when_features_source_changes(panel_and_key, monkeypatch):
    """End-to-end: a change to ``gbdt.features`` source (via patched
    ``inspect.getsource``) MUST flow through ``feature_code_signature``
    into ``compute_key`` and yield a different cache key. This is the
    "edit features.py → next run rebuilds" guarantee at the key level."""
    import inspect as _inspect
    from gbdt import features as gbdt_features

    panel, index_df, base_key = panel_and_key

    real_getsource = _inspect.getsource
    def fake_getsource(obj):
        if obj is gbdt_features:
            return real_getsource(obj) + "\n# perturbation\n"
        return real_getsource(obj)

    monkeypatch.setattr("gbdt.feature_cache.inspect.getsource", fake_getsource)

    panel_sig = fc.panel_signature(panel, index_df)
    other = fc.compute_key(panel_sig=panel_sig, **_BASE_KW)
    assert other != base_key
