"""Cross-cell, universe-level feature-matrix cache (task #183).

The per-run cache in :mod:`gbdt.feature_cache` (task #181) keeps a built matrix
beside its run's artifact dir and lets ``--resume`` skip the rebuild. That
helped within a single cell — but a same-universe sweep (e.g. russell1000
across 20 ``(threshold, horizon, drawdown)`` tuples) re-pays the full feature
build per cell (~5 h on russell1000), because each cell lives in its OWN run
dir and the per-cell key includes the target tuple.

The 279-column candidate matrix produced by :func:`gbdt.features.build_feature_matrix`
is **purely a function of the universe panel + index + lookbacks + feature
families** — the target tuple (direction / threshold / horizon / drawdown) is
applied AFTER the build (Phase 3 in the runner). So all 20 sibling cells of a
russell1000 sweep are building the SAME matrix. This module persists that
matrix in a shared location keyed on its actual inputs (target tuple DROPPED),
turning ``N × build`` into ``1 × build + N × labels``.

Design
------
* **What is cached.** The full candidate feature matrix ``X`` returned by
  ``build_feature_matrix(...).dropna(axis=1, how="all")`` — bit-identical to
  what the runner would produce on a cold build. The target ``y`` and
  uniqueness ``sample_weights`` are cheap, deterministic functions of the panel
  + target params and are re-derived per-cell on every run — keeping the shared
  cache to a single artifact per universe-build.
* **Storage location.** ``<data_root>/gbdt_feature_cache/<key>.parquet`` +
  ``<key>.key.json``. ``data_root`` is whatever the runner resolved (the
  ``data/`` symlink → ``/mnt/.../cache_data`` for this checkout per the worktree
  convention). A dedicated subdir keeps the gbdt cache cleanly namespaced
  alongside ``processed.db`` (the data_pipelines SQLite cache).
* **Cache key.** A SHA-256 over a canonical-JSON dict of EVERYTHING that
  determines the matrix and ONLY those things. Differs from the per-cell key
  in :mod:`gbdt.feature_cache` by **excluding the target tuple** (direction /
  threshold_pct / horizon_days / max_drawdown / uniqueness_weighting). Kept
  fields: universe, the full split config, candidate feature set + their
  definition version, ``random_seed`` (carried for future-proofing — current
  features are deterministic-without-seed, but excluding it would break the
  contract the moment a feature gains a seeded subsampler), the feature-code
  version signature + code commit, and a data-snapshot signature (panel rows
  + min/max date + index hash).
* **Atomicity.** Writes are temp-file + ``os.replace`` so a crash mid-write
  never leaves a half-baked parquet that a later run mistakes for a hit.
  Mirrors the :mod:`gbdt.feature_cache` discipline.

Two-level cache flow (runner side, see ``__main__.py``)
-------------------------------------------------------
1. **Try the per-cell cache first** (cheapest hit: same cell + same spec +
   same data ⇒ skip both the build AND the universe-level cache touch).
2. **On per-cell miss, try the universe cache.** Hit ⇒ load the matrix in
   <2 s, then write it into the per-cell cache so a subsequent resume of this
   cell hits the per-cell layer.
3. **On both miss, build.** Then write BOTH caches.

Correctness contract
--------------------
A loaded matrix MUST be identical to what ``build_feature_matrix`` (followed
by ``dropna(axis=1, how="all")``) would produce. Two guards:

1. The key captures every input to the build, so a key match implies the
   inputs are identical. The target tuple is the ONLY thing dropped, and the
   target is applied as a separate column AFTER the build — so dropping it
   cannot affect ``X``'s values.
2. A round-trip is structurally faithful (parquet preserves dtypes + the
   MultiIndex + column order); on load we verify the persisted column count +
   index length match the sidecar, and any read error falls back to a miss
   (caller rebuilds — the conservative, correctness-preserving fallback).

When in doubt, miss: a wrong reuse is unacceptable; a redundant rebuild is
merely slow. This is purely a build-time optimization — it never changes
results, determinism, or the finalization-retrain contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from gbdt import feature_cache as _per_cell_cache

# Bumped on any breaking change to the cache key composition or the persisted
# layout, so a stale cache from an older code version never key-matches.
SCHEMA_VERSION = "v1"

# Default subdir under ``data_root`` where shared matrices land.
DEFAULT_CACHE_SUBDIR = "gbdt_feature_cache"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def cache_root(data_root: str | Path, subdir: str = DEFAULT_CACHE_SUBDIR) -> Path:
    """Resolve the on-disk root for the universe-level feature cache.

    ``data_root`` should be the runner's resolved data root (the realpath of
    the ``data/`` symlink in this checkout). ``subdir`` is split out so tests
    can isolate the cache under ``tmp_path``.
    """
    return Path(data_root) / subdir


def matrix_path(data_root: str | Path, key: str,
                subdir: str = DEFAULT_CACHE_SUBDIR) -> Path:
    """Path to the cached matrix parquet for ``key`` under ``data_root``."""
    return cache_root(data_root, subdir) / f"{key}.parquet"


def key_path(data_root: str | Path, key: str,
             subdir: str = DEFAULT_CACHE_SUBDIR) -> Path:
    """Path to the cache-key sidecar JSON for ``key`` under ``data_root``."""
    return cache_root(data_root, subdir) / f"{key}.key.json"


# ---------------------------------------------------------------------------
# Key composition (DROPS the target tuple)
# ---------------------------------------------------------------------------


def compute_key(
    *,
    universe: str,
    split: dict,
    lookbacks: Any,
    families: Any,
    exclude: Any,
    random_seed: int,
    code_commit: str,
    code_dirty: bool,
    panel_sig: dict,
) -> str:
    """Compute the deterministic universe-level cache key (SHA-256 hex).

    The payload INTENTIONALLY excludes every field of the target tuple
    (direction, threshold_pct, horizon_days, max_drawdown, uniqueness_weighting).
    The target is applied AFTER ``build_feature_matrix`` in the runner (Phase 3
    in ``__main__.py``), so two cells with the same universe + split + features
    + seed + code + data snapshot build the SAME ``X`` regardless of target.

    The key payload is otherwise the same shape as
    :func:`gbdt.feature_cache.compute_key`, sharing the
    :func:`gbdt.feature_cache.feature_code_signature` and
    :func:`gbdt.feature_cache.panel_signature` helpers — those are the
    single sources of truth for "what defines a feature build" + "what defines
    a data snapshot", and we deliberately reuse them so a code change to the
    feature engineering invalidates both caches at once.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "universe": universe,
        # Target tuple is INTENTIONALLY DROPPED — that's the whole point of
        # this cache layer. The target is applied after build_feature_matrix
        # and cannot affect X.
        "split": {
            "train_rows": split.get("train_rows"),
            "val_rows": split.get("val_rows"),
            "eval_rows": split.get("eval_rows"),
            "test_rows": split.get("test_rows"),
            "min_rows_per_ticker": split.get("min_rows_per_ticker"),
        },
        "features": {
            "lookbacks": list(lookbacks),
            "families": (
                families if isinstance(families, str) else sorted(families)
            ),
            "exclude": sorted(exclude or []),
            "code_signature": _per_cell_cache.feature_code_signature(),
        },
        "random_seed": int(random_seed),
        "code_commit": code_commit,
        "code_dirty": bool(code_dirty),
        "panel_signature": panel_sig,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Persistence (atomic, mirrors gbdt.feature_cache)
# ---------------------------------------------------------------------------


def write_cache(data_root: str | Path, X: pd.DataFrame, key: str,
                 subdir: str = DEFAULT_CACHE_SUBDIR) -> Path:
    """Persist ``X`` + a sidecar manifest under ``<data_root>/<subdir>/``.

    Atomic via temp-file + ``os.replace`` — a crash mid-write never leaves a
    half-written parquet that a later load mistakes for a valid cache. Returns
    the matrix path written.
    """
    root = cache_root(data_root, subdir)
    root.mkdir(parents=True, exist_ok=True)
    mpath = matrix_path(data_root, key, subdir)
    kpath = key_path(data_root, key, subdir)

    tmp_m = mpath.with_suffix(mpath.suffix + ".tmp")
    X.to_parquet(tmp_m)
    tmp_m.replace(mpath)

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "key": key,
        "n_rows": int(len(X)),
        "n_cols": int(X.shape[1]),
        "columns": list(map(str, X.columns)),
    }
    tmp_k = kpath.with_suffix(kpath.suffix + ".tmp")
    tmp_k.write_text(json.dumps(sidecar, indent=2))
    tmp_k.replace(kpath)
    return mpath


def load_cache(data_root: str | Path, expected_key: str,
                subdir: str = DEFAULT_CACHE_SUBDIR) -> pd.DataFrame | None:
    """Load the cached matrix iff the persisted key matches ``expected_key``.

    Returns the matrix on a verified hit, or ``None`` on any miss: absent
    files, a key mismatch, a corrupt/unreadable parquet, or a sidecar that
    disagrees with the parquet's shape. ``None`` ⇒ caller rebuilds (the
    conservative, correctness-preserving fallback).
    """
    mpath = matrix_path(data_root, expected_key, subdir)
    kpath = key_path(data_root, expected_key, subdir)
    if not mpath.exists() or not kpath.exists():
        return None
    try:
        sidecar = json.loads(kpath.read_text())
    except (OSError, ValueError):
        return None
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        return None
    if sidecar.get("key") != expected_key:
        return None
    try:
        X = pd.read_parquet(mpath)
    except Exception:
        # A truncated/corrupt parquet — treat as a miss and rebuild.
        return None
    if int(sidecar.get("n_rows", -1)) != len(X):
        return None
    if int(sidecar.get("n_cols", -1)) != X.shape[1]:
        return None
    return X


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_CACHE_SUBDIR",
    "cache_root",
    "matrix_path",
    "key_path",
    "compute_key",
    "write_cache",
    "load_cache",
]
