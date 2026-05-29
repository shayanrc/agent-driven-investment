"""Per-run feature-matrix cache for the agent-driven FS+HP loop (task #181).

The ``agent_file_protocol`` loop is exit-and-resume: every ``--resume`` is a
FRESH process that reloads only the loop-state checkpoint (no model, no
matrix) and would otherwise rebuild the full candidate feature matrix from
scratch. On ``nifty50`` the build is ~5 min (invisible), but on heavy panels
(``sp500``) it is ~3 HOURS — so a 10-iteration agent loop incurs ~30 h of
redundant rebuilds. This module persists the built matrix to the run's
artifact dir and lets ``--resume`` (and re-runs of the same cell) reuse it.

Design
------
* **What is cached.** Only the full candidate feature matrix ``X`` (the
  expensive ``build_feature_matrix`` output, ALL candidate columns). FS
  pruning between iterations is a *column subset* of this same matrix, so we
  cache the full matrix and let the loop subset columns per-iteration — the
  cache is NOT keyed on the per-iteration feature subset. The target ``y`` and
  the uniqueness ``sample_weights`` are cheap, deterministic functions of the
  panel + target params (index-only computations), so they are re-derived on
  every load rather than cached — keeping the cache to a single artifact.
* **File format.** ``<run_dir>/_feature_matrix_cache.parquet`` (the matrix,
  preserving the ``MultiIndex(date, ticker)``) + a sidecar
  ``<run_dir>/_feature_matrix_cache.key.json`` holding the cache key and
  metadata. Mirrors how ``/gbdt-diagnose`` caches ``_insample_matrix.parquet``.
* **Cache key.** A SHA-256 over a canonical-JSON dict of EVERYTHING that
  determines the matrix: universe, the full target tuple, the split config,
  the candidate feature set + their definition version, ``random_seed``, the
  feature-code version signature + code commit, and a data-snapshot signature
  (panel row count + min/max date + a hash of the panel index, plus the index
  series row count + max date). A mismatch (changed seed/threshold/data/code)
  or an absent/corrupt cache forces a rebuild + cache refresh.

Correctness contract
---------------------
The loaded matrix MUST be identical to what ``build_feature_matrix`` (followed
by the ``dropna(axis=1, how="all")`` the runner applies) would produce. Two
guards enforce this:

1. The key captures every input to the build, so a key match implies the
   inputs are identical.
2. A round-trip is structurally faithful: parquet preserves dtypes + the
   MultiIndex + column order; on load we verify the persisted column count +
   index length match the sidecar, and any read error falls back to a rebuild.

When in doubt, rebuild — a wrong reuse is unacceptable; a redundant rebuild is
merely slow. This is purely a build-time optimization: it never changes
results, determinism, or the finalization-retrain contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Bumped on any breaking change to the cache key composition or the persisted
# layout, so a stale cache from an older code version never key-matches.
SCHEMA_VERSION = "v1"

_MATRIX_FILENAME = "_feature_matrix_cache.parquet"
_KEY_FILENAME = "_feature_matrix_cache.key.json"


def matrix_path(run_dir: str | Path) -> Path:
    """Path to the cached matrix parquet under ``run_dir``."""
    return Path(run_dir) / _MATRIX_FILENAME


def key_path(run_dir: str | Path) -> Path:
    """Path to the cache-key sidecar JSON under ``run_dir``."""
    return Path(run_dir) / _KEY_FILENAME


# ---------------------------------------------------------------------------
# Signature helpers (pure)
# ---------------------------------------------------------------------------


def feature_code_signature() -> dict[str, Any]:
    """A signature of the feature-definition code.

    Captures the canonical family list, the default lookbacks, and the
    expected total column count from :mod:`gbdt.features`. Combined with the
    git ``code_commit`` in the key, this invalidates the cache when the
    feature engineering changes — even if the spec (and thus the rest of the
    key) is identical.
    """
    from gbdt import features as gbdt_features

    return {
        "all_families": list(gbdt_features._ALL_FAMILIES),
        "default_lookbacks": list(gbdt_features.DEFAULT_LOOKBACKS),
        "expected_total_cols": int(gbdt_features.EXPECTED_TOTAL_COLS),
    }


def panel_signature(panel: pd.DataFrame, index_df: pd.DataFrame) -> dict[str, Any]:
    """A data-snapshot signature of the loaded panel + index series.

    The signature pins the matrix to the exact data snapshot it was built
    from. It combines coarse summaries (row counts, min/max date) with a hash
    of the full ``(date, ticker)`` index — so a re-cached run on a refreshed
    cache (rows appended, tickers added/dropped, dates shifted) misses the key
    and rebuilds. The hash is computed over the index tuples, which fully
    determine which rows entered the build; the OHLCV values are not hashed
    (cache freshness is governed by the snapshot identity, and re-fetching the
    same dates from the same provider is deterministic per data_pipelines).
    """
    idx = panel.index
    # Stable, order-sensitive hash of the panel's MultiIndex tuples. The panel
    # is ``sort_index``-ed by the loader, so the order is deterministic.
    h = hashlib.sha256()
    for tup in idx.to_flat_index():
        # ``str(tup)`` renders ``(Timestamp, ticker)`` deterministically.
        h.update(str(tup).encode("utf-8"))

    def _date_bounds(frame_index: pd.Index) -> tuple[str, str]:
        try:
            dates = frame_index.get_level_values("date")
        except (KeyError, AttributeError):
            dates = frame_index
        if len(dates) == 0:
            return ("", "")
        return (str(pd.Timestamp(dates.min())), str(pd.Timestamp(dates.max())))

    pmin, pmax = _date_bounds(idx)
    imin, imax = _date_bounds(index_df.index)
    return {
        "panel_rows": int(len(panel)),
        "panel_n_tickers": int(idx.get_level_values("ticker").nunique()),
        "panel_date_min": pmin,
        "panel_date_max": pmax,
        "panel_index_hash": h.hexdigest(),
        "index_series_rows": int(len(index_df)),
        "index_series_date_min": imin,
        "index_series_date_max": imax,
    }


def compute_key(
    *,
    universe: str,
    target: dict,
    split: dict,
    lookbacks: Any,
    families: Any,
    exclude: Any,
    random_seed: int,
    code_commit: str,
    code_dirty: bool,
    panel_sig: dict,
) -> str:
    """Compute the deterministic cache key (a SHA-256 hex digest).

    The key is a hash of a canonical-JSON dict of every input that determines
    the built matrix. Keying on more than strictly determines ``X`` (e.g. the
    target tuple, which only affects ``y``) is intentional and safe: it only
    ever causes an extra correct rebuild, never an incorrect reuse. On
    ``--resume`` the cell + spec are identical, so the key matches and the
    cache hits.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "universe": universe,
        # Full target tuple — every field that can change the labeled cell.
        "target": {
            "direction": target.get("direction"),
            "threshold_pct": target.get("threshold_pct"),
            "horizon_days": target.get("horizon_days"),
            "max_drawdown": target.get("max_drawdown"),
            "uniqueness_weighting": bool(target.get("uniqueness_weighting", True)),
        },
        "split": {
            "train_rows": split.get("train_rows"),
            "val_rows": split.get("val_rows"),
            "eval_rows": split.get("eval_rows"),
            "test_rows": split.get("test_rows"),
            "min_rows_per_ticker": split.get("min_rows_per_ticker"),
        },
        # Candidate feature set + its definition version.
        "features": {
            "lookbacks": list(lookbacks),
            "families": (
                families if isinstance(families, str) else sorted(families)
            ),
            "exclude": sorted(exclude or []),
            "code_signature": feature_code_signature(),
        },
        "random_seed": int(random_seed),
        "code_commit": code_commit,
        "code_dirty": bool(code_dirty),
        "panel_signature": panel_sig,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_cache(run_dir: str | Path, X: pd.DataFrame, key: str) -> Path:
    """Persist the full candidate matrix ``X`` + the cache-key sidecar.

    Writes are best-effort *atomic* via a temp-file rename so a crash
    mid-write never leaves a half-written parquet that a later load mistakes
    for a valid cache. Returns the matrix path written.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mpath = matrix_path(run_dir)
    kpath = key_path(run_dir)

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


def load_cache(run_dir: str | Path, expected_key: str) -> pd.DataFrame | None:
    """Load the cached matrix iff the persisted key matches ``expected_key``.

    Returns the matrix DataFrame on a verified hit, or ``None`` on any miss:
    absent files, a key mismatch, a corrupt/unreadable parquet, or a sidecar
    that disagrees with the parquet's shape. A ``None`` return means the caller
    rebuilds — the conservative, correctness-preserving fallback.
    """
    run_dir = Path(run_dir)
    mpath = matrix_path(run_dir)
    kpath = key_path(run_dir)
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
    # Structural sanity: the persisted shape must match the sidecar.
    if int(sidecar.get("n_rows", -1)) != len(X):
        return None
    if int(sidecar.get("n_cols", -1)) != X.shape[1]:
        return None
    return X
