"""V1.3 Option B — FS-prefit phase (Phase 1.4 of /gbdt-experiment).

Train ONE default-HP fit on the full feature matrix, sort features by
importance, drop everything below the 1% cliff (D11 Q2.A; matches the
cell-5 manual workflow's ~130-of-279 kept-feature outcome). The kept
feature list is the input to Phase 1.5 (scout) and Phase 2 (iter_0).

Cache layout (D6.2.A): kept-feature list cached at universe-snapshot level
keyed by ``(universe, features_source_sha256, snapshot_end,
default_hp_sha256)``. Payload is a ~1 KB JSON of the result dict (kept +
dropped + importance metadata). Atomic temp+rename writes match the
project's standing cache contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FSPrefitResult:
    """One FS-prefit fit's outcome — the kept-feature set for downstream
    scout + iter_0 phases.

    ``kept_features`` and ``dropped_features`` together cover the full input
    feature pool (no overlap; the union equals the input set). The cliff is
    a hard cutoff: a feature is kept iff its importance ≥ ``cliff_threshold``
    where ``cliff_threshold = cliff_pct * top_importance``.
    """

    kept_features: list[str]
    dropped_features: list[str]
    top_importance: float
    cliff_threshold: float
    backend: str
    default_hp_sha256: str
    cliff_pct: float
    fit_seconds: float
    # importance dict (kept-features only) — useful for diagnostics +
    # serialized in the JSON cache payload.
    importance_kept: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept_features": list(self.kept_features),
            "dropped_features": list(self.dropped_features),
            "top_importance": float(self.top_importance),
            "cliff_threshold": float(self.cliff_threshold),
            "backend": str(self.backend),
            "default_hp_sha256": str(self.default_hp_sha256),
            "cliff_pct": float(self.cliff_pct),
            "fit_seconds": float(self.fit_seconds),
            "importance_kept": {
                str(k): float(v) for k, v in self.importance_kept.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FSPrefitResult":
        return cls(
            kept_features=list(payload["kept_features"]),
            dropped_features=list(payload.get("dropped_features", [])),
            top_importance=float(payload.get("top_importance", 0.0)),
            cliff_threshold=float(payload.get("cliff_threshold", 0.0)),
            backend=str(payload.get("backend", "")),
            default_hp_sha256=str(payload.get("default_hp_sha256", "")),
            cliff_pct=float(payload.get("cliff_pct", 0.01)),
            fit_seconds=float(payload.get("fit_seconds", 0.0)),
            importance_kept={
                str(k): float(v) for k, v in (
                    payload.get("importance_kept") or {}
                ).items()
            },
        )


# ---------------------------------------------------------------------------
# Cliff cut
# ---------------------------------------------------------------------------


def cliff_cut(
    importance: pd.Series,
    cliff_pct: float = 0.01,
) -> tuple[list[str], list[str], float, float]:
    """Apply the D11 Q2.A cliff cut: keep features with importance ≥
    cliff_pct * top.

    Returns ``(kept, dropped, top_importance, cliff_threshold)``.

    Edge cases:
    - Empty importance → all-empty.
    - All-zero importance → keep everything (no signal to cliff on).
    - Top importance == 0 → keep everything (avoid divide-by-zero).
    """
    if importance is None or len(importance) == 0:
        return [], [], 0.0, 0.0
    imp = importance.sort_values(ascending=False)
    top = float(imp.iloc[0]) if len(imp) else 0.0
    if top <= 0.0:
        return list(imp.index), [], 0.0, 0.0
    threshold = float(cliff_pct) * top
    kept_mask = imp >= threshold
    kept = list(imp[kept_mask].index)
    dropped = list(imp[~kept_mask].index)
    return kept, dropped, top, threshold


# ---------------------------------------------------------------------------
# Main entry — fit + cliff cut
# ---------------------------------------------------------------------------


def run_fs_prefit(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
    fit_one: Callable,
    backend: str,
    default_hp: dict,
    cliff_pct: float = 0.01,
) -> FSPrefitResult:
    """Phase 1.4: train one default-HP fit → sort by importance → cliff-cut.

    Q1.A: single fit, full feature matrix, default HPs from spec
    (``backend.hp_starting`` after defaults merge).

    Q2.A: cliff at 1% of top importance (default; spec can override via
    ``backend.fs_prefit.cliff_pct``).

    Q3.A: use the spec's declared backend (one method per backend) —
    CatBoost uses native gain importance, XGBoost uses gain importance.

    ``fit_one(hp, X_train, y_train, w_train, X_val, y_val, w_val) ->
    pd.Series`` — the callable is provided by ``walk_forward_train`` so the
    runner stays backend-agnostic at this layer. It MUST return a feature
    name → importance pd.Series (the model's ``feature_importance("native")``
    output).
    """
    t0 = time.time()
    importance = fit_one(
        hp=dict(default_hp),
        X_train=X_train, y_train=y_train, w_train=w_train,
        X_val=X_val, y_val=y_val, w_val=w_val,
    )
    fit_seconds = time.time() - t0

    if not isinstance(importance, pd.Series):
        importance = pd.Series(importance)
    importance = importance.astype(float)

    kept, dropped, top, threshold = cliff_cut(importance, cliff_pct=cliff_pct)
    importance_kept = {f: float(importance[f]) for f in kept}

    return FSPrefitResult(
        kept_features=kept,
        dropped_features=dropped,
        top_importance=float(top),
        cliff_threshold=float(threshold),
        backend=str(backend),
        default_hp_sha256=hp_sha256(default_hp),
        cliff_pct=float(cliff_pct),
        fit_seconds=float(fit_seconds),
        importance_kept=importance_kept,
    )


# ---------------------------------------------------------------------------
# Cache key + load/save (D6.2.A)
# ---------------------------------------------------------------------------


def hp_sha256(hp: dict) -> str:
    """Canonical SHA-256 of an HP dict — used as a cache-key component.

    The HP dict is JSON-canonicalized (sort_keys=True, no whitespace) so
    semantically-equal dicts produce identical hashes.
    """
    payload = json.dumps(hp, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fs_prefit_cache_key(
    *,
    universe: str,
    features_source_sha256: str,
    snapshot_end: str,
    default_hp_sha256: str,
) -> str:
    """D6.2.A: cache key for the FS-prefit kept-feature list.

    Components:
    - ``universe``: target universe name (e.g. ``"nasdaq100"``).
    - ``features_source_sha256``: the gbdt-features build's SHA over the
      raw OHLCV inputs (matches the universe-feature-cache key).
    - ``snapshot_end``: ISO date string pinning the panel end.
    - ``default_hp_sha256``: SHA over the default HP dict the prefit was
      run with — so a spec change to defaults invalidates the cache.
    """
    payload = json.dumps(
        {
            "universe": str(universe),
            "features_source_sha256": str(features_source_sha256),
            "snapshot_end": str(snapshot_end),
            "default_hp_sha256": str(default_hp_sha256),
        },
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _cache_path(cache_root: str | Path, key: str) -> Path:
    return Path(cache_root) / "fs_prefit" / f"{key}.json"


def load_fs_prefit_cache(cache_root: str | Path, key: str) -> FSPrefitResult | None:
    """D6.2.A: load cached kept-feature list. Returns None on miss or
    on any error reading/parsing the cache entry (treated as a miss — the
    caller refits and rewrites).
    """
    path = _cache_path(cache_root, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return FSPrefitResult.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def save_fs_prefit_cache(
    cache_root: str | Path, key: str, result: FSPrefitResult,
) -> Path:
    """Atomic temp+rename write per the project's cache contract.

    Creates the parent directory tree on demand. Returns the path written.
    """
    path = _cache_path(cache_root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tempfile+os.replace = atomic on POSIX (and on Windows for ≥3.3).
    fd, tmp = tempfile.mkstemp(
        prefix=f"{key}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(result.to_dict(), fh)
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup on any failure path.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


__all__ = [
    "FSPrefitResult",
    "cliff_cut",
    "run_fs_prefit",
    "hp_sha256",
    "fs_prefit_cache_key",
    "load_fs_prefit_cache",
    "save_fs_prefit_cache",
]
