"""Forecast-result cache for the forecasters framework.

Content-addressed by ``(preset_name, preset_content_hash, identifier,
data_path, start, end, origin, horizon, seed, data_hash)``. A repeat
invocation of the same key hits the cache and returns the previously written
result; editing the preset YAML or any input key invalidates. ``data_hash``
is a content hash of the fetched-and-sliced input series
(:func:`forecasters.data.data_hash`) — without it the key was blind to the
underlying data's *values*, so a ``data_pipelines`` cache correction under
the same ``(identifier, start, end)`` served a stale forecast
(``docs/forecasters/V2_TBD.md`` #18).

Write is atomic (temp + rename) so a crash mid-write never leaves a
half-formed cache directory.

Cache directory layout:

    <cache_root>/<cache_key>/
        summary.json     # everything except `paths`
        paths.npz        # the paths array, compressed
        warnings.json    # raw warnings list (mirrored from summary for grep)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CACHE_ROOT = Path("results/forecasters/forecasts")


def cache_key(
    *,
    preset_name: str,
    preset_content_hash: str,
    identifier: str | None,
    data_path: str | None,
    start: str | None,
    end: str | None,
    origin: str,
    horizon: int,
    seed: int | None,
    data_hash: str | None = None,
) -> str:
    """Stable 16-hex-char key for this call's inputs.

    The truncation to 16 hex chars (64 bits) keeps directory names short
    while leaving collision probability negligible for the cache size we
    expect.

    ``data_hash`` (V2_TBD #18) is the content hash of the fetched-and-sliced
    input DataFrame (``forecasters.data.data_hash``). It makes the key
    values-aware: a backfill or correction in the underlying data cache under
    the same ``(identifier, start, end)`` yields a different key → a cache
    miss and a recompute, never a silently stale forecast. The field is
    ALWAYS folded into the digest (``None`` hashes as ``"<none>"``), so keys
    computed before this field existed can never collide with new ones —
    old entries invalidate once.
    """
    h = hashlib.sha256()
    parts = [
        ("preset_name", preset_name),
        ("preset_content_hash", preset_content_hash),
        ("identifier", identifier),
        ("data_path", data_path),
        ("start", start),
        ("end", end),
        ("origin", origin),
        ("horizon", str(horizon)),
        ("seed", str(seed) if seed is not None else "<none>"),
        ("data_hash", data_hash),
    ]
    for k, v in parts:
        h.update(k.encode())
        h.update(b"=")
        h.update((str(v) if v is not None else "<none>").encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _serialize_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Copy of `result` with `paths` stripped — saved separately."""
    summary = {k: v for k, v in result.items() if k != "paths"}
    return summary


def write_cached(
    key: str,
    result: dict[str, Any],
    cache_root: str | Path | None = None,
) -> Path:
    """Atomically write a result to the cache; return the directory path."""
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    target = root / key

    # Write to a temp directory sibling, then atomically rename. This avoids
    # the "half-written cache" failure mode if the process is killed mid-write.
    parent_tmp = tempfile.mkdtemp(prefix=f".{key}.", dir=root)
    try:
        tmp_path = Path(parent_tmp)
        (tmp_path / "summary.json").write_text(
            json.dumps(_serialize_summary(result), indent=2, default=_json_default)
        )
        np.savez_compressed(tmp_path / "paths.npz", paths=result["paths"])
        (tmp_path / "warnings.json").write_text(
            json.dumps(result.get("warnings", []), indent=2)
        )
        # Final atomic move. If target already exists (someone else wrote it
        # mid-call), discard our work — the existing one is good enough.
        if target.exists():
            shutil.rmtree(tmp_path)
        else:
            os.rename(tmp_path, target)
    except BaseException:
        # Roll back any partial work.
        shutil.rmtree(parent_tmp, ignore_errors=True)
        raise
    return target


def read_cached(
    key: str,
    cache_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return the cached result for ``key`` if present; else None.

    Reads `summary.json` + `paths.npz` and reassembles them into the dict
    shape the dispatcher returned.
    """
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    target = root / key
    summary_p = target / "summary.json"
    paths_p = target / "paths.npz"
    if not summary_p.is_file() or not paths_p.is_file():
        return None
    summary = json.loads(summary_p.read_text())
    paths = np.load(paths_p)["paths"]
    summary["paths"] = paths
    return summary


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy scalars / ndarrays in summary fields."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
