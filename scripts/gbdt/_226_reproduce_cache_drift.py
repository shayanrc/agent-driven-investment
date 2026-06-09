"""Bug #226 — reproduce russell1000 cross-cell cache-key drift, attempt 1.

Hypothesis being probed: the universe-level cache (``gbdt.universe_feature_cache``)
holds 3 distinct russell1000 keys for what should be the same matrix, plus
two same-day sp500 keys 4 h apart. If something in the payload — most likely
``panel_signature`` (the panel's row count / date bounds / index hash) — drifts
WITHIN a single sweep window, sibling cells miss the shared cache and pay the
~5 h rebuild each.

This script:
  1. Loads the russell1000 panel via ``gbdt.data.load_panel`` (cache-only,
     same path the runner uses).
  2. Computes ``panel_signature`` (the data-snapshot fingerprint).
  3. Computes the ``universe_key`` with the canonical russell1000 cell spec
     (`russell1000_up_10pct_25d_dd5pct.yaml`) — the universe key DROPS target,
     so any russell1000 cell with the canonical split + features + seed should
     produce the same key.
  4. Sleeps 60 s — letting any background cache mutation expose itself.
  5. Repeats steps 1-3.
  6. Diffs the two payloads. If anything differs, that's our discriminator.

Then walks the 3 russell1000 sidecar JSONs at
``${SCRATCH_CACHE}/gbdt_feature_cache/`` (keys ``6e716519...``,
``fe67c944...``, ``a20842fc...``) and prints what we CAN extract from them
post-hoc — the pre-#226 sidecars carry only ``key + n_rows + n_cols + columns``,
no payload, which is exactly the limitation P1 fixes for future writes.

Usage::

    export SCRATCH_CACHE=<persistent scratch dir>  # see per-user memory `scratch-cache-path`
    uv run python scripts/gbdt/_226_reproduce_cache_drift.py \
        2>&1 | tee /tmp/v1_226_repro.log

**FROZEN ONE-SHOT.** Bug #226 was diagnosed + fixed in a downstream PR. The
script is preserved for traceability; re-running requires ``SCRATCH_CACHE``
set per per-user memory ``scratch-cache-path`` AND the historical sidecar
JSONs still present in the cache.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from gbdt import data as gbdt_data
from gbdt import feature_cache as gbdt_feature_cache
from gbdt import features as gbdt_features
from gbdt import universe_feature_cache as gbdt_universe_feature_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "configs" / "gbdt" / "experiments" / \
    "russell1000_up_10pct_25d_dd5pct.yaml"
DEFAULT_SPEC_PATH = REPO_ROOT / "configs" / "gbdt" / "default.yaml"
CACHE_DIR = Path(os.environ.get("SCRATCH_CACHE", "<SET-SCRATCH_CACHE>")) / "gbdt_feature_cache"

RUSSELL1000_HISTORIC_KEYS = [
    "6e716519e109697b25e51daa6624acab4a7c78765b32ca8162f13a0f7aafc545",
    "fe67c9445328bbf91fbf3dc4a8850e1ca6f823b957583b447b27e49d3d898e89",
    "a20842fc24c090a2cacd9b35c7deddc1f03b2095be6e54a6fe94bf2e2f6c3e6c",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_merged_spec() -> dict:
    """Mirror runner spec-load: default.yaml + per-cell spec on top."""
    with DEFAULT_SPEC_PATH.open() as f:
        spec = yaml.safe_load(f) or {}
    with SPEC_PATH.open() as f:
        cell_spec = yaml.safe_load(f) or {}

    # Shallow merge — same as how the runner stacks defaults + per-cell.
    for k, v in cell_spec.items():
        if isinstance(v, dict) and isinstance(spec.get(k), dict):
            spec[k].update(v)
        else:
            spec[k] = v
    return spec


def _build_payloads(spec: dict, label: str) -> dict:
    """Re-implement the runner's payload construction (P1-flavour) and return it.

    Returns a dict with ``panel_signature``, ``cell_payload``,
    ``universe_payload``, ``matrix_key``, ``universe_key``, plus wall-time
    breakdown for the load_panel + compute_key steps.
    """
    target = spec["target"]
    dr = spec.get("date_range", {}) or {}
    split_d = spec.get("split", {}) or {}
    min_rows = split_d.get("min_rows_per_ticker", 1600)
    fcfg = spec.get("features", {}) or {}
    lookbacks = tuple(fcfg.get(
        "lookback_windows", gbdt_features.DEFAULT_LOOKBACKS,
    ))
    families = fcfg.get("candidates", "all")
    exclude = fcfg.get("exclude") or []
    seed = int(spec.get("random_seed", 42))

    print(f"[{label}] {_now()} — loading russell1000 panel "
          f"(min_rows={min_rows}, start={dr.get('start')!r}, "
          f"end={dr.get('end')!r}) …", flush=True)
    t1 = time.time()
    panel_obj = gbdt_data.load_panel(
        target["universe"],
        start=dr.get("start"),
        end=dr.get("end"),
        min_rows=min_rows,
        repo_root=REPO_ROOT,
    )
    panel_load_sec = time.time() - t1
    print(f"[{label}] panel loaded in {panel_load_sec:.1f}s "
          f"rows={len(panel_obj.panel)} tickers_kept={len(panel_obj.tickers_kept)} "
          f"index_rows={len(panel_obj.index_series)}", flush=True)

    t1 = time.time()
    panel_sig = gbdt_feature_cache.panel_signature(
        panel_obj.panel, panel_obj.index_series,
    )
    matrix_key = gbdt_feature_cache.compute_key(
        universe=target["universe"],
        target=target,
        split=split_d,
        lookbacks=lookbacks,
        families=families,
        exclude=exclude,
        random_seed=seed,
        panel_sig=panel_sig,
    )
    universe_key = gbdt_universe_feature_cache.compute_key(
        universe=target["universe"],
        split=split_d,
        lookbacks=lookbacks,
        families=families,
        exclude=exclude,
        random_seed=seed,
        panel_sig=panel_sig,
    )
    keys_sec = time.time() - t1

    cell_payload = {
        "schema_version": gbdt_feature_cache.SCHEMA_VERSION,
        "universe": target["universe"],
        "target": {
            "direction": target.get("direction"),
            "threshold_pct": target.get("threshold_pct"),
            "horizon_days": target.get("horizon_days"),
            "max_drawdown": target.get("max_drawdown"),
            "uniqueness_weighting": bool(target.get("uniqueness_weighting", True)),
        },
        "split": {
            "train_rows": split_d.get("train_rows"),
            "val_rows": split_d.get("val_rows"),
            "eval_rows": split_d.get("eval_rows"),
            "test_rows": split_d.get("test_rows"),
            "min_rows_per_ticker": split_d.get("min_rows_per_ticker"),
        },
        "features": {
            "lookbacks": list(lookbacks),
            "families": (
                families if isinstance(families, str) else sorted(families)
            ),
            "exclude": sorted(exclude or []),
            "code_signature": gbdt_feature_cache.feature_code_signature(),
        },
        "random_seed": seed,
        "panel_signature": panel_sig,
    }
    universe_payload = {
        "schema_version": gbdt_universe_feature_cache.SCHEMA_VERSION,
        "universe": target["universe"],
        "split": cell_payload["split"],
        "features": cell_payload["features"],
        "random_seed": seed,
        "panel_signature": panel_sig,
    }

    print(f"[{label}] panel_signature + compute_key in {keys_sec:.2f}s", flush=True)
    print(f"[{label}] matrix_key   = {matrix_key}", flush=True)
    print(f"[{label}] universe_key = {universe_key}", flush=True)
    print(f"[{label}] panel_signature = "
          f"{json.dumps(panel_sig, indent=2, default=str)}", flush=True)

    return {
        "label": label,
        "captured_at": _now(),
        "panel_load_sec": panel_load_sec,
        "keys_sec": keys_sec,
        "panel_signature": panel_sig,
        "cell_payload": cell_payload,
        "universe_payload": universe_payload,
        "matrix_key": matrix_key,
        "universe_key": universe_key,
    }


def _diff_dicts(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Recursive shallow diff: list dotted-paths where ``a`` and ``b`` differ."""
    out: list[str] = []
    keys = set(a) | set(b)
    for k in sorted(keys):
        path = f"{prefix}.{k}" if prefix else k
        if k not in a:
            out.append(f"{path}: MISSING in run-1")
        elif k not in b:
            out.append(f"{path}: MISSING in run-2")
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out.extend(_diff_dicts(a[k], b[k], path))
        elif a[k] != b[k]:
            out.append(f"{path}: {a[k]!r}  ->  {b[k]!r}")
    return out


def _analyse_historic_sidecars() -> None:
    """The 3 russell1000 sidecars at the documented keys carry only
    minimal info (no payload — that's exactly what P1 fixes going forward).
    We document what we CAN extract: mtime, sidecar fields, file sizes."""
    print("\n" + "=" * 72, flush=True)
    print("Post-hoc inspection of the 3 known russell1000 universe-cache keys",
          flush=True)
    print("=" * 72, flush=True)
    for key in RUSSELL1000_HISTORIC_KEYS:
        kpath = CACHE_DIR / f"{key}.key.json"
        mpath = CACHE_DIR / f"{key}.parquet"
        if not kpath.exists():
            print(f"  [{key[:12]}…] sidecar MISSING at {kpath}", flush=True)
            continue
        kstat = kpath.stat()
        mstat = mpath.stat() if mpath.exists() else None
        try:
            sidecar = json.loads(kpath.read_text())
        except Exception as exc:  # pragma: no cover
            print(f"  [{key[:12]}…] sidecar unreadable: {exc!r}", flush=True)
            continue
        # mtime: when the sidecar was written. Same proxy as when the matrix
        # was built (atomic temp+replace = one moment in time).
        mtime = datetime.fromtimestamp(kstat.st_mtime, tz=timezone.utc)
        print(f"\n  KEY {key}", flush=True)
        print(f"    sidecar:   {kpath.name}  ({kstat.st_size} B)", flush=True)
        print(f"    parquet:   "
              f"{mpath.name if mstat else '(missing)'}  "
              f"({mstat.st_size if mstat else 0} B)", flush=True)
        print(f"    written:   {mtime.isoformat()}  (sidecar mtime)", flush=True)
        print(f"    schema:    {sidecar.get('schema_version')!r}", flush=True)
        print(f"    n_rows:    {sidecar.get('n_rows')}", flush=True)
        print(f"    n_cols:    {sidecar.get('n_cols')}", flush=True)
        cols = sidecar.get("columns") or []
        if cols:
            print(f"    cols(5):   {cols[:5]} … ({len(cols)} total)", flush=True)
        # Is the payload field present? (Only true for NEW writes post-#226.)
        has_payload = "payload" in sidecar
        print(f"    payload:   "
              f"{'PRESENT (post-#226)' if has_payload else 'ABSENT (pre-#226)'}",
              flush=True)


def main() -> None:
    print(f"# Bug #226 reproduction — started {_now()}", flush=True)
    print(f"# spec: {SPEC_PATH}", flush=True)
    print(f"# default: {DEFAULT_SPEC_PATH}", flush=True)
    print(f"# repo_root: {REPO_ROOT}", flush=True)
    print(f"# pid: {os.getpid()}", flush=True)

    spec = _load_merged_spec()
    print(f"# merged target = {json.dumps(spec.get('target', {}), default=str)}",
          flush=True)
    print(f"# merged split  = {json.dumps(spec.get('split', {}), default=str)}",
          flush=True)
    print(f"# merged features = {json.dumps(spec.get('features', {}), default=str)}",
          flush=True)
    print(f"# random_seed   = {spec.get('random_seed', 42)}", flush=True)
    print(f"# date_range    = {json.dumps(spec.get('date_range', {}), default=str)}",
          flush=True)

    run1 = _build_payloads(spec, label="run-1")
    print(f"\n# sleeping 60 s to expose any background cache mutation …",
          flush=True)
    time.sleep(60)
    run2 = _build_payloads(spec, label="run-2")

    print("\n" + "=" * 72, flush=True)
    print("DIFF — universe_payload run-1 vs run-2", flush=True)
    print("=" * 72, flush=True)
    diffs = _diff_dicts(run1["universe_payload"], run2["universe_payload"])
    if not diffs:
        print("  IDENTICAL — universe_payload unchanged across the 60-s window.",
              flush=True)
        print("  universe_key matches:", run1["universe_key"] == run2["universe_key"],
              flush=True)
    else:
        print(f"  {len(diffs)} field(s) differ:", flush=True)
        for d in diffs:
            print(f"    - {d}", flush=True)

    print("\nDIFF — cell_payload (full target tuple) run-1 vs run-2", flush=True)
    cell_diffs = _diff_dicts(run1["cell_payload"], run2["cell_payload"])
    if not cell_diffs:
        print("  IDENTICAL — cell_payload unchanged across the 60-s window.",
              flush=True)
        print("  matrix_key matches:", run1["matrix_key"] == run2["matrix_key"],
              flush=True)
    else:
        print(f"  {len(cell_diffs)} field(s) differ:", flush=True)
        for d in cell_diffs:
            print(f"    - {d}", flush=True)

    print("\nKey comparison summary:", flush=True)
    print(f"  run-1 universe_key = {run1['universe_key']}", flush=True)
    print(f"  run-2 universe_key = {run2['universe_key']}", flush=True)
    print(f"  same?              = {run1['universe_key'] == run2['universe_key']}",
          flush=True)
    print(f"  run-1 matrix_key   = {run1['matrix_key']}", flush=True)
    print(f"  run-2 matrix_key   = {run2['matrix_key']}", flush=True)
    print(f"  same?              = {run1['matrix_key'] == run2['matrix_key']}",
          flush=True)

    print("\nCheck against the 3 historic russell1000 universe-cache keys:",
          flush=True)
    for key in RUSSELL1000_HISTORIC_KEYS:
        match = (
            run1["universe_key"] == key or run2["universe_key"] == key
        )
        marker = "  <-- matches today's key(s)" if match else ""
        print(f"  {key}{marker}", flush=True)

    _analyse_historic_sidecars()

    print(f"\n# Bug #226 reproduction — done {_now()}", flush=True)


if __name__ == "__main__":
    main()
