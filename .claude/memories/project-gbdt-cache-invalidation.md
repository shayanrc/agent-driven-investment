---
name: project-gbdt-cache-invalidation
description: gbdt feature cache invalidates iff src/gbdt/features.py source changes — not git commit, not unrelated module edits. Two-level cache (per-cell + universe), schema v2 post-#190
metadata:
  type: project
---

The gbdt feature cache is two-level and invalidates on a **targeted source-hash of `src/gbdt/features.py`** — not on git commits and not on edits to unrelated modules. This is the post-#190 (2026-05-31) contract; before #190 the cache invalidated on every commit regardless of what the commit touched, which cost the r1k trio ~15 h of unnecessary cold rebuild.

**Two-level cache layout:**

- **Per-cell cache** — `<run_dir>/_feature_matrix_cache.parquet` + `.key.json` sidecar. Target tuple IS in the key, so a single cell's cache survives `--resume` but not retargeting. Owned by `src/gbdt/feature_cache.py`. Atomic via temp-file + `os.replace`.
- **Universe cache** — `<data_root>/gbdt_feature_cache/<key>.parquet` + `.key.json` sidecar. Target tuple is **excluded** from the key; two cells over the same universe + split + features + seed + code + data snapshot share the build. Owned by `src/gbdt/universe_feature_cache.py`. Same atomic write contract.

**Cache key composition (post-#190, `SCHEMA_VERSION = "v2"`):** SHA-256 over a canonical-JSON dict of:

```
{
  schema_version,
  universe,
  split (train_rows / val_rows / eval_rows / test_rows / min_rows_per_ticker),
  lookbacks,
  families,
  exclude,
  random_seed,
  feature_code_signature {  # see below
    source_sha256,            # THE targeted invalidator
    all_families,
    default_lookbacks,
    expected_total_cols,
  },
  panel_signature             # rows + checksum of the source panel
}
```

The per-cell variant adds the target tuple (`direction`, `threshold_pct`, `horizon_days`, `max_drawdown`, `uniqueness_weighting`); the universe variant intentionally drops it.

**`panel_signature` (8 fields, per `src/gbdt/feature_cache.py:128-168`).** The panel signature pins the matrix to the exact data snapshot it was built from, combining both the universe-panel half and the index-series half:

- `panel_rows` — `int(len(panel))`.
- `panel_n_tickers` — `int(idx.get_level_values("ticker").nunique())`.
- `panel_date_min` / `panel_date_max` — `str(pd.Timestamp(...))` of the panel's date level.
- `panel_index_hash` — SHA-256 over the panel's `(date, ticker)` MultiIndex tuples, stringified deterministically (the panel is `sort_index`-ed by the loader, so the order is stable). OHLCV values are NOT hashed — freshness is governed by snapshot identity.
- `index_series_rows`, `index_series_date_min`, `index_series_date_max` — analogous summaries for the index series (the reference market index passed alongside the panel; the runner uses it for index-relative features).

A rows-appended or ticker-added refresh of the cache flips `panel_rows`/`panel_n_tickers`/`panel_index_hash` (and likely the date bounds) → cache misses cleanly. Adding the index series half means a refreshed index series also invalidates.

**The source-hash mechanism:** `feature_code_signature()` in `src/gbdt/feature_cache.py` (line 92, post-#190) calls `hashlib.sha256(inspect.getsource(gbdt_features).encode()).hexdigest()` and stores the result as `source_sha256`. This invalidates the cache iff `src/gbdt/features.py` source text changes byte-for-byte — including whitespace and comment edits, which is the correct conservative behavior (you don't want to inspect AST diffs to decide if a `# fixed denominator guard` comment is semantic). The three sibling fields (`all_families`, `default_lookbacks`, `expected_total_cols`) are coarse shape summaries kept for debugging — when a cache miss is unexpected, diffing two `feature_code_signature` blobs side-by-side beats diffing two 64-char hex digests.

**Practical rules:**

- *This rule depends on `gbdt.features` being the SOLE source of feature-derivation code. If you ever move feature code elsewhere (e.g., a derivation helper into `model.py` or `__main__.py`, or an import from `data_pipelines`), the cache will silently serve stale matrices with NO automatic invalidation. In that case, bump `SCHEMA_VERSION` manually AND extend `feature_code_signature()` to hash the new source location too.*
- *Edits to `feature_cache.py::feature_code_signature()` itself do NOT invalidate the cache via source-hash — the signature helper is unhashed (it hashes other code, not itself). Any change to this helper (e.g., normalizing whitespace before hashing, changing what's hashed, switching algorithms) is schema-bump-worthy: bump `SCHEMA_VERSION` so existing keys miss cleanly.*
- Edits to `src/gbdt/features.py` → cache invalidates → next run cold-rebuilds. Expected and correct.
- Edits to `src/gbdt/model.py`, `loop_protocol.py`, `train.py`, `__main__.py`, `report.py`, `topk_diagnostics.py`, `calibration.py`, etc. → cache STAYS warm. **This is the whole point of #190** — unrelated commits don't invalidate.
- *Bump `SCHEMA_VERSION` when: removing a field from the payload, renaming a field, changing the semantics/units of a field's value, or editing `feature_code_signature()` itself. Don't bump when: adding a new field (the canonical-JSON `sort_keys=True` automatically changes the hash, so existing caches miss cleanly without an explicit bump). Routine `features.py` edits don't need a bump — the source-hash already carries them.*
- *Formatter / rebase invalidation: `black` / `ruff` / `isort` formatting passes on `src/gbdt/features.py` — including whitespace-only edits — flip the source-hash. So does a `git rebase` that reorders commits if it touches `features.py` at all. Schedule `features.py` reformatting carefully: a single rebase post-merge means every machine's cache cold-rebuilds the next time it runs an experiment (~3 h on r1k, ~30 min on nasdaq100).*

**What changed v1 → v2 (#190):** dropped `code_commit` + `code_dirty` from the key payload (the over-strict belt-and-suspenders that caused every commit to invalidate); added `source_sha256` to `feature_code_signature()` as the targeted replacement. SCHEMA_VERSION bumped to `"v2"` so any v1 parquet on disk misses cleanly rather than half-deserializing into a now-invalid key shape.

**Why this exists (canonical incident):** 2026-05-31 r1k agent-loop trio paid ~15 h of cold rebuild after two unrelated PRs (#86 + #87) bumped `code_commit` between launches — neither touched features.py, but the old key composition invalidated the cache anyway. PR #190 replaced commit-based invalidation with source-hash invalidation. PR #182 (zero-denom guards) was the first features.py edit to ride the new invalidation contract — it correctly busted the cache because it actually changed feature definitions. Proof the targeted invalidator works on both sides: legitimate change → bust, unrelated change → keep.

**How to apply:** when editing gbdt code, ask "did I touch `src/gbdt/features.py`?" If yes, expect a cold rebuild on next launch and budget time accordingly. If no, expect warm cache and verify by checking `<out-dir>/loop/progress.log` for a "loaded cached matrix" line vs the multi-hour "building features" phase. When proposing a schema bump, write down in the commit message *what shape change* required it — every routine features.py edit that bumps SCHEMA_VERSION is a foot-gun reverting the #190 fix.

*Concrete operator example: I edit `_safe_log_returns` in `src/gbdt/features.py`. `source_sha256` changes → next launch on russell1000 misses the cache → cold rebuild ~5 h. Verify via `grep -E 'loaded cached matrix|building features' <out-dir>/loop/progress.log` — `building features` means cold, `loaded cached matrix` means warm. If I only meant to add a debug comment and didn't realize features.py was hot, this is when I notice and decide whether to abort or accept the rebuild.*

See `[[project-gbdt-tuning-playbook]]` for the HP-loop semantics that consume this cache, `[[project-agent-loop-wrapper]]` for the launcher whose `--resume` flag depends on the per-cell cache surviving across retries, the docstring at `src/gbdt/universe_feature_cache.py:49-56` for the Two-level cache flow narrative, and the docstring at `src/gbdt/universe_feature_cache.py:133-164` (`compute_key`) for the live key-composition spec.
