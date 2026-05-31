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

**The source-hash mechanism:** `feature_code_signature()` in `src/gbdt/feature_cache.py` (line ~82, post-#190) calls `hashlib.sha256(inspect.getsource(gbdt_features).encode()).hexdigest()` and stores the result as `source_sha256`. This invalidates the cache iff `src/gbdt/features.py` source text changes byte-for-byte — including whitespace and comment edits, which is the correct conservative behavior (you don't want to inspect AST diffs to decide if a `# fixed denominator guard` comment is semantic). The three sibling fields (`all_families`, `default_lookbacks`, `expected_total_cols`) are coarse shape summaries kept for debugging — when a cache miss is unexpected, diffing two `feature_code_signature` blobs side-by-side beats diffing two 64-char hex digests.

**Practical rules:**

- Edits to `src/gbdt/features.py` → cache invalidates → next run cold-rebuilds. Expected and correct.
- Edits to `src/gbdt/model.py`, `loop_protocol.py`, `train.py`, `__main__.py`, `report.py`, `topk_diagnostics.py`, `calibration.py`, etc. → cache STAYS warm. **This is the whole point of #190** — unrelated commits don't invalidate.
- `SCHEMA_VERSION` manual bump only for breaking key-shape changes (dropping a key field, changing payload structure, changing the parquet column convention). Do NOT bump for routine `features.py` edits — the source-hash carries that. The bump exists for the cases the source-hash can't catch (the cache key dict shape itself changing).

**What changed v1 → v2 (#190):** dropped `code_commit` + `code_dirty` from the key payload (the over-strict belt-and-suspenders that caused every commit to invalidate); added `source_sha256` to `feature_code_signature()` as the targeted replacement. SCHEMA_VERSION bumped to `"v2"` so any v1 parquet on disk misses cleanly rather than half-deserializing into a now-invalid key shape.

**Why this exists (canonical incident):** 2026-05-31 r1k agent-loop trio paid ~15 h of cold rebuild after two unrelated PRs (#86 + #87) bumped `code_commit` between launches — neither touched features.py, but the old key composition invalidated the cache anyway. PR #190 replaced commit-based invalidation with source-hash invalidation. PR #182 (zero-denom guards) was the first features.py edit to ride the new invalidation contract — it correctly busted the cache because it actually changed feature definitions. Proof the targeted invalidator works on both sides: legitimate change → bust, unrelated change → keep.

**How to apply:** when editing gbdt code, ask "did I touch `src/gbdt/features.py`?" If yes, expect a cold rebuild on next launch and budget time accordingly. If no, expect warm cache and verify by checking `<out-dir>/loop/progress.log` for a "loaded cached matrix" line vs the multi-hour "building features" phase. When proposing a schema bump, write down in the commit message *what shape change* required it — every routine features.py edit that bumps SCHEMA_VERSION is a foot-gun reverting the #190 fix.

See `[[project-gbdt-tuning-playbook]]` for the HP-loop semantics that consume this cache, `[[project-agent-loop-wrapper]]` for the launcher whose `--resume` flag depends on the per-cell cache surviving across retries, and the docstring at `src/gbdt/universe_feature_cache.py:124-179` for the live spec.
