---
name: project-cache-date-time-component
description: Cache `date` column is stored as 'YYYY-MM-DD 00:00:00' (time-suffixed) — bare `date <= end` string-drops the end day; use a half-open interval. The #182 off-by-one.
metadata:
  type: project
---

The SQLite cache (`data/processed.db` — both `us_equities_data` and `nse_equities_data`) stores the `date` column as **`'YYYY-MM-DD 00:00:00'`** (with a midnight time suffix), not a bare `'YYYY-MM-DD'`.

**The foot-gun (the #182 off-by-one):** a bare `WHERE date <= 'YYYY-MM-DD'` is a *string* comparison, and `'2026-06-16 00:00:00' > '2026-06-16'` (the longer, time-suffixed string sorts *after* the bare date). So `date <= end` **silently drops the end day** — hiding the most recent bar from every caller. This made the daily `/daily-predictions` cadence log "yesterday" forever (it could never include the latest complete day with `--end today`) until fixed in #182.

**The rule:** when filtering the `date` column in raw SQL, use a **half-open interval** `date >= start AND date < (end + 1 day)` on normalized day boundaries (or wrap in `DATE(date)`). NEVER `date <= 'YYYY-MM-DD'`. The **start side is safe** as a bare date (`date >= '2026-06-01'` includes `'2026-06-01 00:00:00'`, which sorts after) — only the **end** is the trap.

**Already handled** (don't re-fix): `gbdt.data._cache_read` uses the half-open interval since #182, with regression test `tests/gbdt/test_data_loader.py::test_cache_read_end_date_inclusive_with_time_component`; `load_panel` + `data_pipelines.fetch` go through it. `scripts/backtests/daily_forward_predictions._cache_max_date` uses `date < today` *intentionally* — the same string ordering, in the EXCLUDE direction, to drop today's in-progress partial bar.

**Why / how to apply:** the time-suffix is a property of the stored data, so the #182 fix to `_cache_read` does NOT immunize new code — any fresh ad-hoc SQL, loader, or date filter against the `date` column can re-introduce the off-by-one. Always use the half-open form. Note also that a heartbeat/diagnostic `WHERE date = 'YYYY-MM-DD'` (exact match) returns **0 rows** for the same reason — use `date LIKE 'YYYY-MM-DD%'` or `DATE(date) = …`. Relates to [[project-gbdt-cache-invalidation]] (cache mechanics) and the `_019` daily cadence.
