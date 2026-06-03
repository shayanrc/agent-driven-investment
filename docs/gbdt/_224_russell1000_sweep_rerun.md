# Task #224 — russell1000 sweep re-run (canonical R-Precision@K rows)

**Date**: 2026-06-03.
**Branch**: `gbdt-russell1000-sweep-rerun`.
**Data**: `results/gbdt/data/_224_russell1000_sweep_rerun_data.json` (machine-readable headline table + compute notes).
**Canonical CSV**: 13 new rows added to `results/gbdt/data/r_precision_at_k.csv` (now 57 data rows, up from 44).
**Closes**: #225 (russell1000 universe-shape gap in canonical CSV).

## Headline

The russell1000 sweep was re-run end-to-end on the post-V1.3 codebase to materialize `predictions/test.csv` for every cell — the 2026-05-30 #188 run pre-dated #194's predictions-persistence convention, so its `test.csv` files were never written and the russell1000 universe was entirely absent from the canonical R-Precision@K registry. This re-run executes all 20 cells under sweep mode (`fs_hp_loop.max_iterations: 3`, no agent in the loop) against `32c849b` (V1.3 bugfix). All 20 cells completed; 13 emitted non-empty `test.csv` (5d / 10d / 25d / 50d horizons) and were appended to the canonical CSV; 7 cells (all H ≥ 100) hit the known bug #129 (per-ticker test window is `≤100` rows, so any `H ≥ 100` horizon eats the entire test split, leaving `test.csv` empty). With this run, the canonical CSV now carries **50 of the 57 reachable sweep cells** across the four universes (nasdaq100, sp500, russell1000, nifty50); the remaining 7 cells are blocked on the #129 fix.

## Headline table (13 cells, sorted by R-Precision@1 descending)

Raw values from `predictions/test.csv` via `scripts/gbdt/regenerate_r_precision_at_k_csv.py` (per-day tie-break `(p_calibrated desc, ticker asc)` stable mergesort; per-day denominator `min(K, R_q)` over days with `R_q > 0`). No lift columns per CLAUDE.md reporting conventions — `base_rate` is the column readers compare against to compute lift on demand.

| cell | rows | Q_days | base_rate | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| russell1000_up_20pct_50d_dd10pct | 44450 | 50 | 0.1530 | 0.6816 | 0.3800 | 0.3467 | 0.3520 | 0.3260 | 0.3090 |
| russell1000_up_10pct_50d_dd5pct | 44450 | 50 | 0.2988 | 0.5056 | 0.3800 | 0.3533 | 0.3160 | 0.2500 | 0.2230 |
| russell1000_up_10pct_25d_dd5pct | 66675 | 75 | 0.2851 | 0.5865 | 0.3467 | 0.4178 | 0.4347 | 0.4187 | 0.3987 |
| russell1000_up_20pct_25d_dd10pct | 66675 | 75 | 0.1019 | 0.7297 | 0.3333 | 0.4356 | 0.3787 | 0.3800 | 0.3473 |
| russell1000_up_10pct_10d_dd5pct | 80010 | 90 | 0.1256 | 0.6935 | 0.3222 | 0.4000 | 0.3844 | 0.3344 | 0.3433 |
| russell1000_up_10pct_5d_dd5pct | 84455 | 95 | 0.0498 | 0.7624 | 0.2632 | 0.2596 | 0.2737 | 0.2358 | 0.2373 |
| russell1000_up_20pct_10d_dd10pct | 80010 | 90 | 0.0249 | 0.8178 | 0.2556 | 0.2704 | 0.2222 | 0.2222 | 0.2534 |
| russell1000_up_50pct_50d_dd25pct | 44450 | 50 | 0.0248 | 0.8575 | 0.1400 | 0.2400 | 0.2600 | 0.1800 | 0.2211 |
| russell1000_up_40pct_50d_dd20pct | 44450 | 50 | 0.0451 | 0.8033 | 0.1200 | 0.4600 | 0.4000 | 0.3560 | 0.2970 |
| russell1000_up_20pct_5d_dd10pct | 84455 | 93 | 0.0070 | 0.8327 | 0.1075 | 0.1649 | 0.1593 | 0.2084 | 0.2943 |
| russell1000_up_40pct_25d_dd20pct | 66675 | 75 | 0.0179 | 0.8463 | 0.0533 | 0.2044 | 0.1787 | 0.1921 | 0.2445 |
| russell1000_up_40pct_10d_dd20pct | 80010 | 69 | 0.0024 | 0.8613 | 0.0290 | 0.1111 | 0.1816 | 0.2650 | 0.3451 |
| russell1000_up_50pct_25d_dd25pct | 66675 | 74 | 0.0087 | 0.8929 | 0.0270 | 0.1036 | 0.0815 | 0.1500 | 0.2350 |

R-p@1 lifts (in prose, not in table): `20pct_5d` = **15.4×**, `40pct_10d` = **12.2×**, `20pct_10d` = **10.3×**, `50pct_50d` = **5.6×**, `10pct_5d` = **5.3×**, `20pct_25d` = **3.3×**, `50pct_25d` = **3.1×**, `40pct_25d` = **3.0×**, `40pct_50d` = **2.7×**, `10pct_10d` = **2.6×**, `20pct_50d` = **2.5×**, `10pct_50d` = **1.3×**, `10pct_25d` = **1.2×**. The top three top-1 lifts cluster on **short horizon × moderate-to-high threshold × rare-event** cells — the same archetype that wins on nasdaq100 and sp500.

## 7 cells with empty test.csv (bug #129)

All four 100d cells and all three 200d cells emit zero-row `predictions/test.csv` because the walk-forward split allocates 100 trailing rows per ticker for test, and per-ticker targets at offset `H ≥ 100` are NaN within that window:

- `russell1000_up_10pct_100d_dd5pct`
- `russell1000_up_10pct_200d_dd5pct`
- `russell1000_up_20pct_100d_dd10pct`
- `russell1000_up_40pct_100d_dd20pct`
- `russell1000_up_40pct_200d_dd20pct`
- `russell1000_up_50pct_100d_dd25pct`
- `russell1000_up_50pct_200d_dd25pct`

The runner emits a `[data] WARNING: Test segment expected to be EMPTY: horizon_days=N >= split.test_rows=100` line for each, so these failures are well-flagged in the per-cell `report.md` rather than silent. The fix (e.g. bump `split.test_rows` to `max(100, 2 * horizon_days)` or move to an expanding-window split) is tracked as bug #129 and is **not** scoped into this PR.

## Cross-universe shape comparison

The russell1000 R-Precision@K profile closely tracks the nasdaq100 + sp500 shape established in `_188`, `_192`, `_193`, and `_177`:

- **Short-horizon × rare-event = best lift.** russell1000 `20pct_5d` (15.4×) and `40pct_10d` (12.2×) are the top two lifts here. The matched nasdaq100 cells in the canonical CSV: `nasdaq100_up_20pct_5d_dd10pct` (base 0.013, R-p@1 0.156, **12.0×**) and `nasdaq100_up_40pct_10d_dd20pct` (base 0.007, R-p@1 0.030, **4.2×**); the matched sp500 cells: `sp500_up_20pct_5d_dd10pct` (base 0.006, R-p@1 0.220, **36.5×**) and the 40pct_10d cell is not in sp500's sweep family. So at the +20%/5d corner the lift ordering is sp500 > russell1000 > nasdaq100; at +40%/10d russell1000 beats nasdaq100. The narrow read: **rare-event cells reward the wider panel at short horizon** (more independent draws → cleaner top-1), but the russell1000 panel's index-membership turnover (see `_188` § Pattern 4) flattens this versus sp500.

- **Top-tail nasdaq cells vs matched russell1000.** `nasdaq100_up_40pct_25d_dd20pct` has AUC 0.724 / R-p@1 0.161 (lift 3.77×) vs `russell1000_up_40pct_25d_dd20pct` AUC 0.846 / R-p@1 0.053 (lift 2.99×) — russell1000 has the higher AUC but lower top-1 hit-rate because its base rate is 2.5× nasdaq's (1.8% vs 4.3%), so the top-1 pick has fewer positive candidates to choose from per day; AUC is panel-invariant and the rank quality is genuinely better on russell1000, but top-1 is normalized by `min(K, R_q)` so a sparser positive distribution depresses the R-p@1 numerator. This is the same pattern flagged in the project R-Precision@K methodology memory: **R-p@1 is a top-pick measure, not a rank-quality measure** — for cross-universe rank quality, AUC is the cleaner comparison.

- **The `10pct_50d` and `10pct_25d` cells fall into the anti-AUC band on russell1000** (AUC 0.506 and 0.587, both in [0.45, 0.55] for the 50d cell). By the compound rule (AUC ∈ [0.45, 0.55] AND R-p@10 lift < 1.2×), `10pct_50d` is **null** on test (R-p@10 lift = 0.84×); the 25d cell is over the AUC bar at 0.587 so the compound rule doesn't trigger. These match the same boundary the nasdaq100 and sp500 +10% families cross (see `_188` § 1, `_193`); russell1000 crosses at the same horizon-step.

## Methodology

CatBoost backend, sweep mode (`fs_hp_loop.max_iterations: 3`, no agent loop, plateau check on val Brier). All 20 cells ran against the current codebase at HEAD `32c849b` (the V1.3 bugfix commit) — behaviorally unchanged from earlier sweep-mode runs since the V1.3 agent-loop changes only affect the agent-in-loop path, not the sweep path. Per-cell `metrics.json` and `report.md` are the runner's standard artifacts; `predictions/{train,val,eval,test}.csv` carry the columns `date,ticker,p_raw,p_calibrated,y_true,sample_weight`. Canonical R-Precision@K is computed via `scripts/gbdt/regenerate_r_precision_at_k_csv.py` over the freshest `predictions/test.csv` across all worktrees per the project methodology.

## Compute notes

- **Per-cell wall-clock (warm)**: ~9 min, mean 549s over the 19 warm cells (range 480s–632s). Matches the `_188` per-warm-cell budget exactly (mean ~547s there).
- **Cold cell** (`russell1000_up_10pct_100d_dd5pct`, built the universe feature cache for this re-run): 2620s = **~43.7 min**. Faster than the `_188` cold cell (4h24m) because the cache infrastructure introduced post-#188 amortizes the panel-load phase even in the cold case.
- **Total wall-clock**: 12963.9s = **~3h 36m** for all 20 cells (sum of `wall_time_total_sec` across the 20 `metrics.json` files); the wall-clock window of the sweep launcher was ~3h38m end-to-end. **2× faster than the `_188` total of 7h18m**, dominated by the much shorter cold-cell time.
- **Anomaly — #183 cross-cell feature sharing did NOT engage on this re-run**. Each cell wrote its own 6.2 GB `_feature_matrix_cache.parquet` instead of sharing one universe-wide cache as the #183 design intended. Parent agent has already cleaned these up. Tracked as bug **task #226** for a separate plan — root cause likely a cache-key drift between this code revision and the cache directory layout assumed by #183. The sweep ran correctly; only the disk-efficiency property regressed.

## Cross-references

- **PR #112** — V1.3 Option A implementation (eval R-p@K, anti-AUC flag, degenerate-sink warning).
- **PR #114** — V1.3 bugfix (calibrated eval R-p@K, decouple tie_band from plateau_threshold) — the commit (`32c849b`) this sweep ran against.
- **`_223`** — V1.3 A4 re-validation on cell-5 (the cell that motivated V1.3); merged 2026-06-02.
- **`_188`** — original russell1000 sweep memo (2026-05-30, legacy weighted R-precision body table, post-2026-06-01 R-Precision@K addendum for 1 of 20 cells). This memo's 13 rows + `_188`'s 16 remaining cells together cover the full russell1000 sweep landscape; `_188` is the source for the cross-cell narrative (signal/null per cell, eval→test decay patterns, etc.).
- **Bug task #226** — #183 cross-cell feature-cache sharing regression on this codebase revision.
- **Bug task #129** — H ≥ 100 test-split fix (the 7 empty-test cells here).

## Next steps

- Canonical CSV coverage moves from **37/57 → 50/57** sweep cells across the four universes. The remaining 7 cells are all H ≥ 100 cells across russell1000 (4 cells), nasdaq100 (1 cell — `nasdaq100_up_10pct_100d_dd5pct`), sp500 (2 cells), all blocked on #129. No further work is needed on the russell1000 universe-shape gap until #129 lands.
- Closes #225.
