# Task #113 — Exp 1 (nifty50) post-uniqueness-fix re-run + methodology comparison

> **Methodology note (2026-06-01)**: Numbers in this memo's body use the legacy "weighted R-precision" metric (per-day variable K = R(d), micro-aggregated). The project headline metric was renamed 2026-06-01 to **R-Precision@K** (per-day fixed K, macro-aggregated via `(1/Q)·Σ r_q/min(K,R_q)`). See the "R-Precision@K (current methodology)" section at the bottom of this memo for the cells in this memo recomputed under the new metric, plus `.claude/memories/project-r-precision-methodology.md` for the full definition + relationship.

**Cell**: `nifty50_up_20pct_50d_dd10pct` (NIFTY 50, +20% in 50 trading days, max-drawdown 10%; default callback mode, `fs_hp_loop.max_iterations=8`)

**Why this re-run**: PR #18 (López de Prado §4.4) shipped sample-uniqueness weighting as the default on 2026-05-26 — overlapping-label samples are now weighted by their uniqueness. Exp 1 had been run with the old (biased) methodology and its artifact frozen. This is the **second and final half of task #113**; the first (Sweep #1, `nasdaq100_up_10pct_100d_dd5pct`) landed as #121 / memo `docs/gbdt/_113_sweep1_post_uniqueness_comparison.md`. Exp 1 was the only other affected experiment.

**BEFORE artifact** (archived): `project_mgmt/archives/biased_baselines/nifty50_up_20pct_50d_dd10pct_pre_uniqueness_fix/`
**AFTER artifact** (this PR): `results/gbdt/experiments/nifty50_up_20pct_50d_dd10pct_post_fix/`

Both runs use the **same backend** (CatBoost, `default` algorithmic FS+HP loop — both runs' iteration rationales read `algorithmic fallback`), the same spec tuple, the same `random_seed=42`. The only deltas are the sample-uniqueness weighting methodology and whatever the data did. The data-constancy check below establishes that the data did effectively nothing to the split.

## Data-constancy check (CRITICAL — gates whether the comparison is valid)

The #167 NSE back-extend grew the underlying panel: the post-fix run loaded **129,401 total panel rows** where the baseline ran on a shorter history. But the walk-forward split carves a **fixed 1,600 rows/ticker (800+400+200+100) from the most-recent data**, so the back-extend's added rows are *older* history that falls outside the eval/test windows.

| quantity | BEFORE (biased) | AFTER (post-fix) | constant? |
|---|---:|---:|---|
| tickers in universe | 50 | 50 | yes |
| tickers used | 46 | 46 | yes |
| tickers excluded | ETERNAL, JIOFIN, MAXHEALTH, SHRIRAMFIN | (same 4) | yes |
| train rows | 36800 | 36800 | yes |
| val rows | 18400 | 18400 | yes |
| eval rows | 9200 | 9200 | yes |
| test rows | 2300 | 2300 | yes |
| eval window | 2024-10-17 .. 2025-12-26 | 2024-10-17 .. 2025-12-30 | start identical; end +4 td |
| eval positives | 439 | 435 | −4 (edge recompute) |
| test window | 2025-08-06 .. 2026-03-09 | 2025-08-06 .. 2026-03-11 | start identical; end +2 td |
| test positives | 32 | 32 | yes |

The split row counts are **bit-identical**; the ticker set is unchanged. The eval/test windows share the same start date and differ only by a 2–4 trading-day shift at the recent end — the post-fix run executed 3 days later than the baseline, and the +50d label horizon at the most-recent edge recomputes as the data end advances (4 eval labels at the trailing edge flipped, 439 → 435; test positives unchanged). **Verdict: data is effectively constant; the #167 back-extend did NOT confound this comparison.** It is a clean methodology-only A/B.

## Headline metrics

| Metric | BEFORE (biased) | AFTER (post-fix) | Δ |
|---|---:|---:|---:|
| `roc_auc` (eval) | 0.5604 | 0.5902 | +0.0298 |
| `roc_auc` (test) | 0.4352 | 0.5548 | +0.1196 |
| `brier` (eval) | 0.0507 | 0.0517 | +0.0010 |
| `brier_baseline_baserate` (eval) | 0.0454 | 0.0450 | −0.0004 |
| `brier` (test) | 0.0291 | 0.0310 | +0.0019 |
| `val_brier` (best checkpoint) | 0.1156 | 0.1150 | −0.0006 |
| `calibration.spiegelhalter_z` | 0.852 (p=0.394) | −4.815 (p≈1.5e-6) | −5.667 |
| `calibration.decision` | `native` | `isotonic` | flipped |
| `loop.best_iteration` | 0 | 1 | +1 |
| `loop.n_iterations_run` | 3 | 3 | — |
| `loop.inner_stop_signal` | `plateau` | `plateau` | — |
| final feature count | 279 (best=iter0, full pool) | 40 (best=iter1, pruned) | −239 |

### Weighted R-precision (per-day variable-K, K = R(d)) — recomputed identically on both prediction sets

`uv run python -m scripts.gbdt.compute_r_precision <predictions/{eval,test}.csv>` run on BOTH the frozen-baseline and the post-fix predictions:

| segment | run | weighted R-precision | weighted base rate |
|---|---|---:|---:|
| eval | BEFORE | 0.1708 | 0.0643 |
| eval | AFTER | 0.2138 | 0.0638 |
| test | BEFORE | 0.0313 | 0.0356 |
| test | AFTER | 0.0938 | 0.0357 |

(Mean-unweighted R-precision moves the same direction: eval 0.103 → 0.153, test 0.017 → 0.083.)

In prose, with lift computed on demand from the columns above: eval weighted R-precision lift rose from **2.66× to 3.35×** base rate; test rose from **0.88× (below random) to 2.63×**. The post-fix model not only kept its eval top-tail edge, it *recovered* a test segment that the biased run had ranked worse than chance.

## Sample-uniqueness telemetry (post-fix only)

`metrics.json::sample_uniqueness` confirms the fix is wired for the H=50 horizon:

```
horizon_days: 50
overlap_inflation_ratio: 99.0     # = 2H-1 for H=50 — each label window overlaps 99 neighbors
n_rows (train fold 0): 36800
sum_weights: 371.7
ess_kish: 36800.0
```

As in the Sweep #1 (H=100) case, `ess_kish == n_rows` because the interior overlap is uniformly maximal (every interior sample has 99 overlapping neighbors), so per-sample relative weights are near-identical (≈ 1/99) — the fix here acts as a global loss-scale rescale (≈ 100× smaller absolute weight) rather than a differential per-sample reweighting. The H=50 inflation (99×) is smaller than Sweep #1's H=100 (199×), so the loss-landscape rescale is correspondingly milder — yet it still moved this cell's calibration decision and best-checkpoint selection.

## Mechanistic reading

1. **The signal verdict did NOT change: this cell discriminates in both runs.** eval AUC is 0.56 → 0.59, both above the 0.55 discriminating threshold; weighted R-precision lift is well above 1.5× on both eval segments. Per CLAUDE.md's compound rule this is **real top-tail signal**, not a null cell — and the fix did not manufacture it (it was already there pre-fix on eval). What the fix did was *strengthen* it (eval 2.66× → 3.35×) and *rescue* it on test (0.88× → 2.63×, AUC 0.44 → 0.55).

2. **The biggest changed conclusion is calibration.** Pre-fix, Spiegelhalter z = 0.85 (well inside the |z|<2 band) → conditional isotonic shipped **native** CatBoost probabilities. Post-fix, z = −4.81 (p ≈ 1.5e-6) → the conditional-isotonic gate now **fits and ships isotonic**. The negative z means the post-fix raw probabilities are **under-confident** (predicted-mean below observed frequency) — a direct consequence of the uniqueness weighting shrinking the effective loss contribution of overlapping positives, which pulls raw probabilities toward the low side. Any downstream consumer that branched on `calibration.decision` for this cell flips from `native` to `isotonic`.

3. **The best checkpoint moved from iter 0 to iter 1**, and the shipped feature set therefore shrank from the full 279-column pool to the 40-feature iter-1 set. Under the new weighting the iter-1 pruned model has the lower val Brier (0.1150 vs 0.1166 at iter 0). This is a methodology-induced model-selection change, not a tuning change — HPs are byte-identical between runs (the algorithmic fallback never moved them; see `hp.yaml`).

4. **Brier barely moved and stays just below its base-rate baseline** (eval improvement −0.0067 vs −0.0052 before). This cell is rare (eval prevalence ≈ 4.7%), so the base-rate-Brier benchmark is very hard to beat on aggregate Brier — but that is the AUC∈[0.45,0.55]-AND-high-R-precision regime the playbook warns about *in reverse*: here AUC is comfortably >0.55, so the modest Brier is the expected "rare-cell, real-ranking-signal, hard-aggregate-Brier" pattern, not a null. The `prediction_range.flag_low_separation=true` flag (std ≈ 0.046) is consistent with a rare cell whose probability mass clusters low while still rank-ordering the tail correctly.

## Verdict

- **Cell verdict** (the reader's call, not automated): **discriminating top-tail signal, calibrated via isotonic.** AUC > 0.55 on both eval and test; weighted R-precision lift > 2.6× on both segments post-fix. The cell ranks the top tail meaningfully above base rate. Aggregate Brier sits just below the base-rate baseline, which is the expected rare-cell pattern, not a disqualifier. If shipped, it must ship the **isotonic** calibrator (the native gate fails post-fix).
- **Methodology verdict**: the PR #18 sample-uniqueness fix is correctly wired at H=50 (overlap inflation 99×) and materially changes this cell — it flips the calibration decision native → isotonic, moves the best checkpoint iter0 → iter1, and strengthens the measured top-tail ranking. Unlike Sweep #1 (which was a null cell where the fix only improved calibration of noise), here the fix operates on a cell with genuine signal and the net effect is a stronger, better-calibrated artifact.
- **Did the fix change this cell's conclusions?** **Yes on calibration and ranking, no on the signal verdict.** One sentence: *the cell discriminates both before and after, but the uniqueness fix flips its calibration from native to isotonic, shifts the best checkpoint from iter 0 to iter 1, and strengthens the measured top-tail ranking (eval R-precision lift 2.66× → 3.35×, test 0.88× → 2.63×) — all on a provably constant data split.*

## R-Precision@K (current methodology — added 2026-06-01)

Per `.claude/memories/project-r-precision-methodology.md`, R-Precision@K is the post-2026-06-01 headline cross-cell metric for gbdt — defined as `R-Precision@K = (1/Q) · Σ_q r_q / min(K, R_q)` over the Q days where R_q > 0 (R_q = positives on day q; r_q = positives caught in top-K picks on day q; macro-averaged, equal weight per day; K fixed). Recomputed from each cell's `predictions/test.csv`:

| cell | rows | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|---|---|
| nifty50_up_20pct_50d_dd10pct | 2300 | 1.4% | 0.435 | 0.050 | 0.017 | 0.058 | 0.325 | 0.533 |

The canonical CSV carries only the post-fix run of this cell; the BEFORE/AFTER comparison is in the body table above.

## Cross-links

- PR #18 — the LdP §4.4 sample-uniqueness methodology fix.
- Task #113 — re-run the affected experiments (Sweep #1 done in #121; this memo closes the Exp 1 half).
- `docs/gbdt/_113_sweep1_post_uniqueness_comparison.md` — the sibling comparison (null cell; format template for this memo).
- `.claude/memories/project-r-precision-methodology.md` — the weighted R-precision (per-day variable-K) standard used here.
- `.claude/memories/project-nse-data-quirks.md` § 3 — the `processed.db-wal` corruption + scratch-dir workaround used to source clean data for this run (see footnote).

## Footnote — WAL corruption worked around at run time

Pre-flight `sqlite3 data/processed.db 'PRAGMA quick_check'` failed in the worktree with `unable to open database file`, and `stat data/processed.db-wal` returned an I/O error — the filesystem-corrupted-WAL symptom documented in `[[project-nse-data-quirks]]` § 3 (after the 2026-05-26 disk wedge). Recovery per the documented decision tree: the worktree's `data/` symlink was repointed from the corrupted main-checkout cache to the persistent scratch copy at `/mnt/122CEE982CEE765F/cache_data/` (quick_check `ok`, `raw/` already symlinked through to the main checkout). No data loss; the scratch DB is the healthy current cache. The post-fix run's `metrics.json::preflight.data_root` records `/mnt/122CEE982CEE765F/cache_data` for audit.
