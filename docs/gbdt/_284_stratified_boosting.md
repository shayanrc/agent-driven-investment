# _284 — Group-stratified boosting: mixed-family trees recover the top-of-book

**Date:** 2026-07-08 · **Branch:** `gbdt-stratified-boosting` · **Cell:** sp500 +20%/50d (dd10) · **Backend:** xgboost (custom loop)
**Status:** val+eval complete — **test still BLIND** (one-shot commit pending). NOT promoted, NOT in `/daily-predictions`.

## Question

Can a **decoupled** cell — high AUC but near-chance top pick — recover R-p@1/@3 *with all features kept* (no FS), by structurally forcing every tree to be a **mixed-family weak learner** instead of a correlated same-family stack?

Cell selection: the registry's strongest tunable decoupling. sp500 +20%/50d has AUC 0.69–0.73 across arms but R-p@1 ≈ 0.14–0.16 on test ≈ the 0.13–0.15 base rate — real bulk ranking skill, top pick at chance. Common event (13.6% full-panel prevalence) + mid horizon = the tunable regime per `_282`.

## Setup

- **Maximal opt-in pool** via the new `all_fundamentals_vwap_calendar2` token (this branch): technical F1–F16 (279) + F18 fundamentals (13) + F20 vwap-dev (14) + F21 calendar2 (4) = **310 columns**. Spec: `configs/gbdt/experiments/sp500_up_20pct_50d_dd10pct_maxtune.yaml`.
- **date_aligned** split, train_start 2019-01-01 (NYSE): train 2019-01-02→2022-03-04 (357,688 rows, prevalence 0.148), val 2022-03-07→2023-10-06 (184,725, 0.111), eval 2023-10-09→2024-07-25 (92,657, 0.150). Test 2024-07-26→2024-12-16 **blind**.
- Snapshot 2026-07-06. All manual fits share one harness (`scripts/gbdt/stratified_boosting.py`); baseline faithfulness vs the runner's iter-0 confirmed earlier (val_brier 0.1164 manual vs 0.1184 runner).

## Prior findings that motivated the design

1. **iter-0 group importance** (`scripts/gbdt/feature_importance_by_group.py`): the default-HP model is a vol/drawdown-regime machine — volatility 40.7% of total gain + drawdown/regime 12.3%; `garman_klass_50` alone is **18.8%**. Opt-in families are riders (F18 4.4%, F20 2.8%, F21 1.8% — though F21 has the highest per-feature mean). Two dilutive blocks (F16 persistence 54 feats → 3.9%, cross-sectional 65 → 10.2%) hold 38% of columns for ~14% of gain.
2. **COVID-exclusion probe** (user hypothesis, rejected): dropping 2020-02-01→2021-01-01 from train hurt val on every metric (val_brier 0.1164→0.1336, AUC 0.755→0.732, R-p@1 0.325→0.315, R-p@3 0.345→0.297). The COVID rows are the model's richest vol/drawdown examples and val contains the 2022 bear. Keep the full window.
3. **Rule 14 context** (`_282`): slow-eta smoothing on a full pool anti-selects the top book by spreading splits into correlated same-family ladders. FS bounds it by *removing* features; this experiment instead blocks the pile-up **structurally** while keeping all 310.

## Design (user-specified constraint set)

Custom boosting loop — standard additive boosting via `base_margin` continuation, but each tree trains on its own structured column subset (~35 of 310):

| block | rule | per tree |
|---|---|---|
| 8 lookback-ladder families (volatility, returns, drawdown, cross-sectional, persistence/F16, volume, trend, F20 vwap) | **≤2 distinct features** per tree | 2 random each = 16 |
| F18 fundamentals | **exception — uncapped** (13 genuinely distinct ratios) | all 13, every tree |
| calendar (F15+F21) | sin/cos pairs **always intact** (a phase encoding is meaningless split alone) | 2 of 5 pairs = 4 + 2 of 4 flags |

800 trees, eta 0.05, depth 6, seed 42. Note: a tree may still split the *same* feature at multiple thresholds (that's refinement of one signal, not ladder pile-up — within the constraint's spirit).

**Control (the load-bearing comparison):** identical budget — 800 trees, eta 0.05, depth 6, `colsample_bytree` ≈ 35/310 — but random subsets: pairs can break, vol can pile up. If stratified beats control, *structure* is what helps, not subsampling. **Baseline:** iter-0 default HP (100 trees, depth 6, eta 0.3), full 310 cols, same harness.

## Result — val + eval (test blind)

val (prevalence 0.111):

| model | Brier | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 |
|---|--:|--:|--:|--:|--:|--:|
| baseline_iter0 | 0.1164 | 0.755 | 0.325 | 0.345 | 0.337 | 0.314 |
| control_rand35 | 0.1093 | 0.777 | 0.370 | 0.363 | 0.342 | 0.324 |
| **stratified** | **0.0997** | **0.778** | **0.408** | **0.377** | **0.354** | **0.330** |

eval (prevalence 0.150):

| model | Brier | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 |
|---|--:|--:|--:|--:|--:|--:|
| baseline_iter0 | 0.1311 | 0.705 | 0.380 | 0.322 | 0.309 | 0.329 |
| control_rand35 | 0.1273 | 0.741 | 0.355 | 0.355 | 0.339 | 0.338 |
| **stratified** | **0.1236** | **0.750** | **0.490** | **0.440** | **0.384** | **0.352** |

Machine-readable: `results/gbdt/data/_284_data.json`.

## Reading

1. **Stratified wins every metric on both segments — zero regressions.** eval R-p@1 0.380→**0.490**, R-p@3 0.322→**0.440**, AUC 0.705→0.750; val R-p@1 0.325→0.408.
2. **First config this arc that does NOT trade top-of-book for bulk.** Every prior lever (schedules, FS-tunes, feature additions — `_282`, `_283`) improved AUC/@10/@20 at the cost of @1/@3. This improves both.
3. **The control isolates the mechanism.** Same budget, no structure → AUC gains arrive (0.741) but the top pick *degrades* vs baseline (eval R-p@1 0.355 < 0.380). Slow-eta smoothing alone reproduces rule-14 anti-selection; only the family-stratified version sharpens the top. Rule 14's mechanism confirmed from the constructive side: block the same-family pile-up and smoothing helps the whole book.
4. Constraint verified: 310/310 features covered across the ensemble; avg **8.4 distinct families per tree**.
5. Brier caveat: stratified val Brier 0.0997 sits just above the constant-predictor floor (0.0990) — calibrated-but-modest sharpness, not degeneracy (AUC 0.778 / R-p@1 0.41).

## Tree-anatomy analysis

*(pending — `scripts/gbdt/analyze_stratified_trees.py` over the per-tree artifacts: ensemble importance vs iter-0 family shares, availability-normalized usage rates, same-path cross-family co-occurrence, calendar-pair integrity, F18 path-partners. To be appended.)*

## Caveats / discipline

- **One cell, one window-pair.** val+eval agree (the `_282` sweet-spot regime where eval tracks test) but the standing rule holds: one-shot test commit, then an independent second-window replication before any adoption talk (the `_272`→`_273` pattern; `_283`'s standalone-vs-faithful reversal is the fresh cautionary tale).
- **Custom ensemble** — per-tree feature masks are not producible by the current runner; adoption needs a backend/runner extension (parked in `V1.9_TBD`).
- Champion / `/daily-predictions` unchanged.

## Artifacts

- `scripts/gbdt/stratified_boosting.py` — 3-arm harness (exact recipe, seed 42)
- `scripts/gbdt/analyze_stratified_trees.py` — tree-anatomy analysis
- `scripts/gbdt/feature_importance_by_group.py` — iter-0 group-importance table
- `runs/gbdt/stratified/<cell>/{results.json,artifacts.pkl,model.pkl}` (gitignored; model.pkl enables exact-ensemble test scoring later)
