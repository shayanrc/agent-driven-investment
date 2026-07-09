# _284 — Group-stratified boosting: mixed-family trees recover the top-of-book

**Date:** 2026-07-08 · **Branch:** `gbdt-stratified-boosting` · **Cell:** sp500 +20%/50d (dd10) · **Backend:** xgboost (custom loop)
**Status:** complete arc — one-shot test spent 2026-07-08 (ordering replicated), **w2-confirmed at champion level**, backtested (+92.4% vs SPX +37.0%), board-campaign round 1 (H=200 incumbents hold; their test looks banked). NOT promoted, NOT in `/daily-predictions`.

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

Re-run with identical seed persisting per-tree artifacts; determinism verified exactly (val Brier 0.0997 / eval 0.1236 reproduce to 4dp — the analyzed ensemble IS the scored one). `scripts/gbdt/analyze_stratified_trees.py`.

**1. The model rebuilt itself around fundamentals.** Family gain shares, stratified vs iter-0 (same classifier both sides):

| family | n_feat | stratified | iter-0 | shift |
|---|--:|--:|--:|--:|
| F18 fundamentals | 13 | **41.9%** | 4.4% | +37.5 |
| volatility | 54 | 21.2% | 40.7% | −19.5 |
| drawdown | 12 | 10.5% | 12.3% | −1.8 |
| returns | 54 | 6.7% | 17.5% | −10.9 |
| calendar (F15+F21) | 14 | 6.6% | 4.5% | +2.1 |
| trend | 6 | 4.1% | 1.6% | +2.5 |
| cross-sectional | 65 | 3.9% | 10.2% | −6.4 |
| vwap | 14 | 2.7% | 2.8% | −0.1 |
| persistence (F16) | 54 | 1.6% | 3.9% | −2.3 |
| volume | 24 | 1.0% | 2.2% | −1.2 |

All 13 F18 columns land in the top 16 by gain (`fund_rev_ttm_yoy` #1 at 6.0%); `garman_klass_50`'s 18.8% iter-0 monopoly is gone (top vol feature is now `parkinson_200` at 4.7%). Gain is far flatter — top feature 6% vs 18.8%. Being uncapped + offered every tree, fundamentals became the trunk; the capped ladders became regime conditioners.

**2. Availability-normalized usage separates signal from dead weight.** Long-window ladder features are used **100% of the time they're offered** (`parkinson_200`, `realized_vol_100`, `beta_200`, `index_runup_200`, `vol_of_vol_50/100`, `stock_return_zscore_200`); F18 gets used in 78–96% of all 800 trees. At the bottom, the **F16 persistence family is dead weight**: many `*_outside_band_*` columns are used in **0%** of the trees that offered them (25–36 offers, zero splits) — the family holds 54 of 310 columns for 1.6% of gain.

**3. What actually goes together on a path (the interaction structure):**
- **F18 × trend** is the dominant cross-family interaction: `fund_sales_yield × sma_distance_200` co-paths in **52%** of trees offering both (140/267); nearly every F18 column pairs with `sma_distance_100/200` at 37–43%. The model's unit of reasoning: *valuation/growth conditioned on where price sits vs its long trend*.
- **F18 × calendar**: every top fund column pairs with `moy_cos`/`moy_sin` at ~29–34% — seasonally-phased fundamentals.
- **Within-F18** (the uncapped exception earning its keep): `fund_rev_ttm_yoy × fund_sales_yield` on the same path in **49% of all 800 trees** — the growth×valuation (GARP) interaction; yields × their own xs-ranks follow at 40–46%.
- Family×family path-share: F18×F18 21.3%, F18×vol 10.7%, F18×returns 8.3% — F18 rows dominate every interaction row; no non-F18 pair exceeds 1.7%.
- Top non-F18 partners of fundamentals: `sma_distance_200` (1,330 path-pairs), `moy_cos` (1,227), `moy_sin` (1,006), then `vwap_dev_zscore_200` (707) and `drawdown_200` (656) — vwap's contribution is real but runs *through* fundamental paths.

**4. Calendar-pair integrity — the intact-pair rule matters for exactly two pairs:**

| pair | offered | both used | one used | same path |
|---|--:|--:|--:|--:|
| moy | 344 | 189 | 128 | 57 |
| qoy | 324 | 79 | 161 | 6 |
| dom | 292 | 54 | 122 | 21 |
| moq | 333 | 20 | 117 | 0 |
| dow | 307 | **1** | 25 | 0 |

`moy` genuinely works as a pair (55% both-used when offered). `dow` is dead on a 50-day horizon (1 tree in 307 used both), `moq` near-dead — candidates to drop from the rotation in refinement.

**Refinement candidates surfaced by the anatomy** (not yet acted on): (a) drop or down-weight the persistence slot — 54 columns of near-zero conditional usage; (b) drop the `dow` (and maybe `moq`) pair; (c) bias ladder sampling toward long windows (the 100%-usage end); (d) the flip-side risk to watch — 41.9% of gain now rides on 13 quarterly-updated F18 columns, so the test window will say whether that concentration is regime-robust.

## One-shot blind test (committed 2026-07-08)

Pre-declared, single look, all three arms; test 2024-07-26→2024-12-16 (46,400 rows, Q=100 days, base_rate 0.1273 — matches the registry rows on this window):

| model | Brier | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline_iter0 | 0.1154 | 0.672 | 0.130 | 0.217 | 0.230 | 0.237 | 0.258 |
| control_rand35 | 0.1115 | 0.695 | 0.190 | 0.290 | 0.294 | 0.291 | 0.290 |
| **stratified** | **0.1106** | **0.702** | **0.260** | **0.363** | **0.390** | **0.375** | **0.332** |

Registry bar on the same window: `f18xgb` (all_fundamentals, default HP) AUC 0.687, R-p@1 0.44, @3 0.393, @5 0.340, @10 0.317.

**Reading:**
1. **The structural claim replicated blind.** stratified > control > baseline on every metric — the eval-window ordering carried to test with no anti-selection. The relative gaps *grew*: stratified doubles its own pool's baseline at @1 (0.260 vs 0.130) and beats it +67% at @3 (0.363 vs 0.217).
2. **The maximal pool itself was a handicap — stratification mostly rescued it.** On this exact window, default HP on the cleaner `all_fundamentals` pool (f18xgb, 293 cols) scores @1 0.44 / @3 0.393; the same default HP on our 310-col maximal pool collapses to 0.130 / 0.217 (the `_283`-style dilution — +31 mostly-noise columns wreck the default-HP top-of-book). Stratification recovers most of the damage (0.260 / 0.363) *while keeping all features*, but does not fully close to the cleaner-pool bar at @1/@3. It DOES beat f18xgb at @5 (0.390 vs 0.340) and @10 (0.375 vs 0.317) and on AUC (0.702 vs 0.687).
3. **Verdict vs leaderboard: below the same-window champion at the sharp top, above it in the mid-book.** The natural next variant writes itself: stratified sampling over a *pruned* pool (drop the anatomy's dead weight — persistence, dow/moq, low-usage members) — i.e., the offer-efficiency FS. That variant is judged on val+eval and confirmed on the **w2 window** (2025-01-24→2025-06-17, already defined in the registry) per the `_272`→`_273` pattern — this window's look is spent.

## Refinement 1: within-family usage-prune + dow drop — REJECTED

Pre-declared variant (user-directed): drop members with conditional usage < 10% at ≥20 offers (persistence 54→29, flags 4→1 — the rule auto-pruned the India-legacy `diwali/budget/fiscal_year_end` flags on this US universe, keeping `fomc_week`; cross-sectional −4, volatility −1, volume −1) + drop the `dow` pair. Pool 310→274; recipe otherwise identical (seed 42, 800×eta0.05×depth6).

Result vs the original stratified arm (val+eval; test untouched):

| segment | metric | original | pruned | Δ |
|---|---|--:|--:|--:|
| val | AUC | 0.778 | 0.769 | −0.009 |
| val | R-p@1 | 0.408 | 0.370 | −0.038 |
| val | R-p@3 | 0.377 | 0.331 | −0.046 |
| eval | AUC | 0.750 | 0.744 | −0.006 |
| eval | R-p@1 | 0.490 | 0.420 | −0.070 |
| eval | R-p@3 | 0.440 | 0.435 | −0.005 |
| eval | R-p@10 | 0.352 | 0.364 | +0.012 |

Worse nearly everywhere, decisively at the sharp top on both segments. **Mechanistic read: in stratified sampling, dead features act as implicit family down-weights.** A tree offered 2 dead persistence columns simply declines the family — the slot's *effective* weight self-tunes to ~0. Pruning the dead members concentrates the family's 2 draws on its live-but-mediocre members, so weak families get *more* splits than before, diluting the top book. "Cleaning the pool" re-weights weak families UP — the opposite of the intent. (Draw-sequence noise from the changed pool contributes, but the sharp-top direction is consistent across both segments.)

Standing result: **the original 310-col stratified arm remains the _284 model.** Untested parked variant: family-level pool cut (stratified over `all_fundamentals` only, dropping vwap+cal2 wholesale) — motivated by the test-window pool-ablation finding, but each additional variant spends val/eval looks; revisit alongside the w2 confirmation if pursued.

## w2 confirmation + classification-first comparison (the primary lens)

Per the standing convention, R-p@K + AUC are compared BEFORE any PnL claim. The scored OOS window contains the **pre-registered w2 confirmation window** (2025-01-24 → 2025-06-17, 100 days) — the same window the cell's w2 registry arms were tested on (their train ends 2022-08-31; the stratified ensemble's train ends 2022-03-04, six months staler).

**w2 window, same cell, same 100 days (our slice base_rate 0.1430 vs registry 0.1404):**

| model | backend | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 |
|---|---|--:|--:|--:|--:|--:|
| w2cbbase | catboost | 0.820 | **0.560** | 0.467 | 0.463 | 0.454 |
| w2cbdef | catboost | 0.820 | 0.540 | **0.503** | 0.463 | **0.457** |
| **stratified (_284)** | xgb custom | 0.816 | 0.520 | 0.480 | **0.474** | 0.445 |
| w2fbase | xgboost | 0.673 | 0.430 | 0.380 | 0.386 | 0.393 |
| w2ffund | xgboost | 0.685 | 0.350 | 0.397 | 0.401 | 0.422 |

**The stratified ensemble replicates at champion level on the independent second window** — within noise of the two catboost champions (AUC 0.816 vs 0.820; @1 0.520 vs 0.540/0.560; @3 between them; @5 above both) and decisively above the same-backend xgboost arms (@1 +0.09 to +0.17, @3 +0.08 to +0.10). This is the two-window bar the F17/F18/F20 feature arcs failed: w1 one-shot (ordering replicated, below the f18xgb sharp top) + w2 (champion-level). Split-but-strong: it does not *beat* the cb champions; it matches them with a staler train cutoff and a different mechanism (mixed-family trees on the maximal pool).

**Full 22.5-month OOS realized classification** (2024-07-26 → 2026-04-21 labelable, 435 days, base_rate 0.1446): AUC **0.715**, R-p@1 0.382, @3 0.402, @5 0.410, @10 0.398.

**Board-top models' registry rows for context** (each on its own test window — not directly commensurable): russell1000 +50%/200d (base 0.152): AUC 0.697, @1 0.737, @3 0.642; ndx40_mix (base 0.037): AUC 0.923, @1 0.333, @3 0.313, @10 0.735; sp500 +50%/50d f18cb (base 0.010): AUC 0.934, @1 0.132. The long-horizon/rare-event board leaders live in structurally different (base_rate, horizon) regimes.

## Backtest placement (champion strategy config, long OOS window)

Scored the exact saved ensemble over **2024-07-26 → 2026-06-12** (472 trading days; everything after 2024-12-16 is pure forward-OOS never seen by any decision) and ran the standard harness (`run_backtest_cell`, champion config: rank selection / equal sizing / K=3, calibrator refit on our val split; run dir `runs/backtests/strat_284_oos`):

| | total return | max DD |
|---|--:|--:|
| **stratified (rank/equal K=3)** | **+92.4%** | −22.7% |
| SPX buy-hold (same window) | +37.0% | −18.9% |
| EW basket | +38.2% | −18.9% |
| EW top-K unmanaged | +119.1% | −41.2% |

222 entries / 82 tickers; exposure 0.84; exits: 93 DD-stops, 83 target-hits, 38 horizon. Excess over index **+55.4 pts over ~22.5 months**.

Board context (windows differ — total returns are not directly commensurable): the board top is russell1000 +50% champion **+135.9%** (excess +72.1 over ~35 months) and ndx40_mix **+121.7%** (excess +84.2 over ~12 months — the strongest per-month excess). Our +55.4/22.5mo sits between them on excess-per-month, on a cell (sp500 +20%/50d, common-event mid-horizon) whose registry arms never made the board. Same-cell precedent: the deployed sp500_20 champion did +58.1% vs SPX +8.4% on its (older, shorter) window. DD runs deeper than index (ungated, as all board rows; `_017`'s SMA200 gate would cap it). Not a registry row yet; not promoted.

## Board-topper campaign, round 1 (2026-07-09) — incumbents hold

User-directed attack on the leaderboard tops, honest protocol: pre-declared candidates, iteration on val/eval only, ONE sealed test look per candidate spent only when leading on eval. Two shots fired at the H=200 giants (both share the aligned windows: train 2018-01-02→2021-03-08, eval 2022-10-07→2023-07-26, test 2023-07-27→2024-10-03):

**Shot 1 — sp500 +50%/200d** (incumbent `aligned_cbagent`: test @1 0.930/@3 0.617; its eval, agent-selected so optimistically biased: @1 0.845/@3 0.738/@5 0.644/@10 0.536):

| eval segment | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 |
|---|--:|--:|--:|--:|--:|
| v1 (dropna harness, 2019+ train) | 0.708 | 0.755 | 0.623 | 0.579 | 0.523 |
| v2 (NaN-tolerant, +2018 train, row-faithful) | 0.726 | 0.740 | 0.625 | 0.530 | 0.472 |

Behind at every K both variants (v2 improves bulk only). **Test look NOT spent.** Note v2's harness discovery: the runner never drops feature-NaN rows — the dropna harness had silently shed all of 2018 (F18 coverage + warmup) and ~5% of segment rows; NaN-tolerant restores faithfulness (kept for all future shots).

**Shot 2 — russell1000 +50%/200d** (backtest board #1, +135.9%; incumbent `aligned` cb single-fit — eval computed from its published predictions, unbiased: AUC 0.722, @1 0.745/@3 0.595/@5 0.571/@10 0.534, base 0.125):

| eval segment | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 |
|---|--:|--:|--:|--:|--:|
| stratified v1 (NaN-tolerant) | 0.727 | 0.415 | 0.370 | 0.344 | 0.342 |

AUC matches; top-of-book far behind. **Test look NOT spent.**

**Round-1 read:** the `_284` recipe does NOT transfer as-is to long-horizon (H=200) cells — the same boundary `_282` drew for the finetunes (H≥100 resists) and consistent with `_277`/`_278` (CatBoost systematically dominates long-horizon cells at every K). The board tops hold, honestly. Round-2 levers (parked under the campaign task): **cb-backend stratified trees** (combine the incumbents' backend edge with our structure — the highest-prior lever), larger tree budgets, long-window-biased draws, and the nasdaq cells (ndx40_mix, the R-p@3 0.756 anti-AUC cell). Both sealed test looks remain banked.

## Board-topper campaign, round 2 (2026-07-09) — cb backend; incumbents hold again

Round-1's highest-prior lever, executed: the identical pre-declared recipe
(seed 42, 800 trees, eta 0.05, depth 6, same caps/pairs) with **only the
per-tree learner swapped to CatBoost** (1-iteration fits chained via `Pool
baseline`, Plain boosting, has_time pinned) — testing whether the incumbents'
backend (`_277`/`_278`: CatBoost owns long horizons) closes the H=200 gap.
NaN-tolerant harness; same aligned segments; test → JSON unread.
`scripts/gbdt/stratified_cb_attack.py`.

**Shot 1 — sp500 +50%/200d** (incumbent agent-biased eval @1 0.845/@3 0.738):

| eval segment | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 |
|---|--:|--:|--:|--:|--:|
| cb-stratified | 0.719 | 0.745 | 0.663 | 0.568 | 0.456 |
| xgb-stratified (round 1, v1) | — | 0.755 | 0.623 | — | — |

**Shot 2 — russell1000 +50%/200d** (incumbent unbiased eval AUC 0.722 /
@1 0.745 / @3 0.595):

| eval segment | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 |
|---|--:|--:|--:|--:|--:|
| cb-stratified | 0.734 | 0.605 | 0.523 | 0.503 | 0.454 |
| xgb-stratified (round 1) | 0.727 | 0.415 | 0.370 | 0.344 | 0.342 |

Neither candidate leads its incumbent's eval decision metric — **both test
looks stay banked; both incumbents hold.**

**Round-2 read:** the backend swap is *real but insufficient*. It moved the
top-of-book materially on both cells (sp500 @3 0.623→0.663; r1k @1
0.415→0.605, @3 0.370→0.523 — closing over half of round-1's r1k gap) while
leaving @1 short of the board line on both. So CatBoost's long-horizon edge
composes with the stratified structure, but the incumbents' @1 dominance at
H=200 is not explained by backend + per-tree feature stratification alone.
Ops note: the first r1k attempt was OOM-killed at 36.7GB RSS — retaining 800
fitted CatBoost objects is the leak; the fix (models not retained, recipe +
seed persisted instead — the ensemble is rng-deterministic) is in
`scripts/gbdt/stratified_cb_attack.py`.

## Caveats / discipline

- **One cell, one window-pair.** val+eval agree (the `_282` sweet-spot regime where eval tracks test) but the standing rule holds: one-shot test commit, then an independent second-window replication before any adoption talk (the `_272`→`_273` pattern; `_283`'s standalone-vs-faithful reversal is the fresh cautionary tale).
- **Custom ensemble** — per-tree feature masks are not producible by the current runner; adoption needs a backend/runner extension (parked in `V1.9_TBD`).
- Champion / `/daily-predictions` unchanged.

## Artifacts

- `scripts/gbdt/stratified_boosting.py` — 3-arm harness (exact recipe, seed 42)
- `scripts/gbdt/analyze_stratified_trees.py` — tree-anatomy analysis
- `scripts/gbdt/feature_importance_by_group.py` — iter-0 group-importance table
- `runs/gbdt/stratified/<cell>/{results.json,artifacts.pkl,model.pkl}` (gitignored; model.pkl enables exact-ensemble test scoring later)
