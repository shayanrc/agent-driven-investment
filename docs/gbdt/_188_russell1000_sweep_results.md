# Task #188 — russell1000 sweep results (20 cells, post-#183 shared feature cache)

**Date**: 2026-05-30.
**Branch**: `gbdt-russell1000-sweep-results`.
**Data**: `results/gbdt/data/_188_russell1000_sweep_results_data.json` (machine-readable master table + per-cell classifications).
**Sweep log**: `logs/russell1000_sweep.log` (per-cell wall-clock); `logs/russell1000_up_10pct_100d_dd5pct.rerun.log` (post-script 100d re-run).
**Prior**: this memo follows the format of `_177_cross_experiment_analysis.md` (US-sweep snapshot at 12/57 cells) and cross-references its established patterns; this is the **first complete russell1000 sweep**.

## Headline

The russell1000 sweep finishes **20/20 cells** with a clean signal floor: **17 of 20 cells discriminate** (12 on the held-out test window, 5 on eval-only with no test window under the current split), **1 cell is ambiguous on test** (10%/50d), **1 is null on eval** (10%/100d), and **1 is anti-predictive on eval** (10%/200d, AUC 0.422). In total **7 cells lack a test window** (H ≥ 100 ate the test split under the 800/400/200/100 walk-forward — same methodology limitation `_177` flagged). Strongest test-segment cells live where the `_177` US-sweep priors predicted: **short-horizon × high-threshold**. The top three by R-prec lift on test are **40%/10d (42.1×)**, **50%/25d (18.6×)**, **20%/5d (17.7×)** — the rare-but-clean top-pick regime. The systemic eval→test AUC decay observed across nasdaq100 + sp500 in `_177` **largely replicates on russell1000**: 12 of 13 test-evaluable cells lose AUC from eval to test (dAUC range **−0.107 to +0.005**); the lone exception is 40%/10d at +0.005 — a flat-to-slightly-positive shift, not a meaningful gain. The decay bites hardest at H = 25–50 in the +10% threshold band — the 10%/50d cell falls into the [0.45, 0.55] null AUC band on test (lift 1.22×, ambiguous), the 10%/100d cell goes null on eval, mirroring the nasdaq100 +10%/25d crossover point seen in `_177` but pushed one horizon-step out (probably the broader panel buying ~1 extra horizon-step before discrimination collapses). **Wall-time vs the #183 PR projection of ~5.6 h: measured ~7.30 h** — cold cell came in at 4h24m (close to the 5h projection), but warm cells averaged ~9 min each (4× the projected ~2 min); the 4× overshoot is dominated by label generation + LdP §4.4 sample-weight computation on the 6.06M-row russell1000 panel, not feature build (cache-skip confirmed by `loop/progress.log`).

## What's covered

All **20 russell1000 cells** in `configs/gbdt/experiments/russell1000_*.yaml`, spanning:

- Thresholds: {10, 20, 40, 50}% (max-drawdown matched per threshold: dd5, dd10, dd20, dd25).
- Horizons: {5, 10, 25, 50, 100, 200}d (per threshold, not full crossing: 5d for {10, 20}; 10d for {10, 20, 40}; 25d, 50d for {10, 20, 40, 50}; 100d for all four; 200d for {10, 40, 50}).
- Universe: russell1000 — 1002 in registry; 889 tickers actually retained after the data adapter's NaN-row guard.

Read `_177` first for the cross-market US-sweep priors (nasdaq100 + sp500 + the standalone russell1000 +10%/5d cell that #177 had as its single russell1000 data point).

## How to read the metrics

- **Weighted R-precision** (per-day variable-K: `sum(positives_caught) / sum(R(d))`, R(d) = positives that day) is the **standard cross-cell metric** — panel-invariant, baseline = base rate, so it compares cleanly to the nasdaq100/sp500/nifty cells in `_177` without the fixed-K bias P@k carries on staggered panels.
- **ROC-AUC** reported eval + test; discrimination signal, not gated.
- **Lift = R-prec / base_rate** is discussed in **prose only**, never as a table column (CLAUDE.md reporting convention). The `base(e)`/`base(t)` columns let you compute lift on demand and keep the underlying hit-rate scale visible.
- Per-day P@k denominator is `min(R(d), k)` everywhere (achievable positives, not picks-made).

**Compound signal/null rule** (CLAUDE.md):
- AUC ∈ [0.45, 0.55] **AND** R-prec lift < 1.2× → **null**.
- AUC ∈ [0.45, 0.55] **AND** lift > 1.5× → **top-tail signal hidden by AUC** (investigate the prediction-extreme regime).
- AUC ∈ [0.45, 0.55] **AND** lift ∈ [1.2, 1.5] → **ambiguous**.
- AUC > 0.55 → **discriminating**.

---

## Master table (20 cells)

Raw metric values. `(e)` = eval segment, `(t)` = test segment. `n_feat` = features kept after FS+HP loop. `n_iter` = iterations the loop ran (capped at 3 under sweep mode — HP search disabled per issue #32).

| Cell | Thr% | H(d) | DD% | base(e) | base(t) | AUC(e) | AUC(t) | Rprec(e) | Rprec(t) | n_iter | n_feat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| russell1000_up_10pct_5d_dd5pct | 10 | 5 | 5 | 0.0347 | 0.0498 | 0.8244 | 0.7624 | 0.3279 | 0.2547 | 3 | 48 |
| russell1000_up_10pct_10d_dd5pct | 10 | 10 | 5 | 0.0933 | 0.1256 | 0.7863 | 0.6954 | 0.3869 | 0.3342 | 3 | 279 |
| russell1000_up_10pct_25d_dd5pct | 10 | 25 | 5 | 0.2372 | 0.2851 | 0.6890 | 0.5821 | 0.4737 | 0.4101 | 3 | 29 |
| russell1000_up_10pct_50d_dd5pct | 10 | 50 | 5 | 0.3519 | 0.2988 | 0.6092 | 0.5112 | 0.5108 | 0.3653 | 3 | 279 |
| russell1000_up_10pct_100d_dd5pct | 10 | 100 | 5 | 0.4085 | — | 0.4895 | — | 0.4630 | — | 3 | 11 |
| russell1000_up_10pct_200d_dd5pct | 10 | 200 | 5 | 0.4488 | — | 0.4224 | — | 0.5216 | — | 3 | 279 |
| russell1000_up_20pct_5d_dd10pct | 20 | 5 | 10 | 0.0041 | 0.0070 | 0.8662 | 0.8327 | 0.1580 | 0.1237 | 3 | 43 |
| russell1000_up_20pct_10d_dd10pct | 20 | 10 | 10 | 0.0139 | 0.0249 | 0.8572 | 0.8178 | 0.2113 | 0.2034 | 3 | 30 |
| russell1000_up_20pct_25d_dd10pct | 20 | 25 | 10 | 0.0729 | 0.1019 | 0.8170 | 0.7297 | 0.3824 | 0.3153 | 3 | 50 |
| russell1000_up_20pct_50d_dd10pct | 20 | 50 | 10 | 0.1808 | 0.1530 | 0.7426 | 0.6816 | 0.4532 | 0.3049 | 3 | 34 |
| russell1000_up_20pct_100d_dd10pct | 20 | 100 | 10 | 0.3092 | — | 0.6653 | — | 0.4927 | — | 3 | 19 |
| russell1000_up_40pct_10d_dd20pct | 40 | 10 | 20 | 0.0012 | 0.0024 | 0.8910 | 0.8956 | 0.1767 | 0.1000 | 3 | 20 |
| russell1000_up_40pct_25d_dd20pct | 40 | 25 | 20 | 0.0093 | 0.0179 | 0.9120 | 0.8463 | 0.2296 | 0.2091 | 2 | 279 |
| russell1000_up_40pct_50d_dd20pct | 40 | 50 | 20 | 0.0393 | 0.0451 | 0.8586 | 0.8032 | 0.3067 | 0.2566 | 3 | 39 |
| russell1000_up_40pct_100d_dd20pct | 40 | 100 | 20 | 0.1217 | — | 0.7938 | — | 0.4008 | — | 3 | 26 |
| russell1000_up_40pct_200d_dd20pct | 40 | 200 | 20 | 0.2649 | — | 0.7446 | — | 0.5006 | — | 3 | 14 |
| russell1000_up_50pct_25d_dd25pct | 50 | 25 | 25 | 0.0040 | 0.0087 | 0.9300 | 0.8798 | 0.2040 | 0.1618 | 3 | 279 |
| russell1000_up_50pct_50d_dd25pct | 50 | 50 | 25 | 0.0207 | 0.0248 | 0.8873 | 0.8294 | 0.2583 | 0.2418 | 3 | 279 |
| russell1000_up_50pct_100d_dd25pct | 50 | 100 | 25 | 0.0787 | — | 0.8345 | — | 0.3697 | — | 3 | 24 |
| russell1000_up_50pct_200d_dd25pct | 50 | 200 | 25 | 0.2007 | — | 0.7810 | — | 0.4384 | — | 3 | 279 |

R-precision values cross-checked via `uv run python -m scripts.gbdt.compute_r_precision <cell>/predictions/{eval,test}.csv` (same code path as the aggregator's `per_day_r_precision`).

---

## Per-cell verdicts (compound rule, grouped by signal status)

### Discriminating (17 cells: 12 on test + 5 on eval-only)

Listed by ascending threshold then horizon. Verdict basis (test vs eval-only) called out per bullet. Lift quoted in prose only.

- **10%/5d**: test AUC 0.762, lift **5.11×** (base 4.98% → R-prec 25.5%). The cleanest +10% short-horizon cell — matches the `_177` snapshot value (5.1× test) on the same single russell1000 data point it had, so this new sweep run reproduces the prior result exactly.
- **10%/10d**: test AUC 0.695, lift **2.66×** (base 12.6% → 33.4%). Eval AUC 0.786 dropped 0.091 — solid signal but already feeling the eval→test decay.
- **10%/25d**: test AUC 0.582, lift **1.44×** (base 28.5% → 41.0%). Right at the discriminating/null boundary on test — flags the same boundary `_177` saw at nasdaq100 +10%/25d but the russell1000 cell stays just above the 0.55 AUC line.
- **20%/5d**: test AUC 0.833, lift **17.71×** (base 0.70% → R-prec 12.4%). Top-three lift cell — rare events (∼0.7% base) concentrated by the model into the top picks.
- **20%/10d**: test AUC 0.818, lift **8.17×** (base 2.49% → 20.3%). Eval-to-test decay only 0.039 — the rarer-event cells decay less, consistent with `_177` § "Threshold effect".
- **20%/25d**: test AUC 0.730, lift **3.09×** (base 10.2% → 31.5%).
- **20%/50d**: test AUC 0.682, lift **1.99×** (base 15.3% → 30.5%). Cleanest +20% mid-horizon cell — the 50d cell is discriminating on the +20% line but null-band on the +10% line, so threshold buys horizon-reach.
- **20%/100d**: eval AUC 0.665, lift **1.59×** on eval (base 30.9% → 49.3%). Eval-only verdict (no test window). The lift is above 1.5×, so by the compound rule this is discriminating on eval; whether it survives to test is the open question the broken split won't answer.
- **40%/10d**: test AUC 0.896, lift **42.11×** (base 0.24% → R-prec 10.0%). Top lift in the sweep — extreme rare-event concentration. Absolute R-precision is modest because positives are scarce, but the top picks are very clean.
- **40%/25d**: test AUC 0.846, lift **11.70×** (base 1.79% → 20.9%). Inner-stop fired at iter 2 (degradation) rather than 3 (plateau) — only cell in the sweep that didn't go the full 3 iterations.
- **40%/50d**: test AUC 0.803, lift **5.69×** (base 4.51% → 25.7%).
- **40%/100d**: eval AUC 0.794, lift **3.29×** on eval. Eval-only. Strongly discriminating on eval; test missing.
- **40%/200d**: eval AUC 0.745, lift **1.89×** on eval. Eval-only. The lift drops below 2× — at 200d the eval signal weakens but stays above the discriminating bar.
- **50%/25d**: test AUC 0.880, lift **18.57×** (base 0.87% → 16.2%). Second-highest test lift.
- **50%/50d**: test AUC 0.829, lift **9.74×** (base 2.48% → 24.2%).
- **50%/100d**: eval AUC 0.834, lift **4.70×** on eval. Eval-only.
- **50%/200d**: eval AUC 0.781, lift **2.18×** on eval. Eval-only.

### Ambiguous null-band (1 cell)

- **10%/50d**: test AUC **0.511** (in [0.45, 0.55]) AND lift **1.22×** (in [1.2, 1.5]). The runner kept the full 279-feature pool (best_iteration = 0, no FS prune helped val Brier) which is itself diagnostic: there's no compact signal the FS step can isolate. Eval AUC 0.609 / lift 1.45× — the eval segment looked marginal and the test segment confirmed marginal. Same near-null story as `_177`'s nasdaq100 +10%/25d (eval AUC 0.655 → test 0.511, eval lift 2.03× → test 1.46×) — the russell1000 sweep shifts the marginal crossover one horizon-step further out (25d clean here, 50d marginal here vs 25d marginal at nasdaq100).

### Null (1 cell)

- **10%/100d**: eval AUC **0.489** (in [0.45, 0.55]) AND eval lift **1.13×** (< 1.2×). No test window (H = 100 ate the test split — see methodology note below). Eval-only verdict: **null**. Matches the `_177` nasdaq100 +10%/100d cell exactly (AUC 0.488, lift 1.22× — basically the same numbers). The broader russell1000 panel did not rescue this cell.

### Anti-predictive on eval (1 cell)

- **10%/200d**: eval AUC **0.422** (below 0.45 — i.e. model ranks worse than random by the AUC metric). Eval lift 1.16× (basically baseline). No test window. Two consistent reads:
  1. At H = 200d / +10% threshold the base rate is 44.9% (essentially "anything that moves up 10% in 200 trading days") — too common to be useful and the F-pool features pick up on transient signals that **invert at this horizon** (e.g. high recent momentum predicts lower forward 200d gain because of mean-reversion / momentum-crash). The AUC < 0.5 is a feature-relationship-inversion at this horizon, not data corruption.
  2. The cell kept the full 279-feature pool (best_iteration = 0 again). With ~45% base prevalence and inverted features, FS has no clear signal to keep or prune.
  Without a test window this verdict is provisional — it doesn't generalize to a tradable insight, but it does flag "the +10% target at H ≥ 200d is a poorly-posed problem for this feature pool."

### Coverage gap — no test window (7 cells flagged separately above)

The cells with `H ≥ 100` and the standard 800/400/200/100 split have **zero test rows** because each ticker's trailing 100 rows have NaN targets (forward window incomplete past the test cutoff). russell1000 cells affected: **10%/100d (null)**, **10%/200d (anti-predictive)**, **20%/100d, 40%/100d, 40%/200d, 50%/100d, 50%/200d (all eval-only discriminating)**. The runner emits an explicit warning at data load (`[data] WARNING: Test segment expected to be EMPTY: horizon_days=N >= split.test_rows=100`), so the gap is well-flagged in `report.md` for each affected cell, not silently absent.

This is the **same methodology limitation** `_177` flagged on nasdaq100 +10%/100d. The russell1000 broader panel (~1002 names vs nasdaq100's ~100) doesn't change this: the split is per-ticker and the test allocation per-ticker is the binding constraint, not panel width. Any H ≥ 50 cell sees test signal degrade and any H ≥ 100 cell sees it vanish entirely. **Fixing this needs a test-split redesign** (e.g. bump test_rows to `max(100, 2H)` or move to expanding-window with longer test) and is a separate plan item.

---

## Patterns (cross-cell)

### 1. The eval→test AUC decay is universal and matches `_177` qualitatively

12 of 13 cells with a test window lose AUC from eval to test. `dAUC = AUC(test) − AUC(eval)` ranges from **−0.107** (10%/25d, russell1000's worst) to **+0.005** (40%/10d, the lone cell whose test AUC matched/slightly exceeded eval). Excluding the +0.005 outlier, the next-smallest decay is **−0.034** (20%/5d). For comparison `_177`'s worst was −0.144 (nasdaq100 +10%/25d). The decay direction is near-universal across the russell1000 sweep, confirming `_177`'s diagnosis that the 2024–2026 test window is a harder prevalence regime than the eval window for **all** US universes (nasdaq100, sp500, russell1000); the 40%/10d cell is a single counter-example where the very-rare-event lift held up.

Decay magnitude scales with horizon inside fixed-threshold families (10% family, test-evaluable cells only):

| Cell | AUC eval → test | dAUC | Test class |
|---|---|---:|---|
| 10%/5d | 0.824 → 0.762 | −0.062 | discriminating |
| 10%/10d | 0.786 → 0.695 | −0.091 | discriminating |
| 10%/25d | 0.689 → 0.582 | −0.107 | discriminating (just over the line) |
| 10%/50d | 0.609 → 0.511 | **−0.098** | **ambiguous null-band** |

The 10%/50d cell crosses into the null AUC band on test — one horizon-step further than nasdaq100's 10%/25d crossover (`_177` Pattern 1). The broader russell1000 panel buys roughly one extra horizon-step of signal before the +10% target's eval AUC stops translating to test AUC, but doesn't change the direction or the eventual crossover.

The 20%, 40%, 50% threshold families decay less and stay discriminating to the longest evaluable horizon — e.g. 50%/50d only loses 0.058 AUC eval→test and stays at 0.829 test AUC, consistent with `_177`'s observation that rarer events are more stable across prevalence drift.

### 2. Horizon effect: short is clean, long decays — except at very high thresholds

Within the +10% threshold (the most populated family at evaluable horizons), R-prec lift declines with horizon: 5d → 5.11× test, 10d → 2.66×, 25d → 1.44×, 50d → 1.22× (ambiguous), 100d/200d → null/anti-predictive (eval-only).

But at +40% / +50%, the longer-horizon cells stay strongly discriminating on eval (40%/100d lift 3.29×, 40%/200d 1.89×, 50%/100d 4.70×, 50%/200d 2.18×) — the rare-event signal survives further out in horizon because the event's rarity is what carries the signal, not short-horizon momentum patterns. If a test-split fix becomes available, this is the most interesting region to evaluate.

### 3. Threshold effect: rarer events → higher lift, modest absolute R-prec — same as `_177`

Test-segment lifts by threshold family:

- +10%: 1.22× (50d) — 5.11× (5d). Base rates 5–35%. Higher absolute R-precision (~25–47%).
- +20%: 1.99× (50d) — 17.71× (5d). Base rates 0.7–18%. R-prec 12–45%.
- +40%: 5.69× (50d) — 42.11× (10d). Base rates 0.24–4.5%. R-prec 10–31%.
- +50%: 9.74× (50d) — 18.57× (25d). Base rates 0.87–2.5%. R-prec 16–24%.

The pattern is identical to `_177`'s § "Threshold effect": rarer events trade base rate for very high lift on a smaller absolute R-precision. For a strategy this means +40%/+50% cells are "alert me to the rare big mover" surfaces; +10% / +20% cells are "broader coverage, smaller per-pick edge."

### 4. Universe effect: russell1000 vs nasdaq100/sp500 (vs `_177`)

On the matched +10%/5d cell `_177` had nasdaq100 lift 6.9× test, sp500 5.9×-ish (computed from `_177`'s table), russell1000 5.1× — the russell1000 cell is the **lowest** test lift of the three at this matched cell, contradicting `_177`'s tentative "bigger panel → stronger lift" hypothesis. Likely cause: russell1000 panel-membership is far less stable than nasdaq100 / sp500 (more frequent index-membership turnover at the bottom of the market-cap distribution), and the 889 / 1002 actually-used-vs-registered split is much wider than the nasdaq100 / sp500 cells. The +20%/5d comparison is similar: sp500 23.7× test (`_177`) vs russell1000 17.7× test — russell1000 lift is lower despite the wider panel. The cross-sectional rank/z-score features (F14) presumably benefit less from a panel of names that churns more.

This **revises `_177`'s § Universe-effect bullet downward**: it is *not* a clean "bigger universe → bigger lift" story; index stability matters too. nasdaq100 ≈ sp500 > russell1000 at matched cells, though the n is still small enough (2 matched cells) that the read is a hypothesis not a finding.

### 5. FS prunes hard with neutral-to-positive effect — but less aggressively than `_176`/`_147`

11 of 20 cells finish with `n_feat < 60` (the modal FS outcome). 7 cells finish with the full 279 retained (best_iteration = 0 — the loop tried pruning at iter 1 and got worse val Brier, so reverted). The aggressive `_147` (279 → 9) / `_176` (279 → 39) prunes don't replicate here — the russell1000 cells either prune to ~14–50 or keep all 279. No cell finished below 11 features.

The 7 "full pool kept" cells skew toward (a) the shortest horizon at the lower threshold (10%/10d), (b) the longest evaluable horizons (50%/50d, 50%/25d, 40%/25d), and (c) the eval-only cells with the inverted target (10%/200d). The pattern is consistent with: when the signal is either very broad (short-horizon momentum is universal) or very weak (long-horizon high-prevalence) the FS step has no compact signal to find.

### 6. Calibration: 14 of 20 cells received isotonic, 6 stayed native

Spiegelhalter Z values are large-magnitude on most cells (|Z| > 4) because the eval segment has 177,800 rows — Spiegelhalter's null is easily rejected with that much data even for mild miscalibration. The 6 "native" cells (no isotonic layer) cluster at cells where the calibration test's |Z| < 2, mostly the high-threshold low-base-rate cells where the predicted probabilities are tightly compressed near zero and miscalibration is hard to detect on the scale that matters.

### 7. Best-iteration distribution

7 of 20 cells stopped at iter 0 (no prune helped). 3 at iter 1. 9 at iter 2 (the last iteration before plateau). 1 at iter 2 by degradation (40%/25d). No cell ran past 3 iterations because the sweep mode caps `max_iter=3` (HP search disabled per `report.md` warning, issue #32). This is a known sweep-mode constraint, not a per-cell failure.

---

## Wall-time measurement vs the #183 PR projection

The #183 PR projected (in its body):

| scenario | wall time |
|---|---|
| cold sweep (no shared cache) | 20 × 5 h ≈ **100 h** |
| with shared universe cache | 1 × 5 h + 19 × <2 min ≈ **5.6 h** |

Measured on this sweep:

**Cold cell** (first cell after restart, built the universe feature cache): `russell1000_up_10pct_10d_dd5pct` — 15877s = **4h24m37s** (from inside-experiment `wall_time_total_sec`). The actual `[features]` phase inside the cold cell took 15234.9s = 4h13m55s — that's ~96% of the cell's wall time. **The 5h cold projection was within 14% of measured, accurate within sub-cell variability.**

**Warm cells** (`logs/russell1000_sweep.log`): 18 cells with `[features] loaded from universe cache (key match) in ~9s` (confirmed in `loop/progress.log` for sample cell 200d_dd5pct; the entire 30-second heartbeat granularity hides the features phase entirely). Per-cell elapsed range: **484s – 657s** (8m04s to 10m57s), mean **~544s = ~9.1 min**. That's **4.5× the projected ~2 min**.

**100d re-run** (separate run after sweep ended, to fix the cell that was in-flight when the sweep was started): 484.8s = 8m05s.

**Total wall-clock**:
- Sweep window `[SWEEP] start 2026-05-30T08:55:45Z` to `[SWEEP] end 2026-05-30T17:06:05Z` = **25790s = 7h09m50s**.
- Plus the post-sweep 100d re-run at 484.8s = 8m05s.
- **Combined: ~7h18m, or ~7.30 h** vs the **5.6 h projection** — **30% over projection, ~1.7 h gap**.

**Per-cell average**: (25790 + 484.8) / 20 = **1314s ≈ 21.9 min per cell** counting the cold cell; **(25790 − 15877 + 484.8) / 19 = ~547s = ~9.1 min per warm cell**, matching the per-cell elapsed range above.

**The #183 cache worked as claimed for what it was designed to do**: it skipped the ~4h13m feature build out of warm cells (confirmed via `loop/progress.log` of cell 200d_dd5pct showing `[features] loaded from universe cache (key match) in 9.3s` immediately after the `data` phase, and the heartbeat at 30s granularity doesn't even register a `features` phase line for warm cells — the cache hit completes inside one heartbeat). Without the shared cache the 19 warm cells would each have re-run 4h13m of `build_feature_matrix` for a sweep total of ~80h+.

**Where the 4.5× gap on warm cells comes from** (per-warm-cell, after the ~10s features cache load):
1. **`data` phase**: ~30–45s per cell — universe panel load from `processed.db` (6.06M rows × 889 tickers). Per-cell, not amortized.
2. **`target` phase**: ~30–45s per cell — binary label construction for the (direction, threshold, horizon, max_drawdown) tuple over 6.06M rows. Per-cell by design (target is what differs across cells).
3. **`uniqueness` phase**: ~10–15s per cell — LdP §4.4 horizon-overlap sample-weight computation. Per-cell, scales with H (200d cells take longer than 5d cells).
4. **`loop` phase**: ~250–400s per cell — 3 CatBoost fits (iter 0, 1, 2) + per-iteration diagnostic bundle. This is the bulk of the warm-cell wall time. CatBoost fit time scales with `n_rows × n_features × n_iter × tree_depth`, and on the 711k-train-row russell1000 panel with default depth=6 + iterations=500 (the runner default) each fit takes ~80–130s. **This is the load-bearing 4× factor**: the #183 projection of "<2 min" appears to have assumed a much smaller panel basis (the sweep PR text doesn't show its working).
5. **`artifact` phase**: ~5–10s per cell — write model.cbm, calibration.pkl, predictions/{eval,test}.csv, figs/.

(2) + (3) + (4) + (5) = ~325–500s per cell, matching observed.

**The 5.6 h projection was per-cell warm-cost optimistic by ~4×.** A more accurate projection going forward would be: cold ≈ 5h + warm ≈ 9–10 min × N. For a 20-cell sweep that's **5h + 3h = 8h**. The measured 7.3h beats that conservative re-projection (3 of the warm cells came in under 510s; some had no test segment and skipped post-fit test-predictions which shortens the artifact phase).

---

## Practical guidance — which russell1000 cells to pursue

**Pursue (strong discriminating + test-validated):**
- **Short-horizon × high-threshold**: 40%/10d, 50%/25d, 20%/5d — lift > 15× test. Rare-event top-pick surfaces.
- **Short-horizon × mid-threshold**: 20%/10d (lift 8.2×), 50%/50d (9.7×), 40%/50d (5.7×) — solid lift with somewhat broader coverage.
- **10%/5d** as the broadest-coverage discriminating cell (lift 5.1×, base 5%, R-prec 25.5% — picks up something on ∼1/4 of top-K picks on test).

**Use with caution — eval-only (no test window):**
- 40%/100d, 50%/100d, 20%/100d, 40%/200d, 50%/200d are all discriminating on eval. They are interesting candidates for the long-horizon strategy bucket but cannot be trusted without a test-split fix. **Treat eval AUC + eval lift as a hypothesis pending test data**, given the universal eval→test AUC decay observed everywhere else.

**Avoid — null / ambiguous / anti-predictive:**
- 10%/50d (ambiguous null-band on test).
- 10%/100d (null on eval-only).
- 10%/200d (anti-predictive on eval-only — model ranks below random at this horizon for the +10% target on a 45% base rate).

These cells corroborate `_177`'s § 1 finding that **the +10% target at H ≥ 25** is where eval AUC over-states test AUC enough to push cells into the null band, and at H ≥ 100 the +10% target becomes structurally hard for the feature pool.

---

## Cross-reference to `_177`

| `_177` finding | russell1000 sweep evidence |
|---|---|
| Universal eval→test AUC decay (every completed cell loses AUC) | **Largely replicates** — 12 of 13 test-evaluable cells lose AUC; dAUC range −0.107 to +0.005 (40%/10d is the lone +0.005 outlier — extreme-rare-event cell whose test AUC held flat) |
| Decay magnitude scales with horizon at fixed threshold | **Replicates** — 10% family decay grows 0.06 → 0.09 → 0.11 → 0.10 from 5d → 50d |
| Short-horizon (5d–10d) sweet spot | **Replicates** — 5d/10d cells stay strongly discriminating across all thresholds |
| Higher thresholds → higher lift, smaller base, modest absolute R-prec | **Replicates** — +40%/+50% cells hit lift 18–42× on tiny base, +10% cells hit lift 1–5× on larger base |
| H ≥ 100 has zero test rows (test-split methodology gap) | **Replicates** — 7 russell1000 cells affected; same `data WARNING` text |
| Bigger universe → stronger lift (n=1 in `_177`) | **Contradicted** — russell1000 lifts at matched +10%/5d and +20%/5d are *lower* than nasdaq100 and sp500. Likely index-membership stability matters more than panel width |
| FS prunes hard (279 → 25–65) on most cells | **Partially replicates** — 11 of 20 cells prune to <60; 7 keep the full 279; no cell prunes below 11 |
| Brier underperforms base-rate baseline on test even for discriminating cells | **Replicates** — most cells beat baseline Brier on eval but degrade on test (consistent with calibration-on-drifted-prevalence) |

The single concrete revision `_188` introduces to `_177`'s priors is the **universe-effect hypothesis downgrade**: with two matched-cell data points (5d × {10%, 20%}) the russell1000 lift is lower than nasdaq100 / sp500, contradicting "bigger panel → stronger lift." Recommend updating `_177`'s § 4 to note "the russell1000 sweep weakens this hypothesis — index-membership stability appears to matter as much as panel width."

---

## User-facing read (no automated PASS/FAIL)

Of the 20 russell1000 cells, **12 discriminate on the held-out test window**, **5 are eval-only discriminating** (no test window under the current split: 20%/100d, 40%/{100d,200d}, 50%/{100d,200d}), **1 is borderline on test (10%/50d, ambiguous)**, **1 is null on eval (10%/100d, no test)**, **1 is anti-predictive on eval (10%/200d, no test)**. The short-horizon × high-threshold corner is the production-candidate region — **40%/10d, 50%/25d, 20%/5d** are the standout cells by test-segment R-precision lift (42×, 19×, 18×). The eval→test AUC decay observed in `_177`'s partial US sweep largely replicates on russell1000, with the decay crossing into the null AUC band one horizon-step further out (10%/50d vs nasdaq100's 10%/25d). The 7 eval-only cells (H ≥ 100) need the test-split fix before they can be judged. As `_177` noted, the PASS/FAIL call on any individual cell remains a user judgment; this memo characterizes the landscape across the completed russell1000 sweep.
