# Task #193 — sp500 sweep results (17 cells, post-#190 + #182 cache — ~30× speedup vs cold)

> **Methodology note (2026-06-01)**: Numbers in this memo's body use the legacy "weighted R-precision" metric (per-day variable K = R(d), micro-aggregated). The project headline metric was renamed 2026-06-01 to **R-Precision@K** (per-day fixed K, macro-aggregated via `(1/Q)·Σ r_q/min(K,R_q)`). See the "R-Precision@K (current methodology)" section at the bottom of this memo for the 12 test-evaluable cells recomputed under the new metric (plus `r_precision_at_k.csv` for the canonical record), plus `.claude/memories/project-r-precision-methodology.md` for the full definition + relationship.

**Date**: 2026-05-31.
**Branch**: `gbdt-sp500-sweep`.
**Data**: `results/gbdt/data/_193_sp500_sweep_results_data.json` (machine-readable master table + per-cell classifications).
**Sweep log**: `logs/sp500_sweep.log` (per-cell wall-clock, single contiguous pass).
**Prior**: this memo mirrors the format of `_192_nasdaq100_sweep_results.md` (the prior US sweep) and `_188_russell1000_sweep_results.md`, and cross-references both for the three-universe comparison. The sp500 sweep covers **17** cells (not 20) — the canonical sweep grid lacks the +10%/{100d, 200d} and +40%/10d cells on sp500 (no configs exist; deliberate gap, not an outage).

## Headline

The sp500 sweep finishes **17/17 cells, 0 failures** with the cleanest signal floor of the three US universes so far: **16 of 17 cells discriminate** (11 on the held-out test window, 5 on eval-only with no test window under the current split), **0 cells are ambiguous, 0 null**, and **1 cell is anti-predictive on test** (10%/50d at AUC 0.400 — below the 0.45 lower bound). In total **5 cells lack a test window** (H ≥ 100 ate the test split under the 800/400/200/100 walk-forward — same methodology limitation `_177`, `_188`, `_192` flagged). Strongest test-segment cells live exactly where `_177`/`_188`/`_192` predicted: **short-horizon × high-threshold**. The top three by R-prec lift on test are **50%/25d (27.45×)**, **20%/5d (26.29×)**, **40%/25d (16.32×)** — all strongly discriminating (AUC 0.85+). The systemic eval→test AUC decay observed across all three US universes in `_177` / `_188` / `_192` **fully replicates on this sp500 sweep**: **all 12 test-evaluable cells lose AUC from eval to test** (dAUC range **−0.166 to −0.019**; no exception). The decay crosses into the anti-predictive band at +10%/50d on test (AUC 0.400 — the only cell in the three US sweeps to drop below 0.45 on test). **Cross-universe comparison vs `_188`'s russell1000 + `_192`'s nasdaq100**: across the 12 matched test cells **sp500 wins lift on 11** and only loses 40%/50d (8.06× nas vs 6.20× sp); sp500 wins AUC(t) on **11 of 12 cells** (only loses 10%/50d where sp's anti-predictive drop to 0.400 leaves rus at 0.511 on top). This **revises** the universe-effect read from `_192` again — sp500 (486-ticker mid-cap-heavy panel between nas's 92 and rus's 889) is the **lift-and-AUC sweet spot** on the matched grid. **Operational headline**: post-#190 source-hash cache key + #182 universe feature cache delivered a **~31× end-to-end per-cell speedup** (cold 9541s → warm mean 303s) and **~1700× on the features phase alone** (cold 9189s → warm 4.4–5.7s). This is the first sweep that benefited from BOTH #182 (cache infrastructure) and #190 (source-hash key avoiding code-commit thrash). The 17-cell sweep ran end-to-end in **4.02 h wall-clock** (cells-sum **3.996 h**); the cold cell alone was 2.65 h (∼2h33m features), the 16 warm cells totalled 1.35 h.

## What's covered

All **17 sp500 cells** in `configs/gbdt/experiments/sp500_up_*pct_*d_dd*pct.yaml` (canonical sweep specs only; `sp500_smoketest.yaml` is excluded by the sweep script's glob filter):

- Thresholds: {10, 20, 40, 50}% (max-drawdown matched per threshold: dd5, dd10, dd20, dd25).
- Horizons: {5, 10, 25, 50, 100, 200}d (per threshold, not full crossing: 5d for {10, 20}; 10d for {10, 20}; 25d, 50d for {10, 20, 40, 50}; 100d for {20, 40, 50}; 200d for {40, 50}).
- Universe: sp500 — **503 in registry**; **486 tickers actually retained** after the data adapter's NaN-row guard. Excluded set is stable across all 17 cells (recently-listed names lacking the 800+400+200+100 walk-forward history).
- **Missing vs nas/rus**: the sp500 grid intentionally lacks `sp500_up_10pct_100d_dd5pct`, `sp500_up_10pct_200d_dd5pct`, and `sp500_up_40pct_10d_dd20pct` — those three (thr, H) tuples have no yaml under `configs/gbdt/experiments/`.

Read `_177` for the cross-market US-sweep priors, `_188` for the first full russell1000 sweep, and `_192` for the prior nasdaq100 sweep that this memo mirrors.

## How to read the metrics

- **Weighted R-precision** (per-day variable-K: `sum(positives_caught) / sum(R(d))`, R(d) = positives that day) is the **standard cross-cell metric** — panel-invariant, baseline = base rate, so it compares cleanly to the nasdaq100/russell1000/nifty cells in `_177` / `_188` / `_192` without the fixed-K bias P@k carries on staggered panels.
- **ROC-AUC** reported eval + test; discrimination signal, not gated.
- **Lift = R-prec / base_rate** is discussed in **prose only**, never as a table column (CLAUDE.md reporting convention). The `base(e)`/`base(t)` columns let you compute lift on demand and keep the underlying hit-rate scale visible.
- Per-day P@k denominator is `min(R(d), k)` everywhere (achievable positives, not picks-made).

**Compound signal/null rule** (CLAUDE.md):
- AUC ∈ [0.45, 0.55] **AND** R-prec lift < 1.2× → **null**.
- AUC ∈ [0.45, 0.55] **AND** lift > 1.5× → **top-tail signal hidden by AUC** (investigate the prediction-extreme regime).
- AUC ∈ [0.45, 0.55] **AND** lift ∈ [1.2, 1.5] → **ambiguous**.
- AUC > 0.55 → **discriminating**.
- AUC < 0.45 → **anti-predictive** (worse than random).

---

## Master table (17 cells)

Raw metric values. `(e)` = eval segment, `(t)` = test segment. `n_feat` = features kept after FS+HP loop. `n_iter` = iterations the loop ran (capped at 3 under sweep mode — HP search disabled per issue #32).

| Cell | Thr% | H(d) | DD% | base(e) | base(t) | AUC(e) | AUC(t) | Rprec(e) | Rprec(t) | n_iter | n_feat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sp500_up_10pct_5d_dd5pct | 10 | 5 | 5 | 0.0291 | 0.0432 | 0.8393 | 0.7815 | 0.3386 | 0.2905 | 3 | 129 |
| sp500_up_10pct_10d_dd5pct | 10 | 10 | 5 | 0.0812 | 0.1130 | 0.8002 | 0.7107 | 0.4011 | 0.3580 | 3 | 49 |
| sp500_up_10pct_25d_dd5pct | 10 | 25 | 5 | 0.2196 | 0.2638 | 0.7036 | 0.5896 | 0.4711 | 0.4080 | 3 | 33 |
| sp500_up_10pct_50d_dd5pct | 10 | 50 | 5 | 0.3469 | 0.2960 | 0.5660 | 0.4002 | 0.4278 | 0.3690 | 3 | 21 |
| sp500_up_20pct_5d_dd10pct | 20 | 5 | 10 | 0.0033 | 0.0060 | 0.8638 | 0.8457 | 0.1266 | 0.1583 | 3 | 279 |
| sp500_up_20pct_10d_dd10pct | 20 | 10 | 10 | 0.0116 | 0.0226 | 0.8800 | 0.8351 | 0.2363 | 0.2753 | 3 | 279 |
| sp500_up_20pct_25d_dd10pct | 20 | 25 | 10 | 0.0657 | 0.0886 | 0.8485 | 0.7755 | 0.4188 | 0.3428 | 3 | 45 |
| sp500_up_20pct_50d_dd10pct | 20 | 50 | 10 | 0.1657 | 0.1367 | 0.7787 | 0.7271 | 0.4683 | 0.3481 | 3 | 31 |
| sp500_up_20pct_100d_dd10pct | 20 | 100 | 10 | 0.3010 | — | 0.7016 | — | 0.5036 | — | 3 | 16 |
| sp500_up_40pct_25d_dd20pct | 40 | 25 | 20 | 0.0079 | 0.0182 | 0.9347 | 0.8964 | 0.2928 | 0.2977 | 3 | 33 |
| sp500_up_40pct_50d_dd20pct | 40 | 50 | 20 | 0.0366 | 0.0402 | 0.9011 | 0.8621 | 0.3259 | 0.2495 | 3 | 28 |
| sp500_up_40pct_100d_dd20pct | 40 | 100 | 20 | 0.1143 | — | 0.8326 | — | 0.4601 | — | 3 | 29 |
| sp500_up_40pct_200d_dd20pct | 40 | 200 | 20 | 0.2656 | — | 0.7813 | — | 0.5378 | — | 3 | 20 |
| sp500_up_50pct_25d_dd25pct | 50 | 25 | 25 | 0.0035 | 0.0091 | 0.9426 | 0.9128 | 0.1199 | 0.2485 | 3 | 36 |
| sp500_up_50pct_50d_dd25pct | 50 | 50 | 25 | 0.0206 | 0.0257 | 0.9257 | 0.9034 | 0.2758 | 0.2788 | 3 | 17 |
| sp500_up_50pct_100d_dd25pct | 50 | 100 | 25 | 0.0740 | — | 0.8657 | — | 0.4535 | — | 3 | 34 |
| sp500_up_50pct_200d_dd25pct | 50 | 200 | 25 | 0.1936 | — | 0.8240 | — | 0.5202 | — | 3 | 29 |

R-precision values cross-checked via `uv run python -m scripts.gbdt.compute_r_precision <cell>/predictions/{eval,test}.csv` (same code path as the aggregator's `per_day_r_precision`).

---

## Per-cell verdicts (compound rule, grouped by signal status)

### Discriminating (16 cells: 11 on test + 5 on eval-only)

Listed by ascending threshold then horizon. Verdict basis (test vs eval-only) called out per bullet. Lift quoted in prose only.

- **10%/5d**: test AUC 0.781, lift **6.73×** (base 4.32% → R-prec 29.1%). Top +10% cell on test by both AUC and lift; one horizon-step earlier than the +10%/10d cell. Eval lift 11.63× → test 6.73× = 58% retention.
- **10%/10d**: test AUC 0.711, lift **3.17×** (base 11.3% → R-prec 35.8%). Solid mid-tier discriminating cell. Eval lift 4.94×; 64% retention to test.
- **10%/25d**: test AUC 0.590, lift **1.55×** (base 26.4% → R-prec 40.8%). Still above the 0.55 discriminating bar — sp500 buys one more horizon-step at +10% than nasdaq's 25d cell (which crossed into ambiguous-null at AUC 0.511) and matches russell's 25d cell (AUC 0.582 disc).
- **20%/5d**: test AUC 0.846, lift **26.29×** (base 0.60% → R-prec 15.8%). **Second-highest test lift in the sweep**. Eval lift 38.94× → test 26.29× = 68% retention; the rare-event compression of top picks holds up well. Full 279-feature pool retained (best_iter=0).
- **20%/10d**: test AUC 0.835, lift **12.19×** (base 2.26% → R-prec 27.5%). Strong mid-threshold short-horizon cell; full 279-feature pool kept.
- **20%/25d**: test AUC 0.775, lift **3.87×** (base 8.86% → R-prec 34.3%). Discriminating with broader coverage than the 20%/5d cell.
- **20%/50d**: test AUC 0.727, lift **2.55×** (base 13.7% → R-prec 34.8%). Strongest +20% mid-horizon cell still above the discriminating bar.
- **20%/100d**: eval AUC 0.702, lift **1.67×** on eval (base 30.1% → R-prec 50.4%). Eval-only verdict (no test window). Lift above 1.5× → discriminating by the compound rule.
- **40%/25d**: test AUC 0.896, lift **16.32×** (base 1.82% → R-prec 29.8%). **Third-highest test lift in the sweep**. Solid rare-event concentration.
- **40%/50d**: test AUC 0.862, lift **6.20×** (base 4.02% → R-prec 24.9%). Discriminating with the highest AUC in the +40% mid-horizon band.
- **40%/100d**: eval AUC 0.833, lift **4.03×** on eval (base 11.4% → R-prec 46.0%). Eval-only. Strongly discriminating on eval; test missing.
- **40%/200d**: eval AUC 0.781, lift **2.03×** on eval. Eval-only. The lift drops above 2× at 200d — the eval signal stays clearly above the discriminating bar.
- **50%/25d**: test AUC 0.913, lift **27.45×** (base 0.91% → R-prec 24.9%). **Top test lift in the sweep**, AUC also tops the sweep at 0.913. Inner-stop fired at iter 2 (degradation) rather than 3 (plateau) — the only degradation-stop cell on sp500.
- **50%/50d**: test AUC 0.903, lift **10.86×** (base 2.57% → R-prec 27.9%). Strong rare-event cell with very high AUC.
- **50%/100d**: eval AUC 0.866, lift **6.13×** on eval. Eval-only.
- **50%/200d**: eval AUC 0.824, lift **2.69×** on eval. Eval-only. Eval AUC stays above 0.8 at 200d horizon.

### Ambiguous (0 cells)

No sp500 cell lands in the ambiguous band on test or eval-only. This is the cleanest sweep of the three US universes (nas had 4 ambiguous, rus had 1 ambiguous + 1 null + 1 anti-predictive).

### Null (0 cells)

No sp500 cell hits the null compound (AUC ∈ [0.45, 0.55] + lift < 1.2×).

### Anti-predictive (1 cell)

- **10%/50d**: test AUC **0.400** (below the 0.45 lower bound) AND lift **1.25×** (in [1.2, 1.5]). The compound-rule strict reading is anti-predictive on AUC. Eval AUC 0.566 / lift 1.23× — the eval segment was barely above the discriminating AUC bar and the test segment crashed well below random. This is the **only AUC < 0.45 cell across all three US sweeps**. Mechanistic read: at H=50d the +10% target's class boundary is dense (base rate 27–35%) and the sp500 feature pool's calibration inverts on the test segment — the predictions rank positives below negatives more often than chance. nasdaq's +10%/50d cell came close (AUC 0.475) but stayed inside the [0.45, 0.55] band; russell's stayed exactly at 0.511. **Flag for follow-up**: dig into the inverted-prediction regime on `predictions/test.csv` — is the inversion confined to specific quarters or tickers, or is it pool-wide? Candidate root cause is regime-shift in the eval→test window (base rate dropped from 34.7% → 29.6%).

### Coverage gap — no test window (5 cells flagged separately above)

The cells with `H ≥ 100` and the standard 800/400/200/100 split have **zero test rows** because each ticker's trailing 100 rows have NaN targets (forward window incomplete past the test cutoff). sp500 cells affected: **20%/100d, 40%/{100d, 200d}, 50%/{100d, 200d}** — all 5 eval-only discriminating. Same methodology limitation `_177`, `_188`, `_192` flagged; same `[data] WARNING: Test segment expected to be EMPTY` runner emission.

---

## Patterns (cross-cell)

### 1. Universal eval→test AUC decay — fully replicates `_177` + `_188` + `_192`, no exception this time

**All 12 test-evaluable cells lose AUC from eval to test.** dAUC ranges from **−0.166** (10%/50d, the worst on this sweep — the cell that crashed below 0.45) to **−0.019** (20%/5d, the smallest decay). Mean dAUC across all 12: **−0.062**. The mean decay is **smaller than nas (−0.110) and rus (≈ −0.07)** — sp500's eval→test gap is the narrowest of the three universes on the matched test cells.

Decay magnitude scales with horizon inside the +10% family (test-evaluable cells only):

| Cell | AUC eval → test | dAUC | Test class |
|---|---|---:|---|
| 10%/5d | 0.839 → 0.781 | −0.057 | discriminating |
| 10%/10d | 0.800 → 0.711 | −0.089 | discriminating |
| 10%/25d | 0.704 → 0.590 | −0.115 | discriminating |
| 10%/50d | 0.566 → 0.400 | −0.166 | **anti-predictive** |

The +10%/50d cell crosses below the anti-predictive bound on test — **first anti-predictive cell across all three US sweeps**. The crossover horizon-step pattern across the three universes is now: nas 10%/25d crosses to ambiguous; rus 10%/50d crosses to ambiguous; sp 10%/50d crosses to anti-predictive. The sp500 panel does not buy more horizon-reach than russell at the +10% line — and at 50d it actively inverts.

### 2. Horizon effect: short is clean, long decays — same as `_177` + `_188` + `_192`

Within the +10% threshold, test-evaluable lift declines monotonically: 5d → 6.73×, 10d → 3.17×, 25d → 1.55×, 50d → 1.25× (anti-predictive). The +20%, +40%, +50% threshold families decay less and stay strongly discriminating to the longest test-evaluable horizon — e.g. 50%/50d at lift 10.86× test, 40%/50d at 6.20×, 20%/50d at 2.55×. The eval-only cells at H ≥ 100 show the same pattern: +50%/100d eval lift 6.13×, +40%/100d 4.03×, +20%/100d 1.67×, +40%/200d 2.03× — rare-event signal survives further out in horizon.

### 3. Threshold effect: rarer events → higher lift, modest absolute R-prec — same as `_177` + `_188` + `_192`

Test-segment lifts by threshold family:

- +10%: 1.25× (50d, anti-predictive) — 6.73× (5d). Base rates 4.3–29.6%. Higher absolute R-precision (~29–41%).
- +20%: 2.55× (50d) — 26.29× (5d). Base rates 0.6–13.7%. R-prec 16–35%.
- +40%: 6.20× (50d) — 16.32× (25d). Base rates 1.8–4.0%. R-prec 25–30%.
- +50%: 10.86× (50d) — 27.45× (25d). Base rates 0.9–2.6%. R-prec 25–28%.

Same shape as `_177` + `_188` + `_192`: rarer events trade base rate for very high lift on a smaller absolute R-precision. The +40%/+50% sp500 cells are the "alert me to the rare big mover" surfaces; +10%/+20% cells are "broader coverage, smaller per-pick edge."

### 4. Universe effect (sp500 vs nasdaq100 vs russell1000): sp500 wins lift AND AUC across the matched grid

On the 12 cells with test windows across all three universes, the head-to-head comparison is:

| Cell | sp AUC(t) | nas AUC(t) | rus AUC(t) | sp lift(t) | nas lift(t) | rus lift(t) | Lift winner |
|---|---:|---:|---:|---:|---:|---:|---|
| 10%/5d | 0.781 | 0.768 | 0.762 | 6.73 | 5.50 | 5.12 | **sp** |
| 10%/10d | 0.711 | 0.657 | 0.695 | 3.17 | 2.89 | 2.66 | **sp** |
| 10%/25d | 0.590 | 0.511 | 0.582 | 1.55 | 1.46 | 1.44 | **sp** |
| 10%/50d | **0.400** | 0.475 | 0.511 | 1.25 | 1.24 | 1.22 | sp (marginal) |
| 20%/5d | 0.846 | 0.756 | 0.833 | 26.29 | 19.17 | 17.71 | **sp** |
| 20%/10d | 0.835 | 0.781 | 0.818 | 12.19 | 7.32 | 8.17 | **sp** |
| 20%/25d | 0.775 | 0.709 | 0.730 | 3.87 | 3.45 | 3.09 | **sp** |
| 20%/50d | 0.727 | 0.665 | 0.682 | 2.55 | 2.13 | 1.99 | **sp** |
| 40%/25d | 0.896 | 0.724 | 0.846 | 16.32 | 7.98 | 11.70 | **sp** |
| 40%/50d | 0.862 | 0.753 | 0.803 | 6.20 | 8.06 | 5.69 | nas |
| 50%/25d | 0.913 | 0.761 | 0.880 | 27.45 | 14.35 | 18.57 | **sp** |
| 50%/50d | 0.903 | 0.748 | 0.829 | 10.86 | 9.31 | 9.74 | **sp** |

**Tally: sp wins lift on 11 of 12 cells**, only loses 40%/50d (nas 8.06× > sp 6.20×). sp wins AUC(t) on **11 of 12 cells** (only loses 10%/50d where sp dropped to 0.400 vs rus 0.511; no ties).

This **revises `_192` § 4** again. `_192`'s reading was "threshold-dependent: nasdaq wins lift on common events, russell wins lift on rare events." With sp500 in the picture, the corrected reading is **"the mid-cap-heavy 486-ticker sp500 panel dominates both the smaller-and-purer nasdaq100 (92 tickers) and the larger-and-noisier russell1000 (889 tickers) on lift AND AUC on the matched grid."** The 50%/25d sp500 cell — sweep top at 27.45× — is **48% higher lift** than russell's 18.57× and **91% higher** than nasdaq's 14.35× on the same target spec. The 20%/5d sp500 cell at 26.29× is 48% higher than russell's 17.71×.

Mechanistic interpretation candidates (not yet investigated; raised for follow-up):
- **Panel composition sweet spot**: sp500 has the breadth to learn rare-event patterns (vs nas) without the heterogeneity that dilutes the feature-relationships (vs rus). 486 tickers is roughly the optimum for the asset-agnostic feature pool.
- **Liquidity floor**: sp500's inclusion criteria filter out the micro-cap noise that russell1000 includes — the rare-positive cells learn cleaner patterns.
- **Index reconstitution churn**: russell1000's wider reconstitution churn introduces survivor/inclusion-bias noise that sp500 (more stable membership) avoids.

The eval-only cells confirm the pattern at H ≥ 100: sp wins eval lift on **5 of 5 matched eval-only cells** (20%/100d sp 1.67× > rus 1.59× > nas 1.49×; 40%/100d sp 4.03× > rus 3.29× > nas 2.87×; 40%/200d sp 2.03× > nas 1.93× > rus 1.89×; 50%/100d sp 6.13× > rus 4.70× > nas 3.90×; 50%/200d sp 2.69× > nas 2.30× > rus 2.18×).

### 5. FS prunes hard with neutral-to-positive effect — pattern consistent with `_188` + `_192`

14 of 17 cells finish with `n_feat < 100`. 2 cells finish with `n_feat = 279` (full pool kept, best_iter=0): 20%/5d and 20%/10d — the two short-horizon +20% cells where the entire feature pool was retained. 1 cell at 129. No cell prunes below 16 features. The pattern is closer to `_188` than `_192` (russell had 4 full-pool cells; nas had 3) — sp500's mid-size panel pulls fewer cells into the "keep everything" regime.

The 2 "full pool kept" cells on sp500 (20%/5d, 20%/10d) overlap partially with `_192`'s full-pool set on nasdaq100 (which included 20%/25d, 40%/100d, 50%/50d): the mid-threshold short-horizon corner consistently wants the full feature pool. The 50%/50d cell that kept the full pool on nasdaq100 prunes to **17 features** on sp500 — the broader panel finds a compact signal where the narrower panel needed the whole pool.

### 6. Calibration: 15 of 17 cells received isotonic, 2 stayed native

Of the 17 cells, 15 layered an isotonic calibration map on top of CatBoost's raw probabilities (Spiegelhalter Z significant); 2 stayed native (20%/100d, 50%/25d). The 2 native cells are at the extreme-rare-event end where probabilities are tightly compressed near zero (50%/25d base = 0.35%) or near the eval-only mid-base regime (20%/100d base = 30.1%). Same pattern as `_188` (14/20 isotonic) and `_192` (17/20 isotonic) — sp500's calibration take-rate of 88% is the highest of the three.

### 7. Best-iteration distribution

2 of 17 cells stopped at iter 0 (best — no prune improved val Brier; both are the full-pool 20%/{5d, 10d} cells). 6 at iter 1. 9 at iter 2. **1 cell stopped via degradation** (the inner-stop catching val Brier going up): 50%/25d — sweep top by lift, where the inner-stop caught degradation at iter 2 and reverted to iter 1 (which itself happened to be the best). The other 16 cells stopped via plateau. No cell ran past 3 iterations because the sweep mode caps `max_iter=3` (HP search disabled per `report.md` warning, issue #32). Same constraint as `_188` + `_192`.

---

## Wall-time measurement vs the post-#190+#182 cache projection

The task brief identified this as the first sp500 sweep to benefit from **both** #182 (universe feature cache) **and** #190 (source-hash cache key avoiding code-commit thrash). The headline operational finding is the **~30× end-to-end speedup** on warm cells.

Measured (sum of per-cell `wall_time_total_sec` across all 17 cells): **3.996 h** = 14385 s. Wall-clock from the sweep log: **14464 s** = 4.02 h (78 s overhead from script orchestration + the 2-minute sweep restart).

Per-cell breakdown:

- **Cold cell** (first run, built the universe feature cache): `sp500_up_10pct_10d_dd5pct` — 9541 s = **2h39m**. Cold features build alone took **9189.1 s = 2h33m** (`[features] complete in 9189.1s shape=(3638785, 279)`); the remaining 352 s was target + uniqueness + loop + artifact.
- **Warm cells** (16 of them got a `[features] loaded from universe cache (key match) in 4.4–5.7s` line): per-cell elapsed range **258 s–481 s**, mean **~303 s = ~5.0 min**. The 481 s outlier is `sp500_up_10pct_5d_dd5pct` (129-feature post-FS pool, more iterations on the high-prevalence target).
- **Pre-existing cells** (0 of 17): none — this sweep ran every cell fresh on the post-#190 cache key.

**Cache speedup**:
- **Features-phase speedup**: cold 9189.1 s vs warm cache hit 4.4–5.7 s = **~1700× speedup on the features phase alone**.
- **End-to-end per-cell speedup**: cold 9541 s / warm mean 303 s = **~31.5× end-to-end** (features + target + uniqueness + loop + artifact).
- **Compute saved on the 16 warm cells**: 16 × (9189.1 s − 5 s) ≈ **40.8 hours saved** vs the no-cache counterfactual (where every cell would have rebuilt the 3.64M-row × 279-feature matrix from scratch).

**Comparison to `_188` russell1000 + `_192` nasdaq100 timing**:

| Universe | Panel rows | Cold cell | Warm-cell mean | Cache speedup (end-to-end) |
|---|---:|---:|---:|---:|
| nasdaq100 (`_192`) | 645k | 42m20s | ~98 s | ~26× |
| sp500 (`_193`) | 3.64M | 2h39m | ~303 s | ~31.5× |
| russell1000 (`_188`) | 6.06M | 4h24m | ~544 s | ~29× |

The cold-cell scaling tracks panel size cleanly: 645k → 3.64M → 6.06M rows = 1× / 5.6× / 9.4× — and cold features scale 2226 s → 9189 s → 15840 s = 1× / 4.1× / 7.1× (sub-linear because of fixed per-cell I/O overhead). Warm-cell scaling is also panel-size-dominated: 98 s → 303 s → 544 s = 1× / 3.1× / 5.5×. The cache speedup ratio is **higher on sp500 (31.5×) than on either nas (26×) or rus (29×)** — sp500 sits in the sweet spot where the cold features cost is large enough to dwarf the per-cell overhead but the warm loop+target cost is still modest.

**The combined picture**: post-#182 shared feature cache + post-#190 source-hash cache key pay off cleanly on the mid-size panel. The 16 warm cells totalled ~1.35 h of compute (vs ~40.8 h without the cache). On sp500's 3.64M-row panel, the **per-cell overhead** (data load + target + uniqueness + artifact ≈ 295 s per warm cell minus the 5 s cache lookup = ~290 s) is now the dominant time sink in the warm regime — the FS+HP loop itself runs in a few minutes thanks to the small post-prune feature counts.

**Note on sweep wall-clock vs cells-sum wall**: the sweep was launched once, restarted once (the initial launch at 10:59:06Z was killed at the [SKIP] check; the actual sweep resumed at 11:01:21Z), then ran contiguously to 15:00:10Z. Total wall-clock 4h01m, cells-sum 3.996h — the 78 s discrepancy is script orchestration overhead between cells.

---

## Practical guidance — which sp500 cells to pursue

**Pursue (strong discriminating + test-validated):**
- **Short-horizon × high-threshold**: 50%/25d (lift 27.45× test, base 0.91%), 20%/5d (26.29×, base 0.60%), 40%/25d (16.32×, base 1.82%), 50%/50d (10.86×, base 2.57%), 20%/10d (12.19×, base 2.26%) — rare-event top-pick surfaces with AUC 0.83–0.91. Despite lower per-pick absolute R-precision (16–30%), the lift over baseline is very high.
- **Mid-threshold mid-horizon**: 40%/50d (lift 6.20×, R-prec 25.0%, AUC 0.86), 20%/25d (3.87×, R-prec 34.3%, AUC 0.78) — solid lift with broader coverage.
- **10%/5d** as the broadest-coverage discriminating cell (lift 6.73×, base 4.32%, R-prec 29.1%, AUC 0.78 — picks land in ∼3/10 of top-K picks on test on the most common-event cell that still discriminates clearly).

**Use with caution — eval-only (no test window):**
- 50%/100d, 40%/100d, 50%/200d, 40%/200d, 20%/100d are all discriminating on eval (lift 1.67×–6.13×, AUC 0.70–0.87). Candidates for the long-horizon strategy bucket but cannot be trusted without a test-split fix. **Treat eval AUC + eval lift as a hypothesis pending test data**, given the universal eval→test AUC decay observed everywhere else.

**Avoid — anti-predictive:**
- 10%/50d (test AUC 0.400 — the first sub-0.45 cell across all three US sweeps; needs the predictions-test-csv-inversion investigation flagged in the verdict block above).

**Ambiguous: none on sp500.** This is the cleanest sweep of the three US universes — sp500 either discriminates clearly or fails clearly, no in-between cells on the matched grid.

---

## Cross-reference to `_177` + `_188` + `_192`

| `_177`/`_188`/`_192` finding | sp500 sweep evidence (this memo) |
|---|---|
| Universal eval→test AUC decay (every completed cell loses AUC) | **Fully replicates** — 12 of 12 test-evaluable cells lose AUC; dAUC range −0.166 to −0.019. Mean −0.062, the narrowest of the three US universes |
| Decay magnitude scales with horizon at fixed threshold | **Replicates** — 10% family decay −0.057 → −0.089 → −0.115 → −0.166 from 5d → 50d (monotonic, steeper than nas/rus) |
| Short-horizon (5d–10d) sweet spot | **Replicates** — 5d/10d cells stay strongly discriminating across all thresholds |
| Higher thresholds → higher lift, smaller base, modest absolute R-prec | **Replicates** — +40%/+50% cells hit lift 6–27× on 0.9–4% base, +10% cells hit lift 1.25–6.73× on 4–30% base |
| H ≥ 100 has zero test rows (test-split methodology gap) | **Replicates** — 5 sp500 cells affected; same `data WARNING` text |
| `_192` "panel-width-vs-lift is threshold-dependent" hypothesis (nas wins common events, rus wins rare events) | **Revised** — sp500 wins lift on 11 of 12 matched cells across all thresholds (only loses 40%/50d to nas), AUC on 11 of 12 (only loses 10%/50d to rus). The mid-cap-heavy 486-ticker sp500 panel is the lift-and-AUC sweet spot across the three US universes |
| `_192` nas +10%/25d crossover into null AUC band on test (AUC 0.511) | **Different on sp500** — sp500 +10%/25d stays clearly discriminating at AUC 0.590; sp500's +10% crossover is one horizon-step later (at 10%/50d) and crashes harder (AUC 0.400 anti-predictive vs nas's 0.475 and rus's 0.511 ambiguous) |
| `_188` rus +10%/50d ambiguous on test (AUC 0.511, lift 1.22×) | **More severe on sp500** — sp500 +10%/50d is the first anti-predictive cell across all three US sweeps (AUC 0.400 < 0.45 bound) |
| FS prunes hard (279 → 20–60) on most cells | **Partially replicates** — 14 of 17 cells prune to <100; 2 keep full 279 (+20%/{5d, 10d}); no cell prunes below 16 |
| Calibration: 14/20 (rus) → 17/20 (nas) isotonic | **Highest take-rate** — 15/17 (88%) on sp500 |

The two key concrete revisions this memo introduces:
1. **`_192` § 4 (universe effect) is wrong now too**: with sp500 in the picture, the mid-cap-heavy 486-ticker panel dominates BOTH nas and rus on lift across all thresholds (not just rare or common events), and on AUC on 11 of 12 cells. The panel-size-vs-lift relationship is **non-monotonic** — sp500 is a local optimum between nas (too small) and rus (too noisy). This is the **single most actionable cross-universe finding** so far.
2. **The +10%/50d cell hits anti-predictive on sp500 for the first time across all three US sweeps** (AUC 0.400 vs nas's 0.475 / rus's 0.511 in [0.45, 0.55] ambiguous). The +10% target at H ≥ 50 has now spent test cells in every band: discriminating (rus 10%/25d at 0.582), ambiguous (rus 10%/50d, nas 10%/25d-50d), and now **anti-predictive (sp 10%/50d)**. The crossover horizon-step is universe-dependent and the failure mode (where the cell lands once it crosses) is also universe-dependent.

---

## User-facing read (no automated PASS/FAIL)

Of the 17 sp500 cells, **11 discriminate on the held-out test window** (10%/{5d, 10d, 25d}, 20%/{5d, 10d, 25d, 50d}, 40%/{25d, 50d}, 50%/{25d, 50d} — strongest lift first within group), **5 are eval-only discriminating** (no test window under the current split: 20%/100d, 40%/{100d, 200d}, 50%/{100d, 200d}), **0 are ambiguous, 0 null**, and **1 is anti-predictive on test** (10%/50d at AUC 0.400 — the first sub-0.45 cell across all three US sweeps). The short-horizon × high-threshold corner is the production-candidate region — **50%/25d (lift 27.45×), 20%/5d (26.29×), 40%/25d (16.32×)** are the standout cells by test-segment R-precision lift. The eval→test AUC decay observed in `_177` / `_188` / `_192` fully replicates on sp500 with no exception. The cross-universe finding is the headline analytical surprise: **sp500 wins lift on 11 of 12 matched test cells AND AUC on 11 of 12** vs both nasdaq100 and russell1000 — the 486-ticker mid-cap-heavy panel is the lift-and-AUC sweet spot. Operationally, the post-#190+#182 cache delivered **~31.5× end-to-end speedup** on warm cells (cold 2h39m → warm mean 5 min) and **~1700× on features phase alone**, saving ~40.8 h of compute across the 16 warm cells. The 5 eval-only cells (H ≥ 100) need the test-split fix before they can be judged. As `_177` + `_188` + `_192` noted, the PASS/FAIL call on any individual cell remains a user judgment; this memo characterizes the landscape across the completed sp500 sweep and updates the cross-universe panel-size hypothesis to "sp500 dominates" on the matched grid.

---

## R-Precision@K (current methodology — added 2026-06-01)

Per `.claude/memories/project-r-precision-methodology.md`, R-Precision@K is the post-2026-06-01 headline cross-cell metric for gbdt. Recomputed on the 12 test-evaluable sp500 cells from each cell's `predictions/test.csv` (source: `results/gbdt/data/r_precision_at_k.csv`); sorted by AUC descending. The 5 cells without a test window (H ≥ 100) are excluded.

| cell | rows | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|---|---|
| sp500_up_50pct_25d_dd25pct | 36,450 | 0.91% | 0.913 | 0.279 | 0.139 | 0.187 | 0.329 | 0.494 |
| sp500_up_50pct_50d_dd25pct | 24,300 | 2.57% | 0.903 | 0.680 | 0.407 | 0.328 | 0.259 | 0.395 |
| sp500_up_40pct_25d_dd20pct | 36,450 | 1.82% | 0.896 | 0.280 | 0.222 | 0.215 | 0.232 | 0.394 |
| sp500_up_40pct_50d_dd20pct | 24,300 | 4.02% | 0.862 | 0.100 | 0.313 | 0.312 | 0.286 | 0.276 |
| sp500_up_20pct_5d_dd10pct | 46,170 | 0.60% | 0.846 | 0.195 | 0.142 | 0.180 | 0.318 | 0.419 |
| sp500_up_20pct_10d_dd10pct | 43,740 | 2.26% | 0.835 | 0.270 | 0.285 | 0.261 | 0.291 | 0.391 |
| sp500_up_10pct_5d_dd5pct | 46,170 | 4.32% | 0.781 | 0.274 | 0.309 | 0.320 | 0.324 | 0.331 |
| sp500_up_20pct_25d_dd10pct | 36,450 | 8.86% | 0.775 | 0.467 | 0.458 | 0.459 | 0.415 | 0.398 |
| sp500_up_20pct_50d_dd10pct | 24,300 | 13.67% | 0.727 | 0.360 | 0.280 | 0.288 | 0.342 | 0.353 |
| sp500_up_10pct_10d_dd5pct | 43,740 | 11.30% | 0.711 | 0.467 | 0.459 | 0.462 | 0.440 | 0.405 |
| sp500_up_10pct_25d_dd5pct | 36,450 | 26.38% | 0.590 | 0.373 | 0.391 | 0.373 | 0.404 | 0.403 |
| sp500_up_10pct_50d_dd5pct | 24,300 | 29.60% | 0.400 | 0.380 | 0.387 | 0.332 | 0.306 | 0.311 |

The body's "11 discriminate / 0 ambiguous / 0 null / 1 anti-predictive" verdict holds under R-Precision@K: AUC is the dominant classifier and the 10%/50d cell (AUC 0.400, R-p@10 = 0.306) sits clearly in the anti-predictive band — R-Precision@K can't rescue it. **The top-tail story stays strong**: 50%/50d gets the most extreme R-Precision@1 of any cell in the project at 0.680 (1 in ~1.5 top-picks is a hit, base 2.57%) — note that this falls to 0.394 at K=20, showing the model's confidence concentrates sharply in the top 1-3 names per day, not across a wider band.
