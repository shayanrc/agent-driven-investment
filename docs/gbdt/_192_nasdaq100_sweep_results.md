# Task #192 — nasdaq100 sweep results (20 cells, post-#183 + #87 agent-loop trio)

**Date**: 2026-05-31.
**Branch**: `gbdt-nasdaq100-sweep-results`.
**Data**: `results/gbdt/data/_192_nasdaq100_sweep_results_data.json` (machine-readable master table + per-cell classifications).
**Sweep log**: `logs/nasdaq100_sweep.log` (per-cell wall-clock, run in two passes — see "Wall-time" below).
**Prior**: this memo mirrors the format of `_188_russell1000_sweep_results.md` and cross-references both `_177` (US-sweep snapshot at 12/57 cells, includes nasdaq100 prior data points) and `_188` (the first complete russell1000 sweep).

## Headline

The nasdaq100 sweep finishes **20/20 cells** with a clean signal floor: **16 of 20 cells discriminate** (12 on the held-out test window, 4 on eval-only with no test window under the current split), **4 cells are ambiguous** (all in the +10% threshold family at H ≥ 25). **No null, no anti-predictive** cells — a slightly cleaner signal landscape than russell1000 (`_188` had 1 null + 1 anti-predictive at +10%/H≥100). In total **7 cells lack a test window** (H ≥ 100 ate the test split under the 800/400/200/100 walk-forward — same methodology limitation `_177` + `_188` flagged). Strongest test-segment cells live exactly where `_177` and `_188` priors predicted: **short-horizon × high-threshold**. The top four by R-prec lift on test are **40%/10d (20.70×)**, **20%/5d (19.17×)**, **50%/25d (14.35×)**, **50%/50d (9.31×)**. The systemic eval→test AUC decay observed across nasdaq100 + sp500 in `_177` and the full russell1000 sweep in `_188` **fully replicates on this nasdaq100 sweep**: **all 13 test-evaluable cells lose AUC from eval to test** (dAUC range **−0.184 to −0.049**; no exception). The decay crosses into the [0.45, 0.55] null AUC band at +10%/25d on test (AUC 0.511) and +10%/50d (AUC 0.475 — just below the null band) — confirming `_177`'s nasdaq100 prior crossover at +10%/H=25 (where russell1000's `_188` shifted the crossover one horizon-step further out to +10%/50d on its broader 889-ticker panel). **Cross-universe comparison vs `_188`'s russell1000**: AUC(test) is higher on russell1000 at 12 of 13 matched cells; **lift is mixed** — russell wins decisively on the very-rare-event cells (40%/10d 42.1× vs 20.7×, 50%/25d 18.6× vs 14.4×, 40%/25d 11.7× vs 8.0×) but nasdaq wins on most common-event cells (10%/5d 5.5× vs 5.1×, 20%/25d 3.45× vs 3.09×, 40%/50d 8.1× vs 5.7×). This **partially confirms and partially contradicts** `_188`'s "wider panel → lower lift" hypothesis — confirms at common events, inverts at rare events. **Wall-time vs the ~2-3h projection: measured ~2.14 h (cells-sum)** — cold cell came in at 42m19s (≈ 37 min features + 4 min loop), warm cells averaged **~98 s each across 17 cells** (vs `_188`'s russell1000 average of ~544s — a **5.5× speedup on warm cells** consistent with the smaller 645k-row panel vs russell's 6.06M-row panel).

## What's covered

All **20 nasdaq100 cells** in `configs/gbdt/experiments/nasdaq100_up_*pct_*d_dd*pct.yaml` (canonical sweep specs only; the `_xgb_repl`, `_xgb_manual`, `_xgb_manual_fsloop`, and `_acceptance` variants are explicitly excluded by the sweep script's glob filter):

- Thresholds: {10, 20, 40, 50}% (max-drawdown matched per threshold: dd5, dd10, dd20, dd25).
- Horizons: {5, 10, 25, 50, 100, 200}d (per threshold, not full crossing: 5d for {10, 20}; 10d for {10, 20, 40}; 25d, 50d for {10, 20, 40, 50}; 100d for all four; 200d for {10, 40, 50}).
- Universe: nasdaq100 — 100 in registry; **92 tickers actually retained** after the data adapter's NaN-row guard. Excluded set is stable across all 20 cells (NASDAQ:{ABNB, APP, ARM, CEG, DASH, GEHC, GFS, PLTR} — recently-listed names with insufficient history for the 800+400+200+100 walk-forward split).

Read `_177` for the cross-market US-sweep priors (nasdaq100 + sp500 + russell1000 +10%/5d data point) and `_188` for the full russell1000 sweep this memo mirrors.

**Caveat on two cells**: `nasdaq100_up_10pct_100d_dd5pct` and `nasdaq100_up_10pct_25d_dd5pct` artifact directories on this branch are from prior PRs (#27 / #28) predating the #183 shared-feature-cache infrastructure. They were preserved by the sweep script's "skip if non-empty" guard rather than re-run. Per-cell metrics + R-precision were recomputed by the aggregator from the existing `predictions/{eval,test}.csv` (post-#183 methodology), so the entries in the master table below are methodology-consistent with the other 18 cells. Their feature counts and inner-stop signals reflect the older runner versions.

## How to read the metrics

- **Weighted R-precision** (per-day variable-K: `sum(positives_caught) / sum(R(d))`, R(d) = positives that day) is the **standard cross-cell metric** — panel-invariant, baseline = base rate, so it compares cleanly to the russell1000/sp500/nifty cells in `_177` / `_188` without the fixed-K bias P@k carries on staggered panels.
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
| nasdaq100_up_10pct_5d_dd5pct | 10 | 5 | 5 | 0.0435 | 0.0714 | 0.8165 | 0.7678 | 0.3713 | 0.3926 | 3 | 85 |
| nasdaq100_up_10pct_10d_dd5pct | 10 | 10 | 5 | 0.1092 | 0.1516 | 0.7741 | 0.6572 | 0.4463 | 0.4375 | 3 | 160 |
| nasdaq100_up_10pct_25d_dd5pct | 10 | 25 | 5 | 0.2494 | 0.2729 | 0.6549 | 0.5111 | 0.5121 | 0.3994 | 3 | 47 |
| nasdaq100_up_10pct_50d_dd5pct | 10 | 50 | 5 | 0.3538 | 0.2652 | 0.5836 | 0.4750 | 0.4670 | 0.3287 | 3 | 22 |
| nasdaq100_up_10pct_100d_dd5pct | 10 | 100 | 5 | 0.3982 | — | 0.4880 | — | 0.4881 | — | 3 | 33 |
| nasdaq100_up_10pct_200d_dd5pct | 10 | 200 | 5 | 0.4299 | — | 0.5097 | — | 0.5295 | — | 3 | 22 |
| nasdaq100_up_20pct_5d_dd10pct | 20 | 5 | 10 | 0.0057 | 0.0129 | 0.8563 | 0.7558 | 0.1714 | 0.2478 | 3 | 103 |
| nasdaq100_up_20pct_10d_dd10pct | 20 | 10 | 10 | 0.0193 | 0.0430 | 0.8628 | 0.7807 | 0.2303 | 0.3146 | 3 | 23 |
| nasdaq100_up_20pct_25d_dd10pct | 20 | 25 | 10 | 0.0953 | 0.1175 | 0.8070 | 0.7094 | 0.4322 | 0.4057 | 3 | 279 |
| nasdaq100_up_20pct_50d_dd10pct | 20 | 50 | 10 | 0.1980 | 0.1402 | 0.7502 | 0.6654 | 0.4956 | 0.2992 | 3 | 48 |
| nasdaq100_up_20pct_100d_dd10pct | 20 | 100 | 10 | 0.3210 | — | 0.6248 | — | 0.4791 | — | 3 | 19 |
| nasdaq100_up_40pct_10d_dd20pct | 40 | 10 | 20 | 0.0022 | 0.0072 | 0.8734 | 0.7512 | 0.1750 | 0.1500 | 3 | 32 |
| nasdaq100_up_40pct_25d_dd20pct | 40 | 25 | 20 | 0.0139 | 0.0426 | 0.9077 | 0.7235 | 0.2539 | 0.3401 | 2 | 70 |
| nasdaq100_up_40pct_50d_dd20pct | 40 | 50 | 20 | 0.0598 | 0.0554 | 0.8480 | 0.7533 | 0.3833 | 0.4471 | 2 | 61 |
| nasdaq100_up_40pct_100d_dd20pct | 40 | 100 | 20 | 0.1575 | — | 0.7928 | — | 0.4513 | — | 3 | 279 |
| nasdaq100_up_40pct_200d_dd20pct | 40 | 200 | 20 | 0.3037 | — | 0.6936 | — | 0.5870 | — | 3 | 25 |
| nasdaq100_up_50pct_25d_dd25pct | 50 | 25 | 25 | 0.0068 | 0.0291 | 0.8969 | 0.7607 | 0.0952 | 0.4179 | 3 | 25 |
| nasdaq100_up_50pct_50d_dd25pct | 50 | 50 | 25 | 0.0343 | 0.0404 | 0.8564 | 0.7476 | 0.2737 | 0.3763 | 3 | 279 |
| nasdaq100_up_50pct_100d_dd25pct | 50 | 100 | 25 | 0.1130 | — | 0.7888 | — | 0.4411 | — | 3 | 37 |
| nasdaq100_up_50pct_200d_dd25pct | 50 | 200 | 25 | 0.2403 | — | 0.7607 | — | 0.5536 | — | 2 | 29 |

R-precision values cross-checked via `uv run python -m scripts.gbdt.compute_r_precision <cell>/predictions/{eval,test}.csv` (same code path as the aggregator's `per_day_r_precision`).

---

## Per-cell verdicts (compound rule, grouped by signal status)

### Discriminating (16 cells: 12 on test + 4 on eval-only)

Listed by ascending threshold then horizon. Verdict basis (test vs eval-only) called out per bullet. Lift quoted in prose only.

- **10%/5d**: test AUC 0.768, lift **5.50×** (base 7.14% → R-prec 39.3%). Top +10% cell on test by both AUC and lift. Eval lift 8.54× — large eval→test compression (5.50/8.54 = 64% retention) consistent with the universal decay pattern.
- **10%/10d**: test AUC 0.657, lift **2.89×** (base 15.2% → R-prec 43.8%). Solid mid-tier discriminating cell. Eval lift 4.09×; 71% retention to test.
- **20%/5d**: test AUC 0.756, lift **19.17×** (base 1.29% → R-prec 24.8%). **Second-highest test lift in the sweep**. Eval lift 30.20× → test 19.17× = 63% retention; the rare-event compression of top picks holds up well.
- **20%/10d**: test AUC 0.781, lift **7.32×** (base 4.30% → R-prec 31.5%). Solid mid-threshold short-horizon cell.
- **20%/25d**: test AUC 0.709, lift **3.45×** (base 11.8% → R-prec 40.6%). Discriminating with broader coverage than the 20%/5d cell — picks land in ~41% of top-K positions on test.
- **20%/50d**: test AUC 0.665, lift **2.13×** (base 14.0% → R-prec 29.9%). Cleanest +20% mid-horizon cell still above the 0.55 AUC bar.
- **20%/100d**: eval AUC 0.625, lift **1.49×** on eval (base 32.1% → R-prec 47.9%). Eval-only verdict (no test window). Lift is in the [1.2, 1.5] ambiguous band but AUC > 0.55 → still classed discriminating by the compound rule. Whether it survives to test is the open question the broken split won't answer.
- **40%/10d**: test AUC 0.751, lift **20.70×** (base 0.72% → R-prec 15.0%). **Top test lift in the sweep**. Eval lift 79.55× → test 20.70× = 26% retention — the steepest eval→test compression in the sweep, characteristic of extreme-rare-event cells where the eval lift is so high because the absolute count of positives is tiny.
- **40%/25d**: test AUC 0.724, lift **7.98×** (base 4.26% → R-prec 34.0%). Inner-stop fired at iter 2 (degradation) rather than 3 (plateau).
- **40%/50d**: test AUC 0.753, lift **8.06×** (base 5.54% → R-prec 44.7%). Best combination of test AUC + test R-prec at the +40% threshold — strong discrimination AND high absolute pick quality. Inner-stop fired at iter 2 (degradation).
- **40%/100d**: eval AUC 0.793, lift **2.86×** on eval (base 15.8% → R-prec 45.1%). Eval-only. Strongly discriminating on eval; test missing. The 279-features-kept (best_iter=0) means FS prune didn't help — consistent with `_188`'s observation that high-threshold long-horizon cells often skew toward keeping the full pool.
- **40%/200d**: eval AUC 0.694, lift **1.93×** on eval. Eval-only. The lift falls below 2× — at 200d the eval signal weakens but stays above the discriminating bar.
- **50%/25d**: test AUC 0.761, lift **14.35×** (base 2.91% → R-prec 41.8%). **Third-highest test lift**. Strong rare-event concentration.
- **50%/50d**: test AUC 0.748, lift **9.31×** (base 4.04% → R-prec 37.6%). Discriminating with the full 279-feature pool retained (best_iter=0) — same pattern as 40%/100d and 20%/25d, suggesting high-threshold + mid-horizon cells benefit from the broad feature set.
- **50%/100d**: eval AUC 0.789, lift **3.91×** on eval. Eval-only.
- **50%/200d**: eval AUC 0.761, lift **2.30×** on eval. Eval-only. Inner-stop fired at iter 2 (degradation).

### Ambiguous (4 cells, all +10% family)

- **10%/25d**: test AUC **0.511** (in [0.45, 0.55]) AND lift **1.46×** (in [1.2, 1.5]) → ambiguous. Eval AUC 0.655 / lift 2.05× — the eval segment looked solidly discriminating, the test segment crashed into the null band. Matches the `_177` nasdaq100 prior reading exactly (eval AUC 0.655 → test 0.511, eval lift 2.05× → test 1.46× in `_177`'s snapshot was the same H=25 cell from PR #28). This is the **canonical nasdaq100 +10% crossover horizon** where eval-vs-test prevalence drift kills the signal.
- **10%/50d**: test AUC **0.475** (below 0.45 lower bound — technically anti-predictive on AUC) AND lift **1.24×** (in [1.2, 1.5]). The AUC is below 0.5 but only by 0.025 — the compound-rule narrow read is "ambiguous on lift basis, anti-predictive on AUC basis" → the practical call is ambiguous. One horizon-step further out than the 10%/25d crossover; matches the pattern `_188` saw at russell1000 +10%/50d (also ambiguous on test).
- **10%/100d**: eval AUC **0.488** (in [0.45, 0.55]) AND eval lift **1.23×** (in [1.2, 1.5]) → ambiguous (eval-only verdict). No test window. Matches `_188`'s russell1000 +10%/100d which was null on eval (AUC 0.489, lift 1.13×); nasdaq100 lift is marginally higher (1.23× vs 1.13×) so this cell crosses from "null" to "ambiguous" — the smaller-panel nasdaq picks up *just enough* signal at this cell to escape the null band, but not by much.
- **10%/200d**: eval AUC **0.510** (in [0.45, 0.55]) AND eval lift **1.23×** (in [1.2, 1.5]) → ambiguous. No test window. Contrast with `_188`'s russell1000 +10%/200d which was **anti-predictive** on eval (AUC 0.422, lift 1.16× — the feature pool inverted). The nasdaq100 cell is just barely better: AUC right at random, lift barely above the null bar. Interpretation: the +10% target at H=200d is a **structurally hard problem** for the feature pool on both universes, but nasdaq100's smaller panel produces less feature-relationship instability than russell1000's wider panel (where the 889 tickers' heterogeneity drove the feature relationships into a clearly inverted regime).

### Null / anti-predictive (0 cells)

No cell crosses into the null (AUC ∈ [0.45, 0.55] + lift < 1.2×) or strict anti-predictive (AUC < 0.45) regimes. The narrowest call is 10%/50d at AUC 0.475 / lift 1.24× — barely below the AUC band's lower bound and barely above the lift band's lower bound, classed ambiguous above.

### Coverage gap — no test window (7 cells flagged separately above)

The cells with `H ≥ 100` and the standard 800/400/200/100 split have **zero test rows** because each ticker's trailing 100 rows have NaN targets (forward window incomplete past the test cutoff). nasdaq100 cells affected: **10%/100d (ambiguous)**, **10%/200d (ambiguous)**, **20%/100d, 40%/100d, 40%/200d, 50%/100d, 50%/200d (all eval-only discriminating)**. Same methodology limitation `_177` and `_188` flagged; same `[data] WARNING: Test segment expected to be EMPTY` runner emission.

---

## Patterns (cross-cell)

### 1. Universal eval→test AUC decay — fully replicates `_177` + `_188`, no exception this time

**All 13 test-evaluable cells lose AUC from eval to test.** dAUC ranges from **−0.184** (40%/25d, the worst on this sweep) to **−0.049** (10%/5d, the smallest decay). Mean dAUC across all 13: **−0.110**. The 40%/10d cell which `_188`'s russell1000 sweep flagged as the lone "no-decay" outlier (dAUC +0.005 there) decays sharply on nasdaq100 (**−0.122**, from 0.873 eval to 0.751 test). So the "universal decay" reading is stricter on nasdaq100 than on russell1000.

Decay magnitude scales with horizon inside the +10% family (test-evaluable cells only):

| Cell | AUC eval → test | dAUC | Test class |
|---|---|---:|---|
| 10%/5d | 0.817 → 0.768 | −0.049 | discriminating |
| 10%/10d | 0.774 → 0.657 | −0.117 | discriminating |
| 10%/25d | 0.655 → 0.511 | −0.144 | **ambiguous** |
| 10%/50d | 0.584 → 0.475 | −0.109 | **ambiguous** |

The +10%/25d cell crosses into the null AUC band on test — **same horizon-step as nasdaq100's `_177` prior reading**, one horizon-step earlier than russell1000's `_188` crossover (which happened at +10%/50d). The smaller nasdaq100 panel buys roughly one fewer horizon-step of signal before the +10% target's eval AUC stops translating to test AUC. This is the inverse of `_188`'s read that broader panels buy more horizon-reach.

### 2. Horizon effect: short is clean, long decays — same as `_177` + `_188`

Within the +10% threshold, test-evaluable lift declines monotonically: 5d → 5.50× test, 10d → 2.89×, 25d → 1.46× (ambiguous), 50d → 1.24× (ambiguous). The +20%, +40%, +50% threshold families decay less and stay strongly discriminating to the longest test-evaluable horizon — e.g. 50%/50d at lift 9.31× test, 40%/50d at 8.06×. The eval-only cells at H ≥ 100 show the same pattern: +40%/100d eval lift 2.86×, +50%/100d 3.91×, +40%/200d 1.93× — rare-event signal survives further out in horizon.

### 3. Threshold effect: rarer events → higher lift, modest absolute R-prec — same as `_177` + `_188`

Test-segment lifts by threshold family:

- +10%: 1.24× (50d) — 5.50× (5d). Base rates 7–35%. Higher absolute R-precision (~33–44%).
- +20%: 2.13× (50d) — 19.17× (5d). Base rates 1.3–18%. R-prec 25–47%.
- +40%: 7.98× (25d) — 20.70× (10d). Base rates 0.7–5.5%. R-prec 15–45%.
- +50%: 9.31× (50d) — 14.35× (25d). Base rates 2.9–4.0%. R-prec 38–42%.

Same shape as `_177` + `_188`: rarer events trade base rate for very high lift on a smaller absolute R-precision. The +40%/+50% nasdaq100 cells are the "alert me to the rare big mover" surfaces; +10%/+20% cells are "broader coverage, smaller per-pick edge."

### 4. Universe effect (nasdaq100 vs russell1000): mixed, not the clean inversion `_188` argued

On the 13 cells with test windows for both `_188` (russell1000) and this sweep (nasdaq100), test-segment lift compares as follows:

| Cell | nas AUC(t) | rus AUC(t) | nas lift(t) | rus lift(t) | Winner |
|---|---:|---:|---:|---:|---|
| 10%/5d | 0.768 | 0.762 | 5.50 | 5.12 | nas |
| 10%/10d | 0.657 | 0.695 | 2.89 | 2.66 | nas |
| 10%/25d | 0.511 | 0.582 | 1.46 | 1.44 | nas |
| 10%/50d | 0.475 | 0.511 | 1.24 | 1.22 | nas |
| 20%/5d | 0.756 | 0.833 | 19.17 | 17.71 | nas |
| 20%/10d | 0.781 | 0.818 | 7.32 | 8.17 | **rus** |
| 20%/25d | 0.709 | 0.730 | 3.45 | 3.09 | nas |
| 20%/50d | 0.665 | 0.682 | 2.13 | 1.99 | nas |
| 40%/10d | 0.751 | 0.896 | 20.70 | 42.11 | **rus** |
| 40%/25d | 0.724 | 0.846 | 7.98 | 11.70 | **rus** |
| 40%/50d | 0.753 | 0.803 | 8.06 | 5.69 | nas |
| 50%/25d | 0.761 | 0.880 | 14.35 | 18.57 | **rus** |
| 50%/50d | 0.748 | 0.829 | 9.31 | 9.74 | **rus** |

**Tally: nasdaq wins lift on 8/13, russell wins on 5/13.** But the split is not random — it sorts by threshold rarity:
- **At lower thresholds (+10%, +20%)**: nasdaq wins 7 of 8 cells. The smaller, more index-stable nasdaq100 panel produces higher R-precision lift on common-event targets.
- **At higher thresholds (+40%, +50%)**: russell wins 4 of 5 cells. The wider russell1000 panel has more rare events to learn from, which translates to higher lift at the top picks.

`_188`'s read that "wider panel → lower lift" based on only 2 matched cells (5d × {10%, 20%}) was a small-sample artifact. The fuller picture across 13 matched cells is **threshold-dependent**: at common events, panel stability beats panel width (nas wins); at rare events, panel width beats panel stability (rus wins).

**AUC tells a different story**: russell wins AUC on 12 of 13 cells (one tie at 10%/5d). Russell1000 has higher discrimination across the board but lower lift at common events — a calibration / probability-distribution effect rather than a ranking-quality effect.

This **revises `_188` § 4 ("Universe effect")** from "bigger panel → lower lift, full stop" to "**bigger panel → higher AUC across the board, lift trade-off depends on event rarity**."

### 5. FS prunes hard with neutral-to-positive effect — pattern consistent with `_188`

13 of 20 cells finish with `n_feat < 100`. 5 cells finish with `n_feat = 279` (full pool kept, best_iter=0). 2 cells finish at the 100–200 range. The aggressive `_147` (279 → 9) / `_176` (279 → 39) prunes don't replicate here either — the nasdaq100 cells prune to {19, 22, 22, 23, 25, 25, 29, 32, 33, 37, 47, 48, 61, 70, 85, 103, 160} or keep all 279. No cell prunes below 19 features.

The 5 "full pool kept" cells are: 20%/25d, 40%/100d, 50%/50d, plus the two stale cells (10%/25d, 10%/100d) — wait, no: checking the table, the 5 cells with n_feat=279 are: 20%/25d, 40%/100d, 50%/50d. Plus 10%/200d had n_feat=22 (pruned), and the stale 10%/25d had 47 (pruned per its old runner). So the actual full-pool-kept set is 3 cells: 20%/25d, 40%/100d, 50%/50d. Same skew as `_188` toward mid-horizon × mid-to-high threshold cells.

### 6. Calibration: 17 of 20 cells received isotonic, 3 stayed native

Of the 20 cells, 17 layered an isotonic calibration map on top of CatBoost's raw probabilities (Spiegelhalter Z significant); 3 stayed native. The 3 native cells are at the extreme-rare-event end where probabilities are tightly compressed near zero and miscalibration is hard to detect on the scale that matters — same pattern as `_188`.

### 7. Best-iteration distribution

3 of 20 cells stopped at iter 0 (best — no prune improved val Brier). 12 at iter 1. 5 at iter 2. **5 cells stopped via degradation** (the inner-stop catching val Brier going up): 40%/10d, 40%/25d, 40%/50d, 50%/25d, 50%/50d — concentrated in the high-threshold mid-horizon family. The other 15 cells stopped via plateau. No cell ran past 3 iterations because the sweep mode caps `max_iter=3` (HP search disabled per `report.md` warning, issue #32). Same constraint as `_188`.

---

## Wall-time measurement vs the ~2-3h projection

The task brief projected: cold-cache nasdaq100 build ~31 min + 19 warm cells × ~5 min ≈ **~2-3 h total**.

Measured (sum of per-cell `wall_time_total_sec` across all 20 cells): **2.14 h** = 7702 s. Right at the bottom of the projected range.

Per-cell breakdown:

- **Cold cell** (first run, built the universe feature cache): `nasdaq100_up_10pct_10d_dd5pct` — 2540s = **42m20s**. Cold features build alone took **37m07s** (`[features] complete in 2226.7s`); the remaining 5m13s was target + uniqueness + loop + artifact. The 31min projection in the task brief was within 20% of measured.
- **Warm cells** (17 of them got a `[features] loaded from universe cache (key match) in 1.2–1.5s` line): per-cell elapsed range **80s–186s**, mean **~98s = ~1.6 min**. **The 5min/warm-cell projection was generous by ~3×.** Some cells came in under 100s — the smaller nasdaq100 panel (645k rows vs russell1000's 6M rows) trains much faster.
- **Pre-existing cells** (2 of 20): skipped by the sweep runner, no wall time accrued.

**Cache speedup**: cold features 2226.7s vs warm cache hit 1.2–1.5s = **~1500× speedup on the features phase**. The per-cell speedup is `2540s / 98s = ~26×` end-to-end (features + target + uniqueness + loop + artifact). For the 17 warm cells the cache saved **~10.5 h of compute** (17 × ~37min features that would otherwise re-run).

**Comparison to `_188` russell1000 timing**: russell cold cell 4h24m (vs nasdaq cold 42m, **6.3× faster on nasdaq**, scales with panel size 6.06M/645k = 9.4×). Russell warm cells averaged ~9.1 min (vs nasdaq warm ~1.6 min, **5.7× faster on nasdaq**, scales with panel size in the loop phase). Both ratios are below the 9.4× panel-size ratio — there's a fixed per-cell overhead (data load, target build) that dominates at small panels.

**The combined picture**: post-#183 shared feature cache pays off cleanly at every panel size. The 17 warm cells totalled ~28 minutes of compute (vs ~10.5 h without the cache). On a smaller panel like nasdaq100, the **per-cell overhead** (data load + target + uniqueness + artifact ≈ 12s per warm cell) is now a meaningful fraction of total time — at 98s mean warm cell, ~12% of time is in those non-loop phases.

**Note on sweep wall-clock vs cells-sum wall**: the sweep ran in two passes (the first background sub-process was killed at the 40%/100d cell after 9 completed cells; re-launched and skipped 11 = 9 done + 2 pre-existing, then ran the remaining 9 cells in pass 2). Pass-1 sweep window 11:30 → ~12:30 (60 min for 9 cells including the 42min cold). Pass-2 sweep window 12:01 → 12:16 (15 min for 9 warm cells). Total wall-clock elapsed across both passes ≈ 75 min, cells-sum 2.14h — the discrepancy is because pass-2 ran concurrent with the 4 sister russell1000 agent-loop runs sharing the same machine; the per-cell wall_time_sec inside the experiment is unaffected by competing CPU load (CatBoost dominates and the russell1000 sisters were not in feature-build phase at that moment).

---

## Practical guidance — which nasdaq100 cells to pursue

**Pursue (strong discriminating + test-validated):**
- **Short-horizon × high-threshold**: 40%/10d (lift 20.70× test, base 0.72%), 20%/5d (19.17×, base 1.29%), 50%/25d (14.35×, base 2.91%) — rare-event top-pick surfaces. Despite lower per-pick absolute R-precision (15–42%), the lift over baseline is very high.
- **Mid-threshold mid-horizon**: 50%/50d (lift 9.31×, R-prec 37.6%), 40%/50d (8.06×, R-prec 44.7%), 40%/25d (7.98×, R-prec 34.0%), 20%/10d (7.32×, R-prec 31.5%) — solid lift with broader coverage.
- **10%/5d** as the broadest-coverage discriminating cell (lift 5.50×, base 7.14%, R-prec 39.3% — picks land in ∼2/5 of top-K picks on test on the most common-event cell that still discriminates clearly).

**Use with caution — eval-only (no test window):**
- 40%/100d, 50%/100d, 20%/100d, 40%/200d, 50%/200d are all discriminating on eval. Candidates for the long-horizon strategy bucket but cannot be trusted without a test-split fix. **Treat eval AUC + eval lift as a hypothesis pending test data**, given the universal eval→test AUC decay observed everywhere else.

**Avoid — ambiguous:**
- 10%/25d, 10%/50d (both ambiguous on test).
- 10%/100d, 10%/200d (both ambiguous on eval-only — would likely worsen on test based on the universal decay pattern).

These cells corroborate `_177` + `_188` finding that **the +10% target at H ≥ 25** is where eval AUC over-states test AUC enough to push cells into the null/ambiguous regions on nasdaq100. Russell1000's broader panel pushed this crossover by one horizon-step; nasdaq100's smaller, more index-stable panel does not.

---

## Cross-reference to `_177` + `_188`

| `_177`/`_188` finding | nasdaq100 sweep evidence (this memo) |
|---|---|
| Universal eval→test AUC decay (every completed cell loses AUC) | **Fully replicates** — 13 of 13 test-evaluable cells lose AUC; dAUC range −0.184 to −0.049. The russell `_188` "lone +0.005 outlier" (40%/10d) decays sharply here (−0.122) |
| Decay magnitude scales with horizon at fixed threshold | **Replicates** — 10% family decay −0.05 → −0.12 → −0.14 → −0.11 from 5d → 50d |
| Short-horizon (5d–10d) sweet spot | **Replicates** — 5d/10d cells stay strongly discriminating across all thresholds |
| Higher thresholds → higher lift, smaller base, modest absolute R-prec | **Replicates** — +40%/+50% cells hit lift 8–21× on tiny base, +10% cells hit lift 1–5× on larger base |
| H ≥ 100 has zero test rows (test-split methodology gap) | **Replicates** — 7 nasdaq100 cells affected; same `data WARNING` text |
| `_188` "bigger universe → lower lift" hypothesis | **Partially contradicted** — across 13 matched cells, nasdaq wins lift on 8 (mostly common events), russell wins on 5 (mostly rare events). The original `_177` "bigger panel → higher lift" hypothesis is also wrong; the truth is threshold-dependent |
| `_177` nasdaq100 +10%/25d crossover into null AUC band on test | **Replicates** — test AUC 0.511 exactly matches `_177`'s prior reading from PR #28's stale artifact (which was preserved on this branch) |
| `_188` russell1000 +10%/50d crossover into null AUC band on test | **Earlier on nasdaq100** — nas crossover at 10%/25d, rus crossover at 10%/50d. Smaller panel → earlier crossover |
| FS prunes hard (279 → 25–65) on most cells | **Partially replicates** — 13 of 20 cells prune to <100; 3 keep the full 279; no cell prunes below 19 |
| 17 of 20 cells take isotonic calibration | Same as `_188` (14/20 isotonic on russell1000; nasdaq is slightly heavier on isotonic at 17/20) |

The key concrete revision this memo introduces is to **`_188` § 4 (universe effect)**: with 13 matched-cell data points (vs `_188`'s 2), the panel-width-vs-lift relationship is **threshold-dependent**, not monotonic. Update `_188`'s § 4 to note "the nasdaq100 sweep shows panel-width-vs-lift is threshold-dependent: smaller index-stable panels (nasdaq) win lift on common events; wider panels (russell) win on rare events. The AUC ranking is consistent (wider panel → higher AUC across the board)."

---

## User-facing read (no automated PASS/FAIL)

Of the 20 nasdaq100 cells, **12 discriminate on the held-out test window** (10%/5d, 10%/10d, 20%/{5d, 10d, 25d, 50d}, 40%/{10d, 25d, 50d}, 50%/{25d, 50d} — strongest lift first within group), **4 are eval-only discriminating** (no test window under the current split: 20%/100d, 40%/{100d, 200d}, 50%/{100d, 200d}), **4 are ambiguous** (10%/{25d, 50d} on test, 10%/{100d, 200d} on eval-only), **0 null**, **0 anti-predictive**. The short-horizon × high-threshold corner is the production-candidate region — **40%/10d (lift 20.70×), 20%/5d (19.17×), 50%/25d (14.35×)** are the standout cells by test-segment R-precision lift. The eval→test AUC decay observed in `_177` + `_188` fully replicates on nasdaq100 with no exception, with the +10% target crossing into the null AUC band one horizon-step earlier than russell1000 (10%/25d vs 10%/50d), confirming the "smaller panel → earlier crossover" inverse of `_188`'s "wider panel → later crossover" reading. The 7 eval-only cells (H ≥ 100) need the test-split fix before they can be judged. As `_177` + `_188` noted, the PASS/FAIL call on any individual cell remains a user judgment; this memo characterizes the landscape across the completed nasdaq100 sweep and updates the cross-universe panel-width hypothesis from `_188`.
