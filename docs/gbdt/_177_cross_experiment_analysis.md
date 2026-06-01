# Task #177 — Cross-experiment analysis of completed gbdt cells

> **Methodology note (2026-06-01)**: Numbers in this memo's body use the legacy "weighted R-precision" metric (per-day variable K = R(d), micro-aggregated). The project headline metric was renamed 2026-06-01 to **R-Precision@K** (per-day fixed K, macro-aggregated via `(1/Q)·Σ r_q/min(K,R_q)`). See the "R-Precision@K (current methodology)" section at the bottom of this memo for the cells in this memo recomputed under the new metric, plus `.claude/memories/project-r-precision-methodology.md` for the full definition + relationship.

**Date**: 2026-05-29.
**Branch**: `gbdt-sweep-cross-experiment-analysis`.
**Data**: `results/gbdt/data/_177_cross_experiment_analysis_data.json` (machine-readable master table + classifications).

> ## ⚠️ PARTIAL-SWEEP SNAPSHOT — read this first
>
> The 57-cell US sweep (#107) is **mid-run** and infeasible to finish soon (russell1000 cells grind at ~5 h each). This memo is a **snapshot over what is complete as of 2026-05-29**: **12 of 57 sweep cells (~21%)** that have a finished `metrics.json`, plus **4 standalone cells** for cross-market context. The cell currently building (`russell1000_up_20pct_5d_dd10pct`, no `metrics.json` yet) is **excluded**, as is the old `nifty50_up_10pct_20d_pilot` (v1 merge-gate pilot, not a sweep cell). Every pattern below is **provisional**: coverage is heavily weighted toward short-horizon nasdaq/sp500 `+10%`/`+20%` cells; russell1000 has **1** cell, the longest horizon (100d) has **1** cell with **no test window**. Treat universe-effect and long-horizon conclusions as `n=1` data points, not controlled ablations.

## What's covered

**12 completed US sweep cells** (read read-only from the live `wt-us-sweep` worktree; the sweep process was not touched):
nasdaq100 `+10%`×{5d, 10d, 25d, 100d}, `+20%`×{5d, 10d}, `+40%`×10d; sp500 `+10%`×{5d, 10d}, `+20%`×{5d, 10d}; russell1000 `+10%`×5d.

**4 standalone context cells** (committed data on main): `_176` nifty500 `+30%`/50d, and the `_138` H=25 cross-market corpus (`_174` nifty50, nifty100-D, sp500-B).

## How to read the metrics

- **Weighted R-precision** (per-day variable-K: `sum(positives_caught) / sum(R(d))`, R(d) = positives that day) is the **standard cross-cell metric** — panel-invariant, baseline = base rate, so it compares cleanly across markets and universes. Computed via `scripts/gbdt/compute_r_precision.py` on each cell's `predictions/{eval,test}.csv`.
- **ROC-AUC** reported eval + test; discrimination, not gated.
- **Lift = R-prec / base_rate** is discussed in **prose only**, never as a table column (CLAUDE.md reporting convention). The `base(e)`/`base(t)` columns are present so you can compute lift on demand and keep the underlying hit-rate scale.
- Per-day P@k everywhere uses denominator `min(R(d), k)` (achievable positives), not picks-made.

**Compound signal/null rule** (CLAUDE.md): AUC ∈ [0.45, 0.55] **AND** R-prec lift < 1.2× → null; AUC ∈ [0.45, 0.55] **AND** lift > 1.5× → top-tail signal hidden by AUC; lift ∈ [1.2, 1.5] inside the null AUC band → ambiguous; AUC > 0.55 → discriminating.

---

## Master table (12 sweep cells)

Raw metric values. `(e)` = eval segment, `(t)` = test segment. `n_feat` = features kept after FS.

| Universe | Thr% | H(d) | DD% | base(e) | base(t) | AUC(e) | AUC(t) | Rprec(e) | Rprec(t) | n_feat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nasdaq100 | 10 | 5 | 5 | 0.0435 | 0.0756 | 0.8218 | 0.7776 | 0.3675 | 0.3734 | 279 |
| nasdaq100 | 10 | 10 | 5 | 0.1092 | 0.1520 | 0.7724 | 0.6452 | 0.4507 | 0.4056 | 279 |
| nasdaq100 | 10 | 25 | 5 | 0.2494 | 0.2733 | 0.6549 | 0.5111 | 0.5121 | 0.3994 | 47 |
| nasdaq100 | 10 | 100 | 5 | 0.3982 | — | 0.4880 | — | 0.4881 | — | 33 |
| nasdaq100 | 20 | 5 | 10 | 0.0057 | 0.0273 | 0.8662 | 0.7946 | 0.1333 | 0.2124 | 279 |
| nasdaq100 | 20 | 10 | 10 | 0.0193 | 0.0511 | 0.8588 | 0.7902 | 0.2360 | 0.3539 | 25 |
| nasdaq100 | 40 | 10 | 20 | 0.0022 | 0.0198 | 0.8734 | 0.7512 | 0.1750 | 0.1500 | 32 |
| sp500 | 10 | 5 | 5 | 0.0291 | 0.0432 | 0.8388 | 0.7815 | 0.3386 | 0.2905 | 129 |
| sp500 | 10 | 10 | 5 | 0.0812 | 0.1130 | 0.8005 | 0.7146 | 0.4034 | 0.3611 | 62 |
| sp500 | 20 | 5 | 10 | 0.0033 | 0.0070 | 0.8740 | 0.8515 | 0.1709 | 0.1655 | 44 |
| sp500 | 20 | 10 | 10 | 0.0116 | 0.0228 | 0.8799 | 0.8351 | 0.2363 | 0.2753 | 279 |
| russell1000 | 10 | 5 | 5 | 0.0347 | 0.0498 | 0.8244 | 0.7624 | 0.3279 | 0.2547 | 48 |

### Standalone context cells (cross-market)

| Cell | Universe | Thr% | H(d) | DD% | base(t) | AUC(t) | Rprec(t) | source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `_176` | nifty500 | 30 | 50 | 15 | 0.0569 | 0.7155 | 0.1805 | `_176` data JSON |
| `_138`-C/`_174` | nifty50 | 10 | 25 | 5 | 0.1960 | 0.7330 | 0.4160 | `_138`, `_174` |
| `_138`-D | nifty100 | 10 | 25 | 5 | 0.1880 | 0.6890 | 0.4190 | `_138` |
| `_138`-B | sp500 | 10 | 25 | 5 | 0.2640 | 0.5836 | 0.4070 | `_138` |

---

## Signal/null classification (compound rule)

Classified on the **test** segment where a test window exists, else eval:

| Class | Count | Cells |
|---|---:|---|
| discriminating (AUC > 0.55) | 10 | all nasdaq `+10%`/{5d,10d}, all `+20%`, `+40%`; all sp500; russell1000 |
| ambiguous null-band (AUC ∈ [0.45,0.55], lift 1.2–1.5×) | 1 | nasdaq `+10%`/25d (test AUC 0.511, lift 1.46×) |
| no test window (H ≥ 100 ate the test split) | 1 | nasdaq `+10%`/100d |

No cell landed in the **clean-null** (lift < 1.2×) or **hidden-top-tail** (lift > 1.5× in the null band) buckets on test. On **eval**, 11/12 are discriminating; only nasdaq `+10%`/100d is null on eval (AUC 0.488, lift 1.22×) — and that cell has no test window to corroborate. **Bottom line: the completed cells are overwhelmingly real-signal**, with the lone borderline being the longest evaluable horizon at the lowest threshold.

---

## Patterns

### 1. The eval→test AUC decay is **systemic, not cell-specific** — and horizon-scaled

Every single completed cell loses AUC from eval to test. `dAUC` (test − eval) is **always negative**, ranging −0.022 to −0.144. This is not the #185-style "one cell collapses" story — it's a uniform downward shift, consistent with prevalence drift + a harder, more recent test window across the whole sweep.

What's cell-*dependent* is the **magnitude**, and it scales with horizon inside a fixed threshold. The nasdaq `+10%` family:

| H | AUC eval → test | test class |
|---:|---|---|
| 5d | 0.822 → 0.778 (−0.044) | discriminating |
| 10d | 0.772 → 0.645 (−0.127) | discriminating |
| 25d | 0.655 → 0.511 (−0.144) | **falls into null band** |
| 100d | 0.488 → (no test) | null on eval |

The longer the horizon, the more eval AUC over-states test AUC, until at 25d the cell drops into the [0.45, 0.55] null band on test (R-prec lift only 1.46×, eval was 2.03×). This matches the #185 nifty50 H=25 manual-tuning finding *qualitatively* — but with a key contrast: nifty50 H=25 (`_147`/`_174`) actually had test AUC (0.733) **above** eval (0.646), because the nifty50 test window happened to have *higher* prevalence than eval. So the decay direction depends on the prevalence trajectory of the specific test window, but the **eval→test gap existing at all is universal**. The `+20%` cells decay far less (−0.022 to −0.072) and stay strongly discriminating on test — rarer events appear more stable here, though that may be small-sample noise on the tiny positive counts.

**Practical consequence**: do not trust eval AUC alone for any cell, and especially not for H ≥ 25. Judge on test R-precision.

### 2. Horizon effect: short is clean, long decays

R-prec lift is highest at 5d and declines monotonically with horizon within each threshold family. nasdaq `+10%`: 5d → 6.9× test, 10d → 2.7×, 25d → 1.5×, 100d → null/no-test. The 5d–10d band is the sweet spot: strong discrimination *and* the decay is mild enough that test signal survives.

### 3. Threshold effect: rarer events → higher lift, smaller base, modest absolute R-prec

Higher thresholds buy enormous lift on a shrinking base rate. sp500 `+20%`/5d: base 0.7% test, lift **23.7×** test. nasdaq `+40%`/10d: base 2.0% test, lift 7.6×. But absolute R-precision is modest (0.15–0.21) — the top picks concentrate a *rare* event very effectively, but most days have few or no positives. `+10%` cells run lower lift (5–11× at 5d) on a larger base (3–8%) and higher absolute R-precision (~0.30–0.45). For a strategy, `+20%`/`+40%` cells are "rare-but-clean top picks"; `+10%` cells are "more frequent, broader hit rate."

### 4. Universe effect: bigger panel → stronger lift (n is thin)

sp500 (~486 names) shows the strongest lift at matched thresholds — e.g. `+20%`/5d at 23.7× test vs nasdaq `+20%`/5d at 7.8×. The mechanism is the documented one: more positives per day → a deeper cross-section for the F14 rank/z-score features. russell1000 (1 cell) at `+10%`/5d is comparable to nasdaq (lift 5.1× test). The nifty500 `_176` standalone (`+30%`/50d, lift 3.2× test, exceeding nifty50 `_147` H=25's ~2.1×) corroborates "broad universe helps" cross-market. **But this is `n=1` per universe at most thresholds** — not a controlled universe-size ablation.

### 5. The 100d cell has **zero test rows** — a methodology limitation

`nasdaq100_up_10pct_100d_dd5pct` reports `n_rows_test = 0`. The H=100 horizon consumes the entire 100-row-per-ticker test segment under the 800/400/200/100 split (you need ≥ H rows after the test-start to label a single test origin). So the cell has no test metrics at all and its eval AUC (0.488) is null. **Any H ≥ 50 cell needs a larger per-ticker test allocation than the standard split provides** before it is evaluable — flag for the sweep design.

### 6. FS prunes hard with neutral-to-positive effect

Final feature counts cluster at 25–65 on most cells (full 279 retained on only 4 of 12). The aggressive `_176` nifty500 prune (279→39) and `_92` nifty500 (279→39, per task brief, 3–4× R-prec) fit the same pattern: the signal is low-complexity and concentrated in a small feature set (the `_147` finding that 191/279 features had < 0.01 importance). Pruning to ~40 features is the norm, not the exception.

---

## Practical guidance — which regions to pursue

**Pursue (reliably discriminating, signal survives to test):**
- **Short-horizon (5d–10d) up-move cells on large US universes.** sp500 ≥ russell1000 ≈ nasdaq by lift. These are the production-candidate cells.
- **Higher-threshold cells (`+20%`, `+40%`) at short horizons** — they trade base rate for very high lift on top picks and stay discriminating on test. Good for "alert me to the rare big mover" use cases; weak for broad coverage.

**Caution / null:**
- **H ≥ 25 at `+10%`** — eval AUC looks decent but decays into the null band on the recent test window. Don't ship on eval AUC; require strong test R-precision.
- **H ≥ 50 / H = 100** — not evaluable under the current split (no test window). Needs a test-split redesign first; the 100d/`+10%` cell is effectively unusable as-is.

**What a data scientist should expect going in:**
1. **Eval AUC over-states test AUC by 0.02–0.14**, worse the longer the horizon. Budget for it.
2. **Ranking (R-precision) survives prevalence drift even when calibrated-probability Brier does not.** Several cells beat baseline on AUC/R-prec while Brier underperforms the base-rate constant on test (the `_176`/`_147` calibration-on-drifted-prevalence artifact). Judge cells on **R-precision under drift, not Brier**.
3. **FS prunes hard** (279 → ~25–65) on most cells with neutral-to-positive effect; the signal is low-complexity.
4. **The signal lives in the prediction-extreme top picks** — on US panels P@k *descends* with k (top picks cleanest); on staggered NSE panels P@k *ascends* with k (an artifact of R(d) < k days, not weak top picks — see `_138` erratum).

---

## User-facing read (no automated PASS/FAIL)

Of the 12 completed cells, **10 are clearly discriminating on the held-out test window, 1 is borderline (nasdaq +10%/25d), and 1 is unevaluable on test (nasdaq +10%/100d)**. The short-horizon US sweep is producing real, tradable top-tail signal; the open risks are (a) the universal eval→test AUC decay — which is a generalization-gap reality to price in, not a bug — and (b) the long-horizon test-split gap that needs fixing before H ≥ 50 cells can be judged. This is a **snapshot of ~21% of the sweep**; whether the russell1000 and remaining-threshold cells hold the same pattern is the open question the full sweep will answer. The PASS/FAIL call on any individual cell remains yours to read from its `report.md` — this memo characterizes the landscape, it does not gate.

## R-Precision@K (current methodology — added 2026-06-01)

Per `.claude/memories/project-r-precision-methodology.md`, R-Precision@K is the post-2026-06-01 headline cross-cell metric for gbdt — defined as `R-Precision@K = (1/Q) · Σ_q r_q / min(K, R_q)` over the Q days where R_q > 0 (R_q = positives on day q; r_q = positives caught in top-K picks on day q; macro-averaged, equal weight per day; K fixed). Recomputed from each cell's `predictions/test.csv`:

| cell | rows | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|---|---|
| nasdaq100_up_10pct_5d_dd5pct | 8740 | 7.1% | 0.768 | 0.278 | 0.304 | 0.369 | 0.456 | 0.607 |
| nasdaq100_up_10pct_10d_dd5pct | 8280 | 15.2% | 0.657 | 0.479 | 0.488 | 0.470 | 0.462 | 0.536 |
| nasdaq100_up_10pct_25d_dd5pct | 6900 | 27.3% | 0.511 | 0.537 | 0.526 | 0.538 | 0.507 | 0.508 |
| nasdaq100_up_20pct_5d_dd10pct | 8740 | 1.3% | 0.756 | 0.156 | 0.226 | 0.343 | 0.473 | 0.589 |
| nasdaq100_up_20pct_10d_dd10pct | 8280 | 4.3% | 0.781 | 0.197 | 0.204 | 0.280 | 0.492 | 0.670 |
| nasdaq100_up_40pct_10d_dd20pct | 8280 | 0.7% | 0.751 | 0.030 | 0.111 | 0.195 | 0.488 | 0.539 |
| sp500_up_10pct_5d_dd5pct | 46170 | 4.3% | 0.781 | 0.284 | 0.309 | 0.339 | 0.336 | 0.333 |
| sp500_up_10pct_10d_dd5pct | 43740 | 11.3% | 0.711 | 0.489 | 0.474 | 0.464 | 0.433 | 0.413 |
| sp500_up_20pct_5d_dd10pct | 46170 | 0.6% | 0.846 | 0.220 | 0.140 | 0.161 | 0.291 | 0.412 |
| sp500_up_20pct_10d_dd10pct | 43740 | 2.3% | 0.835 | 0.382 | 0.260 | 0.252 | 0.313 | 0.397 |
| russell1000_up_10pct_5d_dd5pct | 84455 | 5.0% | 0.762 | 0.263 | 0.260 | 0.274 | 0.236 | 0.237 |
| nifty500_up_30pct_50d_dd15pct (standalone) | 18800 | 5.7% | 0.715 | 0.431 | 0.242 | 0.231 | 0.213 | 0.199 |
| nifty50_up_10pct_25d_dd5pct (standalone, `_138`-C/`_174`) | 3450 | 17.9% | 0.733 | 0.257 | 0.252 | 0.294 | 0.368 | 0.607 |
| sp500_up_10pct_25d_dd5pct (standalone, `_138`-B) | 36450 | 26.4% | 0.590 | 0.333 | 0.427 | 0.421 | 0.449 | 0.426 |
| nifty100_up_10pct_25d_dd5pct (standalone, `_138`-D) | 3525 | 18.8% | 0.689 | 0.225 | 0.268 | 0.347 | 0.400 | 0.509 |

The canonical CSV does not carry the `nasdaq100_up_10pct_100d_dd5pct` cell (the no-test-window 100d cell referenced in the body master table) — the cell completed feature build but the H=100 walk-forward ate the test split, leaving no `predictions/test.csv`. All other body-referenced cells, including `_138`-D nifty100 H=25, are present above.
