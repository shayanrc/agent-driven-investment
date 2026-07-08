# _281 — VWAP-deviation (F20) matched sweep: verdict

**Plan:** `docs/gbdt/V1.8_vwap_deviation_features_plan.md`. **Feature:** F20 VWAP-deviation
family (opt-in tokens `all_vwap`, `all_fundamentals_vwap`). **Question:** does a
volume-weighted-price-deviation feature add predictive skill over the technical baseline,
and does it stack on fundamentals (F18)?

## Verdict

**VWAP-deviation (F20) is NOT a robust predictor — not promoted.** The `vwap − base`
marginal is ≈ 0 on every universe measured, and the cells where it "wins" are **disjoint
across universes** — the signature of noise, not signal. Same fate as F17-macro (`_264`).

The sweep's one durable finding is a **byproduct**: **F18 fundamentals replicates** — the
`fund − base` top-book gain is near-identical on sp500 and nasdaq100 (the first
fundamentals evaluation ever run off sp500).

## Setup

Matched single-fit sweep (default HP, `max_iterations: 1`, date_aligned
`train_start 2019-01-01`, snapshot 2026-07-06 — `_279`-comparable), **xgboost-only**
(warm fits ~7 min vs catboost ~9.3 min; backend already settled by `_277`/`_278`). Four
feature arms differing ONLY in `features.candidates`, so per-cell deltas are clean
(`[[project-gbdt-macro-features-f17]]`): `all` (base), `all_vwap`, `all_fundamentals`
(F18), `all_fundamentals_vwap`. Universes: **sp500 (17 cells) + nasdaq100 (20 cells)
complete; russell1000 base+vwap complete, fund arms parked** (see Open TODO). sp500's F18
arm was reused byte-identically from `_279`'s `f18xgb` (same spec/window/seed). All metrics
are the **test** segment (2024-07-26 → 2024-12-16).

## 1. VWAP marginal (`vwap − base`) — a wash on every universe

Mean Δ (median; # cells positive):

| Universe | ΔR-p@3 | ΔR-p@1 | ΔAUC |
|---|--:|--:|--:|
| sp500 (17) | −0.025 (−.027; 7/17) | +0.007 (.000; 8/17) | −0.004 (6/17) |
| nasdaq100 (20) | +0.006 (+.005; 10/20) | +0.005 (−.012; 9/20) | −0.002 (9/20) |
| russell1000 (19) | +0.004 (+.005; 11/19) | +0.017 (12/19) | +0.001 (10/19) |

No consistent sign, all magnitudes within noise of zero. The decisive test is **whether
the winning cells replicate**. Top-4 `vwap − base` R-p@3 cells per universe:

- **sp500:** 20pct_50d, 20pct_25d, 20pct_5d, 40pct_50d
- **nasdaq100:** 50pct_25d, 10pct_5d, 20pct_100d, 20pct_50d
- **russell1000:** 40pct_25d, 50pct_200d, 40pct_100d, 10pct_100d

**No cell is in all three top-4** (only one weak two-way overlap: 20pct_50d in
sp500+nasdaq). A real structural edge on some (threshold, horizon) regime would light up
the *same* cells everywhere; instead each universe has its own random winners. And the
gains **barely move AUC** (e.g. sp500 20pct_50d: +.053 R-p@3 but −.008 AUC; russell
40pct_25d: +.080 R-p@3, +.002 AUC) — top-tail reshuffling, not better ranking. Both are
textbook noise.

## 2. F18 fundamentals replicates (`fund − base`) — the byproduct

| Universe | ΔR-p@3 | ΔR-p@1 | ΔAUC |
|---|--:|--:|--:|
| sp500 (17) | **+0.019 (12/17)** | +0.039 (10/17) | −0.006 (5/17) |
| nasdaq100 (20) | **+0.020 (12/20)** | +0.014 (10/20) | +0.001 (11/20) |

Near-identical top-book gain on both universes (R-p@3 ≈ +0.02, 12/17 *and* 12/20). This is
the **first fundamentals read off sp500**, and it confirms the sp500 F18 top-book benefit
cross-universe — the opposite of VWAP. (russell1000's `fund` confirmation is the parked
TODO.) Consistent with the standing F18 read: helps the top book, hurts nothing, modest.

## 3. VWAP on top of fundamentals (`fundvwap − fund`) — weak / mixed

sp500 R-p@3 +0.003 (7/17); nasdaq100 +0.014 (10/20). R-p@1 weakly positive on both
(~+0.02, 10/17 & 10/20) but the per-cell winners don't replicate (sp500's biggest stack,
50pct_200d +.143, is absent from nasdaq's leaders). No robust stacking.

## 4. Follow-up — finetune the strong-*absolute* cells

Independent of the (null) marginal, a few nasdaq cells are strong **absolute** models
clearing R-p@3 > 0.5 AND AUC > 0.5, worth pushing past the single-fit HP envelope
(`docs/gbdt/_282`, tasks 17 + 20):

- **vwap arm:** nasdaq 50pct_25d (0.716/0.937), 20pct_50d (0.513/0.724) *(user-selected)*
- **fund arm:** nasdaq 50pct_25d, 50pct_50d, 40pct_200d, 40pct_25d, 20pct_50d
- **fund+vwap arm:** nasdaq 50pct_25d, 50pct_50d, 40pct_200d, 40pct_25d
- sp500: none clear the bar; russell1000: pending (its fund arms are parked).

## Open TODO — russell1000 fund arms

russell1000 `base` + `vwap` are complete (and already fold into §1 above — same
noise pattern). Its `fundxgb` (6/20) + `fundvwapxgb` (0/20) arms were **parked** (user
deprioritization 2026-07-08) to move to the finetunes. **Resume:**
`bash scripts/gbdt/run_vwap_sweep_uni.sh russell1000 2026-07-06` (skip-if-done preserves
the 46 completed fits). On completion: (a) third-universe `fund − base` F18 confirmation,
(b) apply the R-p@3>0.5 & AUC>0.5 filter for any russell finetune qualifiers. Tracked as
task 19.

## Status

**F20 NOT promoted; no champion or `/daily-predictions` change.** F20 stays an opt-in
token, byte-identical to `all` when unused. Artifacts: 68 sp500 + 80 nasdaq + 46 russell
`metrics.json` under `results/gbdt/experiments/{universe}_up_*_{base,vwap,fund,fundvwap}xgb`;
tooling `scripts/gbdt/{gen_vwap_sweep_specs.py, run_vwap_sweep.sh, run_vwap_sweep_uni.sh,
run_vwap_crossuni.sh}`. Registry rows to be appended to
`results/gbdt/data/r_precision_at_k.csv`. Related: `_264` (macro non-replication),
`_278`/`_279`/`_280` (F18/F19 arc), `_282` (VWAP/fund finetunes).
