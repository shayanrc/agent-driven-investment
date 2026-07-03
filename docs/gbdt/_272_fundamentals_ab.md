# _272 — F18 fundamentals features: matched-HP A/B vs the sp500 champions

**Question.** Do point-in-time valuation ratios (the new `valuation` panel — PE/PS/
P-FCF yields + cross-sectional ranks + TTM growth) improve the two **deployed sp500
champions**? This is the downstream test of the standing hypothesis that fundamentals
are the next edge for the GBDT models.

**Design — clean matched-HP single-fit A/B** (the `_260` macro sign-flip lesson).
For each champion cell, two specs identical except the feature token:
`base_v2` (`candidates: all` = F1–F16) vs `fund` (`candidates: all_fundamentals`
= F1–F16 **+ F18**). Both `split.mode: date_aligned`, `train_start: 2019-01-01`,
`--snapshot-end 2026-07-02`, `backend.library: xgboost`, `callback_mode: default`,
**`max_iterations: 1` (single fit)**, seed 42. So the ONLY difference is the +9
F18 columns — any test-metric delta isolates the fundamentals contribution (no
per-arm HP divergence). F18 is opt-in: the default `all` token stays byte-identical
(base_v2 emitted 279 cols, fund 288 = 279 + 9; `all`-path columns are
`assert_frame_equal`-identical, see `tests/gbdt/test_features_fundamentals.py`).

**Test window (date_aligned): 2024-07-26 → 2024-12-16** (48,600 test rows/cell).
Note this is the *conservative* 2024-H2 date-aligned window — **the same window
style where the F17 macro features failed to replicate** (`_264`), not the trailing
window where macro's apparent win came from.

## Results — test R-Precision@K (raw; base_rate for reference, no lift column)

**+50%/50d champion** (test base_rate 0.0097):

| K | base R-p@K | fund R-p@K | Δ |
|---:|---:|---:|---:|
| 1 | 0.1099 | 0.1209 | +0.0110 |
| 3 | 0.1062 | 0.0952 | −0.0110 |
| 5 | 0.1454 | 0.1370 | −0.0084 |
| 10 | 0.2182 | 0.3056 | **+0.0874** |
| 20 | 0.3912 | 0.4832 | **+0.0921** |

AUC 0.9066 → **0.9131**; test Brier 0.00978 → **0.00943**; eval Brier 0.00927 → 0.00956.

**+20%/25d champion** (test base_rate 0.0402):

| K | base R-p@K | fund R-p@K | Δ |
|---:|---:|---:|---:|
| 1 | 0.1000 | 0.2500 | **+0.1500** |
| 3 | 0.1367 | 0.2200 | **+0.0833** |
| 5 | 0.1660 | 0.2080 | +0.0420 |
| 10 | 0.1879 | 0.2113 | +0.0234 |
| 20 | 0.2570 | 0.2263 | −0.0307 |

AUC 0.7725 → **0.7925**; test Brier 0.03758 → **0.03722**; eval Brier 0.04675 → 0.04682.

## Feature usage

All 9 F18 features are split on in both models (single-fit, no FS pruning),
contributing **~3.0%** (50/50d) and **~2.3%** (20/25d) of total tree gain, ranked
mid-pack (#39–#101). Most-used, consistent across both cells: `fund_earnings_yield`,
`fund_earnings_yield_xs_rank`, `fund_sales_yield_xs_rank`, `fund_fcf_yield(_xs_rank)`,
`fund_rev_ttm_yoy_xs_rank`. So the signal is the **yields + their cross-sectional
ranks + revenue growth** — not a single dominant feature.

## Reading

Fundamentals **help, on both champions, on the conservative window**:
- **AUC up on both** (+0.007, +0.020) and **test Brier down on both** — the whole
  probability surface improves, not just the tail. (Contrast macro `_264`, where the
  edge did not survive this window.)
- **Top-of-book R-p@K net-positive but pattern-shifted per cell:** the +50%/50d cell
  gains at K=10/20 (+40%/+24% relative) while dipping at K=3/5; the +20%/25d cell
  gains strongly at the very top (K=1/3/5) while dipping at K=20. Both directions are
  useful for position-sizing, but the inconsistency across K + the small day counts
  (Q_days ≈ 91–100, so R-p@1 rests on ~11 hits) mean the *shape* is noisy even though
  the *aggregate* (AUC/Brier) is cleanly better.
- eval-segment Brier is ~flat (−/+0.0003) — the gain concentrates in the test window.

**This is a materially stronger result than the F17 macro attempt** (which improved
neither AUC nor Brier and evaporated on the date-aligned window). But it is still
**one window**. The `_264` lesson is precisely that a single-window win can be
window-specific.

## Decision

- **Ship the F18 infra + this memo regardless of outcome** (documented positive
  finding), per module convention. F18 stays **opt-in** (`all` byte-identical); no
  existing model changes.
- **Do NOT auto-promote / do NOT wire fundamentals into `/daily-predictions` yet.**
  The gate before a champion swap is a **second independent date-aligned window**
  (e.g. an earlier test period, or the trailing window) confirming AUC + Brier + top-K
  replicate. That re-run is the recommended next step (`V1.7_TBD`). A champion swap is
  a separate, human decision (`_019`).

Data: `results/gbdt/data/_272_fundamentals_ab_data.json`. Registry rows added to
`results/gbdt/data/r_precision_at_k.csv` (4 experiments). Plan:
`docs/gbdt/V1.7_fundamentals_features_plan.md`.
