# _285 — nifty500 F18-IN fundamentals sweep (regime-corrected, calendar2)

**One-liner:** F18-IN (Indian point-in-time valuation ratios) is a **wash-to-slightly-negative** signal on nifty500 overall — matching the US F18 verdict — **but** the effect is horizon-shaped, with a clean, coherent, powered win localized to the **100-day horizon** (20/100d + 30/100d) and real top-of-book damage at 50d. Single-window result; needs second-window replication before any adoption.

## Setup

Matched base-vs-fund A/B across the 20-cell `(threshold, horizon)` lattice, **both arms carrying F21 calendar2** so F18 is isolated:

- **base arm** `all_calendar2` — technical (F1–F16) + F21 moq/qoy seasonality
- **fund arm** `all_fundamentals_calendar2` — the same **+ F18** (10 India-appropriate valuation columns: earnings/sales yield + rev-TTM-YoY, each × {level, xs_rank, xs_zscore} + earnings_yield_chg_63; no fcf — India files cash flow half-yearly)

Identical HP (xgboost, default, single fit, `max_iterations: 1`) — only `features.candidates` differs, so each cell's fund−base delta is a clean read of the F18 contribution (the `_272`/`_273` protocol; avoids the auto-loop per-arm HP-divergence confound).

### Regime correction (the reason this sweep exists)

The first cut (train_start 2019) trained almost entirely on the 2020–2022 COVID rally — target base rates 3–5× higher in training than test. Fixed in three steps:

1. **Back-extended** nse_equities: nifty500 was a fetch artifact (~85% of names seeded only from 2020). yfinance-first back-extend deepened 82→311 names to pre-2015. Tool: `scripts/data_pipelines/backextend_nse_equities.py`.
2. **Re-anchored** the split to `train_start 2015-01-01`, `train_rows 1787` (train now spans pre-rally + rally + normalization, ~64% pre-rally). 315-name deep-history universe (`min_rows_per_ticker 2591`).
3. **Extended** the test window to 2025-07-01 (`test_rows 204`): the 2024-Q4→Jan-2025 window was anomalously low-momentum; 2025-H1 recovered to ~train prevalence. Q 100→205 (double the statistical power).

**Segments** (all cells): train 2015-01-01→2022-03-29 · val →2023-11-09 · eval →2024-09-03 · **test 2024-09-04→2025-07-01**.

Drawdown-gated labels (exact `build_target`, `max_drawdown` per cell). 6 rare short-horizon high-threshold cells are near-degenerate (base_rate ≈0) but symmetric across arms.

## Verdict 1 — overall: a wash

| metric | mean Δ | median Δ | #pos / 20 |
|---|--:|--:|--:|
| AUC | +0.006 | +0.002 | 10 |
| R-p@1 | −0.011 | −0.020 | 8 |
| R-p@3 | −0.004 | −0.006 | 9 |
| R-p@5 | +0.004 | −0.001 | 9 |
| R-p@10 | +0.005 | −0.003 | 10 |

Centered on zero at every K. Consistent with the US F18 arc (`_279`/`_280`: failed two-window replication).

## Verdict 2 — the effect is horizon-shaped (the real finding)

Mean fund−base delta, grouped by horizon across the 4 thresholds:

| horizon | Δ AUC | Δ R-p@1 | Δ R-p@5 | Δ R-p@10 |
|---|--:|--:|--:|--:|
| 10d | −0.004 | −0.028 | +0.020 | +0.056 |
| 25d | −0.009 | +0.038 | +0.014 | −0.012 |
| 50d | +0.019 | **−0.056** | **−0.032** | −0.020 |
| **100d** | **+0.021** | **+0.026** | **+0.028** | **+0.016** |
| 200d | +0.004 | −0.033 | −0.011 | −0.017 |

**100d is the only horizon positive at every K.** Mechanistically sensible: valuation ratios resolve over a quarter or two (~5 months ≈ 100 trading days, the fundamental-refresh cadence) — not in the 10–50d technical/momentum regime, nor over the 200d drift regime.

## Verdict 3 — the two powered wins

Well-powered cells (Q=205, base_rate ≥ 0.05), the standouts:

| cell | base_rate | AUC base | AUC fund | R-p@1 base | R-p@1 fund | R-p@3 base | R-p@3 fund | R-p@10 base | R-p@10 fund |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **30/100d** | 0.143 | 0.550 | **0.614** | 0.215 | **0.283** | 0.207 | **0.285** | 0.163 | **0.297** |
| **20/100d** | 0.233 | 0.517 | **0.539** | 0.234 | **0.415** | 0.229 | **0.337** | 0.210 | **0.251** |

Both improve at **every K** (not a single-K blip) — the signature of real signal, not noise. 30/100d is the cleanest (AUC + every R-p@K up together); 20/100d has the biggest top-1 jump (+0.180).

## Verdict 4 — the anti-pattern to avoid: 20/50d

| cell | AUC base | AUC fund | R-p@1 base | R-p@1 fund | R-p@3 base | R-p@3 fund |
|---|--:|--:|--:|--:|--:|--:|
| 20/50d | 0.562 | **0.595** | 0.361 | **0.171** | 0.363 | **0.205** |

F18 lifts AUC (+0.033) while **collapsing** the top-of-book (R-p@1 −0.190, R-p@3 −0.158). The textbook `_276` trap — bulk ranking improves, the tail that trading consumes flattens. Judge these on the **book**, never AUC.

## Base-model note

The high-AUC cells (30/10d 0.83, 20/10d 0.79, 50/25d 0.79) are **rare-event mirages** — base_rate 0.00–0.02, Q as low as 24 days. High AUC on almost nothing to predict; not tradeable. The tradeable workhorses (10–30% thresholds at 100d) sit at AUC 0.50–0.61.

## Caveats

- **Single window** (test 2024-09→2025-07). Per the F17/F18 history, single-window wins need independent-window replication before adoption — this is exactly `_284`/`_264`'s cautionary pattern.
- **F18 is fundamentals-bound to 2019+** (in_fundamentals floor 2016 + 4-quarter TTM). The back-extend de-confounded the target and technical features, but the F18 valuation signal is still learned only from 2019 on — we can't fully separate "F18 doesn't help pre-rally" from "F18 never saw pre-rally."
- NOT promoted / NOT wired into `/daily-predictions`. A champion swap is a separate human decision.

## Follow-ups (each a separate PR)

1. **FS+HP agent-protocol finetune** of 30/100d + 20/100d (fund arm) — can the loop widen the 100d edge past the single-fit? (task #29)
2. **Stratified-boosting finetune** (`_284` recipe) of the same two cells — 100d/common-base-rate is the regime `_284` said the recipe favors. (task #30)
3. **sp500 second window** — re-run the same regime-corrected lattice on sp500 (us_equities already deep, no back-extend). If "F18 helps at 100d" replicates there, it's a real, adoptable finding, not a nifty500 window artifact. (task #28)

Registry: 40 rows (`nifty500_up_*_{fbase,ffund}`, train_start 2015) in `results/gbdt/data/r_precision_at_k.csv`.
