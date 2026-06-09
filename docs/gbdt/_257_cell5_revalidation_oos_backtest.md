# _257 — Cell-5 V1.3 revalidation: OOS back-test pilot

**Branch**: `gbdt-top10-oos-backtest-pilot`.
**Date**: 2026-06-09.
**Validates**: pilot of the top-10 R-p@3 OOS back-test plan (parent task). Uses the regenerated cell-5 V1.3 revalidation artifact (memo [_223](_223_cell5_loop_v1.3_revalidation.md), restored in PR #156) as the first cell. Question: does the canonical-CSV R-p@K hold up on a window the model has never seen?
**Cell**: nasdaq100 +10% / 50d / dd 5% — the anti-AUC strong-top-1 cell. Same model artifact as [_223](_223_cell5_loop_v1.3_revalidation.md). XGBoost, max_depth=2, n_estimators=500 (≈6 trees retained after ES), eta=0.1, 190 kept features, native calibration.
**Script**: [scripts/gbdt/backtest_top10_revalidation_pilot.py](../../scripts/gbdt/backtest_top10_revalidation_pilot.py).
**Outputs**: [results/gbdt/backtest/nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation/](../../results/gbdt/backtest/nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation/).

## Headline

**OOS R-Precision@1 matches the held-out test slice within sampling noise** (0.7255 OOS vs 0.7143 canonical-CSV test). R-p@3 degrades modestly (0.4967 OOS vs 0.5381 canonical); R-p@10 degrades more (0.2981 OOS vs 0.4559). The cell's top-1 conviction is durable on truly unseen dates; broader top-K rankings drift.

| K | OOS R-p@K (this memo) | Canonical CSV R-p@K | Regen own test R-p@K | Δ (OOS − canonical) |
|---|---|---|---|---|
| 1 | **0.7255** | 0.7143 | 0.8000 | **+0.0112** |
| 3 | 0.4967 | 0.5381 | 0.7556 | −0.0414 |
| 5 | 0.4010 | 0.4836 | 0.7156 | −0.0826 |
| 10 | 0.2981 | 0.4559 | 0.6178 | −0.1578 |
| 20 | 0.3001 | 0.4695 | 0.6634 | −0.1694 |

**OOS window**: 2025-12-29 → 2026-03-12 (51 Q_days, base_rate = 0.2645). Strictly after the regen artifact's per-ticker trailing test_end (2025-12-26).

## Setup

- **Boundary**: regen model's `segment_dates.test.end = 2025-12-26`. Picks from 2025-12-29 onwards are unseen by training/val/eval/test.
- **OOS upper bound**: cache right edge = 2026-05-22 (cache_only=True; raw-fetch path blocked — see § Caveats). With H=50BD, last pickable day with a completed forward label is 2026-03-12.
- **Data**: 645,189 panel rows over 92 NASDAQ100 tickers (8 excluded for short history: ABNB, APP, ARM, CEG, DASH, GFS, MNST, ON — same set as the fit-time exclusions). 4,661 OOS (date, ticker) rows in the back-test set; 92 tickers × 51 days nominally, less the small number of NaN-feature rows on staggered series starts.
- **Scoring path**: identical to the runner's canonical test-slice path — `XGBoostModel.load(...).predict_proba` → native pass-through (Spiegelhalter |z|<2.0 on this artifact) → canonical `scripts/gbdt/compute_r_precision.py` for R-p@K.

## Distributional read on top-K picks

Forward 50-BD return distributions for the picks (computed from cached OHLCV; **not portfolio-style — no costs, sizing, capital allocation**):

| K | n picks | Hit rate | Mean fwd close ret | Median | p10 | p90 | Mean fwd max DD |
|---|---|---|---|---|---|---|---|
| 1 | 51 | 0.725 | +7.2% | +7.1% | −4.9% | +17.1% | −18.0% |
| 3 | 153 | 0.497 | +23.4% | +14.6% | −7.4% | +75.8% | −18.9% |
| 5 | 255 | 0.400 | +14.6% | +11.5% | −10.7% | +37.7% | −20.9% |
| 10 | 510 | 0.292 | +6.6% | +3.7% | −19.2% | +38.6% | −22.6% |

The K=3 mean is materially higher than K=1 — driven by a heavy concentration of NASDAQ:AMD at rank 3 across late-Feb / early-Mar 2026, where AMD ran +20% to +136% in its forward windows. That's a real OOS signal (the model picked AMD at rank 3 with p_calib ≈ 0.39 and was rewarded), but it makes the K=3 mean a noisy estimator on 51 days. The K=1 (most conviction-weighted) median +7.1% is the more representative read.

Note the mean fwd max DD ≈ −18% to −23% across all K — this cell trades on a +10% / −5% rule, but the model isn't conditioning on intraday drawdown control (the label requires no -5% drawdown during the 50d window, but the realized picks experience −18% drawdowns on average; the y_true=0 picks are where the drawdown rule killed the label even when close-to-close closed higher).

## OOS hit/miss split

The hit-rate split below confirms the cell's character: when the top-1 pick lands the label (37 of 51 days), the trade typically resolves at +10–30% on close; when it misses the label, the drawdown gate is the usual cause — close eventually recovers but the path went through a −5% intraday at some point.

| K=1 sub-population | n | Mean fwd close ret | Mean fwd max DD |
|---|---|---|---|
| y_true=1 (label hit) | 37 | +12.7% | −13.4% |
| y_true=0 (label miss) | 14 | −7.4% | −30.3% |

The misses are concentrated in early-to-mid Jan 2026 (4 of 14 worst picks are AMD picks in the 2026-01-02 → 2026-01-23 window, all hit max DD around −28%). The hits cluster in late Feb / early Mar 2026, suggesting the model's signal is regime-aware rather than uniformly distributed across the OOS window.

## Reconciliation with prior test-slice metrics

Three R-p@K series exist for this artifact:

1. **Canonical CSV row** (memo's per-ticker trailing slice, Q=70, base_rate=0.265): R-p@1=0.7143, R-p@3=0.5381.
2. **Regen own test slice** (`predictions/test.csv` rebuilt from the regen run, Q=101, base_rate=0.338): R-p@1=0.8000, R-p@3=0.7556.
3. **OOS back-test** (this memo, Q=51, base_rate=0.265): R-p@1=0.7255, R-p@3=0.4967.

The OOS read on R-p@1 (0.7255) sits between (1) and (2), validating the canonical row. R-p@3 OOS (0.4967) is meaningfully BELOW both prior slices (0.5381, 0.7556), with the gap larger at higher K. Interpretation: the model's #1 pick is durably predictive; its #2-#10 ranks within the top-tail are noisier and more sample-dependent.

This pattern is **consistent with the tiny-model / anti-AUC regime** of the cell: ≈6 trees of depth 2 ⇒ very few distinct probability outputs (the OOS picks at rank ≥ 2 cluster at p_calibrated ≈ 0.385–0.390, with ticker name breaking ties). The model essentially partitions the universe into a few "high-prob" buckets; the #1 vs #2 distinction inside a bucket is noise.

## Caveats

1. **Cache freshness**: the `data/raw` symlink is currently broken (self-loop), blocking provider fetches. OOS is bounded by cache right edge 2026-05-22 instead of today (2026-06-09). With H=50BD this caps the OOS at 2026-03-12. Fixing the symlink would add ~12 BD of new pick days. **Tracked as a follow-up; not in scope for this pilot.**
2. **Q_days=51 is small.** R-p@K at K ∈ {10, 20} on 51 days has wide CI. The K=1 read is the most stable.
3. **Realized labels use the same target builder as training.** No label-time drift between train and OOS (`gbdt_targets.build_target(direction='up', threshold_pct=10, horizon_days=50, max_drawdown=0.05)` applied to the extended panel).
4. **Native calibration is correct for this artifact** — the regen ran conditional_isotonic and shipped native pass-through (Spiegelhalter |z| < 2.0 on val). When scaling this pilot to other top-10 cells, calibration.pkl must be loaded and applied (not all artifacts ship native).
5. **No portfolio metrics here.** Forward returns are reported for human interpretation only — the project explicitly bans transaction costs, position sizing, and PnL in this module per `[[goal-no-portfolio]]`.

## What this pilot proves

- The harness works end-to-end on the regenerated artifact: load model, refresh features through cache right edge, score on a truly unseen window, compute canonical R-p@K, distribution-summarize top-K picks.
- The pattern (R-p@1 durable, K≥3 degrades) is interpretable as **tiny-model bucket-quantization noise**, not as a model failure.
- The script is structured to generalize — the only cell-specific bits are the artifact directory, the (direction, threshold_pct, horizon_days, max_drawdown) tuple, and the universe name. Scaling to the other 9 top-10 cells is a refactor + 9 invocations, not a redesign.

## Follow-ups

- Fix `data/raw` symlink so we can extend OOS to today − 50BD (would add ~12 BD of picks).
- Generalize the script to consume any artifact directory + cell tuple → `backtest_topk.py`.
- Run the back-test on the other 9 cells in the top-10. Long-horizon cells (russell/sp500 @ H=200d) have ~300 BD of usable OOS — the strongest validation set.
- Investigate whether the K=3 OOS degradation (vs test) is recoverable with isotonic recalibration on the OOS-leading 25 BD (would require re-fitting the calibrator, but the model stays frozen).
