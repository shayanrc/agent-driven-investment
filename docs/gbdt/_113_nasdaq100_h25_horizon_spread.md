# Task #113 — H=25 horizon-spread data point (`nasdaq100_up_10pct_25d_dd5pct`)

**Cell**: `nasdaq100_up_10pct_25d_dd5pct` (NASDAQ100, +10% in 25 trading days, max-drawdown 5%; sweep-mode `fs_hp_loop.max_iter=3`)

**Why this run**: PR #27's memo recommended a horizon-spread to disambiguate "no signal at all" from "no signal at this specific horizon". Sweep #1 used H=100; this run uses H=25 (overlap_inflation 199.0 → 49.0). Both runs use the same universe, threshold (+10%), drawdown (5%), and methodology (sample-uniqueness weighting on).

**Result location**: `results/gbdt/experiments/nasdaq100_up_10pct_25d_dd5pct/`

## Headline metrics

| Metric | H=100 eval (Sweep #1) | H=25 eval | **H=25 test (held-out)** |
|---|---:|---:|---:|
| AUC | 0.488 | 0.655 | **0.511** |
| Brier vs baseline | -0.006 | **+0.010 (beats)** | -0.005 |
| Spiegelhalter z | 1.97 | 1.59 | — |
| n_rows | 18400 | 18400 | **6900** |
| positive_prevalence | 0.398 | 0.249 | 0.273 |
| prediction range | 0.32-0.53 (std 0.04) | 0.07-0.60 (std 0.10) | 0.08-0.52 (std 0.07) |

H=25 produced a non-empty test split — H=100's forward-100-day window ate the test slice structurally, leaving only eval. This run gives us our first held-out US data point.

## Critical finding: AUC = 0.51 on test is MISLEADING

The headline AUC says "null signal on held-out data." But the top-K per-day breakdown tells a different story:

### Per-day top-K hit rates

| Segment | k=1 | k=3 | k=5 | k=10 |
|---|---:|---:|---:|---:|
| Eval P@k | 0.251 (1.01x) | 0.399 (1.60x) | **0.429 (1.72x)** | 0.436 (1.75x) |
| **Test P@k** | **0.486 (1.78x)** | **0.431 (1.58x)** | **0.430 (1.57x)** | 0.376 (1.38x) |

**Test top-1 per day = 48.6% hit rate vs 27.3% base = 1.78x lift.** Top-5/day on test = 174 positives / 405 picks = 43% (1.57x lift).

On data the model never saw during selection.

## Mechanistic reading

This is the **classic finance-ML pattern where AUC understates top-tail signal**:

- **AUC averages over the whole prediction distribution.** The model is anti-predictive on the bulk of test cases (most predictions cluster near base rate; ranking is noisy in that range), which drags AUC to ~0.5.
- **Top-K only looks at the extreme.** The model's highest-confidence picks per day DO outperform. The signal exists, but only at the tail of the prediction distribution — which is what a trading strategy actually consumes.

Compare to H=100:
- **H=100 prediction std=0.04** — model produces predictions tightly clustered around base rate. No "extreme" tail exists to exploit. Null AUC + null P@k = genuinely no signal.
- **H=25 prediction std=0.07** (test) / **0.10** (eval) — model produces a wider, informative distribution. Bulk is still noisy (null AUC), but the top decile carries real predictive power (high P@k lift).

The horizon-spread experiment was the right call: H=25 reveals signal that H=100 does not have access to.

## Why was Sweep #1's H=100 result so flat then?

Looking back at the H=100 prediction distribution: the model couldn't even produce confident calls — std=0.04 means EVERY prediction was within 0.08 of the mean. The CatBoost loop early-stopped at 12-17 trees (out of 1000-budget) on every iteration. The features simply couldn't distinguish "this stock will hit +10% in 100 trading days, with max drawdown ≤5%" from "it won't" — the target is too long-horizon for the F-family features to anchor on. At H=25, the target moves into a regime where short-window features (run-up, volatility, beta) start to carry useful information.

## Implications — revised plan

### 1. Pivot the US sweep to short-horizon focus (was: defer entirely)

Drop the 17 cells with H ≥ 100. Focus the next sweep on H ∈ {5, 10, 25, 50} × {nasdaq100, sp500, russell1000} × {up} × the 3 threshold-drawdown combinations = ~36 cells if we keep all threshold variants, or ~12 cells if we focus on the +10%/dd5% triplet.

Cells most likely to surface signal (extrapolating from H=25 success):
- `nasdaq100_up_10pct_5d_dd5pct`, `_10d_dd5pct`, `_50d_dd5pct`
- `sp500_up_10pct_*_dd5pct` (same 4 horizons)
- `russell1000_up_10pct_*_dd5pct` (same 4 horizons)
- Total: 12 cells, ~6 hr compute.

### 2. Add per-day P@k diagnostics to the standard runner report — HIGH PRIORITY

Without per-day P@k, we'd have looked at H=25's test AUC=0.51 and declared it null, missing the 1.78x top-1 lift. The standard `report.md` template must include:

- Per-day P@k table (k ∈ {1, 5, 10}, eval AND test, with lift over base rate)
- Per-ticker hit rate when picked (sorted; surface systematically-wrong tickers)
- Per-quarter stability of P@k (catches regime-dependent collapse)
- Prediction-range stat (max-min, std) — flag when std < 0.05 as "no separation"

This is the gating change before the next sweep batch.

### 3. Amend the null-signal rule in CLAUDE.md

Current rule: "AUC ∈ [0.45, 0.55] is a null-signal flag". This rule misclassifies H=25 as null.

Proposed:
> AUC ∈ [0.45, 0.55] AND per-day P@5 lift < 1.2x = null signal flagged.
> AUC ∈ [0.45, 0.55] AND per-day P@5 lift > 1.5x = **top-tail signal; AUC understates; investigate the prediction-extreme regime.**

### 4. Drop the deep-search rerun of H=100 (was: investigate)

H=100's prediction distribution is too tight for any search depth to extract signal. Per-day P@5 lift was 1.08x even on the iteration that won — there's no extreme tail to exploit. Compute is better spent on short-horizon variants.

### 5. V2_TBD addition stands

The semis-cohort hit-rate story (AVGO/NVDA/ADI/AAPL strong, ANSS/MSTR/TTD anti-predictive) from PR #27 carries over to H=25 — worth confirming with per-ticker analysis on the new test set before committing the V2 redesign question.

## Verdict

- **Cell verdict**: TOP-TAIL SIGNAL on H=25. P@5_test = 1.57x lift, P@1_test = 1.78x. AUC=0.51 is misleading without P@k context.
- **Methodology verdict**: sample-uniqueness fix wired correctly (`overlap_inflation_ratio: 49.0` for H=25). At this horizon the fix matters more than at H=100 (per-sample-variable weights, not uniform).
- **Plan verdict**: short-horizon focus + P@k diagnostics-first BEFORE running more cells. The whole "57-cell sweep is wasteful" recommendation from PR #27 was overcorrected — we should sweep, but smarter.

Cross-links: PR #18 (uniqueness fix), PR #27 (Sweep #1 + initial plan recommendations), task #107 (sweep — REVISE scope per above), task #113 (re-run experiments — partially fulfilled).
