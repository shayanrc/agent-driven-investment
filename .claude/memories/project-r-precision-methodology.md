---
name: project-r-precision-methodology
description: R-precision (per-day variable K) is the headline cross-cell metric for gbdt experiments — replaces fixed-K P@k because it's panel-invariant and matches actual trading rule semantics. AUC is misleading when used alone for top-tail signal detection.
metadata:
  type: project
---

When comparing gbdt experiment cells **across different universes/markets** (e.g. nasdaq100 vs nifty50 vs sp500), use weighted R-precision as the headline metric, not fixed-K P@k or AUC alone.

## What R-precision is

For each day in the test (or eval) segment:
- `R(d)` = number of actual positives that day
- Sort items by `p_calibrated` descending (tie-break: `ticker` ascending, stable mergesort — matches the runner's per-day P@k convention)
- Take the top R(d) items
- `r_precision(d)` = correct_picks / R(d)

Two aggregates:
- **Mean unweighted**: `mean over days of r_precision(d)`
- **Weighted**: `sum(correct@R) / sum(R)` = global recall@R = global precision@R (they're equal when k=R per day)

The **weighted R-precision lift** = `weighted_rprec / base_rate_weighted` is the comparable cross-cell number.

## Why it's better than P@k for cross-cell work

- **Panel-invariant**: nasdaq100 has ~25 positives/day, nifty50 has ~8, sp500 has ~50+. Fixed P@5 captures a wildly different fraction of available signal in each. R-precision adapts K per day.
- **Matches trading semantics**: "pick R(d) items each day sized to the day's signal" is closer to a real strategy than "always pick exactly 5".
- **Self-normalizing**: range [0,1]; baseline (random picker) = `R(d)/n(d)` = per-day base rate. Lift is directly meaningful.

## When P@k is still useful (don't fully demote)

- **P@1** is a sanity check on the *single most-confident pick per day*. When P@1 lift < 1.0 AND R-prec lift > 1.5×, that's diagnostic of a specific over-pick cohort (see memo #138's HDFCBANK/WIPRO/COALINDIA pattern — model overconfident on low-volatility large-caps).
- **P@k** is still in the runner's standard report (PR #33). Use as **secondary** diagnostic, with R-precision as primary headline.

## Operational recipe

Post-hoc computation script: `scripts/gbdt/compute_r_precision.py`
- Takes a `predictions/{test,eval}.csv` path
- Emits JSON or human-readable summary
- Uses stable mergesort for tie-breaking (matches `src/gbdt/topk_diagnostics.py`)

Cross-cell anti-predictive analysis: `scripts/gbdt/nse_anti_predictive_cross_cell.py`
- Per-ticker pick/hit/anti-score
- Intersect across cells for recurring patterns

## Future runner integration (deferred)

Memo #138 recommended baking R-precision into `src/gbdt/topk_diagnostics.py` alongside the existing P@k computation, so every future run auto-emits R-precision in `metrics.json::segment_diagnostics`. Not done as of 2026-05-27 (separate PR work). Until then, compute post-hoc via the script.

## The CLAUDE.md compound rule

Per the gbdt § What-not-to-do section:
- AUC ∈ [0.45, 0.55] **AND** weighted R-prec lift < 1.2× = null signal flagged
- AUC ∈ [0.45, 0.55] **AND** weighted R-prec lift > 1.5× = **top-tail signal hidden by AUC** — investigate, don't dismiss

The old "AUC ∈ [0.45, 0.55] = null" rule misclassified nasdaq H=25 (AUC=0.51, R-prec lift 1.46×) as null. PR #28 caught this; memo #138 formalized R-precision as the fix.

## Discovered

2026-05-27 — H=25 cross-market memo work surfaced the cross-cell P@k apples-to-oranges problem. nifty50/100 P@1 lift looked anti-predictive (0.5-0.7×); nasdaq P@1 lift looked dominant (1.78×). Under R-precision, all 3 cells showed comparable signal magnitude (1.46–2.12× lift) and the picture inverted.

See:
- `docs/gbdt/_138_h25_cross_market_combined.md` — the memo this finding came from.
- `[[project-gbdt-uniqueness-weights]]` — related methodology fix (LdP §4.4 weighting).
- `[[feedback-agent-pkill-antipattern]]` — process-coordination lesson from the same memo's experimental work.
