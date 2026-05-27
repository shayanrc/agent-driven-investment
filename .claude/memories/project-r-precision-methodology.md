---
name: project-r-precision-methodology
description: R-precision (per-day variable K) is the headline cross-cell metric for gbdt experiments. P@k as a secondary diagnostic MUST use min(R(d), k) denominator. Always report raw values + base rate, never lift columns in tables.
metadata:
  type: project
---

When comparing gbdt experiment cells **across different universes/markets** (e.g. nasdaq100 vs nifty50 vs sp500), use weighted R-precision as the headline metric, not fixed-K P@k or AUC alone.

## What R-precision is

For each day in the test (or eval) segment:
- `R(d)` = number of actual positives that day
- Sort items by `p_calibrated` descending (tie-break: `ticker` ascending, stable mergesort — matches the runner's per-day P@k convention)
- Take the top R(d) items
- `r_precision(d) = correct_picks / R(d)`

Two aggregates:
- **Mean unweighted**: `mean over days of r_precision(d)`
- **Weighted** (preferred for cross-cell comparison): `sum(correct@R) / sum(R)` = global recall@R = global precision@R (they're equal when k=R per day)

## P@k (when reported as secondary diagnostic)

For each day:
- Same sort + tie-break as R-precision
- Take top k items
- `p_at_k(d) = (positives in top k) / min(R(d), k)` — **the `min(R(d), k)` denominator is mandatory**

Two aggregates:
- **Weighted** (preferred): `sum(positives_in_top_k) / sum(min(R(d), k))`
- Mean unweighted: average of per-day values, skipping zero-denominator days

## The `min(R(d), k)` denominator is mandatory — pre-2026-05-28 bug

Earlier P@k code (the original `src/gbdt/topk_diagnostics.py` + the original `scripts/gbdt/compute_r_precision.py` P@k path + the original H=25 memo's tables) used `denominator = min(k, n_tickers_in_day)` — count of picks actually made. On staggered panels where R(d) often falls below k, this dragged P@k down for reasons unrelated to model skill: a day with 1 ticker and 0 positives would still get hits=0, picks=1 in the aggregate, when R(d)=0 means there was nothing to find.

The correct denominator is `min(R(d), k)` — count of achievable positives. When R(d) ≥ k it equals k (standard P@k). When R(d) < k it equals R(d) (recall-at-R for that day). Aggregates as `sum(positives_in_top_k) / sum(min(R(d), k))`.

**Concrete impact** of the bug on the H=25 cross-market memo (corrected 2026-05-28):
- nifty50 test P@1 OLD: 0.119 (looked "anti-predictive" at 0.67× base). CORRECTED: 0.257 (1.44× base — actual signal).
- nifty50 test P@10 OLD: 0.222. CORRECTED: 0.426 (since 124/151 test days had R(d) < 10).
- The original memo's "NSE anti-predictive top / skip top 1-2 trading rule" was a bug artifact and is withdrawn.
- sp500 was unaffected (R(d) mean = 128, always > k for our k ∈ {1,3,5,10}).
- R-precision was always correct (denominator IS R(d) by definition).

## Why R-precision is the cross-cell headline (not P@k)

- **Panel-invariant**: nasdaq100 has ~20 positives/day, nifty50 has ~9, sp500 has ~128. Fixed P@5 even with the correct denominator still depends on whether k is above or below R(d). R-precision adapts K per day, so cross-cell comparison is apples-to-apples.
- **Matches trading semantics**: "pick R(d) items each day sized to the day's signal" is closer to a real strategy than "always pick exactly 5".
- **Self-normalizing**: range [0,1]; baseline (random picker) = `R(d)/n(d)` = per-day base rate.

## Reporting conventions (project-wide, codified in CLAUDE.md)

- Tables show **raw metric values + base rate**, NOT lift columns. Lift compresses two values into one and loses the scale.
- Lift is OK in narrative prose ("nasdaq P@1 was 1.97× base rate").
- Every memo that reports top-K metrics should include a **"how to read this + formulas" subsection** near the top so the reader knows the exact denominator and tie-break convention used.
- All P@k computations (memo, runner, post-hoc scripts) MUST use `min(R(d), k)`. Anything else is a bug.

## Operational recipe

Post-hoc computation script: `scripts/gbdt/compute_r_precision.py`
- Takes a `predictions/{test,eval}.csv` path
- Emits R-precision (always correct) and P@k (with corrected denominator post-2026-05-28)
- Uses stable mergesort for tie-breaking (matches `src/gbdt/topk_diagnostics.py`)

Cross-cell anti-predictive analysis: `scripts/gbdt/nse_anti_predictive_cross_cell.py`
- Per-ticker pick/hit/anti-score
- Intersect across cells for recurring patterns

## Runner integration

`src/gbdt/topk_diagnostics.py::compute_top_k_metrics` was fixed in the same PR as this memo update (2026-05-28). All metrics.json files from runs BEFORE that PR have the buggy `p_at_k` values — re-compute post-hoc via `compute_r_precision.py` rather than trusting the cached `metrics.json::segment_diagnostics::top_k_metrics::per_day::*::p_at_k` field for pre-fix runs.

`segment_diagnostics::top_k_metrics::per_day::*` includes a `formula_version` field after the fix: `"v2_min_R_d_k"` (corrected). Pre-fix artifacts implicitly have `"v1_picks_made"` (buggy).

## The CLAUDE.md compound rule

Per the gbdt § What-not-to-do section:
- AUC ∈ [0.45, 0.55] **AND** weighted R-prec lift < 1.2× = null signal flagged
- AUC ∈ [0.45, 0.55] **AND** weighted R-prec lift > 1.5× = **top-tail signal hidden by AUC** — investigate, don't dismiss

The old "AUC ∈ [0.45, 0.55] = null" rule misclassified nasdaq H=25 (AUC=0.51, R-prec lift 1.46×) as null. Compound rule held up across all 4 H=25 cells in memo #138.

## Discovered + revised

2026-05-27 — original H=25 cross-market memo surfaced the cross-cell P@k apples-to-oranges problem and established R-precision as the primary metric.

2026-05-28 — user-flagged that the P@k formula in the original memo (and in the runner) used the wrong denominator. Revision: corrected formula, withdrew the "NSE inverted" narrative, codified "raw not lift" reporting convention in CLAUDE.md, fixed runner code.

See:
- `docs/gbdt/_138_h25_cross_market_combined.md` — the memo (post-correction).
- `[[project-gbdt-uniqueness-weights]]` — related methodology fix (LdP §4.4 weighting).
- `[[feedback-agent-pkill-antipattern]]` — process-coordination lesson from the same memo's experimental work.
