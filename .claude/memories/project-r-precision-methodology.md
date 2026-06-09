---
name: project-r-precision-methodology
description: R-Precision@K (per-day fixed K, macro-averaged) is the headline cross-cell metric for gbdt experiments. Renamed 2026-06-01 from the prior weighted variant. Compound (AUC, R-Precision@10 lift) null/signal rule (V1.3 D4 thresholds [0.46, 0.54] + 1.2×/1.8×). Runner emits canonical macro post-PR #144 (2026-06-08).
metadata:
  type: project
---

When comparing gbdt experiment cells **across different universes/markets** (e.g. nasdaq100 vs nifty50 vs sp500), use **R-Precision@K** as the headline metric, not AUC alone.

## Definition — R-Precision@K (post-2026-06-01)

Per-day fixed K, macro-averaged across days:

```
R-Precision@K = (1 / Q) · Σ_{q=1..Q}  r_q / min(K, R_q)
```

where for each test day q:
- `R_q` = number of actual positives that day (`y_true == 1`)
- `r_q` = positives caught in the top-K picks on day q (sorted by `p_calibrated` descending, tie-break `ticker` ascending, stable mergesort — matches the runner's per-day P@k convention)
- `Q` = number of days with `R_q > 0` (days with no positives are skipped — `min(K, 0)` is ill-defined and the day contributes no information)
- `K` is a fixed integer; the **standard reporting K is `{1, 3, 5, 10, 20}`**.

Notes:
- The denominator `min(K, R_q)` is what makes this **R-Precision** rather than plain P@K — on days where `R_q < K`, you can't catch more than `R_q` positives even in principle, so the achievable precision ceiling is `R_q / min(K, R_q) = 1`. Penalizing days with `R_q < K` for not catching `K` positives would mis-normalize on staggered panels.
- The aggregation is **macro** (mean of per-day ratios) — each day gets equal weight. This matches the question "how reliable is the model on a typical day?"

## Relationship to the prior "weighted R-precision" (now legacy)

Pre-2026-06-01, the project used a single "weighted R-precision" headline computed as:

```
weighted R-precision = Σ_q (positives caught in top R_q) / Σ_q R_q  =  Σ_q r_q^{K=R_q} / Σ_q R_q
```

— per-day **variable** K (always K = R(d)), with **micro** (sum/sum) aggregation. It's a different metric:

| Property | weighted R-precision (legacy) | R-Precision@K (current) |
|---|---|---|
| K per day | variable, K = R(d) | fixed (typically 1, 3, 5, 10, 20) |
| Aggregation | micro (Σ/Σ) | macro (mean of per-day ratios) |
| Per-day weight | proportional to R(d) | equal across days |
| Interpretation | average picked-positive rate over Σ R(d) picks | average per-day precision-at-K |

**They are NOT comparable as a single number** — high-R(d) days get heavy weight in the legacy metric and equal weight in the current one. For most cells the two land within ~30% of each other, but the direction of divergence depends on whether high-R(d) days are easier or harder for the model.

**All numbers in memos written before 2026-06-01 use the legacy formula** unless explicitly updated. New memos use R-Precision@K. The post-hoc CSV at `results/gbdt/data/r_precision_at_k.csv` carries R-Precision@{1,3,5,10,20} for every cell that has predictions in `results/gbdt/experiments/*/predictions/test.csv`, allowing apples-to-apples comparison across the historical record.

## The P@K denominator (still mandatory) — pre-2026-05-28 bug

Earlier P@K code used `denominator = min(k, n_tickers_in_day)` — count of picks actually made. This silently mis-normalized on staggered panels where R(d) < k for many days. The correct denominator is `min(R(d), k)` — count of achievable positives. Aggregates as `sum(positives_in_top_k) / sum(min(R(d), k))` for the micro form, or as the per-day-ratio mean for the macro form.

**Concrete impact** on the H=25 cross-market memo (corrected 2026-05-28):
- nifty50 test P@1 OLD: 0.119 (looked "anti-predictive"). CORRECTED: 0.257 (signal).
- The "NSE anti-predictive top / skip top 1-2" rule was a bug artifact and was withdrawn.

The R-Precision@K formula above already bakes in the correct denominator.

## Why R-Precision@K is the cross-cell headline

- **Trades off K precisely**: panel-invariance comes from K being a real trading-rule knob — "how many names per day do I want to size for?" Reporting at K ∈ {1, 3, 5, 10, 20} gives the full precision-vs-K curve, which is what a portfolio manager actually needs to set position sizing against.
- **Day-equal weighting matches the implicit decision unit**: each trading day is one decision occasion. Macro averaging avoids overweighting calendar windows that happen to be positive-dense.
- **Self-normalizing**: range [0, 1]; baseline (random picker) = base rate; **lift = R-Precision@K / base_rate** is reported in narrative prose.

## Reporting conventions (codified in CLAUDE.md)

- Tables show **raw R-Precision@K values + base rate**, NOT lift columns. Lift compresses two values into one and loses the scale.
- Lift is OK in narrative prose ("nasdaq H=25 R-Precision@10 was 1.86× base rate").
- Every memo that reports top-K metrics should include a **"how to read this + formulas" subsection** near the top so the reader knows the exact denominator, K values, and tie-break convention used.
- Standard K set: `{1, 3, 5, 10, 20}`. Other K values may be added for specific analyses; never DROP a K from the standard set without justification.
- P@K = R-Precision@K (same denominator). Use "R-Precision@K" consistently — the legacy "P@k" terminology survived in some code paths but the corrected formula is identical.

## Operational recipe

Canonical post-hoc computation script: `scripts/gbdt/compute_r_precision.py`
- Takes a `predictions/{test,eval}.csv` path
- Emits R-Precision@K at K ∈ {1, 3, 5, 10, 20} (macro)
- Also emits the legacy weighted R-precision for cross-walk to pre-2026-06-01 memos
- Uses stable mergesort for tie-breaking (matches `src/gbdt/topk_diagnostics.py`)

Canonical cell-by-cell registry: `results/gbdt/data/r_precision_at_k.csv` — all completed experiments × {R-Precision@1, @3, @5, @10, @20} + AUC + base_rate + Q_days. Regenerate after any new experiment via:
```bash
uv run python -m scripts.gbdt.regenerate_r_precision_at_k_csv
```

Cross-cell anti-predictive analysis: `scripts/gbdt/nse_anti_predictive_cross_cell.py`
- Per-ticker pick/hit/anti-score
- Intersect across cells for recurring patterns

## Runner integration

`src/gbdt/topk_diagnostics.py::compute_top_k_metrics` was fixed in PR #28 (2026-05-28) to use the correct `min(R(d), k)` denominator. metrics.json files from runs BEFORE that PR have buggy `p_at_k` values — re-compute post-hoc via `compute_r_precision.py` rather than trusting the cached field. Post-fix artifacts have `formula_version: "v2_min_R_d_k"`; pre-fix implicitly have `"v1_picks_made"` (buggy).

The runner emits **both** aggregations from 2026-06-08 onward (PR #144 / task #252):

- Legacy micro at `metrics.json::segment_diagnostics::<seg>::top_k_metrics::per_day::*::p_at_k` — preserved for back-compat with scripts that consumed the legacy path.
- Canonical macro at `metrics.json::segment_diagnostics::<seg>::r_precision_at_k` with `formula_version: "macro_per_day_fixed_k"` provenance — matches the post-hoc CSV byte-for-byte for new artifacts.

For NEW cells, memos can pull R-Precision@K directly from `metrics.json`. For artifacts produced BEFORE PR #144 the metrics.json doesn't have the canonical block — fall back to the post-hoc CSV (whose source-of-truth status is unchanged; it remains the cross-cell registry).

## The CLAUDE.md compound rule (post-rename)

Per the gbdt § What-not-to-do section:
- AUC ∈ [0.46, 0.54] **AND** R-Precision@10 lift < 1.2× = null signal flagged
- AUC ∈ [0.46, 0.54] **AND** R-Precision@10 lift > 1.8× = **anti-AUC strong-top-1 cell** — top-tail signal hidden by AUC; investigate the prediction-extreme regime, don't dismiss. V1.3 Option A auto-disables the L1 tie-break + val_brier auto-plateau when the iter_0 anti_auc_flag fires (visible in `loop/checkpoint.json::auto_disabled`).

The old "AUC ∈ [0.45, 0.55] = null" rule misclassified nasdaq H=25 (AUC=0.51, R-Precision@10 = 0.507 / base 0.273 = 1.86× lift) as null. Under the **tightened** [0.46, 0.54] + 1.8× rule the H=25 cells still classify as anti-AUC strong-top-1 — the new bound is marginally above nasdaq's 1.86× but stays inside the band, and memo #138's compound-rule conclusion is preserved.

**Threshold calibration history**: the original (1.2×, 1.5×) thresholds + [0.45, 0.55] AUC band were calibrated against the legacy weighted R-precision metric. V1.3 plan D4 (2026-06-XX, see `docs/gbdt/V1.3_option_a_loop_anti_auc_integration_plan.md`) **tightened to (1.2×, 1.8×) + [0.46, 0.54]** so the agent loop's anti-AUC auto-disables (L1 tie-break + val_brier plateau gate) don't false-positive on cells that are marginal but still meaningfully discriminating. The COMPOUND form (AUC band + top-tail metric) is the durable lesson; the specific thresholds may want further re-calibration as more cells accumulate.

## Discovered + revised

2026-05-27 — original H=25 cross-market memo surfaced the cross-cell P@k apples-to-oranges problem and established R-precision (per-day variable K) as the primary metric.

2026-05-28 — user-flagged that the P@k formula in the original memo (and in the runner) used the wrong denominator. Revision: corrected formula, withdrew the "NSE inverted" narrative, codified "raw not lift" reporting convention in CLAUDE.md, fixed runner code.

2026-06-01 — user-flagged that the project's "weighted R-precision" (per-day variable K, micro aggregation) was NOT the metric they had been mentally computing. Renamed to R-Precision@K with fixed K and macro aggregation per the user's stated formula. All memos + CLAUDE.md updated; legacy weighted R-precision preserved for cross-walk.

2026-06-08 — PR #144 (task #252) unified the runner's segment_diagnostics output with the canonical macro: `metrics.json::segment_diagnostics::<seg>::r_precision_at_k` now matches `results/gbdt/data/r_precision_at_k.csv` byte-for-byte. The legacy micro `top_k_metrics::per_day::p_at_k` block remains for back-compat. Memo authors no longer need the post-hoc CSV detour for new artifacts. Same PR fixed the PR #139 review-time confusion that misframed the runner-vs-canonical divergence as a tie-break drift (it was always micro-vs-macro aggregation).

See:
- `docs/gbdt/_138_h25_cross_market_combined.md` — the memo (post-correction, with R-Precision@K appended 2026-06-01).
- `results/gbdt/data/r_precision_at_k.csv` — canonical cell registry.
- `[[project-gbdt-uniqueness-weights]]` — related methodology fix (LdP §4.4 weighting).
- `[[feedback-agent-pkill-antipattern]]` — process-coordination lesson from the H=25 memo's experimental work.
