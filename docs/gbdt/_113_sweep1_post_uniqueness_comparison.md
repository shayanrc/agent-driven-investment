# Task #113 — Sweep #1 post-uniqueness-fix re-run + methodology comparison

**Cell**: `nasdaq100_up_10pct_100d_dd5pct` (NASDAQ100, +10% in 100 trading days, max-drawdown 5%; sweep-mode `fs_hp_loop.max_iter=3`)

**Why this re-run**: PR #18 (LdP §4.4) shipped sample-uniqueness weighting as the default. Sweep #1 had been run on 2026-05-26 12:00 with the old (biased) methodology. This re-run is the first half of task #113 (Exp 1 nifty50 still pending). It also serves as the first signal-quality data point for whether to launch the full 57-cell US sweep (task #107).

**BEFORE artifact** (archived): `project_mgmt/archives/biased_baselines/nasdaq100_up_10pct_100d_dd5pct_pre_uniqueness_fix/`
**AFTER artifact** (this PR): `results/gbdt/experiments/nasdaq100_up_10pct_100d_dd5pct/`

## Headline metrics

| Metric | BEFORE (biased) | AFTER (post-fix) | Δ | Interpretation |
|---|---:|---:|---:|---|
| `roc_auc` (eval) | 0.4932 | 0.4880 | -0.0052 | null in both — slightly worse than coin-flip |
| `brier` (eval) | 0.2532 | 0.2456 | -0.0076 | mild improvement, still below baserate |
| `brier_baseline_baserate` (eval) | 0.2396 | 0.2396 | 0 | (data hash matched; baseline unchanged) |
| `brier_improvement_vs_baseline` | -0.01356 | -0.00599 | +0.00757 | post-fix Brier is closer to baseline (still negative — model worse than constant) |
| `log_loss` (eval) | 0.9126 | 0.6843 | **-0.2283** | major improvement; pre-fix was overfit |
| `calibration.spiegelhalter_z` | 2.257 (p=0.024) | 1.972 (p=0.049) | -0.285 | better calibrated post-fix |
| `calibration.decision` | `isotonic` | `native` | — | model is now self-calibrating; conditional isotonic chose to leave it alone |
| `loop.best_iteration` | 0 | 1 | +1 | post-fix loop preferred iter 1 over iter 0 |
| `loop.inner_stop_signal` | `plateau` | `plateau` | — | both runs hit max_iter without finding a clear min |

**Spec hash**: changed (`d673b44…` → `1f47e6a…`) — expected, since `uniqueness_weighting=true` is now baked into the spec metadata. **Data hash**: matched (`1576366…` → `1576366…`) — same underlying panel; apples-to-apples.

## Sample-uniqueness telemetry (post-fix only)

The `metrics.json::sample_uniqueness` block confirms the fix is wired:

```
horizon_days: 100
overlap_inflation_ratio: 199.0   # = 2H-1 for H=100 — every label window overlaps with 199 others
n_rows: 73600  (train fold 0)
sum_weights: 369.85
ess_kish: 73600.0
```

**Observation**: `ess_kish == n_rows` because `overlap_inflation_ratio` is at its theoretical maximum (every interior sample has 199 overlapping neighbors), so every per-sample weight is **identical** (≈ 1/199). The relative weights don't differ, but the absolute weight magnitude is now ≈ 200× smaller than naive — which is what changes the loss landscape and is responsible for the calibration improvement.

This is the boundary regime for sample-uniqueness weighting: when overlap is uniformly maximal, the fix becomes equivalent to a global learning-rate / loss-scale rescale rather than a per-sample reweighting.

## Mechanistic reading

1. **There is no signal in nasdaq100 +10%/100d/dd5%.** Both runs show AUC ≈ 0.49 ± 0.01. The pre-fix biased run wasn't masking signal — it was overconfidently miscalibrated noise. Sweep #1 was the **priority #1** cell (highest base rate among the 57); this is concerning evidence that the whole 57-cell US sweep may produce mostly null signals.

2. **The fix correctly improved calibration without surfacing hidden signal.** The Spiegelhalter z moved toward 0 (well-calibrated) and conditional isotonic flipped to `native` — meaning the model's raw probabilities now match observed frequencies well enough that no post-hoc correction is needed. Pre-fix, conditional isotonic was applying correction to mask the overconfidence.

3. **The log_loss collapse (0.91 → 0.68)** is consistent with (1) + (2): the pre-fix model was making confident-and-wrong predictions; the post-fix model is making conservative-and-right-about-uncertainty predictions. Brier captures less of this than log_loss because Brier is bounded ≤ 1 and noise predictions cluster near baseline-Brier; log_loss is unbounded above and severely punishes overconfident errors.

4. **The fix matters most exactly where overlap_inflation_ratio is high.** Sweep #1 has `H=100`, which is the longest horizon in the sweep — and consequently the most overlap. Shorter-horizon cells (H=20, H=50) will have smaller `overlap_inflation_ratio` and a less-uniform weight distribution, where the *relative* per-sample weights matter (not just the magnitude). Those are where the fix could surface signal that the biased methodology obscured.

## Verdict

- **Cell verdict**: NULL SIGNAL. AUC ∈ [0.45, 0.55] per CLAUDE.md's null-flag rule. Both pre-fix and post-fix runs concur.
- **Methodology verdict**: PR #18 sample-uniqueness fix is correctly wired and produces the expected calibration improvement at this overlap regime.
- **Downstream consumers**: any code that branches on `calibration.decision` should be re-examined — the post-fix world has `native` as a common case where it used to default to `isotonic`.

## Recommended next moves (sequencing for #107)

Given Sweep #1 is null, the full 57-cell sweep should be priority-reordered. Specifically:

1. **Don't** mechanically launch all 57 cells in priority order. Sweep #1 was priority #1; null result is a red flag for the whole sweep design.
2. **Do** run a smaller exploratory batch first — 3-5 cells with different `horizon_days` (e.g., H=20, H=50, H=100 across one universe). If horizon-shorter cells also show null, the issue is upstream (features, base-rate distribution, target-construction).
3. **Investigate** before sweeping: check the F-family feature importances from this run (`figs/feature_importance_final.png`). If the top-ranked features are noise, the feature set needs rework before any sweep is informative.

Cross-links: PR #18 (the methodology fix), PR #23 (universe YAML format spec), task #113 (re-run sweep — Exp 1 nifty50 portion still pending), task #107 (57-cell sweep — recommend deferring pending the above investigation).

## Footnote — archived baseline spec.yaml is corrupted

The archived `project_mgmt/archives/biased_baselines/nasdaq100_up_10pct_100d_dd5pct_pre_uniqueness_fix/spec.yaml` contains nifty50 universe registry content (with the long-removed `ticker_prefix` field), NOT the nasdaq100 spec. This is a pre-existing artifact-snapshotting bug in the runner that affected the prior run — the actual experiment ran on the real nasdaq100 spec (proven by `data_hash` match + ticker list in `metrics.json::data.cache_age_days_by_ticker` matching NASDAQ100). The `metrics.json` and `report.md` in the archive are trustworthy; the `spec.yaml` in the archive is not. The post-fix run does NOT exhibit this bug — its `spec.yaml` snapshot is correct. Worth filing as a follow-up if not already in `docs/gbdt/V1.1_TBD.md`.
