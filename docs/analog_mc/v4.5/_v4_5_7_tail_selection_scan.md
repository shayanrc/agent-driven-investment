# V4.5.7 — Tail-positive selection generality

**Question.** V4.5.4 showed that at 2020-03-16, both v2.4 and A2.1 assign less-than-uniform probability mass to same-side tail analogs (lift < 1). Does this hold at the other 14 fat-tail anchors? Are the two matchers' weaknesses **correlated** (suggesting a structural feature gap) or **complementary** (suggesting ensemble is the answer)?

**Method.** For each anchor, compute `lift = mass(same-sign-tail with |realized_60d| ≥ |anchor realized|) / (count_in_tail / n_eligible)`. Lift < 1 means the matcher avoids the right-direction tail relative to uniform random selection. Lift > 1 means over-selection.

Script: [`scripts/analog_mc/v4_5/tail_selection_scan.py`](../../../scripts/analog_mc/v4_5/tail_selection_scan.py) · Data: [`v4_5_7_tail_selection_scan.json`](../../../results/analog_mc/data/v4_5_7_tail_selection_scan.json)

## Per-anchor lift table

| Anchor | Real % | v24 lift | A2 lift | v24 mass | A2 mass | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 1991-03-26 | −1.6 | 0.42 | **1.88** | 11.5% | 51.7% | A2 finds tail |
| 2010-04-23 | −10.4 | 0.36 | **1.54** | 4.6% | 19.6% | A2 finds tail |
| 2010-11-10 | +7.4 | **1.10** | **1.16** | 41.3% | 43.4% | Both find tail |
| 2012-03-14 | −5.5 | 0.87 | 0.70 | 17.6% | 14.1% | **Neither** |
| 2025-07-02 | +8.2 | 0.84 | **1.19** | 27.3% | 38.6% | A2 finds tail |
| 1990-09-24 | +12.2 | **1.97** | **1.21** | 36.5% | 22.3% | Both find tail |
| 2001-04-04 | +33.5 | 0.53 | 0.08 | 1.6% | 0.2% | **Neither** |
| 2001-10-02 | +38.6 | **0.00** | 0.73 | **0.00%** | 1.5% | **Neither** |
| 2000-04-03 | −7.5 | ≈1.00 | 0.87 | 9.8% | 8.7% | Both ≈ uniform |
| 2008-10-03 | −18.3 | **6.56** | 0.69 | 43.1% | 4.5% | **v24 only** |
| 2017-06-01 | +0.1 | 1.06 | 1.00 | 72.2% | 67.9% | Both find (trivial) |
| 2018-10-08 | −12.7 | **1.83** | 0.48 | 14.4% | 3.8% | **v24 only** |
| 2020-03-16 | +43.8 | **0.00** | 0.25 | **0.00%** | 0.1% | **Neither** |
| 2022-03-01 | −14.7 | 0.16 | 0.83 | 1.0% | 5.3% | **Neither** |
| 2026-02-19 | +17.5 | 0.81 | **1.30** | 7.2% | 11.4% | A2 finds tail |

## Three structural cohorts

**Cohort 1 — At least one matcher finds the tail (10/15).** v2.4 *and* A2.1 have complementary strengths: A2.1 covers V-recovery anchors (1991-03-26, 2010-04-23, 2025-07-02, 2026-02-19), v2.4 covers negative-extremity anchors (2008-10-03 lift 6.56!, 2018-10-08 lift 1.83). Both find easy cases (2010-11-10, 1990-09-24, 2017-06-01).

This is the **strongest possible argument for ensemble** of the two distances. The complementarity is direct: where v2.4 wins, A2.1 fails (2008-10-03: 6.56 vs 0.69); where A2.1 wins, v2.4 fails (1991-03-26: 0.42 vs 1.88). A path-level average of forecasts from both would inherit each anchor's stronger matcher.

**Cohort 2 — Neither matcher finds the tail (5/15).** These are the structural fail cases:
- 2012-03-14 (−5.5%): both lifts ~0.7–0.9, dispersion not concentrated enough.
- 2001-04-04 (+33.5%): v2.4 lift 0.53, A2 lift 0.08. The +33%+ analogs exist (the post-LTCM 1998 windows) but neither matcher selects them.
- 2001-10-02 (+38.6%): v2.4 puts **zero** mass on +38%+ analogs; A2.1 only 1.5%. Yet A2.1 wins CRPS at this anchor — because v2.4 is so bad (0.00%) that anything ≥ 0 is an improvement, not because A2.1 is good.
- 2020-03-16 (+43.8%): zero v2.4 mass, 0.14% A2.1 mass. The COVID case from V4.5.4.
- 2022-03-01 (−14.7%): v2.4 lift 0.16, A2.1 lift 0.83. Negative-side analog of the COVID case.

The pattern in Cohort 2: **extreme realized magnitudes** (|real| > 13%) where no current feature representation aligns the target with the historical V-recovery / capitulation regime. These need a new feature (drawdown depth, regime indicator, multi-L corrwindow).

**Cohort 3 — Trivially easy (2017-06-01)** where realized is near zero and most of the pool is "same side." Not informative.

## V5 implications

This is the most actionable v4.5 finding to date. Two clear v5 candidates:

### V5.A — **Distance ensemble (Tikhonov mix or path-level average)**

Given the complementarity, an ensemble combining v2.4 weighted-Euclidean and A2.1 corrwindow distances should rescue **most of Cohort 1** without re-introducing v4's regressions. Two implementation options:

1. **Tikhonov-mixed distance** with grid-searched α: `d = (1-α) d_eu + α d_cw`, search α ∈ {0, 0.25, 0.5, 0.75, 1.0} per fold. Single canonical run.
2. **Path-level ensemble**: run forecast() twice with each matcher; concatenate the path arrays. Same n_paths total (e.g., 500 + 500), each contributing half. Embarrassingly simple, no search needed.

Path-level ensemble is **cheaper and more diagnostic-friendly** — separates "matcher choice" from "blend coefficient" and lets v5.A.1 / v5.A.2 attribute regressions to specific matchers. **My recommendation: V5.A.2 first as a 6-hour pilot, then V5.A.1 if the pilot motivates the Tikhonov search.**

### V5.B — **Feature augmentation for Cohort 2**

Add a new feature `recent_drawdown_60d_normalized = (close[t] − min(close[t-60:t])) / std(returns[t-60:t])`. This captures the "extreme drawdown velocity" signature that 2020-03-16, 2001-10-02, 2022-03-01 all share but neither z-scores nor corrwindow capture. Weighted-Euclidean over (z_20, z_50, z_200, drawdown_norm) is a 4-feature distance — minor change to the existing pipeline. Search now has 4-weight grid instead of 3 (the v2.4 search infrastructure assumes 3; would need to extend `weight_grid_resolution` × 4D, possibly with reduced resolution).

This is bigger — modifies the feature pipeline — but **directly targets the 5 anchors in Cohort 2**, which include the COVID anchor that all v4 experiments missed.

### V5.C — **Delay-coordinate distance (B2 from v4 backlog)**

Cohort 2 might also be addressable by Takens-embedded matching: `embed(returns, dim=5, tau=10)` and match in 5-D space. Captures momentum + reversal in one shot. Originally deferred at v4 in favor of A2.1; V4.5.7 + V4.5.4 strengthen the case to revisit.

## Verdict

**Complementarity confirmed.** Ensemble is the cleanest v5 lever — addresses 10/15 anchors with minimal new code. The remaining 5 anchors need either feature augmentation or delay-coordinate distance.

The v5 plan should sequence: **V5.A.2 (path-level ensemble) first** → if successful, **V5.A.1 (Tikhonov)** to refine → in parallel, prototype **V5.B (drawdown feature)** for the Cohort-2 anchors.
