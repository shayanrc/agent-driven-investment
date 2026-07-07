# V4.5.5 — Cross-experiment mechanism map

**Purpose.** Synthesize V4.5.1–4 + 6/7 into a single 15-anchor × failure-mechanism matrix, then map mechanisms to v5 candidate experiments. This is what v5 plans against.

Script: [`scripts/analog_mc/v4_5/mechanism_map.py`](../../../scripts/analog_mc/v4_5/mechanism_map.py) · Data: [`v4_5_5_mechanism_map.json`](../../../results/analog_mc/data/v4_5_5_mechanism_map.json)

## Mechanism definitions

| ID | Name | Diagnostic | Source |
|---|---|---|---|
| **M1** | Over-concentration | A2.1 top-1 prob ≥ 0.4 | V4.5.2 Mode 1 |
| **M2** | Bimodal mis-match | A2.1 top-2 ≥ 0.4 AND both forwards opposite-sign to realized | V4.5.2 Mode 2 |
| **M3** | Path-construction tightness | A2 cum-σ-growth < 0.7 × v24's | V4.5.6 |
| **M4** | Tail under-selection | v24 lift < 1 AND A2 lift < 1 | V4.5.7 |
| **M5** | B1 over-correction | B1 horizon drift magnitude > 10% | V4.5.3 |

## Per-anchor matrix

| Anchor | Real % | A2.Δ% | B1.Δ% | a2_top1 | a2_lift | v24_lift | σ-ratio | b1_drift% | **A2 mech** | **B1 mech** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| 1990-09-24 | +12.2 | +7.2 | **+156.1** | 0.118 | 1.21 | 1.97 | — | **+17.92** | — | **M5** |
| 1991-03-26 | −1.6 | +7.1 | +8.1 | 0.179 | 1.88 | 0.42 | 0.90 | +1.90 | — | — |
| 2000-04-03 | −7.5 | −19.6 | +3.1 | 0.197 | 0.87 | 0.99 | — | — | **M4** | **M4** |
| 2001-04-04 | +33.5 | +41.3 | +39.7 | 0.102 | 0.08 | 0.53 | 1.24 | +4.09 | **M4** | **M4** |
| 2001-10-02 | +38.6 | −47.8 | −16.4 | 0.164 | 0.73 | 0.00 | 1.37 | +13.14 | **M4** | **M5,M4** |
| **2008-10-03** | **−18.3** | **+122.2** | +8.1 | 0.288 | 0.69 | 6.56 | — | +2.02 | **M2** | — |
| 2010-04-23 | −10.4 | −47.0 | −23.9 | 0.096 | 1.54 | 0.36 | 1.24 | −3.70 | — | — |
| 2010-11-10 | +7.4 | +12.7 | −0.2 | 0.118 | 1.16 | 1.10 | 0.97 | — | — | — |
| 2012-03-14 | −5.5 | +27.0 | −51.1 | 0.272 | 0.70 | 0.87 | 0.89 | −7.75 | **M4** | **M4** |
| 2017-06-01 | +0.1 | +6.2 | +3.4 | 0.297 | 1.00 | 1.06 | 0.62 | — | **M3** | — |
| **2018-10-08** | **−12.7** | **+31.1** | −2.0 | **0.500** | 0.48 | 1.83 | — | — | **M1,M2** | — |
| **2020-03-16** | **+43.8** | −18.6 | +3.6 | **0.507** | 0.25 | 0.00 | 0.82 | — | **M1,M4** | **M4** |
| **2022-03-01** | **−14.7** | **+33.7** | +1.1 | 0.150 | 0.83 | 0.16 | **0.40** | — | **M3,M4** | **M4** |
| 2025-07-02 | +8.2 | −19.0 | +31.7 | 0.317 | 1.19 | 0.84 | — | −3.31 | — | — |
| 2026-02-19 | +17.5 | +13.1 | −2.1 | 0.168 | 1.30 | 0.81 | — | — | — | — |

## V5 candidate coverage of regressions

| Candidate | Mechanism it addresses | Cost | A2.1 regrs covered (of 10) | B1 regrs covered (of 5) |
|---|---|---|---:|---:|
| **V5.A.1** Tikhonov mix `d = (1-α)d_eu + α d_cw` | M1, M2 partial | 1 canonical w/ α-grid | 2 (2008, 2018) | 0 |
| **V5.A.2** Path-level ensemble (half v24 + half A2.1) | M1, M2, M3 partial | 1 canonical, no new code | 2 + halo on Mode-3 | 0 |
| **V5.A.3** Conditional corrwindow re-matching | M3 directly | ~250 LOC + tests + canonical | 2 (2017, 2022) | 0 |
| **V5.B** Feature augmentation: drawdown-depth | M4 | New feature + 4-D grid search | 3 (2001-04, 2012, 2022) | 1 (2001-04) |
| **V5.C** Delay-coordinate (Takens) distance | M4 | New distance + canonical | 3 (same as V5.B) | 1 (same) |
| **V5.D** B1 shrinkage parameter | M5 | 1 config knob | 0 (no A2.1 mech) | 1 (1990-09-24) |

## Reading

### Five "structural" anchors

The most informative subset is the intersection of A2.1 regressions × identified mechanism. Five anchors have **multiple confirmed mechanisms**, accounting for 4 of the 5 catastrophic regressions (>30% CRPS):

| Anchor | A2.Δ% | Mechanisms | Best v5 candidates |
|---|---:|---|---|
| **2008-10-03** | **+122%** | M2 (bimodal: 1998 + 1992 V-recovery analogs) | V5.A.2 / V5.A.1 (ensemble or mix) |
| **2018-10-08** | +31% | M1 (top-1 = 50%) + M2 | V5.A.2 / V5.A.1 |
| **2022-03-01** | +34% | M3 (σ-ratio 0.40) + M4 | V5.A.3 (cond. corrwindow) **AND** V5.B/C |
| **2012-03-14** | +27% | M4 | V5.B / V5.C |
| **2001-04-04** | +41% | M4 (both lifts < 1) | V5.B / V5.C |

These five regressions cover the cases where promotion-bar accounting is most affected (CRPS >25% regression). Two of three v5 ensemble candidates (V5.A.2, V5.A.1) plus V5.B/C *together* would address all five.

### Four "unclassified" A2.1 regressions

| Anchor | A2.Δ% | Diagnostic snapshot | Hypothesis |
|---|---:|---|---|
| 1990-09-24 | +7% | A2 lift 1.21, no M1/M2/M3 | Borderline noise. Within the 5% threshold's neighborhood. |
| 1991-03-26 | +7% | A2 lift 1.88, no signal | Borderline noise. |
| 2010-11-10 | +13% | Top-1 = 0.118 (diffuse), σ-ratio 0.97 | Path-construction near baseline; mean shift between A2 (+4.4%) vs v24 (+5.2%) is too small to be the mechanism. May be CRPS noise on a +7% rally that both matchers approximately capture. |
| 2026-02-19 | +13% | Top-1 = 0.168, lift 1.30 (good!) | A2 actually finds the tail (lift > 1), yet regresses. Mean fwd is similar to v24. Likely path-construction noise — A2 wins on tail-mass but loses on dispersion shape. |

**V5.A.2 (path-level ensemble)** is the only candidate that could address these via averaging — without a target mechanism, no specific lever. **Worth noting**: these four anchors regress *modestly* (7–13%). Together they contribute 4 to A2.1's "10/15 regressions" count. If V5.A.2 averages their CRPS halfway to v2.4's, three to four of them drop below the 5% threshold by construction.

### B1 regressions

- **1990-09-24 (+156%)**: M5 (over-correction). **V5.D (shrinkage)** directly targets this. Expected: shrinkage 0.3 reduces +17.9% drift to +5.4%, likely recovering most of the lost coverage.
- **2001-04-04 (+40%)**: M4 (tail under-selection). Same v5 fix as A2.1 — feature augmentation.
- **2001-10-02 (−16% — actually a WIN)**: M5 + M4. Drift +13% helps here; shrinkage would erode the win.
- **2008-10-03 (+8%)**: Small regression. No clean B1 mechanism (drift only +2%).
- **2025-07-02 (+32%)**: Drift −3.3%, wrong sign. Shrinkage helps marginally.
- **1991-03-26 (+8%)**: Drift +1.9%, mostly noise.

The **V5.D shrinkage tradeoff**: shrinks the +156% catastrophe (huge gain) at the cost of attenuating the +13% drift at 2001-10-02 (where the drift correction was actually right). Empirical question whether the bar passes.

## Aggregated v5 forecast

**Optimistic prediction** (assuming all v5 candidates land their stated coverage):

| Scenario | A2.1 regrs reduced from 10 → | Failures recovered (of 5) | New regrs (off-target) | Promotion bar? |
|---|---:|---:|---:|---|
| V4 A2.1 alone (current) | 10 | 2 | (baseline) | ❌ |
| V5.A.2 path-ensemble | 6–7 | 3–4 (incl. preserved 2010-04-23) | small | maybe |
| V5.A.2 + V5.B drawdown feat. | 3–4 | 3–4 | risk: new feature → search instability | likely |
| V5.A.2 + V5.A.3 + V5.B | 1–2 | 4–5 | possible α-search instability | yes |
| Full stack (A.1 + A.2 + A.3 + B + D) | 0–1 | 5/5 | combinatorial blowup | uncertain |

The cheapest path that *plausibly* passes the bar is **V5.A.2 + V5.B**. V5.A.3 is a 2x cost addition (~250 LOC). V5.D is a tiny config knob worth folding in regardless.

## Recommended v5 sequencing

In priority/cost order:

1. **V5.A.2 — Path-level ensemble.** Cheapest, no new code (just merge two existing canonical runs' paths). 1-day implementation, 1 canonical-equivalent compute (already have B1 + v24 + A2.1 cached; just need ensemble forecasts.npz). Most likely to pass bar by itself.

2. **V5.B — Drawdown-depth feature augmentation.** Adds `drawdown_60d_norm` as a 4th feature. Requires extending the search grid from 3-D to 4-D. Targets Mode-4 anchors (2001-04, 2012, 2020, 2022).

3. **V5.D — B1 shrinkage.** Lightest of all — single config knob. Bundled with V5.B if V5.A.2 + V5.B doesn't pass alone.

4. **V5.A.3 — Conditional corrwindow.** Most ambitious. Only run if V5.A.2 + V5.B doesn't pass — addresses the residual M3 regressions (2017, 2022).

5. **V5.A.1 (Tikhonov), V5.C (delay coords).** Held in reserve. V5.A.1 is largely subsumed by V5.A.2. V5.C is an alternative to V5.B if drawdown feature underperforms.

## What's still uncertain

- **V5.A.2's interaction with A2.1's wins**: ensemble averages CRPS — wins (2010-04-23, 2001-10-02, 2020-03-16) get half-attenuated. Need to verify the ensemble preserves enough of those wins to keep the failure-recovery count.
- **Drawdown-feature search resolution**: 4-D grid at the current 0.1 resolution = 11⁴ = 14,641 weight combinations per fold. May need coarser resolution or restricted region. Could blow up compute budget by 4×.
- **The 4 unclassified regressions** (1990-09-24, 1991-03-26, 2010-11-10, 2026-02-19): no mechanism, no targeted fix. Risk of being persistent regression noise in v5.

## Verdict and next step

The v5 experiment plan is now **mechanism-aware**: each candidate experiment is tied to specific failure modes seen in v4. The leading candidates are V5.A.2 (path ensemble) and V5.B (drawdown feature). Both are short-build experiments that together cover the bulk of v4's regressions.

Next step: V4_5_RESULTS.md synthesizes V4.5.1–7 into the v5 plan structure, and V5_EXPERIMENTS_PLAN.md formally specs V5.A.2 + V5.B + V5.D as the v5 canonical sequence with V5.A.3 as a stretch goal.
