# V4.5.1 — A2.1 gate-signal validation

**Question.** Does `corrwindow val_crps > k × cross-fold median` discriminate A2.1's catastrophic folds (regime-coverage regressions) from its wins (failure-anchor recoveries)?

**Method.** Read per-fold val_crps from A2.1v1 canonical (`runs/analog_mc/20260521T061730Z`, 76 folds). Threshold-sweep k ∈ {1.2, 1.5, 2.0, 3.0}. For each fat-tail anchor, identify its fold; if the fold is "gated" (val_crps > threshold), substitute v2.4's cell into the hypothetical V5.1 panel. Score against the v4 promotion bar (≥3/5 failures recovered AND ≤2/15 regressions >5%).

Script: [`scripts/v4_5/validate_gate_signal.py`](../../../scripts/v4_5/validate_gate_signal.py) · Data: [`v4_5_1_gate_signal.json`](../../../results/analog_mc/data/v4_5_1_gate_signal.json)

## Headline

| k | threshold val_crps | folds gated | anchors gated | failures recovered /5 | regressions >5% /15 | promotion bar |
|---:|---:|---:|---:|---:|---:|:---|
| 1.2 | 0.05537 | 25 | 8 | **0** | 6 | ❌ |
| 1.5 | 0.06921 | 16 | 6 | **1** | 7 | ❌ |
| 2.0 | 0.09228 | 12 | 5 | **1** | 7 | ❌ |
| 3.0 | 0.13841 | 6 | 4 | **1** | 8 | ❌ |

Reference: A2.1 alone is **2/5 recovered, 10/15 regress**. v2.4 alone is **0/5 recovered, 0 regress (definitionally)**.

No threshold passes the bar. Worse: at all thresholds the gate *loses* failure recoveries that A2.1 actually had, because the catastrophic-regression folds and the success folds overlap in val_crps space.

Val CRPS stats (A2.1v1): median 0.04614, mean 0.06237, min 0.01858, max 0.35653.

## Per-anchor disposition at k=1.5 (the V4_RESULTS recommendation)

| Anchor | Fold | val_crps | Gated? | A2.1 Δ CRPS | V5.1 Δ CRPS | A2.1 in90 | V5.1 in90 | Failure? |
|---|---:|---:|:---:|---:|---:|---:|---:|:---:|
| 1991-03-26 | 2 | 0.060 | no | +7.1% | +7.1% | 60 | 60 | |
| **2010-04-23** | 42 | 0.068 | no | **−47.0%** ✅ | **−47.0%** ✅ | 57 | 57 | **YES** |
| 2010-11-10 | 43 | 0.052 | no | +12.7% | +12.7% | 57 | 57 | |
| 2012-03-14 | 46 | 0.036 | no | **+27.0%** ❌ | **+27.0%** ❌ | 41 | 41 | |
| 2025-07-02 | 74 | 0.047 | no | −19.0% | −19.0% | 60 | 60 | |
| 1990-09-24 | 1 | 0.097 | yes | +7.2% | 0% | 60 | 55 | |
| 2001-04-04 | 23 | 0.292 | yes | +41.3% | 0% | 53 | 55 | |
| **2001-10-02** | 24 | 0.357 | **yes** | **−47.8%** ✅ | **0%** ❌ | 53 | 44 | **YES** |
| 2000-04-03 | 21 | 0.146 | yes | −19.6% | 0% | 59 | 56 | |
| **2008-10-03** | 39 | 0.207 | **yes** | **+122.2%** ✅gated | **0%** | 7 | 52 | |
| 2017-06-01 | 57 | 0.025 | no | +6.2% | +6.2% | 59 | 59 | |
| **2018-10-08** | 60 | 0.036 | no | **+31.1%** ❌ | **+31.1%** ❌ | 10 | 10 | **YES** |
| **2020-03-16** | 63 | 0.078 | **yes** | −18.6% | 0% | 11 | 38 | **YES** |
| 2022-03-01 | 67 | 0.055 | no | +33.7% | +33.7% | 47 | 47 | |
| **2026-02-19** | 75 | 0.026 | no | +13.1% | +13.1% | 38 | 38 | **YES** |

## Reading

**The gate is decoupled from the test failures.** Folds with high val_crps tend to be those whose **val window itself contains a fat-tail event** — and ironically, those are also the folds where A2.1 either wins big (2001-10-02) or fails big (2008-10-03). The gate cannot distinguish the two from val_crps alone.

Two structural reasons:

1. **Val_crps is dominated by val-window volatility, not by corrwindow's calibration quality.** A val window straddling a 2001-style V-recovery produces high val_crps for *every* model — A2.1, B1, even v2.4. The gate fires on regime-volatility presence, not on matcher pathology.
2. **The catastrophic regressions concentrate at low-val_crps regime-coverage anchors** (2018-10-08 val_crps 0.036, 2022-03-01 val_crps 0.055, 2012-03-14 val_crps 0.036, 2026-02-19 val_crps 0.026 — all below the median 0.046). These are anchors where corrwindow confidently picks shape-similar windows whose forwards diverge — exactly the V4_RESULTS "shape-similar wrong-forward" mode — but the val window preceding the test slice was *calm*, giving misleadingly low val_crps.

The V4_RESULTS V5.1 design is **invalidated**. The gate-signal must come from a different observable.

## Decision: drop val_crps gate; promote V4.5.2's temporal-cluster signal as the candidate

The natural replacement is a **per-anchor concentration signal** computed at inference time: if corrwindow's top-K analogs cluster heavily in a single year (Herfindahl H > 0.4), the matcher is over-confidently pulling from one regime. That's the V4.5.2 hypothesis and the next investigation.

Lower-priority but possible: a 2-D gate combining val_crps with a separate "shape-confidence" metric (e.g., top-K distance entropy at the test origin). Deferred unless V4.5.2 also fails.

## Open follow-up

- **k=1.2 misses ALL failures** because A2.1's biggest wins (2010-04-23 val_crps 0.068, 2001-10-02 val_crps 0.357) sit at very different val_crps levels; no single threshold contains both. Confirms the signal is genuinely not monotonic.
- The 4 failures-not-recovered at k=1.5 are: 2010-04-23 (not gated, retained as A2.1 win ✅), 2001-10-02 (gated, lost win ❌), 2018-10-08 (not gated, retained as A2.1 regression ❌), 2026-02-19 (not gated, retained as A2.1 regression ❌). The arithmetic of "1 retained win + 1 inadvertent re-pull to baseline-passing" yields the single recovery at k=1.5.

## Verdict

**val_crps is the wrong gate signal.** No v5 experiment should be designed around it. Move to V4.5.2 for the temporal-clustering alternative.
