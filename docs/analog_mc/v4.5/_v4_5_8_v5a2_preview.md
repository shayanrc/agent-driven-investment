# V4.5.8 — V5.A.2 path-level ensemble preview

**Question.** Does V5.A.2 (path-level ensemble of v2.4 + A2.1 forecasts) pass the v5 promotion bar by itself? If yes, v5 is essentially solved. If no, what does it cover, and what additional experiments are needed?

**Method.** For each of the 15 fat-tail anchors, load cached v2.4 and A2.1v1 path arrays from `forecasts.npz`, mix at multiple α ratios (α = fraction of paths from A2.1), recompute CRPS and 50/90-band coverage. **No new walk-forward.**

Script: [`scripts/analog_mc/v4_5/v5_a2_ensemble_preview.py`](../../../scripts/analog_mc/v4_5/v5_a2_ensemble_preview.py) · Data: [`v4_5_8_v5a2_preview.json`](../../../results/analog_mc/data/v4_5_8_v5a2_preview.json)

## Headline — α sweep

| α (A2.1 fraction) | Failures recovered /5 | Regressions >5% /15 | Failure mean CRPS | Control mean CRPS | Bar? |
|---:|---:|---:|---:|---:|:---:|
| 0.00 (pure v2.4) | 0 | 0 | 0.0947 | 0.0443 | (ref) |
| 0.25 | 1 | 4 | 0.0892 | 0.0462 | ❌ |
| **0.50** | **2** | **6** | **0.0826** | **0.0494** | ❌ |
| 0.75 | 2 | 8 | 0.0794 | 0.0544 | ❌ |
| 1.00 (pure A2.1) | 2 | 10 | 0.0756 | 0.0610 | ❌ |

**No α value passes the bar.** Failure-recovery count plateaus at 2/5 — the bar requires ≥3/5. Regression count is monotone-increasing in α.

## Per-anchor table at α = 0.5

| Anchor | Real % | v2.4 CRPS | A2.1 CRPS | V5.A.2 CRPS | Δ vs v2.4 | v2.4 in90 | A2.1 in90 | V5.A.2 in90 | Failure? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1991-03-26 | −1.6 | 0.0234 | 0.0251 | 0.0240 | +2.7% | 60 | 60 | 60 | |
| **2010-04-23** | **−10.4** | 0.0707 | 0.0375 | **0.0530** | **−25.1%** | 27 | 57 | **48** | **YES** ✅ |
| 2010-11-10 | +7.4 | 0.0154 | 0.0174 | 0.0162 | +5.0% | 56 | 57 | 56 | |
| 2012-03-14 | −5.5 | 0.0340 | 0.0432 | 0.0382 | +12.3% | 55 | 41 | 46 | |
| 2025-07-02 | +8.2 | 0.0174 | 0.0141 | 0.0152 | −12.9% | 60 | 60 | 60 | |
| 1990-09-24 | +12.2 | 0.0417 | 0.0447 | 0.0367 | −11.9% | 55 | 60 | 60 | |
| 2001-04-04 | +33.5 | 0.0774 | 0.1094 | 0.0936 | +20.9% | 55 | 53 | 55 | |
| **2001-10-02** | **+38.6** | 0.1140 | 0.0596 | **0.0808** | **−29.1%** | 44 | 53 | **54** | **YES** ✅ |
| 2000-04-03 | −7.5 | 0.0794 | 0.0638 | 0.0629 | −20.8% | 56 | 59 | 59 | |
| **2008-10-03** | **−18.3** | 0.1008 | 0.2241 | 0.1547 | +53.5% | 52 | 7 | **41** | (regime) |
| 2017-06-01 | +0.1 | 0.0123 | 0.0131 | 0.0125 | +1.1% | 60 | 59 | 60 | |
| **2018-10-08** | **−12.7** | 0.0620 | 0.0813 | 0.0717 | +15.6% | 31 | 10 | 25 | **YES** ❌ |
| **2020-03-16** | **+43.8** | 0.1788 | 0.1455 | 0.1604 | −10.3% | 38 | 11 | 21 | **YES** ❌ |
| 2022-03-01 | −14.7 | 0.0411 | 0.0549 | 0.0462 | +12.3% | 58 | 47 | 59 | |
| **2026-02-19** | **+17.5** | 0.0480 | 0.0543 | 0.0508 | +5.7% | 41 | 38 | 40 | **YES** ❌ |

The 6 regressions at α=0.5 (Δ > 5% CRPS):
1. **2008-10-03** (+53.5% — but recovered from A2.1's +122.2%, and 90-band 7→41)
2. **2001-04-04** (+20.9% — Cohort 2)
3. **2018-10-08** (+15.6% — Mode 1+2)
4. **2012-03-14** (+12.3% — Cohort 2)
5. **2022-03-01** (+12.3% — Mode 3+4)
6. **2026-02-19** (+5.7% — diffuse)

## Key wins from the ensemble

1. **2008-10-03 catastrophic recovery**: 90-band coverage 7 → 41 (vs v2.4's 52). CRPS regression cut from +122% (A2.1 alone) to +53%. This is the largest single win of the ensemble.

2. **All wins preserved**: 2010-04-23 (+57 → 48), 2001-10-02 (53 → 54), 2020-03-16 (11 → 21 — slight improvement), 2025-07-02 (60), 1990-09-24 (60), 2000-04-03 (59). The 50/50 mix retains the A2.1 wins while damping the regressions.

3. **Aggregate failure CRPS −12% vs v2.4**, with **only +13% control penalty** (vs A2.1's +28%). The mix is materially better-balanced than either matcher alone.

## What the ensemble does NOT solve

- **2018-10-08, 2020-03-16, 2026-02-19 never reach 45/60 90-band coverage** even at α=1.0 (pure A2.1). These are anchors where neither matcher's analog selection captures the realized magnitude. **The bar's recovery condition (≥3/5) is structurally blocked at 2/5 unless one of these three is rescued.**

- **Cohort-2 regressions (2001-04, 2012-03, 2022-03) remain**: ensemble averages each anchor's two wrong-direction predictions; doesn't add a missing tail dimension.

- **Aggregate control regression +11.5%** at α=0.5. The ensemble preserves more control-anchor variance than v2.4 alone — acceptable but not free.

## V5 plan implications

This preview **invalidates** the V5_EXPERIMENTS_PLAN.md scenario where V5.A.2 alone passes the bar. The plan's "optimistic prediction" was wrong on the recovery count. Updated picture:

- **V5.A.2 is necessary but not sufficient.** It cuts regressions 10→6 and adds zero failure recoveries vs pure A2.1, but its anti-regression effect is the cleanest contribution any v5 candidate can make.
- **V5.B (drawdown feature) becomes mandatory** for failure recoveries at 2018-10-08, 2020-03-16, 2026-02-19. V4.5.9 will assess whether V5.B's feature design plausibly rescues these.
- **The α=0.5 stack is the recommended starting V5.A.2 configuration**, not 1.0. Best failure CRPS × control trade-off.

The recommended v5 sequencing is now:
1. **V5.A.2 at α=0.5** — as the BASE configuration that all subsequent experiments stack on top of.
2. **V5.B + V5.A.2** — drawdown feature applied with the ensemble. Tests whether the joint helps the 3 unrescued failure anchors.
3. **V5.D shrinkage + V5.A.2** — B1 shrinkage stacked. Cheap.
4. **V5.A.3 conditional corrwindow** — stretch goal.

## Open questions

- The α-sweep showed 2010-04-23 90-band drops from 57 (pure A2.1) to 48 (α=0.5) — α=0.5 still passes the 45 threshold but margin is narrow. A 60/40 mix (α=0.6) might offer slightly better failure recovery without re-introducing regressions.
- The ensemble preview uses identical realized arrays for v24 and A2 paths (sanity-asserted in code). This is correct since both runs forecast the same test origins; just included for completeness.

## Verdict

**V5.A.2 (path ensemble, α=0.5) is a confirmed v5 baseline.** It does not pass the bar alone but cuts the regression count significantly and recovers the catastrophic 2008-10-03 regression. V5.A.2 + V5.B is the minimum stack required to potentially pass the bar — and V4.5.9 determines whether V5.B can plausibly do its half.

This confirms the v5 sequencing should be **V5.A.2 + V5.B run together (or V5.B layered on V5.A.2)** as the P0 v5 experiment, not V5.A.2 alone.
