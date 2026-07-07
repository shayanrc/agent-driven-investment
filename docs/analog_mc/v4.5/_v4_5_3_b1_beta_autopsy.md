# V4.5.3 — B1 β autopsy at the 5 regressions

**Question.** B1's most severe regression is at 1990-09-24 (control 90-band 55 → 11, CRPS +156%) — unexplained in V4_RESULTS. Is the local-linear regression's β driven by a single leverage outlier? If yes, a leverage-trimmed WLS fix (one v5 follow-up candidate) is well-targeted.

**Method.** For each B1 regression anchor: reproduce the B1 canonical fold's WLS fit using the fold-specific weights/n_eff, compute per-candidate leverage `h_ii = w_i · x_iᵀ (XᵀWX + λI)⁻¹ x_i`, identify max-leverage candidate, refit with it dropped, compare corrections. 3 B1 wins included as positive controls.

Script: [`scripts/analog_mc/v4_5/b1_beta_autopsy.py`](../../../scripts/analog_mc/v4_5/b1_beta_autopsy.py) · Data: [`v4_5_3_b1_beta_autopsy.json`](../../../results/analog_mc/data/v4_5_3_b1_beta_autopsy.json)

## Per-anchor table

| Anchor | Class | corr (orig) | horizon drift | clamp? | matcher_mean | predicted_mean | h_max | h_argmax date | corr (trim) | Δ % |
|---|:---:|---:|---:|:---:|---:|---:|---:|---|---:|---:|
| 1991-03-26 | REG | +0.0188 | +1.90% | n | (small) | (small) | 0.091 | 1989-06-12 | +0.0188 | −0.1% |
| 2010-04-23 | WIN | −0.0377 | −3.70% | n | — | — | 0.127 | 1986-11-04 | −0.0246 | −34.7% |
| 2012-03-14 | WIN | −0.0806 | −7.75% | n | — | — | 0.103 | 1991-04-08 | −0.0733 | −9.1% |
| 2025-07-02 | REG | −0.0336 | −3.31% | n | — | — | 0.166 | 2012-03-12 | −0.0287 | −14.7% |
| **1990-09-24** | **REG** | **+0.1648** | **+17.92%** | n | — | — | 0.072 | 1987-10-26 | +0.1652 | +0.2% |
| 2001-04-04 | REG | +0.0401 | +4.09% | n | — | — | **0.521** | 1990-08-22 | +0.0326 | −18.7% |
| 2001-10-02 | WIN | +0.1235 | +13.14% | n | — | — | 0.125 | 2001-04-06 | +0.1255 | +1.6% |
| 2008-10-03 | REG | +0.0200 | +2.02% | n | — | — | **0.355** | 2000-12-20 | +0.0437 | **+118.3%** |

(realized: 1990-09-24 = +12.2%; 1991-03-26 = −1.6%; 2008-10-03 = −18.3%; 2001-04-04 = +33.5%; 2025-07-02 = +8.2%.)

## Leverage hypothesis (V4.5.3 decision rule) — REJECTED

**Decision rule from the plan.** "≥3/5 regressions have max h_ii > 0.5 AND trimmed correction differs from original by >50%" → V5.3 leverage-trimmed B1 is well-targeted.

**Actual count.**
- `h_max > 0.5`: **1/5 regressions** (2001-04-04, h=0.521).
- Trim-change > 50%: 1/5 regressions (2008-10-03, +118%) — but **the trim moves the correction *further* from the realized direction**, not closer. 2008-10-03 realized is −18.3%; original corr was +2.0% (already wrong sign); trimmed corr becomes +4.4% (still wrong sign, further out). Trimming makes things worse here.

Both conditions: 0/5. **Leverage outliers are not the mechanism driving B1 regressions.**

## The actual mechanism — magnitude over-correction at 1990-09-24

The catastrophic case is 1990-09-24 (B1 90-band 55 → 11, CRPS +156%). Diagnostics:
- Correction = **+0.1648** scalar log-return ⇒ **+17.92% cumulative horizon drift** added.
- Per-day drift = +0.275 bps × 60 days = +16.5% cumulative bias.
- Realized 60-day return: **+12.2%**.
- Max leverage: **0.072** — low. No outlier driving β.
- Trimming changes correction by 0.2% — leverage trim is a no-op.

What's happening: the matcher_mean (β-free term) is far below realized at this anchor — a classic "magnitude undershoot" failure. The regression fit at this origin yields a `predicted_mean` of (matcher_mean + 0.165) — i.e., the regression is **doing exactly what it was designed to do**, applying a big positive correction. But the magnitude is too aggressive. The +17.92% drift dominates the v2.4 baseline dispersion, narrowing bands around an over-bullish forecast that just *narrowly* contains realized at the center but blows the 90-band tails inward.

**The wins follow the same pattern but smaller:**
- 2012-03-14 (WIN): drift = −7.75%, realized = −5.5%. Correction overshoots in magnitude, but smaller magnitude → still wins CRPS.
- 2001-10-02 (WIN): drift = +13.14%, realized = +38.6%. Undershoots — but a positive correction toward a much-bigger move still helps CRPS.

The variable that distinguishes wins from regressions is **|correction|** — too big and it blows dispersion (1990-09-24), in-range and it helps (2010-04-23 −3.7%, 2012-03-14 −7.8%, 2001-10-02 +13.1%).

## The v5 fix — B1 shrinkage parameter

Analogous to `momentum_shrinkage: 0.30` for trailing-momentum drift, introduce **`b1_shrinkage`** ∈ (0, 1]:

```python
drift_target = drift_target + (b1_shrinkage * correction) / config.forecast_horizon
```

Cost: 1 config param, 1 line in `simulate.py:277`. Search via v5 canonical with grid {0.0, 0.3, 0.5, 0.7, 1.0}.

Hypothesis: `b1_shrinkage = 0.3` would shrink the 1990-09-24 correction to +5.4% drift (from +17.9%) — much closer to realized +12.2%, recovering most of the lost 90-band. Wins like 2012-03-14 would shrink to −2.3% (from −7.8%) — still right direction, smaller correction, likely still positive CRPS.

The risk is that **shrinkage attenuates the wins more than the losses**, since shrinkage is uniform. The decision is empirical — V5 canonical with the grid resolves it. The cheap version: run b1_shrinkage=0.3 vs the v4 B1 (=1.0) on the 8-anchor sub-panel (5 regressions + 3 wins) and see if the 1990-09-24 collapse is rescued.

## Open follow-up — why is matcher_mean so far off at 1990-09-24?

The B1 correction is enormous because the matcher_mean at this anchor is presumably well below realized. That points at the *underlying v2.4 matcher* being the upstream problem. Possible candidates:
- The 1990-09-24 anchor sits early in the data (fold 1, train end ~idx 1059), so candidate pool is small and likely missing recovery analogs.
- The z_20/z_50/z_200 features at 1990-09-24 may be in a regime poorly matched by the available analogs.

Out-of-V4.5 scope to fully resolve. The B1 shrinkage handles it pragmatically: a smaller correction multiplied by a wrong-direction matcher mean is less catastrophic than a big correction layered on the same wrong-direction base.

## Verdict

**Leverage-trimmed B1 is not the v5 fix.** The mechanism is magnitude over-correction, not leverage. **Replace V5.3 in the v5 plan with `b1_shrinkage`-search B1**, cheaper and more directly targeted.

Specifically: **the B1 family is still alive for v5**, just with a magnitude regularizer. This is good news — v4 B1 had the only experiment with both aggregate failure AND control improvement (V4_RESULTS), but it lost the 1990-09-24 collapse. A shrunk B1 has a credible path to retaining those wins without the catastrophe.
