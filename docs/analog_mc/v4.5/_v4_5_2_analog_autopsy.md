# V4.5.2 — A2.1 analog autopsy at the 15 fat-tail anchors

**Question.** V4_RESULTS attributes A2.1's 10 regressions to "shape-similar wrong-forward" — corrwindow confidently selects historical windows that look like the target but whose realized forwards diverge. Does temporal clustering of the top-K analogs actually discriminate A2.1 wins from regressions, and would a cluster-based gate work better than `val_crps` (which V4.5.1 ruled out)?

**Method.** For each of the 15 fat-tail anchors, reproduce the matcher's probability layer under A2.1 (corrwindow L=100, n_eff=50) AND v2.4 (weighted-Euclidean with each anchor's fold-specific weights/n_eff). Compute on top-20 analogs:
- **Year-Herfindahl** H = Σ p²_year (concentration in single year)
- **Top-1 / Top-3 cumulative probability mass** (concentration in few analogs)
- **Mass-weighted 60-day forward return** vs realized
- **|Δfwd|** = absolute disagreement between A2.1 and v2.4 mass-weighted forwards (an ensemble-disagreement signal)

Script: [`scripts/v4_5/analog_autopsy_a2.py`](../../../scripts/v4_5/analog_autopsy_a2.py) · Data: [`v4_5_2_analog_autopsy.json`](../../../results/analog_mc/data/v4_5_2_analog_autopsy.json)

## Headline per-anchor table

(WIN = A2.1 CRPS Δ < −5% vs v2.4; REG = A2.1 CRPS Δ > +5%. Both bands per V4.5.1's classification.)

| Anchor | Class | Real % | A2.wfwd | v24.wfwd | \|Δfwd\| | A2.top1 | A2.top3 | A2.H | v24.H |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1991-03-26 | REG | −1.6 | −2.7 | +7.0 | 9.7 | 0.179 | 0.346 | 0.309 | 0.487 |
| 2010-04-23 | **WIN** | −10.4 | +1.8 | +3.5 | 1.7 | 0.096 | 0.274 | 0.129 | 0.304 |
| 2010-11-10 | REG | +7.4 | +4.4 | +5.2 | 0.7 | 0.118 | 0.300 | 0.101 | 0.169 |
| 2012-03-14 | REG | −5.5 | +8.3 | +2.7 | 5.5 | 0.272 | 0.415 | 0.198 | 0.103 |
| 2025-07-02 | **WIN** | +8.2 | +5.7 | +4.3 | 1.4 | 0.317 | 0.446 | 0.275 | 0.087 |
| 1990-09-24 | REG | +12.2 | +2.6 | +9.2 | 6.6 | 0.118 | 0.303 | 0.397 | 0.689 |
| 2001-04-04 | REG | +33.5 | +8.0 | +14.1 | 6.1 | 0.102 | 0.266 | 0.154 | 1.000 |
| 2001-10-02 | **WIN** | +38.6 | +6.5 | +11.4 | 4.9 | 0.164 | 0.345 | 0.138 | 0.737 |
| 2000-04-03 | **WIN** | −7.5 | +5.3 | +4.0 | 1.3 | 0.197 | 0.346 | 0.203 | 0.263 |
| **2008-10-03** | **REG** | −18.3 | **+12.4** | −9.8 | **22.2** | 0.288 | 0.550 | 0.292 | 0.328 |
| 2017-06-01 | REG | +0.1 | +5.8 | +5.3 | 0.5 | 0.297 | 0.450 | 0.241 | 0.159 |
| 2018-10-08 | REG | −12.7 | +4.2 | +2.1 | 2.1 | **0.500** | **0.555** | **0.600** | 0.150 |
| 2020-03-16 | **WIN*** | +43.8 | +0.5 | +5.4 | 5.0 | **0.507** | **0.569** | **0.650** | 0.199 |
| 2022-03-01 | REG | −14.7 | +0.8 | +4.5 | 3.6 | 0.150 | 0.353 | 0.142 | 0.191 |
| 2026-02-19 | REG | +17.5 | +3.8 | +4.7 | 1.0 | 0.168 | 0.443 | 0.156 | 0.103 |

\* 2020-03-16 is a "win" only on CRPS; the 90-band coverage collapses 38→11. Treat as effectively a regression for gating purposes.

## Signal discrimination

Range overlap is severe for every univariate signal:

| Signal | WIN range | REG range | Cleanly separates? |
|---|---|---|---|
| Year-Herfindahl H | [0.129, **0.650**] | [0.101, **0.600**] | No — wins and regs span same range |
| Top-1 probability | [0.096, **0.507**] | [0.102, **0.500**] | No |
| Top-3 probability | [0.274, **0.569**] | [0.266, **0.555**] | No |
| \|Δfwd\| (A2 vs v24) | [1.3, **5.0**] | [0.5, **22.2**] | Partial — `> 5.5` catches some regs |

**No threshold on H, top-1, or top-3 cleanly separates the populations.** Wins span the full clustering range (2010-04-23 has H=0.129 = diffuse; 2020-03-16 has H=0.650 = highly concentrated). Regressions span the same range (2010-11-10 has H=0.101 = diffuse; 2018-10-08 has H=0.600 = highly concentrated).

## Mechanism per anchor — read from the top-K dates

A close reading of each regression's actual top-10 reveals **three distinct failure modes**, not the single "shape-similar wrong-forward" mode hypothesized in V4_RESULTS:

### Mode 1 — Confident single-analog lock-on (2018-10-08, 2020-03-16)

| Anchor | Top-1 analog | Top-1 prob | Top-1 fwd | Realized |
|---|---|---:|---:|---:|
| 2018-10-08 | 2015-03-25 | **50.0%** | +4.3% | −12.7% |
| 2020-03-16 | 2018-10-26 | **50.7%** | −0.9% | +43.8% |

The matcher commits half its probability mass to a single analog whose forward is wrong. The 2015-03-25 window does look shape-similar to Oct-2018 — but it preceded a calm 4% rally, not the 13% Q4-2018 selloff. The 2018-10-26 window preceded a stall before the COVID crash, with nothing comparable to the COVID +44% rally.

Diagnostic signature: **top-1 ≥ 0.4**.

### Mode 2 — Bimodal cluster on regime-similar but era-different precedents (2008-10-03)

| Top-3 analog | Prob | Year | Forward |
|---|---:|---:|---:|
| 1998-09-04 | 28.8% | 1998 (post-LTCM) | +35.4% |
| 1992-09-30 | 22.0% | 1992 (post-1991 recession) | +14.2% |
| 2003-12-24 | 4.1% | 2003 | −5.5% |

Top-2 analogs together hold **51% of probability**, both from V-recovery eras after market wobble. Neither shaped like a GFC-style cascade. The matcher confidently picks "this is like 1998 or 1992" and predicts +12.4% — but Oct-2008 became −18.3%.

Diagnostic signature: **top-2 sum ≥ 0.4 AND top-2 forwards same-sign AND opposite to realized**.

### Mode 3 — Diffuse selection, modest forward-mean error (2012-03-14, 2010-11-10, 2022-03-01, 2026-02-19, 1990-09-24, 2001-04-04, 1991-03-26, 2017-06-01)

Top-K is well-spread (top-3 < 0.5), the mass-weighted forward looks reasonable (small Δfwd vs v2.4), but A2.1 still loses CRPS or coverage. Examples:
- 2022-03-01: top-3 = 0.353, A2.wfwd = +0.8% (v24 +4.5%). The matcher disperses across non-shape-matching candidates and the dispersion happens to span too narrow a range.
- 2026-02-19: top-3 = 0.443, A2.wfwd = +3.8% (v24 +4.7%). Almost identical mean predictions; A2.1 regresses by 13% CRPS. The regression is in *dispersion shape*, not in mean.

Diagnostic signature: **top-3 < 0.5 AND \|Δfwd\| < 4 AND CRPS still regresses**. This is the largest mode (8/10 regressions) and the one **no concentration-based gate can address**.

## What this means for v5

The temporal-clustering hypothesis (V4_RESULTS) explains only **2 of the 10 regressions** cleanly (2018-10-08, 2008-10-03 — and the latter not by year but by bimodal-era). Eight regressions fail by Mode 3 — diffuse top-K with dispersion-shape problems — which a concentration gate cannot fix.

**Three implications.**

1. **No simple per-anchor gate signal exists.** Neither val_crps (V4.5.1) nor temporal-clustering metrics separate wins from regressions. V5.1 (gated A2.1) as conceived in V4_RESULTS is **not viable**.

2. **Ensemble / blending is the better v5 candidate.** Mode-3 regressions look like A2.1 producing a *slightly different* forecast that happens to lose CRPS at the test anchors. A continuous blend `d = (1-α)·d_eu + α·d_cw` with α-search would smoothly downweight A2.1 where weighted-Euclidean wins, without needing a hard gate. This is **V5.2 (Tikhonov-mixed)** from V4_RESULTS, now the leading candidate.

3. **Mode-1 and Mode-2 specifically suggest an n_eff regularizer.** A2.1's top-1=50%+ at 2018-10-08 and 2020-03-16 reflects the matcher's effective sample size collapsing despite n_eff=50 nominally. Adding a **floor on top-K dispersion** (e.g., reject distance vectors where top-1 prob > 1/n_eff × 5) would force corrwindow to diversify or fall back. Cheaper than full blending; v5 sub-candidate.

## Open follow-up

- **Mode 3 ("diffuse but still wrong") deserves its own investigation.** The matcher selects reasonable-looking analogs and produces reasonable-looking forwards, yet CRPS regresses. The mechanism is likely in *path construction* (block sampling under corrwindow disables conditional re-matching, so block-0 distances are reused across all blocks — see `simulate.py:308`). This is an **A2.1 implementation choice**, not a fundamental distance problem. **Add V4.5.6: A2.1 path-construction inspection** to test whether re-enabling conditional re-matching under corrwindow would close Mode-3 regressions.

- **v24's H is very high at the negatives** (2001-04-04: H=1.000; 2001-10-02: H=0.737). v24 *also* clusters heavily on those anchors — yet doesn't catastrophically fail. The matcher's *what it does with the cluster* matters more than the clustering itself. Suggests dispersion mechanics (vol scaling, FHS-like residual sampling) are the bigger lever than distance choice.

## Verdict

**No clean gate signal.** V5.1 (gated A2.1) is dropped. V5 must shift to either continuous-blend (V5.2 Tikhonov) or path-construction fixes (new V4.5.6).
