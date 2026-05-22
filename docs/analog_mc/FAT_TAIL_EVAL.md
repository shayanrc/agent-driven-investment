# Fat-tail evaluation set

**Mandatory benchmark for every v4+ experiment.** This document defines the canonical **15-anchor set** (5 positive extreme-z + 3 negative extreme-z + 7 regime-coverage) on which forecast calibration must be reported for any new modeling change. Companion to [`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md).

## Purpose

The v3 phase shipped a 9.7% mean-CRPS improvement under aggregate metrics (CRPS, PIT slope across 4,552 origins) — but ad-hoc plotting on individual anchors revealed that the model **systematically misses regime-onset rallies** (COVID-2020, Q4-2018, 2026 forecasts). Aggregate metrics average those misses away. Fat-tail evaluation pins them down.

Every v4 experiment is expected to compare its forecasts to baseline (v2.4 Cell-D-s30) on these exact 15 anchors and report the per-anchor coverage table.

## Anchor selection methodology

**Reproducible via** [`scripts/select_fat_tail_anchors.py`](../../scripts/select_fat_tail_anchors.py); persisted at [`results/analog_mc/data/fat_tail_eval_anchors.json`](../../results/analog_mc/data/fat_tail_eval_anchors.json).

### Feature scaling note

This pipeline's `zscore_50` feature is computed as `mean(returns) / std(returns)` — *not* divided by `√N`. The "classical" z-score interpretation (where ±3 is a tail event) requires multiplying by `√50 ≈ 7.07`. The selection script does this conversion explicitly; all `z₅₀` values quoted below are on the **classical** scale.

### Selection rules

The eval set has **two strata**:

**Stratum A — extreme z₅₀ (programmatic, 8 anchors)**

1. Compute classical `z₅₀ = (50-day rolling mean / 50-day rolling std) × √50`.
2. Filter to anchors inside the canonical run's walk-forward test windows (`runs/analog_mc/20260520T045525Z`).
3. **Positive side**: take all `|z| > 3` anchors, cluster-pick by 120-trading-day min-gap, keep the most-extreme per cluster → **5 anchors**.
4. **Negative side**: no anchors in the data reach `-3` (the most-negative classical `z₅₀` ever observed in NDX 1986-2026 is `-3.00` once, never below). Take all `|z| > 2` anchors, cluster-pick by 120-day min-gap, keep the **3 most-extreme** → **3 anchors**.

The negative side's asymmetry reflects a true property of equity returns: bullish-momentum extremes are more common than equally-extreme bearish-momentum extremes (drawdowns are sharper but shorter; rallies grind for longer with positive drift).

**Stratum B — hand-curated regime coverage (7 anchors)**

Hand-picked anchors that span major macro regimes (crashes, calm bull, post-crash bottoms, recent OOD data). These are **not** captured by the `|z₅₀| > 3` rule because some of these regimes coincide with moderate z₅₀ while still being highly stressful for the matcher. They are the exact dates we used while exploring v2.4's failure pattern; pinning them as eval anchors preserves that diagnostic surface for v4 comparisons.

## The 15 anchors

### Stratum A.1 — Positive momentum (|z₅₀| > 3)

| Anchor | z₅₀ | Anchor close | 60d realized | Regime |
|---|---:|---:|---:|---|
| 1991-03-26 | +3.08 | 264.7 | −1.7% | Post-Gulf War recovery rally |
| 2010-04-23 | +3.78 | 2,055.3 | −10.4% | Pre-flash-crash peak |
| 2010-11-10 | +3.38 | 2,187.7 | +6.9% | QE2 announcement |
| 2012-03-14 | +3.65 | 2,708.4 | −5.5% | Early-2012 melt-up peak |
| 2025-07-02 | +3.14 | 22,641.9 | +7.8% | Recent bull continuation |

### Stratum A.2 — Negative momentum (top 3 most-extreme |z₅₀|, all > 2)

| Anchor | z₅₀ | Anchor close | 60d realized | Regime |
|---|---:|---:|---:|---|
| 1990-09-24 | −2.56 | 177.6 | +12.2% | Gulf War / 1990 recession bottom |
| 2001-04-04 | −2.62 | 1,370.8 | **+33.5%** | Dotcom bottom (April 2001) |
| 2001-10-02 | −2.18 | 1,159.4 | **+38.6%** | Post-9/11 bottom |

### Stratum B — Regime-coverage (hand-curated, 7 anchors)

| Anchor (requested → in-window) | Anchor close | 60d realized | Regime |
|---|---:|---:|---|
| 2000-04-03 | 4,077.0 | −7.5% | Dotcom peak |
| 2008-10-03 (← 2008-09-15) | 1,470.8 | **−18.3%** | Post-Lehman / GFC |
| 2017-06-01 | 5,816.5 | +0.1% | Calm bull market |
| 2018-10-08 (← 2018-10-01) | 7,352.8 | −12.7% | Q4-2018 selloff onset |
| 2020-03-16 | 7,020.4 | **+43.8%** | COVID crash bottom |
| 2022-03-01 | 14,006.0 | −14.7% | Russia-Ukraine + Fed tightening |
| 2026-02-19 | 24,797.3 | +17.5% | Recent rally (last available 60-day window) |

Two anchors were snapped to the nearest in-window date when the requested date fell in a walk-forward val (not test) window. The selection script handles this automatically and records the original requested date in the JSON.

## Baseline (v2.4 Cell-D-s30) coverage

Each entry is the number of days (out of 60) the realized price stayed inside the forecast band. Nominal coverage: 50% band expects 30/60 days, 90% band expects 54/60.

### Stratum A — Extreme z₅₀

| Anchor | z₅₀ | 60d realized | 50% band | 90% band | Verdict |
|---|---:|---:|---:|---:|---|
| 1991-03-26 | +3.08 | −1.7% | 48/60 | 60/60 | ✅ excellent — mean-reversion called |
| 2010-04-23 | +3.78 | −10.4% | 3/60 | 27/60 | ❌ flash crash blew out band |
| 2010-11-10 | +3.38 | +6.9% | 47/60 | 56/60 | ✅ continuation called |
| 2012-03-14 | +3.65 | −5.5% | 28/60 | 55/60 | ✅ mean-reversion called |
| 2025-07-02 | +3.14 | +8.2% | 59/60 | 60/60 | ✅ near-perfect |
| 1990-09-24 | −2.56 | +12.2% | 28/60 | 55/60 | ✅ bounce called |
| 2001-04-04 | −2.62 | +33.5% | 30/60 | 55/60 | ✅ caught by huge 90% band (1700–4100) |
| 2001-10-02 | −2.18 | +38.6% | 6/60 | 44/60 | ❌ realized rallied above band |

### Stratum B — Regime-coverage

| Anchor | 60d realized | 50% band | 90% band | Verdict |
|---|---:|---:|---:|---|
| 2000-04-03 (dotcom peak) | −7.5% | 33/60 | 56/60 | ✅ well-calibrated, called direction |
| 2008-10-03 (post-Lehman) | −18.3% | 7/60 | 52/60 | ✅ wide 90% band caught the crash; median under-shot |
| 2017-06-01 (calm bull) | +0.1% | 43/60 | 60/60 | ✅ perfect, narrow band correctly held |
| 2018-10-08 (Q4'18 selloff) | −12.7% | 4/60 | 31/60 | ❌ realized fell out the bottom — model expected sideways |
| 2020-03-16 (COVID crash) | +43.8% | 6/60 | 38/60 | ❌ realized rallied out the top — model expected continued decline |
| 2022-03-01 (UKR + Fed) | −14.7% | 29/60 | 58/60 | ✅ called the drawdown direction |
| 2026-02-19 (recent) | +17.5% | 32/60 | 41/60 | ❌ second half blew through 90% band (see §Re-forecast experiment below) |

### Aggregate coverage on the 15-anchor set

- **50%-band mean** = 26.9/60 (nominal 30) — under-dispersed by ~10%
- **90%-band mean** = 49.9/60 (nominal 54) — under-dispersed by ~8%
- **Pass rate**: 10/15 anchors pass, 5/15 fail
- **Failures**: 2010-04-23 (flash crash), 2001-10-02 (post-9/11 rally), 2018-10-08 (Q4 selloff), 2020-03-16 (COVID rally), 2026-02-19 (recent rally)

The five failures all share a **regime-transition** character that the matcher could not anticipate from its prior 10-day analog blocks.

## Charts

> **v4 reference panel.** A canonical v2.4 baseline panel (15 charts + per-anchor coverage/CRPS table + v3.5 failure/control annotations) lives at [`experiments/_v24_fat_tail_baseline.md`](experiments/_v24_fat_tail_baseline.md). Each v4 experiment compares its own panel against that one. The charts below are the older single-render set retained for in-place context inside this doc.

### Stratum A.1 — Positive momentum (|z₅₀| > 3)

![1991-03-26](figs/forecast_19910326.png)

![2010-04-23](figs/forecast_20100423.png)

![2010-11-10](figs/forecast_20101110.png)

![2012-03-14](figs/forecast_20120314.png)

![2025-07-02](figs/forecast_20250702.png)

### Stratum A.2 — Negative momentum (top 3 most-extreme)

![1990-09-24](figs/forecast_19900924.png)

![2001-04-04](figs/forecast_20010404.png)

![2001-10-02](figs/forecast_20011002.png)

### Stratum B — Regime-coverage

![2000-04-03 — dotcom peak](figs/forecast_20000403.png)

![2008-10-03 — post-Lehman](figs/forecast_20081003.png)

![2017-06-01 — calm bull](figs/forecast_20170601.png)

![2018-10-08 — Q4-2018 selloff](figs/forecast_20181008.png)

![2020-03-16 — COVID crash](figs/forecast_20200316.png)

![2022-03-01 — Russia-Ukraine + Fed](figs/forecast_20220301.png)

![2026-02-19 — recent rally](figs/forecast_20260219.png)

### Re-forecast experiment (2026-03-26)

For the most recent anchor (2026-02-19), the realized rallied above the 90% band starting at t=45. To stress-test the v2.4 model's *response to receiving more data*, we re-forecast from the date the realized first crossed below the original 50% band (2026-03-26). This is the most-recent-state forecast production would have made one month into the window. Only 35/60 forecast days have realized data available.

![2026-03-26 re-forecast](figs/reforecast_from_dip.png)

**Re-forecast 35-day coverage:** 2/35 in 50% band, 8/35 in 90% band. Even one month into the unprecedented rally, the model continued to under-call its magnitude.

## Observations (v2.4 baseline)

### Pattern 1 — Bull-momentum extremes: mostly well-calibrated

4 of 5 positive anchors stay inside the 90% band ≥55/60 days. The model correctly identifies that extreme positive z₅₀ is followed by *some* mean-reversion (median drifts down) but with wide uncertainty. The 2025-07-02 case is essentially perfect: 59/60 in the 50% band.

**The exception is 2010-04-23** — the pre-flash-crash peak. The model said "more rally" (median up), realized fell −10.4%. The flash crash itself (May 6, 2010) was a single-day anti-momentum event that no analog match could anticipate. *This is the only failure case from a bull-momentum extreme.*

### Pattern 2 — Bear-bottom extremes are the matcher's blind spot

All three negative anchors had positive realized returns: +9.8%, +32%, +38.6%. The model correctly called the *direction* (median above anchor on all three), but **2 of 3 forecasts under-shot the magnitude of the rally**. The 2001-10-02 case is the worst: realized +38.6%, forecast 90% upper bound only +12% — model could not produce a path with a 38% rally in its analog pool.

The 2001-04-04 case stayed inside the 90% band only because the band was enormous (1,700–4,100 — a 2.4× spread). The matcher's vol-injection mechanism is *event-of-the-moment dependent* — when very recent vol is high, the band widens automatically. So 2001-04-04 (mid-bear, very volatile) got a usable band; 2001-10-02 (a year of bear absorption later, lower realized vol) got a narrower band that the realized still escaped.

### Pattern 3 — Regime-coverage stratum: 5/7 calm regimes pass, 2/7 fail on V-recovery

Stratum B's regime-coverage anchors split cleanly into two groups:

**Calm or mean-reverting regimes (5/7 pass)** — 2000-04 (gradual peak), 2008-10 (GFC crash with wide bands), 2017-06 (calm bull), 2022-03 (UKR drawdown), 2026-02-19 (first half before the rally). The 2008-10 case is especially instructive: realized fell −18.3% (a huge move) but the matcher's recent-vol-driven band was wide enough (52/60 in 90%) to catch it. **Wide-band drawdowns are a model strength, not a weakness.**

**V-recovery regimes (2/7 fail)** — 2018-10-08 (Q4-2018 selloff *out of a sideways forecast*) and 2020-03-16 (COVID rally *out of a continued-decline forecast*). These are the same failure shape as Stratum A.2's 2001-10-02. Plus 2026-02-19's second half (the rally that the model under-shot).

### Pattern 4 — The structural ceiling: "regime-transition into a sharp move"

Across all 15 anchors, the 5 failures collectively define the analog primitive's structural ceiling:

| Anchor | Failure shape | Realized 60d |
|---|---|---:|
| 2010-04-23 | Bull-momentum peak → flash crash | −10.4% |
| 2001-10-02 | Bear-bottom → V-rally | +38.6% |
| 2018-10-08 | Sideways forecast → Q4 selloff | −12.7% |
| 2020-03-16 | Continued-decline forecast → COVID rally | +43.8% |
| 2026-02-19 | First-half flat → second-half rally | +17.5% |

All five share a common structural feature: **the realized 60-day path is shaped by a regime transition** the matcher cannot infer from local features (z₂₀, z₅₀, z₂₀₀, ewma_vol). The analog matcher draws from historical 10-day blocks. Both ±10% and ±30%+ moves over 60 days are present in the historical pool, but they are rare and only get sampled when the matcher *finds analogs whose past behaviour matched the current state vector*. When the current state is "bear bottom" or "Fed pivot moment," the matcher pulls analogs from prior "bear continuation" or "tightening cycle" cases — which played out very differently.

**This is the limit of the analog primitive at regime transitions — and is what v4's B1 (Platzer local-linear correction) is the candidate fix for.** B1's Jacobian-bias correction is theoretically targeted exactly at high-Lyapunov regimes (regime onsets), which is the failure surface this eval set characterizes.

## v4 mandatory deliverable

For every v4 experiment that produces a forecast (A1 FHS baseline, B1 Platzer local-linear, A2 OFTER max-corr, B2 delay-coords, B3 Dirichlet weights):

1. **Render the 15-anchor panel** using `scripts/plot_forecast_from_date.py --date <ISO>` for each anchor in `fat_tail_eval_anchors.json` (`positive`, `negative`, and `regime_coverage` sections). Save under `docs/analog_mc/experiments/figs/<exp_id>_fat_tail/`. The v2.4 reference panel at [`experiments/_v24_fat_tail_baseline.md`](experiments/_v24_fat_tail_baseline.md) is the side-by-side comparison target.
2. **Produce the coverage table** (15 rows × {50% band days, 90% band days, verdict}) and a per-anchor diff vs the v2.4 baseline coverage table above.
3. **Aggregate metrics**: per-anchor mean CRPS over the 60-day horizon, plus the aggregate mean across all 15 anchors. Diff against the v2.4 numbers stored at [`results/analog_mc/data/fat_tail_baseline_v24.json`](../../results/analog_mc/data/fat_tail_baseline_v24.json) (generated by `scripts/compute_fat_tail_baseline_v24.py`).
4. **Verdict**: did the experiment improve `bear-bottom + rally` anchors (the systematic miss) without regressing the well-calibrated bull-momentum anchors? This is the headline question for B1 specifically. The promotion bar is recovery of ≥3 of the 5 V3.5 failure anchors at 90%-band ≥45/60 days.

An experiment that improves aggregate CRPS but regresses on >2 fat-tail anchors should not be promoted to default without explicit justification.

### v4 outcome (closed 2026-05-22)

The three v4 P0 experiments shipped: B1, A2.1, B5. **None promote** — full synthesis at [`V4_RESULTS.md`](V4_RESULTS.md). Headline counts vs the bar above:

| Experiment | Failures recovered (90 ≥45) | Anchors regressing >5% CRPS | Promotion |
|---|---:|---:|---|
| B1 (Platzer local-linear) | 1/5 | 5/15 | ❌ |
| A2.1 (corrwindow L=100) | 2/5 | 10/15 | ❌ |
| B5 (joint) | 2/5 | 10/15 | ❌ |

A2.1's strongest single-anchor win was **2010-04-23 (90-band 27 → 57)** — the cleanest evidence yet that the matcher distance is the right v5 lever, gated to avoid the 2008-10-03 catastrophic regression (+122% CRPS).

## How to reproduce

```bash
# 1. Generate/refresh the anchor list (after a new canonical run, the
#    walk-forward windows may shift slightly).
uv run python scripts/select_fat_tail_anchors.py

# 2. Render the 15 charts from the canonical run.
for d in 1991-03-26 2010-04-23 2010-11-10 2012-03-14 2025-07-02 \
         1990-09-24 2001-04-04 2001-10-02 \
         2000-04-03 2008-10-03 2017-06-01 2018-10-08 \
         2020-03-16 2022-03-01 2026-02-19; do
    uv run python scripts/plot_forecast_from_date.py --date "$d"
done
```

The plot script writes to `docs/analog_mc/figs/forecast_<YYYYMMDD>.png` by default.

## Carry-forward

If a future canonical run shifts the walk-forward fold boundaries such that one of these 15 anchors falls outside any test window, **re-anchor to the nearest in-window date**. The selection script handles this automatically for Stratum B (regime-coverage) anchors via nearest-in-window snapping; for Stratum A (extreme-z) anchors, the JSON pins the current selection (re-running the script on a new canonical may produce different picks if fold boundaries shift). Do not silently change the eval set without updating this document and re-stating the v2.4 baseline numbers.
