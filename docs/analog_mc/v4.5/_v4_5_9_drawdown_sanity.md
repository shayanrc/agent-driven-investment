# V4.5.9 — Drawdown-feature sanity

**Question.** V5.B proposes a `drawdown_60d_norm` feature: `log(close[t] / max(close[t-59:t+1])) / std(log_returns[t-59:t+1])`. The hypothesis is that this captures the "extreme recent drawdown velocity" signature that Cohort-2 anchors (2012-03-14, 2001-04-04, 2001-10-02, 2020-03-16, 2022-03-01) share but that z-scores and corrwindow miss. At each Cohort-2 anchor, do the **top-20 historical candidates by drawdown-distance alone** include the expected V-recovery / capitulation precedents?

**Method.** Compute drawdown_60d_norm causally. For each Cohort-2 anchor, find top-20 candidates by `|dd_target − dd_cand|`. Inspect:
- Top-5 nearest historical dates.
- Mean and median 60-day forward returns of top-20.
- Fraction of top-20 with forward sign matching realized.
- Year-cluster distribution.

Script: [`scripts/analog_mc/v4_5/drawdown_feature_sanity.py`](../../../scripts/analog_mc/v4_5/drawdown_feature_sanity.py) · Data: [`v4_5_9_drawdown_sanity.json`](../../../results/analog_mc/data/v4_5_9_drawdown_sanity.json)

## Anchor-by-anchor verdict

### 2001-04-04 (realized +33.5%, dd_target = −18.71) ✅

| Top-5 analogs | Forward 60d % |
|---|---:|
| 1990-09-25 | +12.2% |
| 1990-09-26 | +10.9% |
| 1990-08-20 | −4.2% |
| 1990-10-10 | +13.4% |
| 1990-09-28 | +16.4% |

Top-20 mean forward +8.4%, **85% same-sign as realized**, all 20 from 1990. The 1990 recession-recovery analogs are textbook precedents for the 2001 dotcom-bottom rally. Drawdown feature directly pulls them in.

### 2001-10-02 (realized +38.6%, dd_target = −16.49) ✅

| Top-5 analogs | Forward 60d % |
|---|---:|
| 1990-09-19 | +4.5% |
| 1990-10-17 | +18.6% |
| 1990-09-18 | +3.7% |
| 1990-09-14 | +5.1% |
| 1990-09-17 | +5.2% |

Top-20: 16/20 from 1990, mean +6.0%, **75% same-sign**. Same story as 2001-04-04. Excellent feature behavior.

### 2022-03-01 (realized −14.7%, dd_target = −9.31) ✅ (moderate)

| Top-5 analogs | Forward 60d % |
|---|---:|
| 1990-01-24 | +1.6% |
| 2001-01-01 | −25.9% |
| 1992-06-10 | +0.6% |
| 1990-02-05 | +0.7% |
| 2000-12-29 | −32.0% |

Top-20: years dispersed (1992, 1990, 2000, 2008, 2002 each 2). Mean forward −5.2%, **60% same-sign**. Reasonable — picks up moderate-drawdown precedents that include both wobbles and full bears. Mean is the right direction (negative), so a matcher using this feature would have a sensible mean prediction.

### 2020-03-16 (realized +43.8%, dd_target = −10.33) ⚠️ BIMODAL

| Top-5 analogs | Forward 60d % |
|---|---:|
| 2008-10-30 | **−8.3%** |
| 2012-11-20 | +6.5% |
| 1986-07-22 | −4.1% |
| **1987-11-11** | **+10.2%** |
| 2015-08-21 | +2.7% |

Top-20: years dispersed (2008×4, 1987×3, 2012×2, 1986×2, 2000×2). Mean +1.7%, **55% same-sign**. **The COVID drawdown was so extreme that the closest analogs in dd-space are other crisis bottoms — both continuation cases (2008-10 post-Lehman continued falling, fwd −8%) and recovery cases (1987-11 post-crash recovery, fwd +10%).**

The feature **does** pull in V-recovery precedents (1987-11-11), but **not exclusively** — it also pulls in continuation cases. Pure drawdown distance is bimodal at COVID's extreme drawdown level. Without a second feature to disambiguate "continuing crisis" vs "V-recovery," drawdown alone won't give COVID a confident enough V-recovery signal.

### 2012-03-14 (realized −5.5%, dd_target = 0.00) ❌

| Top-5 analogs | Forward 60d % |
|---|---:|
| 1991-08-27 | −0.8% |
| 2000-03-22 | −15.7% |
| 1991-11-13 | +14.9% |
| 1991-11-12 | +17.2% |
| 1991-11-11 | +18.7% |

Top-20: 11/20 from 1991, mean +2.7%, **45% same-sign**. Wrong direction. **2012-03-14 is at the peak (dd=0), so the drawdown feature has no signal here — it just pulls all other "near peak" historical moments, which are mostly bullish continuations.** This anchor's regression isn't a drawdown-feature problem.

## Cross-anchor summary

| Anchor | Drawdown-feature sanity | Forward sign agreement | Notes |
|---|:---:|---:|---|
| 2001-04-04 | ✅ Strong | 85% | Best case |
| 2001-10-02 | ✅ Strong | 75% | Best case |
| 2022-03-01 | ✅ Moderate | 60% | Right direction |
| 2020-03-16 | ⚠️ Bimodal | 55% | Pulls right region but doesn't disambiguate |
| 2012-03-14 | ❌ No signal | 45% | Target dd=0; feature inert here |

## V5 plan implications

**V5.B is justified** for the three anchors where the feature provides a clean signal (2001-04, 2001-10, 2022-03). At 2020-03-16 the feature is necessary but not sufficient — needs a co-feature to separate V-recovery from continuation in the extreme-drawdown regime. At 2012-03-14 the feature is inert; this anchor needs a different fix.

Three refinements to V5.B:

1. **Run V5.B as planned** with the corrected log-ratio drawdown formula. The 3 strong-sanity anchors are worth pursuing.

2. **For 2020-03-16 specifically, V5.B may need supplementation.** Two candidate co-features:
   - **Trailing realized volatility** (separate from σ for vol scaling): high vol + deep drawdown = recovery regime; low vol + deep drawdown = secular bear.
   - **"Time since last peak" or "drawdown duration"**: COVID's drawdown was very fast (30 days); 2008 was slow. This isn't captured by 60-day window stats alone.

3. **2012-03-14 is out of V5.B's reach.** Add a note to V5_EXPERIMENTS_PLAN that 2012-03-14 is a residual regression V5.B will not address; either accept it (it's a −5.5% modest move) or queue it as v5+ open work.

## Updated V5.B specification

The drawdown formula should be:
```python
drawdown_60d_norm = log(close[t] / max(close[t-59:t+1])) / std(log_returns[t-59:t+1])
```
**Not** the naive linear-difference version (V4.5.9's first iteration was broken — it produced values in absolute price units, ~−85000 at COVID; corrected to dimensionless log-ratio normalized by vol).

A test in `tests/analog_mc/test_drawdown_feature.py` should assert:
- Causality: `dd[t]` depends only on `close[t-59:t+1]`.
- At a peak: `dd = 0` (since `close = max`).
- Symmetric scaling: when std doubles, |dd| halves for the same drawdown magnitude.
- NaN at the first 60 indices.

## Verdict

**V5.B is sound but incomplete.** Top-20 sanity confirms the feature works at 3/5 Cohort-2 anchors. COVID needs an additional disambiguation feature (likely a vol-regime or drawdown-duration co-feature). 2012-03-14 is outside V5.B's scope.

The v5 plan should:
- Keep V5.B as P1.
- Add a **V5.B.2 stretch** that ensembles V5.B with a vol-regime co-feature — only if V5.B's COVID coverage doesn't meet expectations in the canonical run.
- Acknowledge 2012-03-14 as a likely residual regression beyond v5 scope.
