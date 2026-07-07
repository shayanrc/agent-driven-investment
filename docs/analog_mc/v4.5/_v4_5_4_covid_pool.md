# V4.5.4 — COVID 2020-03-16 pool sufficiency

**Question.** All three v4 experiments fail at 2020-03-16 (best: A2.1 90-band 11/60, realized +43.8%). V3.5.2 established the pool contains 179 candidates with +30%/60d returns at this fold (≥3 — pool is NOT structurally empty). Are these candidates **being selected** by any v4 matcher, or are they being ignored?

**Method.** At the 2020-03-16 anchor, compute probability mass concentration above the +30%/+40%/+50% realized-forward thresholds under each matcher (v2.4 weighted-Euclidean, A2.1 corrwindow L=100). Compute **lift over uniform** = `mass(above-T) / count(above-T) × N`. If lift < 1, the matcher is *avoiding* tail-positive candidates relative to random selection.

Script: [`scripts/analog_mc/v4_5/covid_pool_sufficiency.py`](../../../scripts/analog_mc/v4_5/covid_pool_sufficiency.py) · Data: [`v4_5_4_covid_pool.json`](../../../results/analog_mc/data/v4_5_4_covid_pool.json)

## Headline

| Matcher | n eligible | count >+30% | mass >+30% | uniform | **lift** | count >+40% | mass >+40% | lift >+40% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2.4 weighted-Euclidean | 8361 | 179 (2.14%) | **0.60%** | 2.14% | **0.28×** | 63 | 0.000% | 0.00× |
| A2.1 corrwindow L=100 | 8361 | 179 (2.14%) | **1.02%** | 2.14% | **0.48×** | 63 | 0.158% | 0.21× |

**Both matchers actively AVOID high-return candidates relative to uniform random selection.** v2.4 puts 28% of the mass that uniform random would put on the +30% tail. A2.1 is slightly better at 48% but still well below uniform. At the +40% tail, v2.4 assigns essentially zero (0.000%) and A2.1 assigns 21% of uniform.

## Top-3 highest-realized candidates (the 1998 V-recovery analogs)

| Date | Forward 60d % | v2.4 prob | v2.4 rank | A2.1 prob | A2.1 rank |
|---|---:|---:|---:|---:|---:|
| 1998-10-08 | +68.6% | 0.0000 | 3216 / 8361 | 0.0000 | 2830 / 8361 |
| 1998-10-09 | +64.0% | 0.0000 | 4284 / 8361 | 0.0000 | 1263 / 8361 |
| 1998-10-13 | +63.5% | 0.0000 | 3703 / 8361 | 0.0000 | 7094 / 8361 |

These three windows (Oct 1998, post-LTCM-crisis V-recovery) are **exactly the pattern** COVID 2020-03-16 needed. They are present in the pool. Both matchers rank them around the **middle of the distribution** — neither tail nor top. They receive negligible probability mass.

## Reading — the COVID failure is a matcher problem, not a tail-inflation problem

V4_RESULTS framed COVID as "the pool likely has the rally instances but no analog has the COVID-specific dispersion." V4.5.4 sharpens that:
- **The dispersion is in the pool.** 179 candidates at >+30%/60d exist; 63 at >+40%.
- **The matchers fail to find them.** Mass assignment is *below uniform random*. Both v2.4 (z-score state) and A2.1 (corrwindow shape on L=100) consider the 1998-10 V-recovery moments *less* similar to 2020-03-16 than the average pool member.

The mechanism: 2020-03-16 sits in a regime characterized by *recent extreme drawdown velocity* — a feature neither matcher captures. v2.4's z-scores are bounded by the rolling normalization; the 2020-03 z-scores are not extreme (z50_classical = −0.96, well within ±1). A2.1's corrwindow with L=100 captures the last 100 trading days' return shape — but the 1998-10 100-day shape doesn't look like the COVID 100-day shape (LTCM was a slow grind, COVID was a 30-day cliff).

What feature WOULD distinguish "recent-shock-recovery-ahead" candidates? Three plausible v5 directions:

1. **Short-window momentum / drawdown depth.** Add a feature like `max_drawdown_60d` or `(close[t] − min(close[t-30:t])) / std(returns[t-30:t])`. The 1998-10-08 anchor would have a similar drawdown-from-peak as 2020-03-16. A weighted-Euclidean over a 4-or-5-feature vector with this added might pull V-recovery analogs in.

2. **Variable-L corrwindow ensemble.** Currently L=100 is fixed. An ensemble averaging corrwindow distances at L ∈ {20, 60, 100, 250} would let each anchor self-select the relevant time scale. 2020-03-16's distinctive shape is short-window (the 30-day cliff); a shorter L would catch it.

3. **Delay-coordinate (Takens-style) features (the v4 B2 deferred experiment).** Embed each window as `[r_t, r_{t-k}, r_{t-2k}, ...]` for some lag k; match in the embedding space. Lyapunov-aware; captures momentum + reversal in one shot. Originally deferred at v4 in favor of A2.1; V4.5.4 makes the case to revisit it.

## A weaker conclusion — A2.1 is still partially better than v2.4

A2.1's lift at +30% is 0.48× vs v2.4's 0.28× — almost 2× better at finding tail-positive analogs. And at +40%, A2.1 has measurable mass (0.16%, lift 0.21×) where v2.4 has none. This explains A2.1's CRPS improvement at 2020-03-16 (−18.6%, the only CRPS win at this anchor) — it's pulling more probability into the tail. But the *coverage* still collapses (11/60 90-band) because A2.1 simultaneously over-concentrates on a single bad analog (2018-10-26, top-1 50.7%, see V4.5.2 Mode 1).

So 2020-03-16 is **two compounding failures**: (a) under-weighting of true V-recovery analogs, and (b) over-concentration on a single visually-similar-but-wrong analog. Mode-1 from V4.5.2 explains (b). Mode (a) needs a different feature set.

## Verdict

**COVID 2020-03-16 is matcher-addressable, not tail-inflation-bound.** Drop the V4_RESULTS recommendation to defer tail inflation as v5+. **Promote B2 (delay-coordinate features) or a feature-augmented weighted-Euclidean to the v5 plan.** These are now the two leading candidates for *unlocking* the V-recovery analogs that already exist in the pool.

This finding also implies a sub-investigation: does the same "matcher avoids tail-positive analogs" hold at other failure anchors? Specifically the V3.5 failure set — 2001-10-02 (+38.6%), 2026-02-19 (+17.5%). If yes, the same v5 fix could rescue multiple anchors. Worth a quick V4.5.7.
