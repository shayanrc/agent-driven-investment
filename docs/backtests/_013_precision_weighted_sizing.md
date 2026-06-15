# _013: precision-weighted sizing — raw-p weighting can't concentrate (it's a de-risker)

## TL;DR

`_012` ended with: *optimal K tracks R-p@1; high precision wants concentration*. The
hoped-for continuous generalization was **precision-weighted sizing** — size each top-K
position ∝ its calibrated `p`, so the book would **auto-concentrate** when `p` is peaked
(high-precision days) and spread when flat, removing the need to tune K. We implemented it
(`sizing_mode="prob_weight"`, weights ∝ p normalized to `fractional_c·gross_cap`, never
zeroing sub-breakeven picks so rare cells still deploy) and swept K∈{10,20} against `_012`'s
equal-weight champions.

**The auto-concentration hypothesis FAILS.** Calibrated `p` on rare-event cells is too
**flat** (≈ base rate) to produce peaked weights, so prob_weight spreads across nearly the
**entire universe** (sp500: 486 entries, r1k: **890** — every name) and becomes a low-vol
closet-index. Result:

- **Drawdowns collapse everywhere** (nifty +50%/25d K=20 DD −1.6%; sp500 −4.3%; r1k −8%) —
  it's the strongest **de-risker** we've tested.
- **But it destroys the edge on the high-R-p@1 cells** it was supposed to help most:
  sp500_50 **92% → 38%** of windows beat; r1k_50 **90% → 4%** (median excess **−13%**).

So prob_weight is **not** a universal "auto-K" — it's an aggressive diversifier. It mildly
helps the low-R-p@1 NSE cells (risk-adjusted) and **wrecks** the high-R-p@1 cells. This
**reinforces `_012`**: concentration is what monetizes a high-precision signal, and raw-`p`
weighting can't deliver concentration when `p` isn't peaked.

## The comparison (ret / maxDD / %windows-beat / median-excess / entries)

| Cell | R-p@1 | equal K=3 (champ) | equal K=10 | **prob_weight K=10** | **prob_weight K=20** |
|---|---|---|---|---|---|
| nifty +50%/50d | 0.09 | −9% / −37 / 44% / −1.1 / 30 | −6% / −28 / 45% / −0.7 / 86 | −10% / **−11** / 56% / +0.6 / 279 | −6% / **−6** / 63% / +0.8 / 232 |
| nifty +50%/25d | 0.06 | −56% / −60 / 35% / −3.0 / 40 | −1% / −22 / 60% / +0.8 / 139 | −3% / **−7** / 50% / −0.1 / 319 | −1% / **−2** / 52% / +0.2 / 238 |
| nifty +20%/25d | 0.20 | +34% / −26 / 57% / +1.6 / 65 | −1% / −20 / 48% / −0.1 / 180 | −9% / −12 / 46% / −0.8 / 241 | −4% / −4 / 50% / +0.0 / 268 |
| sp500 +50%/50d | 0.43 | +34% / −10 / **92%** / +7.3 / 9 | +43% / −13 / 85% / +6.8 / 35 | +9% / −4 / **38%** / −1.0 / 486 | −1% / −5 / 38% / −6.9 / 486 |
| r1k +50%/200d | 0.48 | +136% / −22 / **90%** / +14.2 / 22 | +96% / −25 / 68% / +4.9 / 68 | +6% / −10 / **4%** / −13.2 / 890 | +1% / −8 / 4% / −15.4 / 891 |

## Why it fails to concentrate (the mechanism)

Weights ∝ `p` only concentrate if `p` is **peaked across the day's candidates**. On these
cells calibrated `p` ≈ the (low) base rate for *every* name — rank-1's p is barely above
rank-20's in absolute terms — so the normalized weights are near-uniform, the half-equal-slice
dust floor barely trims, and with multi-day holds the book accumulates **the whole universe**
(890/890 for r1k). That is the opposite of concentration. The entry counts make it concrete:
prob_weight enters 240–890 names vs equal-K=3's 9–65.

The de-risking is the flip side of the same coin: a near-equal book over hundreds of names is
very low variance (DDs of −2% to −11%), but for a high-R-p@1 cell that variance *was the edge*
— the few genuinely-high-precision top picks. Spreading into the universe averages them away
(r1k → 4% of windows beat a strong +63.8% bull).

## What this says, with `_011`/`_012`

- **Concentration is the active ingredient for high-R-p@1 cells**, and you get it from a hard
  top-K cut (equal K=3), **not** from raw-`p` weighting — because `p` isn't peaked enough to
  concentrate on its own.
- **prob_weight's real use is risk reduction on low-precision cells**: it took the NSE +50%
  cells to their lowest drawdowns yet (−1.6% to −7%) with slightly-positive median excess —
  but still ~breakeven return, confirming (again) there's no monetizable edge to extract there.
- To make precision-weighting *concentrate*, the weight must be a **sharper** function than
  raw p — rank-based (1/rank) or `p^α` with α≫1, or restricting the pool to a high-p quantile.
  That's the natural follow-up.

## Bug-fix note

`prob_weight` shares the K-aware dust floor from `_012` (half the equal slice), so it has no
zero-entry pathology. Two regression tests added (`test_prob_weight_sizes_proportional_to_p`
asserts weights ∝ p with the dust pick dropped; `test_prob_weight_flat_p_approximates_equal`
asserts flat p → equal slices). 42 strategy tests pass. The merged equal/kelly/rank_kelly
paths are untouched (`prob_weight` is a new branch).

## Caveats

- **Single NSE down-regime; small US window-counts** (sp500 13, r1k 105) — carried from
  `_008`/`_009`. The directional verdict (prob_weight diversifies, doesn't concentrate) is
  consistent across all 5 cells and both K.
- **Zero costs.** prob_weight's turnover is enormous (240–890 entries); real costs would hit
  it far harder than equal-K=3 — *strengthening* "concentrate when precision is high."
- **prob_weight ≠ Kelly.** Deliberately weights ∝ raw p (not Kelly-on-p), so it deploys on
  sub-breakeven rare cells; that's why it can spread so widely. A Kelly/breakeven variant
  would zero most NSE picks (the `_004` gate) — a different (already-explored) failure mode.

## Reproducibility

- Branch `backtests-v17-precision-weight`. New `sizing_mode="prob_weight"` +
  `run_rolling_validation --sizing-mode prob_weight --k {10,20}`. Artifacts under
  `results/backtests/_013_probweight/`; registry rows 036–045.

## Open questions / follow-ups

- **Sharper precision weighting**: `p^α` (α≫1), rank-decay (1/rank), or top-quantile pooling —
  the version that actually concentrates. This is the real test of "precision-weighted sizing."
- **Cross-cell precision allocation**: weight a *portfolio of cells* by each cell's R-p@1
  (the original cell-level reading of "bet ∝ R-p@1"), distinct from within-cell weighting.
- The recurring conclusion stands: the tradeable edge lives in the **high-R-p@1 cells traded
  concentrated (K≈3)**; everything else is risk management around a signal that isn't there.
