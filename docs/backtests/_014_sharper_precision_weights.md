# _014: sharper precision weights — the edge is in SELECTION, not weighting

> **⚠️ CORRECTION (mechanism only — see [_023](_023_probweight_trim_fix.md)).** The "α never
> concentrates — entries stay full-universe (486/890) for every α" *mechanism* below is
> **bug-contaminated** by the same ratchet-down-trim churn as `_013`. Fixed in `_023`, prob_weight
> holds ~K-scale books (sp500_50 K=20 → 24 names, not 486). **The blanket verdict *"no weighting
> scheme beats equal-weight top-3"* survives** the fix: the corrected rolling re-run has equal-K=3
> ≥ prob_weight on every cell incl. r1k (an apparent r1k reversal was retracted once re-run on the
> model-bearing champions). The α-insensitivity framing is moot once the book is no longer the
> whole universe; read the mechanism here as superseded by `_023`, the conclusion as intact.

## TL;DR

`_013` showed raw-`p` precision weighting (`prob_weight`, α=1) is too flat to concentrate
and destroys the high-R-p@1 edge. The proposed fix was a **sharper** transform — weight ∝
`p^α` (α>1) — to amplify the ranking's gaps and recover concentration. We added
`prob_weight_alpha` and swept α∈{2,4,8} at K=20. **It does not work.**

- **Sharpening does NOT recover the edge.** sp500_50 stays at **8–15%** of windows beating
  (vs equal@K=3's **92%**); r1k_50 is **pinned at 4%** for every α (vs 90%). No improvement
  over the α=1 failure.
- **Why: it never concentrates.** Entry counts stay at the **full universe** (sp500 486,
  r1k 890) for *every* α — even α=8. Calibrated `p` within a day's top-20 is so flat that
  `p^8` still can't separate the weights enough to drop the tail below the dust floor, so the
  book keeps the whole universe.
- **The deeper reason: the edge lives in tight SELECTION (top-3), and re-weighting a wide
  (K=20) selection can't recover it.** Down-weighting ranks 4–20 isn't the same as *excluding*
  them — they keep nonzero weight and dilute. Concentration has to come from selecting fewer
  names, not from sharpening weights over many.

**The precision-weighting thread is closed:** no weighting scheme (equal, raw-`p`, or `p^α`)
at wide K beats the simple **equal-weight top-3** champion. Concentration via tight selection
is the only thing that monetizes a high-precision signal.

## The α-sweep (%windows-beat / median-excess / maxDD / entries, all K=20)

| Cell | R-p@1 | equal K=3 (champ) | pw α=1 | pw α=2 | pw α=4 | pw α=8 |
|---|---|---|---|---|---|---|
| nifty +50%/50d | 0.09 | 44% / −1.1 / −37 / 30 | 63% / +0.8 / −6 / 232 | 63% / +0.9 / −6 | 57% / +0.5 / −9 | 63% / +1.0 / −6 |
| nifty +50%/25d | 0.06 | 35% / −3.0 / −60 / 40 | 52% / +0.2 / −2 | 52% / +0.2 / −3 | 54% / +0.3 / −8 | 51% / +0.1 / −10 |
| nifty +20%/25d | 0.20 | 57% / +1.6 / −26 / 65 | 50% / +0.0 / −4 | 50% / +0.0 / −3 | 50% / −0.0 / −6 | 51% / +0.2 / −4 |
| sp500 +50%/50d | 0.43 | **92% / +7.3 / −10 / 9** | 38% / −6.9 / −5 | **8%** / −5.4 / −8 | **8%** / −4.8 / −9 | **15%** / −3.1 / −9 |
| r1k +50%/200d | 0.48 | **90% / +14.2 / −22 / 22** | 4% / −15.4 / −8 | **4%** / −15.5 / −10 | **4%** / −12.9 / −10 | **4%** / −13.1 / −11 |

(NSE entries 213–304; sp500 486; r1k 890 — i.e. the entire universe, at every α.)

## The mechanism (why `p^α` fails to concentrate)

Concentration via magnitude weighting needs `p` to be **peaked across the day's candidates**.
On these cells the top-20's calibrated `p` are nearly tied (all ≈ base rate), so even `p^8`
only mildly tilts the weights — every name stays above the half-equal-slice dust floor
(`0.5·gross_cap/K` = 0.025 at K=20), nothing is pruned, and the book holds the whole universe.
The α-knob moves weight *within* a still-fully-populated book; it never shrinks the book.

Note sp500 α=2 (8%) is actually *worse* than α=1 (38%): mild sharpening churns weight toward
the day's highest-`p` names (high turnover, 486 names cycling) without the precision to back
it — a worse closet-index, not a more concentrated portfolio.

## What actually concentrates

Only two things shrink the book to the high-precision tip:
1. **Tight selection** — a small top-K cut (equal@K=3), which simply *excludes* ranks 4+.
   This is the `_012`/`_008` champion and remains unbeaten.
2. **A rank-based prune** (weight ∝ 1/rank^β, or top-quantile) — deterministic in the
   *ordering*, so it prunes regardless of how flat `p` is. But at any β steep enough to
   concentrate to ~3 effective names it just reproduces equal@K=3; it cannot exceed it,
   because it's selecting the same top names.

Either way the lever is **how many names you hold**, set by selection — not how you weight
them. `p`-magnitude weighting is the wrong knob for flat-`p` rare-event signals.

## For the NSE cells (the de-risking holds)

Across all α the NSE cells stay where `_013` left them: ~breakeven return, low DD (−2% to
−10%), ~50–63% of windows. Sharpening neither helps nor hurts them — there's no edge to
concentrate *or* dilute. Consistent with `_011`: their absolute R-p@1 is too low to trade
concentrated, and no weighting rescue exists.

## Caveats

- **Single NSE regime; small US window-counts** (sp500 13, r1k 105) — carried from `_008`/`_009`.
- **Zero costs** — the α variants turn over the whole universe (486/890 entries); real costs
  would crush them far harder than equal@K=3 (9–22 entries). The cost-adjusted gap is even
  wider than shown.
- **We did not test rank-decay / top-quantile** weighting directly — but the argument above
  (it can at best reproduce equal@K=3) makes it low-value; noted as a follow-up only for
  completeness.

## Bug-fix / hygiene note

Fixed the `_013`-review nit: `run_rolling_validation.py` now writes the actual
`sizing_mode`, `K`, and `prob_weight_alpha` into each run's `summary.json` config block
(was hardcoded `"equal"`), so run provenance is self-describing. `prob_weight_alpha`
defaults to 1.0 (reproduces `_013` exactly); +2 regression tests
(`test_prob_weight_alpha_concentrates`, `test_prob_weight_alpha_default_is_raw_p`);
44 strategy tests pass. equal/kelly/rank_kelly paths untouched.

## Reproducibility

- Branch `backtests-v18-sharper-weights`. New `prob_weight_alpha` param +
  `run_rolling_validation --prob-weight-alpha`. Artifacts under
  `results/backtests/_014_sharper/`; registry rows 046–060.

## Open questions / follow-ups

- The precision-weighting line of inquiry is **exhausted**: the champion is **equal-weight
  top-3 on high-R-p@1 cells**. Next worthwhile directions are *not* sizing tweaks but:
  **(a)** cost + slippage modelling on the equal@K=3 champion (does the edge survive frictions?);
  **(b)** more US OOS / a genuine bear sub-window to firm up the `_010` regime-neutrality;
  **(c)** disjoint-window block bootstrap for an honest effective-N on the 9–22-trade cells.
- Disk: the 120 GB spike during the sweep was transient backtest scratch (freed on completion);
  standing reclaimables are `data.corrupted/` (2 GB) + `runs/` (14 GB, gitignored), optional.
