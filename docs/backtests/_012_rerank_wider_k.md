# _012: re-rank by R-p@1 + wider-K — K should be matched to precision, not widened blindly

## TL;DR

Two parts, following `_011`'s prescription ("rank cells by R-p@1; try wider-K to
rescue the low-top-1 NSE cells"):

- **Part 1 (re-rank).** Ranking all 84 lattice cells by absolute test-split R-p@1
  surfaces a **confounder**: the top cell is the nasdaq cell-5 family (R-p@1 **0.80**,
  lift 2.4) — which `_006` showed has **no rolling edge**. So **test-split R-p@1 is not a
  clean tradeability oracle**: it can be high on a high-base-rate, small-payoff cell
  (+10% targets that can't beat a rising index). The predictor needs *fresh* R-p@1 **and**
  a payoff (threshold) large enough to generate excess.
- **Part 2 (wider-K).** Sweeping K∈{3,10,20} on 3 NSE + 2 US cells: **wider-K is a
  risk-reducer, not an edge-creator, and the optimal K tracks R-p@1 inversely.** It rescued
  the worst NSE blow-up from catastrophe to ~breakeven (risk management), but **hurt every
  cell that already had an edge** (the high-R-p@1 US cells want *concentration*). You cannot
  widen a low-precision signal into positive expectancy that isn't there.

**Refined rule: match K to R-p@1** — concentrate (K≈3) when R-p@1 is high; widening only
diversifies a low-R-p@1 catastrophe toward breakeven.

## Part 2 — the wider-K sweep

| Cell | R-p@1 | K=3 ret/DD/%beat/med | K=10 | K=20 |
|---|---|---|---|---|
| nifty +50%/25d | 0.06 | **−56%** / −60% / 35% / −3.0% | **−1% / −22% / 60% / +0.8%** | −12% / −24% / 49% / −0.1% |
| nifty +50%/50d | 0.09 | −9% / −37% / 44% / −1.1% | −6% / −28% / 45% / −0.7% | −6% / −24% / 49% / −0.4% |
| nifty +20%/25d | 0.20 | **+34% / −26% / 57% / +1.6%** | −1% / −20% / 48% / −0.1% | −8% / −19% / 54% / +0.4% |
| sp500 +50%/50d | 0.43 | +34% / −10% / **92% / +7.3%** | +43% / −13% / 85% / +6.8% | +26% / −14% / 77% / +5.1% |
| r1k +50%/200d | 0.48 | +136% / −22% / **90% / +14.2%** | +96% / −25% / 68% / +4.9% | +133% / −28% / 77% / +10.8% |

(ret = full-OOS strat return, DD = max drawdown, %beat / med = rolling windows beating
the index / median excess.)

### Three findings

1. **Wider-K rescues the worst blow-up — but to breakeven, not profit.** nifty +50%/25d
   (R-p@1 0.06) went from **−56% / −60% DD / 35%** at K=3 to **−1% / −22% DD / 60%** at K=10.
   That is risk management: top-3 on a 6%-precision tip is catastrophically concentrated;
   diluting to 10 names caps any single loser and averages over more of the (rare) winners.
   But the *return* is ~0 — widening removed the disaster, it did not create alpha. And
   **K=20 over-dilutes** (back to −12% / 49%): there is an interior optimum (~K=10), not
   "more is better."

2. **Wider-K HURTS the cells that already work.** Every high-R-p@1 cell is best at K=3:
   sp500_50 (92% of windows → 85% → 77%), r1k_50 (90% → 68% → 77%), and nifty +20% (the
   `_009` winner, +34%/57% → −1%/48%). When the top picks are genuinely the winners,
   diluting with rank 4–20 names washes out the edge. **High R-p@1 wants concentration.**

3. **Dilution is a fragility probe.** nifty +20%'s +34% (K=3) *evaporates* at K=10 (−1%) →
   that `_009` "win" rested on a few concentrated top-3 names (fragile, the small-N concern
   made concrete). By contrast sp500_50/r1k stay **positive across all K** (+5–14% median) —
   robust edges. **Surviving dilution is a real-edge test; nifty +20% fails it, the US cells
   pass.**

Drawdown moves oppositely by regime of precision: for the low-R-p@1 NSE cells wider-K
**reduces** DD (diversifies the blow-up: −60%→−22%); for the high-R-p@1 US cells it
**raises** DD (−10%→−14%, −22%→−28% — the added rank 4–20 names are worse than the top-3).

## The synthesis with `_011`

R-p@1 is the master variable, and it sets the **optimal K**:

| R-p@1 | best K | why | example |
|---|---|---|---|
| ≥0.4 | **3** (concentrate) | top picks are the winners | sp500_50, r1k_50 |
| ~0.2 | 3 | still better concentrated; edge is fragile though | nifty +20% |
| <0.10 | 10 (de-risk) | top-3 too noisy; widen to tame catastrophe → breakeven | nifty +50% |

You can't widen your way to a tradeable edge — if R-p@1 is too low to trade concentrated,
wider-K only converts a catastrophe into a ~breakeven. The genuinely tradeable cells are
the high-R-p@1 ones, and they want K=3.

## Bug found + fixed (D9 dust-floor, K-unaware)

The first sweep showed **0 entries at K=20** for the NSE cells. Root cause: the entry
dust-floor `max(floor_pct_equity=5%, floor_pct_room=10%·room)` is **K-independent**, so an
equal slice of `gross_cap/K` = 5% at K=20 is below the 10%-of-room floor on the first
entry → every candidate dropped → room never shrinks → **0 entries forever** (a flat cash
curve that spuriously "beats" a −4.8% index). Fix: in equal mode, floor at **half the
intended slice** (`0.5·gross_cap/K`) — K-aware, so a full 1/K slice always clears while a
room/cash-squeezed slice is still dropped as dust. Backward-compatible at K=3 (the merged
`_005`/`_008`/`_009` runs: the 33% slice never approached the 10% floor — reproduced
identically). +2 regression tests (`test_equal_sizing_wide_k_enters`,
`test_equal_sizing_floor_still_drops_dust`); 40 strategy tests pass.

*(Correction: an hourly-heartbeat note mid-run flagged "wider-K helps nifty_50_50d, 63%
beat" — that was the buggy 0-entry flat curve, not a real result. The corrected K=20 run is
−6% / 49%.)*

## Caveats

- **Single NSE down-regime / small US window-counts** (sp500 13, r1k 105) carry over from
  `_008`/`_009`; the K-vs-R-p@1 pattern is consistent across 5 cells but not multi-regime.
- **Costs still zero.** Wider-K materially raises turnover (K=20: 150–290 entries vs 9–65);
  realistic costs would penalize the wide-K variants more than shown, *strengthening* the
  "concentrate when you can" conclusion.
- **K=20 over-dilution** may be universe-size-sensitive (376 NSE names); not a fixed number.

## Reproducibility

- Branch `backtests-v16-rerank-widerk`. Re-rank: the survey over `r_precision_at_k.csv`.
  Sweep: `run_rolling_validation --cell <cell> --fresh <fresh.csv> --k {3,10,20}` (new `--k`
  arg). Artifacts under `results/backtests/_012_widerk/`; registry rows 021–035.

## Open questions / follow-ups

- **Precision-weighted sizing** (bet size ∝ R-p@1) instead of equal weight — the principled
  version of "match K to precision."
- **Fresh R-p@1 as the cell-selection metric** (Part 1 showed test R-p@1 is confounded) —
  re-rank the lattice by *fresh-OOS* R-p@1 × payoff and re-pick.
- **Multi-regime** validation of the K-vs-R-p@1 rule before it's a deployment default.
