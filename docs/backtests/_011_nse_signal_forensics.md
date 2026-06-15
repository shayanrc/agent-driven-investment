# _011: NSE signal forensics — it's the base rate, not the AUC

## TL;DR

`_010` redirected to: *why doesn't NSE's high label-AUC convert to tradeable excess?*
Two hypotheses — **(A)** the AUC is overfit and collapses OOS, or **(B)** the moves are
real but happen as uncapturable overnight gaps. We recomputed realized labels
(`gbdt.targets.build_target`, the exact training label) on the fresh OOS and tested both,
with US cells as positive controls. **Both hypotheses are refuted.** The real answer is
the **base rate**:

- **A is out — the NSE AUC is REAL and generalizes.** Fresh-OOS AUC ≈ test AUC for every
  NSE cell (drops of −0.04 to +0.11; nifty +50%/25d even *rose* to 0.867). The model
  genuinely ranks NSE movers out-of-sample.
- **B is out — the moves are capturable.** NSE entry overnight gaps are *smaller* than US
  (median 0.17% vs 0.28%; only 4.6% of NSE picks gap up >2% vs 23% of US). Not a
  microstructure problem.
- **The real gap: absolute top-K precision.** At equal AUC, the NSE +50% cells' top-1 pick
  wins only **6–10%** of the time vs **43–48%** for the US controls — because NSE +50%
  events are **5–20× rarer** (base rate 0.003–0.016 vs US 0.066–0.131). A concentrated
  top-3 long-only strategy can't monetize a 6%-precision tip: the rare winners can't carry
  the 90%+ of picks that miss and ride to the drawdown stop.

**AUC is the wrong yardstick across universes with different base rates. R-Precision@1 —
absolute *and* lift — is what predicts tradeable rank/equal edge,** and it cleanly explains
the whole `_005`–`_009` arc.

## Forensic A — fresh-OOS discrimination (realized labels)

| Cell | base rate | test AUC | fresh AUC | AUC drop | fresh R-p@1 | R-p@1 lift |
|---|---|---|---|---|---|---|
| nifty +50%/50d | 0.0159 | 0.889 | 0.779 | +0.11 | 0.094 | 5.9× |
| nifty +50%/25d | 0.0034 | 0.827 | **0.867** | −0.04 | 0.059 | 17.1× |
| nifty +30%/25d | 0.0235 | 0.814 | 0.798 | +0.02 | 0.096 | 4.1× |
| nifty +20%/25d | 0.0766 | 0.722 | 0.705 | +0.02 | **0.197** | 2.6× |
| nifty +10%/25d | 0.2383 | 0.601 | 0.619 | −0.02 | 0.276 | 1.2× |
| **sp500 +50%/50d** (US) | 0.0664 | 0.899 | 0.831 | +0.07 | **0.429** | 6.5× |
| **r1k +50%/200d** (US) | 0.1308 | 0.726 | 0.779 | −0.05 | **0.484** | 3.7× |

(NSE cells: 107k–116k labeled fresh rows — robust. r1k: 191k. **sp500_50: only 6.8k
labeled rows / ~42 top-K picks — thin**, so its R-p@1 0.429 is directionally right but
imprecise; r1k is the solid US anchor.)

Read the table by **AUC vs R-p@1**: AUC is similar across US and the high-threshold NSE
cells (0.78–0.89), but R-p@1 — the precision of the single pick the strategy leans on
hardest — is an order of magnitude apart (US 0.43–0.48 vs NSE +50% 0.06–0.09). AUC
averages over the whole ranking; R-p@1 is the sharp tip, and the tip is where a
concentrated top-K strategy lives. See `figs/auc_vs_rp1.png`.

## Forensic B — capturability (entry gaps for top-K picks)

| Market | median entry gap | % gap-up >2% | winner median entry gap |
|---|---|---|---|
| NSE | +0.17% | 4.6% | +0.35% |
| US | +0.28% | 23.2% | +0.48% |

The strategy enters at the next open; the overnight gap is return it can't capture. NSE
gaps are *smaller* than US — the NSE moves are not front-loaded into untradeable jumps.
Capturability is not the problem.

## Why this explains the whole arc

Tradeable rank/equal edge needs **both** (i) absolute R-p@1 high enough that the
concentrated top picks win often enough to carry the losers through their drawdown stops,
**and** (ii) R-p@1 lift > 1 (genuine skill, not base rate). Mapping the cells:

| Cell | abs R-p@1 | lift | tradeable? | matches |
|---|---|---|---|---|
| US sp500/r1k +50% | 0.43–0.48 | 3.7–6.5× | **yes** (both high) | `_008` winners |
| nifty +20%/25d | 0.197 | 2.6× | **mild yes** | `_009`'s only NSE winner (+1.6%) |
| nifty +50% (×2/3 cells) | 0.06–0.10 | 4–17× | **no** (abs too low) | `_009` blow-ups |
| nifty +10%/25d | 0.276 | 1.2× | **no** (≈ no skill) | `_009` loser |
| ndx cell-5 family | (low) | ~1× | **no** | `_006` no-edge |

The +50% NSE cells have enormous *lift* (up to 17×) but absolute precision too low to
trade concentrated; nifty +10% has decent absolute precision but ~no lift (it's just the
high base rate). The sweet spot — high absolute R-p@1 **and** lift>1 — is exactly the US
cells (base rates 0.07–0.13) and, marginally, nifty +20% (base 0.077). **This is why the
`_008` cross-market AUC story misled us into `_009`: AUC ignored the base-rate-driven
collapse of absolute top-K precision.**

## Caveats

- **C1: sp500_50 US control is thin** (6.8k labeled rows, ~42 top-K picks — its fresh OOS
  is short and H=50 truncates the labelable tail). Its R-p@1 0.429 is directionally
  consistent with r1k (0.484, 191k rows) but not precise on its own. r1k is the solid anchor.
- **C2: end-of-window label truncation.** Fresh rows within H days of the panel end have no
  realized label and are dropped (build_target → NaN). The labeled set is the early/middle
  of each fresh OOS; the very recent tail is excluded by construction.
- **C3: R-p@1 is itself noisy** at these base rates (the +50%/25d 0.059 is over few positive
  days); the *pattern* across cells is the signal, not any single value.
- **C4: this diagnoses the concentrated top-K, long-only, equal-weight strategy.** A
  wider-K or precision-weighted deployment would interact with these numbers differently
  (see follow-ups) — "untradeable" is specific to the current strategy.

## Net

The NSE rare-event signal is **real and capturable** — it just isn't *concentrated* enough
at the tradeable tip, because the events are far rarer than in the US. The lesson is a
metric correction: **rank rank/equal candidates by absolute R-Precision@1 (with lift>1),
not AUC.** That single change would have predicted every cell outcome in `_006`–`_009`.

## Reproducibility

- Branch `backtests-v15-nse-forensics`. `uv run python -m scripts.backtests.nse_signal_forensics`.
- Realized labels via `gbdt.targets.build_target` on the cached panels; merged with the
  fresh-prediction CSVs (`_008`/`_009`). Outputs under `results/backtests/_011_forensics/`.

## Open questions / follow-ups

- **Wider-K / diversified NSE deployment**: at R-p@1 0.06 but R-p@10 ~0.12, a top-20–50
  basket might average the low precision into a positive expectancy where top-3 can't.
- **Re-rank the whole cell lattice by absolute R-p@1 × (lift>1)** and re-pick which cells
  to trade — likely surfaces moderate-threshold cells over the +50% lottery cells.
- **Position-size by precision** (smaller bets on lower-R-p@1 cells) instead of equal weight.
