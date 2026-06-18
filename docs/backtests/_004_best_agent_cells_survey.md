# _004: best-agent-cells survey (headline sizing c=0.25 + mean)

## TL;DR

Took the **headline sizing established in `_003`** (quarter-Kelly `c=0.25`, mean
selection) and (a) re-ran the `_001` cell on its own window, and (b) surveyed the
**top agent runs from the R-p@K registry** (`results/gbdt/data/r_precision_at_k.csv`)
across distinct universe×target families. The dominant result is a **structural
negative**: of the 5 best agent cells, **only cell-5 trades at all** — the other four
(the rarer-event cells) **never clear the discrete-bounded-loss Kelly breakeven of
`p > 1/3`, so the strategy makes 0 trades on them.** The strategy is **base-rate-gated**:
because every cell here has `threshold = 2×drawdown` → `b=2` → breakeven `p=1/3`, and a
well-calibrated model maps scores to absolute probabilities near the base rate, only
cells with base rate near/above ~1/3 (just cell-5) ever produce a tradeable pick.
Ranking skill (high AUC / R-p@K) does **not** rescue them — Kelly sizing keys off
**absolute calibrated probability, not rank**.

Secondary result (Part 1): re-running the `_001` cell at `c=0.25` lifts it **+6.7% →
+11.5%** (4 → 7 tickers), confirming the `_003` sizing finding — but its window still
carries the 132-day forced-cash gap (exposure stuck at 0.27), so it stays well below
NDX (+25.3%). The `_002` window at `c=0.25+mean` is the clean win (= `_003`, +25.4%).

## The base-rate gate (why 4 of 5 cells don't trade)

Per-cell, after the Bayesian recalibrator (fit on each cell's VAL) — the **maximum
recalibrated `p_mean` tracks the base rate**, and only cell-5 reaches the `1/3` gate:

| Cell | R-p@3 | AUC | base rate | max recal. `p` | % days top-1 `> 1/3` | trades? |
|---|---|---|---|---|---|---|
| nasdaq100 cell-5 (`_001` cell) | 0.756 | 0.52 | 0.338 | 0.452 | **97%** | yes |
| russell1000 +50%/200d/dd25 | 0.559 | 0.73 | 0.152 | 0.260 | 0% | **no** |
| nasdaq100 +40%/50d/dd20 (mix) | 0.490 | 0.74 | 0.055 | 0.144 | 0% | **no** |
| sp500 +50%/50d/dd25 | 0.427 | **0.90** | 0.026 | 0.047 | 0% | **no** |
| sp500 +20%/25d/dd10 | 0.431 | 0.76 | 0.089 | 0.130 | 0% | **no** |

The `event_driven_topk` benchmark (same predictions, same `1/3` gate, no Kelly) is also
0% on all four — independent confirmation it's the gate, not the sizer. The sp500
+50% cell is the sharpest illustration: **AUC 0.90, R-p@1 0.64 (excellent ranking) but
base rate 2.6%**, so its calibrated `p` peaks at 0.047 and it never bets.

## Part 1 — `_001` cell re-run at the new headline sizing

| Config (cell5_revalreg, its own test window) | Total % | Max DD | Avg exp | entries / tickers |
|---|---|---|---|---|
| c=0.25, mean (new headline) | **+11.5%** | −11.6% | 0.27 | 11 / 7 |
| c=0.5, mean (= original `_001`) | +6.7% | −11.6% | 0.27 | 6 / 4 |
| — *NDX buy-and-hold* | +25.3% | −14.2% | 1.00 | 1 |
| — *EW basket* | +18.2% | −14.0% | 1.00 | 92 |

Quarter-Kelly improves return and diversification here too, but **exposure is capped at
0.27 by the 132-day forced-cash gap** in this window — so unlike the `_002` window
(`_003`: +25.4%, beats NDX), the gap-confounded `_001` window stays below passive even
at `c=0.25`. Reading both: the `_001` underperformance was **both** the gap **and**
half-Kelly; quarter-Kelly fixes the sizing half, the gap remains.

## Methodology

`scripts/backtests/run_backtest_cell.py` — cell-agnostic: reads
(universe, threshold_pct, max_drawdown, horizon_days) from each cell's `spec.yaml`,
derives Kelly payoffs + breakeven, picks the benchmark index by universe
(nasdaq100→^NDX, sp500→^SPX, russell1000→^SPX **proxy**, ^RUI uncached), and
back-tests the cell's published `predictions/test.csv` on its scored window
(comparison_end = test_end + horizon BD, clipped to data end). Calibrator fit on each
cell's VAL (leak-free). No data refresh / inference needed — historical windows are
already cached.

## Caveats

- **C1: russell1000 index is a proxy** (^SPX; ^RUI not cached) — moot for the headline
  since that cell makes 0 trades, but its index/basket comparison is approximate.
- **C2: the gate finding is calibrator-conditional but robust.** A different calibrator
  could shift absolute `p`, but any *well-calibrated* probability must sit near the base
  rate, so rare-event cells will generically fail an absolute `1/3` gate. This is a
  property of (payoff geometry × base rate), not of the Beta calibrator specifically.
- **C3: short recent windows** for the nasdaq/sp500 cells (2025-12→2026-03) — same as
  `_002`/`_003`; the russell1000 window is longer (2023-07→2024-10, H=200).
- **C4/C5:** zero costs, DD-not-bounded-at-stop on gap-downs — as prior memos.

## Reproducibility

- Branch `backtests-v11-headline-sizing-survey`.
- `for cell in <5 cells>; for c in 0.25 0.5: uv run python -m scripts.backtests.run_backtest_cell --cell results/gbdt/experiments/<cell> --out results/backtests/_004_survey/<short>_c<c> --name <short>_c<c> --c <c> --selection-bound mean`
- Base-rate-gate table: recompute via the calibrator on each cell's VAL + test (`fit_calibrator` + `cal.transform(...).p_mean`).
- **Action charts** (per run, added retroactively via `scripts/backtests/plot_actions.py`, see `_020`): `results/backtests/_004_survey/<short>_c<c>/figs/actions.png`. Only `cell5_revalreg_*` shows trades; the four rare-event cells render as a **flat $100K cash line** — the 0-trade base-rate-gate finding made visual.

## Open questions / follow-ups (NEW experiment surfaced)

- **Rank / relative sizing for rare-event cells.** The whole agent-cell catalogue beyond
  cell-5 is unreachable by an absolute-probability Kelly gate. To deploy the strong
  *ranking* of those cells, size by **cross-sectional rank** (top-K each day regardless
  of absolute `p`) or use a **base-rate-relative gate** (`p > base_rate × margin`) instead
  of the absolute Kelly breakeven. This is the natural V1.2 experiment — it would let the
  high-AUC sp500 +50% cell (0.90 AUC) actually trade.
- **Promote c=0.25+mean** as the headline sizing default (confirmed helpful on every
  cell that trades; never harmful).
- **Rolling fresh-OOS** for cell-5 as the cache ages (one-command now).
