# _005: rank-sizing — deploying the ranking on rare-event cells (V1.2)

## TL;DR

`_004` showed the strategy is **base-rate-gated**: the strong-ranking agent cells
(high AUC / R-p@K) never trade because their *calibrated* `p` sits far below the
absolute Kelly breakeven `p>1/3`. This experiment fixes that with a **rank mode** —
select the day's top-K by *rank* (no absolute gate), drop the breakeven exit (it would
insta-fire when `p` is always sub-breakeven), and size either equal-weight or by a
**rank-conditional Kelly** (use the cell's *eval* R-p@K as the per-pick win prob).

**The ranking is tradeable.** Under rank + equal-weight, all five cells trade and
**4 of 5 beat their index out-of-sample** — including the cells that made literally 0
trades in `_004`:

| Cell (rank/equal, c=1.0) | Strategy | Index | DD strat/idx | entries/tk | win% |
|---|---|---|---|---|---|
| nasdaq +40%/50d (mix) | **+61.5%** | NDX +36.8% | −7.6% / −12.1% | 6 / 6 | 83% |
| sp500 +20%/25d | **+58.1%** | SPX +8.4% | −7.3% / −9.1% | 13 / 8 | 69% |
| russell1000 +50%/200d | **+49.1%** | SPX +40.2% | −21.6% / −18.9% | 12 / 10 | 67% |
| sp500 +50%/50d | **+12.5%** | SPX +8.4% | −9.3% / −9.1% | 4 / 4 | — |
| nasdaq cell-5 +10%/50d | +12.3% | NDX +25.3% | −5.2% / −14.2% | 9 / 6 | — |

So the absolute-`p` Kelly gate was **discarding deployable signal** — the R-p@K lift
that `_004` measured does translate into realized PnL when you bet on the *rank* instead
of the absolute probability. This is the first evidence the rare-event gbdt cells are
tradeable.

## The two sizing arms

| Cell | rank/equal (c=1.0) | rank_kelly (c=0.5) | eval R-p@3 |
|---|---|---|---|
| cell5_revalreg | +12.3% (DD −5.2%) | +6.6% (DD −13.4%) | 0.568 |
| ndx40pct50d | +61.5% (DD −7.6%) | **0.0% (0 trades)** | 0.338 |
| r1k_50pct200d | +49.1% (DD −21.6%) | +26.7% (DD −19.0%) | 0.732 |
| sp500_50pct50d | +12.5% (DD −9.3%) | **0.0% (0 trades)** | 0.208 |
| sp500_20pct25d | +58.1% (DD −7.3%) | +18.0% (DD −10.1%) | 0.350 |

- **`equal`** (each position = `gross_cap/K`, p-independent) fully unlocks every cell —
  it deploys the rank regardless of absolute probability. It is the headline.
- **`rank_kelly`** (Kelly on the cell's *eval* R-p@K, leak-free) is a **principled +EV
  gate**: it trades only when the rank bucket's historical hit-rate clears breakeven
  (`R-p@3 > 1/3`). It correctly zeroes ndx40 (0.338 ≈ breakeven) and sp500_50 (0.208 <
  1/3) — those buckets are not +EV under the label payoff even though their *ranking*
  (AUC) is strong. Where it trades it is more conservative (lower return, similar/again
  DD). It is the "don't bet a −EV bucket" discipline; `equal` ignores that discipline and
  wins here because the windows were favorable.

## Composition (ruling out one-lucky-pick)

- **ndx40 equal** (+61.5%): 6 picks / 6 names, **5 winners** — AZN +104%, AMAT +42%,
  INTC +33%, AVGO +25%, ANSS +11%. AZN is a large outlier but 5 of 6 won; mean +34%.
- **sp500_20 equal** (+58.1%): 13 picks / 8 names, 69% win — a **semiconductor cluster**
  (INTC +34%, MU +27%, ADI +26%, AMAT +23%×2). A sector bet that paid this window.
- **r1k equal** (+49.1%): 12 / 10, 67% win — META +59%, QXO +55%, MSTR/NVDA +54%.

High win rates align with the cells' high R-p@1 — the ranking is real, not a single
fluke. But small N (4–13 picks), sector concentration, and one big outlier (AZN) mean
these are **single favorable windows, not an alpha claim** (see caveats).

## Methodology

Added to `TopKDailyKellyLabelExit` (backward-compatible, defaults unchanged):
`selection_mode="rank"` (top-K by `p_mean`, no `p>breakeven` gate),
`sizing_mode ∈ {kelly, equal, rank_kelly}`, `rank_kelly_p` (per-pick win prob), and an
auto-off **breakeven exit** in rank mode (else every sub-breakeven position exits
immediately). `run_backtest_cell.py` exposes `--selection-mode/--sizing-mode/--rank-kelly-p`;
`rank_kelly_p` defaults to the cell's **eval** R-p@K (NOT test — leak-free). Exits in rank
mode are DD/target/horizon only. Each cell on its published `predictions/test.csv` window
(OOS by construction of the gbdt loop), calibrator fit on VAL.

## Caveats

- **C1: single windows, small N — NOT a persistent-alpha claim.** 4–13 entries per cell,
  one window each (the russell1000 window is 2023-07→2024-10; the rest 2025-12→2026-03 or
  2025-06→2026-03). The big numbers need **rolling multi-window validation** before any
  deployment claim. This is the immediate follow-up.
- **C2: bull-market tailwind + concentration.** Every window is a rising tape; equal-weight
  c=1.0 rides 4–13 concentrated picks (semis in sp500_20) to their +target exits. Returns
  are high-variance; r1k drew down −21.6%.
- **C3: `equal` has no +EV discipline.** It bets every top-K rank regardless of whether the
  bucket is +EV; it won here because the ranking was good AND the tape rose. `rank_kelly` is
  the safer default for unfavorable regimes — it refuses −EV buckets. The right production
  choice is likely `rank_kelly` (or `equal` with a regime/`R-p@K`-floor gate).
- **C4: russell1000 index is a ^SPX proxy** (^RUI uncached); C5: zero costs, DD-not-bounded
  on gap-downs — as prior memos.

## Reproducibility

- Branch `backtests-v12-rank-sizing`. Full grid: `results/backtests/_005_rank/grid.csv`.
- `uv run python -m scripts.backtests.run_backtest_cell --cell <cell> --out <dir> --name <n> --selection-mode rank --sizing-mode equal --c 1.0` (or `--sizing-mode rank_kelly --c 0.5`).
- **Action charts** (per run, added retroactively via `scripts/backtests/plot_actions.py`, see `_020`): `results/backtests/_005_rank/<short>_{equal,rankkelly}/figs/actions.png` — the `equal` arms show the deployed ranking's buy/sell points; the two −EV `rank_kelly` arms (ndx40, sp500_50) render as a flat cash line (correctly 0 trades).

## Open questions / follow-ups

- **Rolling multi-window validation** (the gate to any alpha claim) — re-score + re-run each
  cell across several disjoint windows; report the distribution, not one number.
- **Per-rank `rank_kelly_p`** (R-p@1 for the #1 pick, R-p@3 bucket for #2–3) instead of one
  scalar — sizes the highest-conviction pick larger.
- **Rank-fallout exit** (exit when a held name drops out of the top-N rank) as a
  rank-native alternative to the breakeven exit.
- **Sector neutralization** for sp500_20 (semis concentration) to test breadth vs sector bet.
- **Cost/turnover sensitivity** for the higher-turnover rank_kelly arm (sp500_20: 35 entries).
