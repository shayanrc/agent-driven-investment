# _291 — champion-matrix pilot: +20%/25d/dd10% across universes (#56)

**Branch:** `champion-matrix-pilot` · **Task:** #56 (pilot row, gating the full matrix)

**One-liner:** The deployed sp500 +20%/25d/dd10% champion **cell** (the target definition)
transfers strongly to the other US universes — nasdaq100 posts the best book of any
universe including sp500 itself, russell1000 matches sp500 — but only weakly to nifty500.
The champion's **HP** (d8/ss0.85) does NOT transfer: default d6 beats it on the nasdaq/
russell book. Verdict for the full matrix: **GO, baseline-only** (cells transfer, configs
don't — so sweep the matrix at the controlled-baseline config and finetune winners
per-universe afterwards).

## Setup

One matrix row: the +20%/25d/dd10% cell on nasdaq100 / russell1000 / nifty500, matched
token `all` (same as the sp500 champion so the universe is the only variable), canonical
explicit-boundary split (train 2015-01-01 → test 2024-07-01→2025-06-30), min_rows 2591,
snapshot 2026-07-06. Two configs per universe via `final_fit_canon`: the controlled
baseline (`d6/mcw1/ss1/cs1`) and the sp500 champion's adopted HP (`d8/mcw1/ss0.85/cs1`,
the #50 winner). sp500 reference: the deployed canon_ft champion.

## Test books (2024-07 → 2025-06)

| universe / config | base_rate | R-p@1 | @3 | @5 | @10 | @20 |
|---|--:|--:|--:|--:|--:|--:|
| sp500 — champ d8/ss0.85 (deployed ref) | 0.048 | 0.321 | 0.313 | 0.311 | 0.299 | 0.330 |
| **nasdaq100 — d6 (winner)** | 0.068 | **0.439** | **0.417** | **0.414** | **0.531** | **0.712** |
| nasdaq100 — champ d8/ss0.85 | 0.068 | 0.377 | 0.380 | 0.401 | 0.491 | 0.705 |
| **russell1000 — d6 (winner, book)** | 0.058 | 0.312 | 0.312 | **0.317** | **0.308** | **0.294** |
| russell1000 — champ d8/ss0.85 | 0.058 | **0.332** | **0.316** | 0.316 | 0.296 | 0.286 |
| **nifty500 — champ d8/ss0.85 (winner)** | 0.077 | **0.201** | **0.173** | **0.180** | **0.207** | **0.258** |
| nifty500 — d6 | 0.077 | 0.181 | 0.150 | 0.165 | 0.192 | 0.233 |

## Pilot question 1 — does the cell carry signal across universes? **YES for US, weak for NSE.**
- **nasdaq100 is the standout**: 0.439 @1 on a 0.068 base rate, with an exceptional deep
  book (0.712 @20) — the best +20%/25d book of any universe, sp500 included.
- **russell1000 ≈ sp500**: 0.31–0.33 @1 on a lower base rate; the cell generalizes.
- **nifty500 is weak**: 0.201 @1 on the HIGHEST base rate (0.077) — modest skill. Consistent
  with `_289`/`_290`: nifty500's edge lives in the F18-IN fundamentals cells, not
  short-horizon technicals.

## Pilot question 2 — does the champion's HP transfer? **NO.**
d6 default beats the sp500-adopted d8/ss0.85 at every K on nasdaq and on the russell book
(@5–@20; champ takes @1/@3 — the classic mid-@1 trade, book rules per the recipe). Only
nifty prefers the champ HP, at every K. This re-confirms the playbook's "no universal
recipe" rule at the cross-universe level: **tune per-universe, transfer the cell.**

## Backtests (2025-07-01 → 2026-07-14, per-universe winner, `--sizing-mode equal`, H=25)

| universe (winner cfg) | strategy | maxDD | EW basket | exits target/DD/horizon |
|---|--:|--:|--:|--:|
| nasdaq100 (d6) | **+40.9%** | −14.7% | +34.9% | 81 / 111 / 7 |
| **russell1000 (d6)** | **+50.9%** | −21.4% | +20.2% | 103 / 141 / 1 |
| nifty500 (champ) | −6.7% | −25.7% | +6.2% | 36 / 82 / 11 |

russell1000 is the backtest standout (+30.7pp over its basket); nasdaq beats its (very hot)
basket by +6pp; nifty500 confirms the weak transfer (negative absolute, worse than basket).
None meets the strict target>DD cut (the 25d/10% geometry churns through DD-stops — same
shape as the deployed sp500_20). NB: the harness's "EW top-K no-Kelly" reference emitted
0 trades on these runs (a harness quirk at H=25 with these artifact dirs, not investigated
here) — the strategy vs EW-basket comparison is the read.

## Full-matrix recommendation: **GO — baseline-only grid**

The pilot's two answers compose into a cheap full-matrix design: run the remaining
champion cells (+50%/50d, +40%/50d, +40%/100d, +50%/200d, +40%/200d-F18) × 4 universes at
the **controlled baseline only** (~16 new single fits + base builds; no HP-transfer arm —
it failed here), judge test books, then `/canonical-finetune` only the winners
per-universe. Registry rows carry `mode=pilot`.

## Caveats
- Single test window per cell (the standing caveat); winners need the usual second-window
  check before any promotion.
- `^NDX` reference benchmark in the backtest harness (esp. misleading for nifty500).
- nasdaq100's small universe (86 deep-history tickers) inflates book metrics at high K
  (min(R,K) denominators are small) — compare @1–@5 across universes, treat @10–@20
  within-universe.
