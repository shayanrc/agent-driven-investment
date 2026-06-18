# _020: capital metering — does spreading the bet thinner capture more edge?

## TL;DR

The deployed champion (`_005`/`_008`: rank-select, equal-weight, **K=3, c=1.0**) puts
**~33% of equity into each of 3 names** — so it is ~fully invested after one signal day
and **cannot take a new signal until one of the three exits**. A slow loser ties up a
third of the book for up to the full horizon. The hypothesis: **meter capital out** —
deploy only **5% per position, ≤15%/day** (still K=3/day), let the book accumulate up to
~20 names — to stop leaving fresh top-ranked signals on the table.

It does capture far more trades. **It also lowers return on both cells, and *erases* the
edge on the rarer one.**

| Cell (OOS 2025-12-30 → 2026-05-22) | Champion 33%/name | Variant 5%/name, 15%/day | SPX b-h |
|---|---|---|---|
| **sp500_20** (+20%/25d) | **+58.1%** (DD −7.3%) | +22.3% (DD −8.1%) | +8.4% |
| **sp500_50** (+50%/50d) | **+12.5%** (DD −9.3%) | **−3.3%** (DD −10.3%) | +8.4% |

The variant takes **7×** the trades (sp500_20: 13 → 92 entries, 8 → 37 names) and runs at
*higher* average exposure (0.55 → 0.66) — the champion genuinely sat idle. But the extra
trades are progressively lower-ranked picks where the edge is weak, so breadth dilutes
return. On **sp500_50** the dilution turns a winner into a loser that underperforms the
index. **Concentration is the source of the edge, not a bug to fix.**

## The single knob

Same runner, same cell, same OOS test window, same geometry (target / stop / horizon),
same rank selection, **no regime gate** (orthogonal; SPX was risk-ON throughout this
window). Only the per-position slice changes. In `equal` sizing mode the slice is
`fractional_c · gross_cap / K`:

| | `c` | slice/name | new $/day (K=3) | book at `gross_cap=1.0` |
|---|---|---|---|---|
| Champion | 1.0 | 33% | ≤100% | ≤3 names (locked day-1) |
| Variant | 0.15 | **5%** | **≤15%** | ≤~20 names (metered in) |

So the variant needs **no new code** — `--c 0.15` gives 5% slices, and the K=3 entry cap
already enforces ≤15% new deployment per day.

## Per-cell turnover + exit mix

| | entries | unique names | avg gross exp | exit triggers |
|---|---|---|---|---|
| sp500_20 champion | 13 | 8 | 0.55 | target 7 / DD 2 / horizon 4 |
| sp500_20 variant | 92 | 37 | 0.66 | target 32 / **DD 34** / horizon 25 |
| sp500_50 champion | 4 | 4 | 0.44 | target 1 / horizon 3 |
| sp500_50 variant | 23 | 22 | 0.47 | DD 3 / target 1 / **horizon 19** |

## Why metering loses — and loses harder on the rarer cell

- **The edge lives in the very top ranks.** These are high-AUC rare-event cells; precision
  falls off steeply below rank ~3 (`_011`/`_012`). The champion bets 33% on each top-3
  name; the variant spreads 5% across ranks 4–20, which are close to noise. This is the
  documented `_012`/`_014` result — *"wider-K is a risk-reducer, not an edge-creator;
  concentration monetizes precision."* The metered book is effectively a wide-K (~13–20
  name) strategy.
- **The DD-stop count is the signature.** sp500_20 DD-exits jump **2 → 34**: the variant
  buys a wide basket of weak names that just drift to the −10% stop.
- **Rarer event → steeper penalty.** sp500_50 (+50% target, base rate ~2.6% vs sp500_20's
  ~8.9%) has an even steeper precision-vs-rank curve, so dilution is more punishing: of 23
  metered positions only **1 hit the +50% target** while **19 timed out at horizon** — the
  book bought low-conviction "+50% candidates" that mostly went nowhere for 50 days. Net
  −3.3%, below the index.
- **It did not even reduce drawdown** (−7.3% → −8.1%; −9.3% → −10.3%) — breadth added
  losers faster than it smoothed the curve.

**What the variant does buy:** less single-name dependence. The champion's +58% rode just
8 names (one/two big winners dominate — a real fragility flagged in `_005`/`_008`); the
variant's +22% spreads over 37. It is breadth-over-conviction — a genuinely different
risk/return profile, just a worse one on these windows. And its turnover is **~10× higher**
(sp500_20: 92 entries + 91 exits + 375 trims vs 13 + 13 + 33), so under realistic costs it
would erode far more than the champion, which `_015` showed is cost-robust at 50 bps/side.

## Action charts (this memo also introduces the tooling)

Per-cell two-panel champion-vs-variant action charts (strategy equity + SPX buy-hold +
labeled buy ▲ / sell ▼ points, exit trigger suffixes `·t` target / `·D` drawdown-stop /
`·h` horizon):

- `results/backtests/_020_capital_metering/sp500_20_actions.png`
- `results/backtests/_020_capital_metering/sp500_50_actions.png`
- Equity-only overlays: `…/sp500_20_champion_vs_variant_equity.png`, `…/sp500_50_champion_vs_variant_equity.png`

The charting is now a reusable verb, **`scripts/backtests/plot_actions.py`**, wired into
`run_backtest_cell.py` so every future single-window back-test emits `figs/actions.png` as
a standard artifact. It was back-filled across the single-window memos `_001`–`_005` (each
run dir's `figs/actions.png`; the 0-trade `_004`/`_005` cells render as a flat cash line —
the gate finding made visual).

## Caveats

- **C1: single window, bull tape.** One OOS window per cell on a rising SPX (+8.4%); the
  concentration advantage is largest exactly when a few top picks run. Rolling/bear
  re-tests (`_008`/`_016`) would be the honest extension before any general claim.
- **C2: zero costs.** Both runs are cost-free. Costs would widen the gap further against
  the high-turnover variant (`_015`).
- **C3: no regime gate.** Orthogonal to sizing and ON throughout this window; included in
  neither run for a clean apples-to-apples.
- **C4: the variant is not strictly dominated** — it is more diversified / less
  luck-dependent and still beats the index on sp500_20. It is a different objective
  (breadth), not simply "worse," but it does not improve risk-adjusted return here.

## Reproducibility

- Branch `backtests-action-charts`.
- Champion / variant (per cell):
  `uv run python -m scripts.backtests.run_backtest_cell --cell results/gbdt/experiments/<cell>_agentloop --out results/backtests/_020_capital_metering/<cell>_champion_c1.0 --name <…> --selection-mode rank --sizing-mode equal --c 1.0`
  and the same with `--c 0.15` → `…_variant_c0.15`. Cells:
  `sp500_up_20pct_25d_dd10pct`, `sp500_up_50pct_50d_dd25pct`.
- Headline metrics: `results/backtests/data/_020_data.json`.
- Action-chart tooling: `scripts/backtests/plot_actions.py` (CLI: `uv run python -m
  scripts.backtests.plot_actions <run_dir> [...]`), auto-emitted by `run_backtest_cell.py`.

## Open questions / follow-ups

- **Breadth↔conviction frontier**: sweep slice size (10% → ~7-name book, 20% → ~5-name)
  to find where added breadth stops costing top-rank edge — a middle ground may keep most
  of the concentration edge with less single-name risk.
- **Cost-loaded re-run** (`--cost-bps 25/50`) to quantify how much the variant's ~10×
  turnover bleeds.
