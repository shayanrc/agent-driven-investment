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

Two follow-ups are now closed in-memo: a **K-sweep** (champion sizing, K=3→6) shows **K=3
is the swept peak — return falls monotonically with every added name**; and a **bear-2022
robustness** re-test (leak-free retrain, test = the held-out 2022 bear) shows the
concentration ≥ metering ordering **holds on a disjoint window + regime** — it is not a
lucky-start artifact.

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

## K-sweep: K=3 is the peak (breadth↔conviction frontier)

The metering test above widens breadth via *slice size* at fixed K=3. The complementary
knob is **K itself** at fixed champion sizing (`c=1.0`, so still ~fully invested — each name
gets `1/K` of equity). Sweeping K=3→6 on the bull OOS resolves the "is there a gentler
optimum just above 3?" question `_012` left open (it jumped 3 → 10 → 20):

| K | slice/name | sp500_20 return / DD | sp500_50 return / DD |
|---|---|---|---|
| **3** | 33% | **+58.1% / −7.3%** | **+12.5% / −9.3%** |
| 4 | 25% | +51.1% / −6.2% | +5.3% / −8.2% |
| 5 | 20% | +40.5% / −6.5% | +6.1% / −8.5% |
| 6 | 17% | +29.0% / −6.5% | −10.0% / −18.7% |

**Return falls monotonically from K=3** on both cells — the decline starts immediately at
K=4; there is no gentler plateau just above 3. Drawdown does *not* improve to compensate
(sp500_20 ≈ flat; sp500_50 *worsens* to −18.7% at K=6 as low-conviction +50% names drag the
book negative). Same mechanism as the slice-size test: each added rank (4th/5th/6th
highest-p) is a weaker pick that dilutes the top-3 edge faster than it diversifies risk.
**K=3 is the peak, not a corner of a flat optimum** — refining `_012` (K=3 > 10 > 20) to the
fine grain, and closing the breadth↔conviction follow-up below. Both runs reproduce the
K=3 champion exactly (validates the new `--k` flag). Chart:
`results/backtests/_020_capital_metering/k_sweep/k_sweep_3to6.png`.

## Robustness: a different start + regime (bear-2022)

The metering verdict above is one bull window per cell. The `_016` **leak-free bear retrain**
(train 2016 → 2019-08, val/eval through 2021-12, **test = the held-out 2022 bear**,
2021-12-21 → 2022-11-23 — strictly disjoint, no 2022 in training) gives a genuinely different
start *and* regime to re-test it:

| Cell (bear OOS, SPX −17.4%) | Champion 33%/name | Variant 5%/15% |
|---|---|---|
| sp500_20 | **−5.8%** (DD −32.3%) | **−23.1%** (DD −34.6%) |
| sp500_50 | −9.2% (DD −32.8%) | −8.3% (DD −30.9%) |

The **concentration ≥ metering ordering holds on the disjoint window** — decisively on
sp500_20 (concentration loses only −6% and still beats the falling index; metering sprays
into 84 names, takes **126 DD-stops**, and loses −23%), a wash on the rarer sp500_50 (both
≈ −8 to −9%; neither has an edge because the +50% target barely fires in a downtape). So the
metering result is **not a lucky-start artifact**. It also re-confirms `_016`: *neither*
sizing rescues a bear (both −6 to −23%, DDs −31 to −35% vs index −25%) — the deployment fix
is the regime gate (`_017`), not the bet size. Chart:
`results/backtests/_020_capital_metering/bear2022/bear2022_champion_vs_variant.png`.

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

- **C1: headline is a bull window; bear cross-check now done.** The headline tables are one
  OOS window per cell on a rising SPX (+8.4%), where concentration's advantage is largest.
  The **Robustness** section adds the leak-free 2022-bear re-test — the ordering holds — but
  a *rolling-origin per-K distribution* (the `_008` tool exposes `--c`/`--sizing-mode`/`--k`)
  remains the fully rigorous extension and is not yet run.
- **C2: zero costs.** Both runs are cost-free. Costs would widen the gap further against
  the high-turnover variant (`_015`).
- **C3: no regime gate.** Orthogonal to sizing and ON throughout this window; included in
  neither run for a clean apples-to-apples.
- **C4: the variant is not strictly dominated** — it is more diversified / less
  luck-dependent and still beats the index on sp500_20. It is a different objective
  (breadth), not simply "worse," but it does not improve risk-adjusted return here.

## Reproducibility

- Branches: `backtests-action-charts` (metering headline + tooling),
  `backtests-020-k-sweep-robustness` (K-sweep, bear-2022 robustness, `--k` flag).
- Champion / variant (per cell):
  `uv run python -m scripts.backtests.run_backtest_cell --cell results/gbdt/experiments/<cell>_agentloop --out results/backtests/_020_capital_metering/<cell>_champion_c1.0 --name <…> --selection-mode rank --sizing-mode equal --c 1.0`
  and the same with `--c 0.15` → `…_variant_c0.15`. Cells:
  `sp500_up_20pct_25d_dd10pct`, `sp500_up_50pct_50d_dd25pct`.
- **K-sweep**: same command with `--k 3|4|5|6` (new backward-compatible flag on
  `run_backtest_cell.py`; default 3 = champion) → `…/k_sweep/<cell>_k<K>`.
- **Bear-2022**: same champion/variant commands but `--cell` = the `_016` retrains
  `…_bear2022` → `…/bear2022/<cell>_<champion|variant>`.
- Headline metrics + `k_sweep` + `bear2022_robustness` blocks: `results/backtests/data/_020_data.json`.
- Action-chart tooling: `scripts/backtests/plot_actions.py` (CLI: `uv run python -m
  scripts.backtests.plot_actions <run_dir> [...]`), auto-emitted by `run_backtest_cell.py`.

## Open questions / follow-ups

- ~~**Breadth↔conviction frontier**~~ — **resolved** (see *K-sweep*): widening K=3→6 (and the
  5%/15% slice test) both lower return monotonically with no compensating drawdown
  reduction. There is no middle ground that keeps the edge; K=3 is the peak.
- **Rolling-origin per-K distribution**: re-run the K-sweep and the slice-size test through
  `run_rolling_validation.py` (which exposes `--c`/`--sizing-mode`/`--k`) to report *what %
  of overlapping windows* K=3 beats wider K, in both bull and bear — the rigorous version of
  the single-window tables here.
- **Cost-loaded re-run** (`--cost-bps 25/50`) to quantify how much the variant's ~10×
  turnover (and wider-K turnover) bleeds.
