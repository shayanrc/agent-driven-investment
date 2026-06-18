# _021: is there an entry-`p` threshold? — and does ranking by raw score help?

## TL;DR

Following the `_020` thread, we asked a simple risk-management question: **for the champion
(rank-select, equal-weight, K=3, c=1.0), what is the entry probability of the trades that
lose — is there a `p` threshold below which we shouldn't enter?** The answer is layered and
ends in a clean **negative result**:

1. **No within-entry threshold exists.** The calibrated `p` the strategy ranks on is quantized
   into a handful of isotonic plateaus, and the top plateau is **~102 names wide** (median),
   so every entry at any realistic K sits on the *same* `p`. Winners and losers share it;
   `p` has zero power to separate them. What decides win/loss is the **regime** (bull 76% /
   bear 34% win-rate at constant `p`), not the probability.
2. **The model's `p` *is* meaningful — but between plateaus**, and the strategy already exploits
   it maximally (K ≪ 102 ⇒ it only ever trades the top plateau).
3. **Within the top plateau the raw score still orders the label** (sp500_20 top raw-quintile
   hits 0.32 vs 0.23 plateau avg; sp500_50 0.24 vs 0.10). So we added a `--rank-by raw` option
   and tested it.
4. **It doesn't pay.** Ranking by raw score hits *more* targets in every cell+regime (the
   hit-rate signal is real) but **return is flat-to-worse, drawdown worse, and risk-adjusted
   return worse in all four cells** — catastrophically so in the sp500_20 bear (**−5.8% →
   −28.8%**). The highest-raw names are the highest-*beta* names: they reach the target more
   often *and* crash hardest. **Keep ranking on calibrated `p`.**

The single most important caveat: these are single bull + single bear OOS windows per cell
(small N), gross of costs — the verdict is "raw ranking is not an improvement here," not a
universal law.

## The question and the four findings

### 1. The calibrated `p` is one wide plateau — no within-entry threshold

Across 36,450 candidate (day, ticker) points in the sp500_20 bull test window there are only
**10 distinct calibrated `p` values** (conditional-isotonic calibration is a step function).
The **top plateau** — the only one the strategy ever enters — holds a **median of 102 names
per day** (range 18–174). Since K (3–20) ≪ 102, *every* entry is on that single top plateau:
sp500_20 bull all `p=0.1304`, bear all `p=0.3307`; sp500_50 bull all `0.0471`, bear all
`0.1362`. The tie-break is `(p desc, ticker asc)`, so within the plateau selection is **purely
alphabetical** — the champion holds the alphabetically-first names (AMAT, AMD, AVGO, BIIB…),
not the highest-conviction ones. K=5 (or K=20) cannot change this; you'd need K > ~102.

### 2. Regime, not `p`, separates winners from losers

With entry `p` constant within a window, win/loss is decided entirely by regime:

| Cell / regime | entry `p` (constant) | win-rate | n |
|---|---|---|---|
| sp500_20 bull | 0.1304 | 76% | (K=5) 21 |
| sp500_20 bear | 0.3307 | 34% | 73 |
| sp500_50 bull | 0.0471 | 43% | 7 |
| sp500_50 bear | 0.1362 | 40% | 25 |

A naive pooled split ("`p ≤ 0.13` → 69% win, `p > 0.13` → 33%") looks like a threshold but is
just a **bull-vs-bear label** — and the higher-`p` (bear) entries did *worse*, the opposite of
a usable filter. Figure: `results/backtests/_021_entry_p_threshold/figs/entry_p_vs_outcome.png`
(wins ▲ and losses ▼ stacked on the same vertical `p` stripe). K=5 confirms it stays one
plateau: `…/figs/k5_single_plateau.png`.

### 3. Between plateaus, the calibration is monotone (the real `p` structure)

Realized label hit-rate rises monotonically with the calibrated-`p` plateau, so the model's
`p` *is* informative — and the strategy already sits on the top plateau (the maximal threshold):

| sp500_20 plateau `p` | n | realized hit-rate |  | sp500_50 plateau `p` | n | realized hit-rate |
|---|---|---|---|---|---|---|
| **0.1304** (traded) | 7168 | **0.226** |  | **0.0471** (traded) | 5543 | **0.100** |
| 0.0698 | 5017 | 0.121 |  | 0.0060 | 5215 | 0.005 |
| 0.0516 | 4409 | 0.080 |  | 0.0045 | 1358 | 0.007 |
| … | … | … → 0.00 |  | … | … | … → 0.00 |

(base rate: sp500_20 0.089, sp500_50 0.026.) sp500_50 is near-binary — top plateau 0.100 then a
cliff to ~0.005. Figure: `…/figs/plateau_calibration.png`.

### 4. Within the top plateau, the *raw* score still orders the label

The calibrated `p` ties the whole top plateau, but the raw model score (`p_raw`, full
resolution) keeps ranking the outcome inside it:

| `p_raw` quintile within top plateau | sp500_20 hit | sp500_50 hit |
|---|---|---|
| Q1 (lowest raw) | 0.165 | 0.005 |
| Q3 | 0.217 | 0.097 |
| Q5 (highest raw) | **0.321** | **0.243** |
| *(plateau avg / alphabetical pick)* | 0.226 | 0.100 |
| corr(`p_raw`, label) | 0.125 | 0.272 |

Figure: `…/figs/within_plateau_raw_threshold.png`. This is the signal the strategy throws away
by ranking on the quantized calibrated `p` — which motivated the experiment below.

## The experiment: rank by `p_raw` vs `p_calibrated`

A new `rank_by` option ranks the entry top-K on the model's raw score instead of the quantized
calibrated `p` (sizing + the breakeven gate stay on calibrated `p` either way). Same champion
config (K=3, c=1.0, rank/equal), both cells, both regimes:

| Cell | Regime | calibrated (current) | raw (new) | targets hit | ret / |DD| |
|---|---|---|---|---|---|
| sp500_20 | bull | **+58.1%** (DD −7.3%) | +56.6% (DD −13.8%) | 7 → **9** | 8.0 → 4.1 |
| sp500_20 | **bear** | **−5.8%** (DD −32.3%) | **−28.8%** (DD −31.4%) | 5 → **12** | −0.18 → −0.92 |
| sp500_50 | bull | +12.5% (DD −9.3%) | +14.2% (DD −15.0%) | 1 → **2** | 1.34 → 0.94 |
| sp500_50 | bear | −9.2% (DD −32.8%) | −12.8% (DD −29.6%) | 0 → 0 | −0.28 → −0.43 |

(SPX: bull +8.4%, bear −17.4%. Calibrated rows reproduce the `_020` champion exactly.) Figure:
`…/figs/rank_by_comparison.png`.

**Verdict: raw ranking captures the within-plateau signal (more +target exits everywhere) but
is not an improvement — return is flat-to-worse, drawdown worse, and risk-adjusted return worse
in all four cells.** The sp500_20 bear is the killer: raw turns −5.8% into −28.8% (30
drawdown-stops vs 19).

**Mechanism.** The highest-`p_raw` names are the highest-conviction-per-model names, which are
also the highest-**beta** names — they hit +20%/+50% more often *and* fall hardest. The
calibrated-`p` alphabetical selection, which looked like a bug in finding 1, is **accidentally a
volatility diversifier**: it samples across the wide plateau instead of concentrating into the
most extreme names. Raw ranking removes that accidental protection. This is the textbook
**"hit-rate ≠ return"** — only the backtest, not the calibration curve, reveals it. (Same shape
as the `_013`/`_014` `prob_weight` dead-ends: a real-looking selection signal that doesn't
monetize.)

## Implementation

- **Strategy** (`src/trading_strategies/topk_daily_kelly_label_exit.py`): new optional
  `rank_scores: dict[Timestamp, dict[ticker, float]]` param. When set, the entry top-K **sort
  key** uses it (missing ticker → falls back to that row's `p_mean`); sizing + breakeven gate
  are untouched. Default `None` ⇒ behavior unchanged (the `_020` champion reproduces exactly).
- **Runner** (`scripts/backtests/run_backtest_cell.py`): `--rank-by {calibrated,raw}` (default
  `calibrated`). `raw` builds `rank_scores` from the predictions CSV's `p_raw` column. Recorded
  in `summary.json::config.rank_by`.
- **Tests**: `+2` in `tests/trading_strategies/test_topk_daily_kelly.py` (override the
  alphabetical tie-break via a higher raw score; missing-ticker fallback to `p_mean`). 33 pass.

## Caveats

- **C1: small-N, single windows.** One bull + one bear OOS window per cell, gross of costs.
  The verdict is "raw ranking is not an improvement on these windows," not a universal claim. A
  rolling-origin per-`rank_by` distribution is the rigorous extension (not run).
- **C2: hit-rate vs return.** Findings 3–4 are realized label hit-rates (leak-free: model raw
  output + a val-fit calibrator). Hit-rate ≠ PnL — the +target magnitude and the drawdown-stops
  mediate return, which is exactly why the raw-ranking edge evaporated.
- **C3: the alphabetical bias is real.** With ~102 names tied on calibrated `p`, the champion's
  per-name picks skew alphabetically (A/B/C-heavy). It is not a selection by per-name skill —
  the "edge" is the *plateau's* edge (0.226 vs base 0.089), sampled alphabetically.
- **C4: bear `bear2022` cells are leak-free** (train 2016→2019-08, test = the held-out 2022
  bear; see `_016`/`_020`) — the bear column is genuinely OOS, not 2022-in-training.

## Reproducibility

- Branch `backtests-rank-by-raw`.
- Champion rank-by sweep (per cell × regime × rank_by):
  `uv run python -m scripts.backtests.run_backtest_cell --cell results/gbdt/experiments/<cell> --out results/backtests/_021_entry_p_threshold/rank_by/<name> --name <…> --c 1.0 --k 3 --selection-mode rank --sizing-mode equal --rank-by {calibrated|raw}`
  Cells: `sp500_up_{20pct_25d_dd10pct,50pct_50d_dd25pct}_{agentloop,bear2022}`.
- Diagnostics (plateau calibration, within-plateau raw quintiles, entry-`p`-vs-outcome): rebuilt
  from each cell's `predictions/test.csv` via the runner's calibrator (`fit_calibrator` +
  `_predictions_dict`); `p_raw`/`p_calibrated`/`y_true` columns.
- Headline metrics + diagnostics: `results/backtests/data/_021_data.json`. Figures under
  `results/backtests/_021_entry_p_threshold/figs/`.

## Follow-ups

- **Raw ranking + a volatility/beta control** (vol-scaled sizing, or a drawdown-aware filter)
  is the only way the within-plateau hit-rate edge could pay — it would have to keep the extra
  targets without inheriting the extra beta. Speculative; a new experiment, not this one.
- **Rolling-origin per-`rank_by` distribution** to convert the single-window verdict into a
  win-fraction (the `_008` tool exposes `--c`/`--sizing-mode`/`--k`; `--rank-by` would need
  wiring there too).
