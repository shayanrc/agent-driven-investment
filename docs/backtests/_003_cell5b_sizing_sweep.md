# _003: cell5b_sizing_sweep (fresh-OOS quarter-Kelly + p_low)

## TL;DR

Follow-up to `_002` on the **same fresh-OOS data** (b_acceptance_agent scored on the
refreshed panel, `2026-03-13 → 2026-06-12`), sweeping two knobs: fractional Kelly
`c ∈ {0.5, 0.25}` and the entry filter `selection_bound ∈ {mean, p_low}`. The
requested variant (quarter-Kelly + p_low) returns **+13.2%** (vs `_002`'s +9.0%) at a
lower drawdown — an improvement. **But the attribution overturns the premise:**
**quarter-Kelly is the entire gain, and p_low selection actively *hurts*.** The best
config is **c=0.25 + mean selection: +25.4%, max-DD −3.6%, avg exposure 0.58 — the
first config that BEATS NDX buy-and-hold (+21.6%) and at less than half its
drawdown.** The headline reframes `_001`/`_002`: the cell-5 underperformance was
substantially a **sizing artifact** — half-Kelly over-concentrates into 1–2 positions
near the gross cap, while quarter-Kelly fits ~2–3× more positions → broad
diversification (25 tickers) that captures the rally's breadth with lower drawdown.

## Sweep result (the key table)

$100,000 start, gross, window `2026-03-13 → 2026-06-12` (comparison_end = data end):

| Config | Total % | Max DD | Avg gross exp | ret/\|DD\| | entries / tickers |
|---|---|---|---|---|---|
| **c=0.25, mean** (headline) | **+25.4%** | **−3.6%** | 0.58 | **7.1** | 36 / 25 |
| c=0.25, p_low (requested) | +13.2% | −5.2% | 0.52 | 2.5 | 20 / 15 |
| c=0.5, p_low | +11.5% | −7.3% | 0.28 | 1.6 | — |
| c=0.5, mean (= `_002`) | +9.0% | −7.3% | 0.41 | 1.2 | 16 / 14 |
| — *NDX buy-and-hold* | +21.6% | −7.4% | 1.00 | 2.9 | 1 |
| — *EW basket* | +15.4% | −6.5% | 1.00 | 2.4 | 92 |
| — *EW top-K (no Kelly)* | +7.1% | −5.2% | 1.00 | 1.4 | — |

**Reading the grid:**
- **`c` is the dominant lever.** Holding selection at `mean`, dropping `c` from 0.5 → 0.25
  takes the result from +9.0% / −7.3% to **+25.4% / −3.6%** — nearly 3× the return at half
  the drawdown. Mechanism: half-Kelly's notional (~0.78·equity at median `p`) lets only
  1–2 positions fit under the gross cap; quarter-Kelly halves each position so ~2–3× more
  fit → 25 names vs 14, exposure 0.58 vs 0.41, and the drawdown drops because no single
  position dominates.
- **`p_low` selection HURTS.** At `c=0.25` it cuts the return from +25.4% → +13.2%; at
  `c=0.5` it raises return slightly (+9.0% → +11.5%) but collapses exposure (0.41 → 0.28).
  The cell's tiny anti-AUC model produces wide credible bands for *everyone*, so filtering
  on `p_low > breakeven` mostly discards winners and starves the book of capital — the
  opposite of the §10-Q10a hypothesis on this data.

## Composition check (c=0.25, mean — ruling out a lucky bet)

36 entries across **25 unique tickers**, median position notional 0.33 (max 0.82), win rate
**68.8%** (22/32 closed). Realized by exit: target ×11 (+15.2% mean), breakeven ×15 (+2.5%),
DD ×6 (−7.8%). Top contributors spread across FTNT +30%, TEAM +24%, AMD +19%, CRWD +14%,
TTWO +13% — diversified, no single name carrying the result.

## Methodology

Identical pipeline to `_002` (inferred fresh predictions, calibrator fit on cell VAL,
TopKDailyKellyLabelExit, next_open engine). One `src/` change: a `selection_bound` knob on
the strategy — the entry filter clears breakeven against `p_low` instead of `p_mean` when
set to `"low"` (ranking and Kelly sizing stay on `p_mean`). Default `"mean"` preserves
`_001`/`_002` behavior. Runner gains `--c` and `--selection-bound`.

## Caveats

- **C1: short, single fresh window** (64 signal days, ~3 months; tail partially
  marked-to-market). +25.4% beating NDX on *one* 3-month OOS window is **not** a claim of
  persistent alpha — it is one clean data point that the *sizing* (not the signal) drove
  `_001`/`_002`'s underperformance. Needs repetition across rolling windows before any
  stronger claim (now a one-command inference + back-test as the cache ages).
- **C2: `c=0.25` is near the gross cap's sweet spot for THIS `p` distribution.** The benefit
  is "more positions fit," which depends on the median Kelly notional vs the cap; a different
  cell / `p` scale could put the optimum elsewhere. Don't hard-code c=0.25 as a default —
  it is data-dependent (cf. `[[project-gbdt-tuning-playbook]]` "no universal recipe").
- **C3: p_low's harm is calibrator-specific.** Wide bands come from the tiny anti-AUC model +
  the Beta posterior; a tighter-band calibrator might make `p_low` selection neutral-to-helpful.
- **C4/C5:** zero costs (turnover modest, verdict cost-insensitive) and DD-not-bounded-at-−5%
  (gap-downs) — same as `_001`/`_002`.

## Reproducibility

- Branch `backtests-v11-qkelly-plow`. Fresh predictions: `results/backtests/_002_fresh/fresh_predictions.csv`.
- Headline: `uv run python -m scripts.backtests.run_fresh_oos --cell <b_acceptance_agent> --predictions results/backtests/_002_fresh/fresh_predictions.csv --out results/backtests/_003_cell5b_qkelly_mean --name cell5b_qkelly_mean --c 0.25 --selection-bound mean`
- Requested variant: same with `--out …_qkelly_plow --c 0.25 --selection-bound low`.
- Artifacts: `results/backtests/_003_cell5b_qkelly_mean/` (headline), `…_qkelly_plow/` (requested).
- **Action charts** (strategy equity + NDX buy-hold + labeled buy/sell points; added retroactively via `scripts/backtests/plot_actions.py`, see `_020`): `results/backtests/_003_cell5b_qkelly_mean/figs/actions.png` (headline) + `…_qkelly_plow/figs/actions.png`.

## Open questions / follow-ups

- **Promote c=0.25 + mean to the headline sizing for future cell-5 back-tests** and re-run
  `_001`/`_002` windows under it to see if the bull-market underperformance was sizing all along.
- **Rolling fresh-OOS series**: re-score monthly and track whether c=0.25 keeps beating NDX, or
  whether this window was favorable.
- **Drop p_low selection** from the candidate-knob list for tiny-anti-AUC cells (it hurts);
  revisit only with a tighter-band calibrator.
