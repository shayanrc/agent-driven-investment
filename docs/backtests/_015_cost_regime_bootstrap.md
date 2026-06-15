# _015: validating the equal@K=3 champion — costs, bear sub-window, block bootstrap

## TL;DR

The `_005`–`_014` arc converged on one champion: **equal-weight top-3 (K=3) on the
high-R-p@1 cells**. This memo stress-tests it three ways — transaction costs, the worst
index sub-window, and a moving-block bootstrap for an honest effective-N. **It holds up,
with one honest demotion:**

- **Costs: PASS.** At a punishing 50 bps/side, the champions barely move (low turnover,
  9–22 trades): sp500_50 +34.2%→+31.7% (still 92% of windows beat), sp500_20 +73.7%→+66.0%
  (94%), r1k_50 +135.9%→+123.2% (87%), ndx40 +121.7%→+115.9% (57%). The high-turnover
  prob_weight K=20 contrast (486 trades) is **destroyed** — −1.4% → −13.9% (**0%** of
  windows beat) at 25 bps, −26.2% at 50 bps. Concentration isn't just better gross; it's
  vastly more cost-efficient.
- **Bootstrap: 3 of 4 PASS.** A moving-block bootstrap (block = horizon) on daily excess
  returns gives a 95% CI that **excludes zero for sp500_50, sp500_20, r1k_50** (bootstrap
  p(mean≤0) = 0.007 / 0.000 / 0.009). **ndx40 does NOT** (CI −16.9%…+125.5%, p=0.08) — the
  bootstrap correctly demotes the weakest cell (57% / 11 trades) to *not significant*.
- **Bear sub-window: PASS, but shallow.** In the index's worst peak-to-trough drawdown on
  each OOS, all four beat the index (sp500 cells lost less; ndx40 +33% vs −12%; r1k −13% vs
  −19%). But the worst available stretches are only −9% to −19% — not a true bear market.

**Net: the edge is cost-robust and statistically real on 3 of 4 cells even after deflating
for window overlap — but on a small effective-N (2–5 blocks) and a shallow bear.** It is a
validated edge with explicit N/regime caveats, not yet proven alpha.

## Cost sweep (per-side bps; rolling %windows-beat / median-excess)

| Cell | trades | 0 bps | 10 bps | 25 bps | 50 bps |
|---|---|---|---|---|---|
| sp500 +50%/50d | 9 | 92% / +7.3 | 92% / +7.0 | 92% / +6.7 | 92% / +6.5 |
| sp500 +20%/25d | 19 | 100% / +9.5 | 94% / +9.3 | 94% / +9.0 | 94% / +8.4 |
| ndx40 +40%/50d | 11 | 57% / +6.6 | 57% / +6.4 | 57% / +6.2 | 57% / +5.8 |
| r1k +50%/200d | 22 | 90% / +14.2 | 90% / +13.9 | 90% / +13.4 | 87% / +11.9 |
| **contrast: sp500_50 prob_weight K=20** | **486** | 38% / −6.9 | — | **0%** / −12.2 | **0%** / −18.4 |

The champions lose ~0.1–0.3 pts of median excess per 25 bps (turnover is tiny). The
diversified variant loses ~5–6 pts per 25 bps and crosses below the index almost immediately.

## Block bootstrap (annualized daily excess, moving block = horizon, 2000 reps)

| Cell | eff-N (blocks) | ann excess | 95% CI | p(mean≤0) | significant? |
|---|---|---|---|---|---|
| sp500 +50%/50d | 2.3 | +49% | [+6.2%, +62.7%] | 0.007 | **yes** |
| sp500 +20%/25d | 4.5 | +106% | [+52.2%, +122.4%] | 0.000 | **yes** |
| ndx40 +40%/50d | 5.1 | +47% | [−16.9%, +125.5%] | 0.080 | **no** |
| r1k +50%/200d | 3.6 | +13% | [+2.5%, +31.2%] | 0.009 | **yes** |

The effective-N is brutal — "13 / 18 / 42 / 105 rolling windows" deflate to **2–5
independent blocks** once you account for the stride-5-vs-horizon overlap. That is the
honest sample size. Three cells still clear significance because their per-block excess is
large; ndx40 (the cell that always lagged) does not, and is honestly demoted.

## Bear sub-window (index's deepest peak→trough on each OOS)

| Cell | index drawdown | index ret | strat ret | strat beat? |
|---|---|---|---|---|
| sp500 +50%/50d | −9.1% | −9.1% | −5.9% | yes (smaller loss) |
| sp500 +20%/25d | −9.1% | −9.1% | −0.3% | yes |
| ndx40 +40%/50d | −12.1% | −12.1% | +32.9% | yes |
| r1k +50%/200d | −18.9% | −18.9% | −13.2% | yes |

Consistent with `_010`'s regime-neutrality (the strategy is defensive-to-positive when the
index falls), but these are **shallow** drawdowns within a bull OOS — a genuine bear test
(2008/2022-scale) is not available in this data and remains the key gap.

See `figs/cost_bootstrap.png`.

## Caveats (what "validated" does and does not mean)

- **Effective-N 2–5 blocks** is the real sample. Significance on 3/4 cells is meaningful but
  rests on few independent episodes; out-of-sample-forward confirmation as the cache ages is
  the right next check (the `ROLLING` node).
- **Shallow bear only.** The worst sub-windows are −9% to −19%; regime-robustness in a real
  bear is suggested by `_010` + these sub-windows but not proven.
- **r1k uses a ^SPX proxy** (^RUI uncached) and a grown universe — as in `_008`.
- **Costs are a flat bps model** (commission+slippage lumped, no market-impact curve, no
  borrow). For a long-only, low-turnover, large-cap strategy 50 bps/side is conservative;
  small/mid-cap impact on r1k could be higher, but turnover is so low it stays immaterial.
- **ndx40 is now explicitly not-significant** — drop it from the "champion" set; the robust
  trio is sp500_50, sp500_20, r1k_50.

## The arc, settled

`_005`→`_015`: the tradeable edge is **high-R-p@1 cells (R-Precision@1, not AUC, selects
them) traded concentrated (equal-weight top-3)**. That edge: beats its index in 87–94% of
rolling windows after 50 bps costs, has a block-bootstrap CI excluding zero on 3 of 4 cells,
and is defensive in the worst sub-window — on a small effective-N and a shallow bear. Every
sizing elaboration (wider K, prob_weight, p^α) only diluted or de-risked it. **The champion
stands; the remaining honest gaps are sample size and a real bear, not the strategy.**

## Reproducibility

- Branch `backtests-v19-cost-regime-bootstrap`. Costs: `run_rolling_validation --cost-bps`
  (new; per-side bps via the engine's `commission_fn`) — also dumps `daily_equity.csv`.
  Bootstrap/bear: `uv run python -m scripts.backtests.bootstrap_regime` (seed 12345, 2000
  reps). Artifacts under `results/backtests/_015_validation/`; registry rows 061–079.

## Open questions / follow-ups

- **Forward OOS** as the cache ages (the `ROLLING` node) — the only way to grow the 2–5
  effective-N honestly.
- **A real bear** (extend the panel back to include 2022, or 2020) to test regime-robustness
  beyond shallow sub-windows.
- The strategy itself needs no further sizing work — this closes the V1.x backtest arc.
