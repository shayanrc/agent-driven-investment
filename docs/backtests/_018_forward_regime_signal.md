# _018: forward-looking regime signals — it was never a lag problem

## TL;DR

`_017` found a trend gate (price > SMA200) can't rescue the bear and blamed **lag** — at the
2022 top the SMA200 was still rising, so the gate kept entering into the decline. The proposed
fix: a **forward-looking / faster-reacting** signal that flips risk-off at or before the turn.
This memo builds three (realized **vol**, **drawdown**-from-high, cross-sectional **breadth**)
and tests them against the SMA200 baseline.

**Result: none rescues the bear — the fast/leading ones are worse. The lag hypothesis is
refuted; the real cause is `_016`'s.** There is no up-move edge anywhere in a bear to gate
*toward*, so no signal — trend, vol, drawdown, or breadth — can manufacture one. Regime gating
is a **risk-reducer, not an alpha-restorer, regardless of signal sophistication.**

- **Bull: every signal preserves the edge** (all ≥89% of windows beat) — they differ only in how
  much they gate (vol 2%, sma200/drawdown ~12%, breadth 31%). All four are safely deployable.
- **Bear: every signal is negative and mostly *worse* than ungated.** The faster the signal, the
  worse the whipsaw: drawdown (reacts after a −5% dip, then re-enters into the bounce) is the
  worst (sp500_50 **−36.3%**); vol next; breadth is the least-bad forward signal but still does
  not beat doing nothing.

The simple **SMA200 gate from `_017` remains the recommended overlay** — not because it rescues
the bear (it doesn't) but because it preserves the bull edge and gives the best bear-drawdown
reduction on the active cell, with the least whipsaw. This **closes the regime-gate thread**.

## Results (equal@K=3; bull = 2025–26 fresh, bear = 2022 retrain)

### sp500 +50%/50d

| signal | bull ret / %beat / %off | bear ret / %beat / maxDD |
|---|---|---|
| ungated | +34.2% / 92% / 0% | −22.2% / 53% / −32.8% |
| SMA200 (_017) | +34.4% / 100% / 11% | −25.9% / 38% / −33.9% |
| vol (20d > 0.20 ann) | +34.2% / 92% / 2% | −30.5% / 28% / −34.6% |
| drawdown (>5% off 60d high) | +34.4% / 100% / 12% | −36.3% / 28% / −40.3% |
| breadth (<50% above 50d MA) | +57.7% / 92% / 31% | −26.4% / 44% / −30.5% |

### sp500 +20%/25d

| signal | bull ret / %beat / %off | bear ret / %beat / maxDD |
|---|---|---|
| ungated | +73.7% / 100% / 0% | −2.3% / 50% / −32.3% |
| SMA200 (_017) | +81.0% / 100% / 11% | −8.3% / 50% / **−22.2%** |
| vol | +76.5% / 100% / 2% | −26.8% / 36% / −30.9% |
| drawdown | +81.0% / 100% / 12% | −26.1% / 45% / −39.0% |
| breadth | +71.8% / 89% / 31% | −8.1% / 50% / −33.3% |

See `figs/forward_regime.png`.

## Why no signal works (the mechanism, settled)

`_017` framed the failure as timing lag. But the forward signals **do** react fast — vol and
drawdown flip risk-off within days of the top, breadth deteriorates before it — and they fail
*harder*. So lag was never the root cause. The root cause is `_016`'s:

> In a sustained bear the target event (a large up-move in N days) **structurally does not
> occur**, so every name the model ranks into its top-3 is a stock that falls. There is no
> regime *within* the bear where the signal has edge.

Given that, a regime gate can only ever do two things: (1) **reduce the number of entries**
(less participation → usually, but not always, less drawdown), and (2) **whipsaw** — any signal
that re-enters on a bear rally walks straight into the next leg down. The faster/leading signals
maximize (2): the drawdown gate re-arms after each −5% bounce and re-enters at the worst moment,
which is why it is the *worst* performer despite "reacting" earliest. There is nothing for any
of them to gate *toward*.

## What this means for deployment

- **Keep the `_017` SMA200 gate.** Across four signal families it is the best risk/whipsaw
  trade-off: preserves the bull edge, gives the largest bear-drawdown reduction on the active
  cell (sp500_20 −32%→−22%), and whipsaws least. The forward signals add complexity and more
  whipsaw without rescuing the bear.
- **The honest deployment story is unchanged from `_016`/`_017`:** bull/range-bound alpha; the
  gate caps — does not eliminate — bear losses; *no* gate makes the bear profitable. The only
  true "fix" is not trading the strategy through a confirmed bear at all — which the SMA200 gate
  approximates but cannot perfect, because calling a bear in real time costs either lag (enter
  the top) or whipsaw (re-enter the rallies).
- **Stop tuning regime signals for this strategy.** The ceiling is set by the absence of a bear
  edge, not by signal quality. Further regime work has no expected payoff here.

## Caveats

- **One bear (2022)** — as in `_016`/`_017`. A 2020-style vol crash might let the vol gate flip
  faster relative to the decline; but breadth/drawdown already react fast here and still fail, so
  the conclusion (no bear edge to capture) is unlikely to flip.
- **Default params** (vol 20d/0.20, drawdown 60d/5%, breadth 50d/50%) were not swept — but the
  bear failure is so uniform and the mechanism so clear that parameter tuning is curve-fitting on
  one bear, explicitly out of scope.
- **Breadth gates 31% of bull days** (vs vol's 2%) — it is the most conservative; on a longer
  bull sample it could clip more of the edge than the short 2025–26 window shows (sp500_20 already
  dipped to 89%). Another reason to prefer the simpler SMA200.
- Survivorship in the 2022 retrains (current membership) flatters the bear, as before.

## The arc, closed on the regime question

`_016`→`_018`: the champion is bull/range-bound alpha. A regime gate (any of four signal
families) preserves that alpha in the bull and reduces — never reverses — bear losses. The
best overlay is the simplest: **SMA200**. The strategy is **bull/range-bound alpha with a
trend-gate that bounds bear risk** — and that is the honest ceiling, set by the structural
absence of an up-move edge in a bear, not by anything tunable.

## Reproducibility

- Branch `backtests-v22-forward-regime`. New `scripts/backtests/regime_signals.py`
  (`risk_on_sma` / `risk_on_vol` / `risk_on_drawdown` / `risk_on_breadth` + `compute_risk_on`
  dispatcher — all causal). `run_rolling_validation` gains `--regime-signal {sma,vol,drawdown,
  breadth}` + `--regime-window` + `--regime-thresh`. 11 unit tests in
  `tests/backtests/test_regime_gate.py` (causality + no-lookahead for every signal, dispatcher
  errors). Runs under `results/backtests/_018_fwd/`; registry rows 096–107. Figure
  `figs/forward_regime.png`.

## Open questions / follow-ups

- The regime-signal question is **closed for this strategy**. The remaining genuinely-open items
  are unchanged and external to the decision policy: **forward OOS** as the cache ages (recurring
  `ROLLING` node) and a **second bear** (2020/2008) to confirm the no-bear-edge result generalizes
  across bear types.
- If a bear edge is ever wanted, it must come from a **different target** (a down-move / short
  cell), not from gating the existing long-only up-move signal — a new module-level direction,
  not a backtest overlay.
