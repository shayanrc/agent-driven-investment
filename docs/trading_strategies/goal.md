# trading_strategies — Goal

This document states what `trading_strategies` is optimizing for and what trade-offs are unacceptable. Read it before editing any file under `src/trading_strategies/`, `tests/trading_strategies/`, or `docs/trading_strategies/`.

For *how* it works (strategy lifecycle, sizer protocols, exit logic), see `V1_PLAN.md`. This file is the *why* and *what success looks like*.

---

## What this module is optimizing for

Provide **concrete, composable `backtesting.Strategy`-conformant strategy classes** that turn a calibrated probability stream into orders, with one defining rule:

> **The strategy must be backend-agnostic in its probability contract.** A strategy class accepts a `dict[Timestamp, list[(ticker, p_mean, p_low, p_high)]]` (or a DataFrame with that shape) — NOT a model object, NOT a fitted calibrator, NOT a backend reference. If a strategy needs to know which predictor produced its inputs, the abstraction is broken.

This rule keeps strategies reusable across prediction backends (gbdt today, analog_mc fan-quantiles tomorrow) and across calibrators (Bayesian today, conformal in v1.1+).

## What success looks like

- **One strategy = one decision policy.** `TopKWithLabelExit` is one policy (top-K selection, label-anchored exits). Different exit rules, different selection rules → different strategy classes. No mega-strategy with feature flags.
- **Sizers are first-class, separable, swappable.** A strategy is constructed with a sizer instance. Swapping `VinceOptimalF` for `FixedFraction` for `DiscreteBoundedLossKelly` is a one-line change at the call site, with no strategy-code changes required.
- **Two sizer protocols, deliberately.** `PortfolioSizer` fits on returns and exposes a single fraction; `PerPredictionSizer` computes a fraction per prediction. Unifying them would force one or the other into an awkward API. The strategy dispatches on `isinstance(sizer, PortfolioSizer)`.
- **No execution, no PnL, no data fetching.** Strategies emit action dicts conforming to `backtesting.spec.md` § 3.1. The engine executes; the strategy decides.
- **Exit anchors are explicit and label-faithful.** A strategy that mirrors a gbdt label MUST anchor its exit rules to the SAME reference price the label uses (signal-day close, not fill open). Mis-anchoring is the most common silent foot-gun.

## What this module is *not*

- **Not a forecaster.** Strategies consume probabilities; they don't produce them.
- **Not an execution engine.** Engine lives in `src/backtesting/`. Strategies are callers of `backtesting.Strategy` Protocol — they do not subclass `Backtest` or call `process_queue` directly.
- **Not a calibrator.** Calibration lives in `src/calibration/`. A strategy is given a calibrator instance (already fit) and a `predictions` DataFrame; the strategy doesn't fit calibrators itself.
- **Not a backtest results processor.** Equity curve / drawdown / Sharpe calculations live in `backtesting/results.py` or per-backtest scripts. Strategies just emit orders.
- **Not where universe selection lives.** The universe roster is an input to the strategy (which 92 tickers to consider); the strategy doesn't build rosters.

## How to apply this when working on the module

- **No predictor imports.** `src/trading_strategies/` should never `import gbdt` / `import analog_mc`. Predictions come in via DataFrames; predictor identity is irrelevant.
- **Backend-agnostic prediction format.** Use `pd.DataFrame` with columns `[date, ticker, p_mean, p_low, p_high]`, or `dict[Timestamp, list[tuple]]` — never a custom predictor class.
- **Exit rules are documented, not implicit.** Every strategy class must document its exit triggers in priority order and its anchor convention in the docstring. No "obvious" assumptions.
- **Sizers compose with strategies, not the other way around.** The strategy takes a `sizer` argument; the sizer doesn't know what strategy it's powering.
- **State carried across `__call__` invocations is the strategy's responsibility.** The engine doesn't help with this — strategies maintain internal dicts of open positions, anchors, etc.

## What not to do

- **Don't import from `gbdt`, `analog_mc`, or `forecasters`.** The probability contract is the only interface.
- **Don't add transaction costs or PnL computation inside strategy code.** Those concerns belong to the engine and downstream reporting.
- **Don't put position-sizing math in the strategy class.** Sizers are separable; strategies dispatch to them. If `topk_label_exit.py` contains a Kelly formula, that's a sign sizing should be extracted.
- **Don't add reward functions, observation/action spaces, or RL hooks.** The `backtesting.Strategy` Protocol is a simple `(state, info) → action` callback. RL adapters belong in a separate module.
- **Don't conflate selection (top-K) with sizing (Kelly fraction).** Two separate concerns: selection filters which tickers to consider; sizing decides how much to allocate. Keep them in separate code paths.
- **Don't bake universe-specific behavior into a strategy.** A strategy that assumes "NASDAQ-100" is broken; it should work on any universe represented in the predictions DataFrame.
