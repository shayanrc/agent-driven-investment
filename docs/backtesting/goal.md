# backtesting — Goal

This document states what `backtesting` is optimizing for and what trade-offs are unacceptable. Read it before editing any file under `src/backtesting/`, `tests/backtesting/`, `configs/backtesting/`, or `docs/backtesting/`.

For *how* it works (architecture, component APIs, execution lifecycle), see `spec.md`. This file is the *why* and *what success looks like*.

---

## What this module is optimizing for

Provide a **correct, configurable, multi-asset backtesting engine** that simulates order execution against historical data with one defining trade-off rule:

> **Look-ahead-bias elimination is structural, not conventional.** The engine must make it impossible to trade on information the caller hasn't observed, rather than relying on the caller to avoid doing so. Any change that weakens this guarantee — even if it simplifies the API or improves performance — is not acceptable.

That single rule shapes the two-phase execution lifecycle (D2 in `spec.md`), the master timeline design (D3), and the correctness constraints (B1–B6).

## What success looks like

- **B1–B6 constraints hold unconditionally.** No combination of valid inputs can violate look-ahead (B1, B2), causal fill (B3), portfolio consistency (B4), deterministic replay (B5), or lot-size integrity (B6). These are tested, not asserted.
- **The step loop is the only interface.** All interaction flows through `Backtest.step()` and `Backtest.reset()`. No backdoor methods that mutate state outside the lifecycle.
- **The engine is asset-class-agnostic.** Equities, crypto, forex, futures, macro — anything expressible as a DataFrame with a date index. Per-instrument lot sizes (whole shares, fractional, round lots) are the only asset-class-specific knob.
- **The engine is strategy-agnostic.** It does not compute rewards, signals, or performance metrics. It executes orders and reports fills. The caller owns strategy logic and evaluation.

## What this module is *not*

- **Not a forecasting engine.** Forecasts are upstream (`analog_mc`, `gbdt`). This module consumes trading signals; it does not produce them.
- **Not an alpha-research tool.** It evaluates strategy performance; it does not discover strategies.
- **Not a live-trading gateway.** The `ExecutionBroker` simulates fills against historical data; it does not connect to any brokerage API.
- **Not an RL environment.** The step-loop is naturally Gym-compatible, but the engine does not subclass `gymnasium.Env`, compute rewards, or define observation/action spaces. A Gym adapter wrapper is a separate concern.
- **Not a margin/risk engine.** v1 has no margin requirements, borrow costs, or position limits. Short selling is unconstrained. These are documented v1 limitations, not design oversights.

## How to apply this when working on the module

- **Every change must preserve B1–B6.** If a change makes any constraint harder to verify or easier to violate, it needs a stronger justification than convenience.
- **Don't add strategy logic.** No reward functions, no Sharpe calculations, no drawdown metrics inside the engine. The caller owns evaluation.
- **Don't add data sourcing.** The engine takes DataFrames. A convenience constructor wrapping `data_pipelines.fetch()` is fine; making `data_pipelines` a hard dependency is not.
- **Don't add intrabar resolution without re-evaluating `current_close`.** The `current_close` execution mode is valid only under the end-of-bar assumption (the caller has seen the close before trading at it). Intrabar support would require rethinking this mode.
- **Don't silently change the execution lifecycle.** The 6-step order in `spec.md` § 5 is the spec. If implementation reveals a problem, surface it and ask before reordering.
