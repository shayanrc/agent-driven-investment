# analog_mc — Goal

This document states what `analog_mc` is optimizing for and what trade-offs are unacceptable. Read it before editing any file under `src/analog_mc/`, `tests/analog_mc/`, `configs/analog_mc/`, `dashboards/analog_mc/`, or `docs/analog_mc/`.

For *how* it works (architecture, stages, constraints), see `IMPLEMENTATION_PLAN.md`. For the step-by-step math, see `ALGORITHM.MD`. This file is the *why* and *what success looks like*.

---

## What this module is optimizing for

Develop a Monte Carlo simulation method that produces **calibrated probabilistic 60-day price-path forecasts** for any daily-price asset, with one defining trade-off rule:

> **Fat-tail performance is a primary objective. Aggregate-CRPS gains that come at the cost of fat-tail anchor regressions are not acceptable trades.**

That single rule shapes nearly every architectural choice in the module. A model that lowers mean CRPS by 10% but blows out the 90% coverage band at 2020-03-16 COVID, 2010-04-23 flash crash, or 2001-10-02 post-9/11 bottom is a worse model, not a better one.

## What "calibrated" means here

The success surface is a **bundle of diagnostics**, not a scalar. A candidate must pass the bundle; CRPS is the tie-breaker between candidates that do.

- **Flat global PIT** histogram
- **Flat conditional PIT** in each vol-regime tercile (low / mid / high σ)
- **Reliability diagram** on track at {0.1, 0.25, 0.5, 0.75, 0.9} quantiles
- **σ-clip-hit fraction < 15%** on either bound
- **Squared-return ACF** tracking realized at lag 1..50, or a structural gap explicitly documented
- **Per-vol-regime CRPS** broken out, not pooled
- **15-anchor fat-tail eval** (8 |z₅₀|>3 programmatic + 7 hand-curated regime anchors) — 50%/90% band coverage and per-anchor CRPS reported. Defined in [`FAT_TAIL_EVAL.md`](FAT_TAIL_EVAL.md). **Mandatory for every v4+ experiment.**

The implementation lives in `decision_rules()` in `diagnostics.py`. When the rules fire, they specify the upgrade that's warranted.

## The decision-rule architecture

Upgrades are gated on diagnostics, not on ideas that sound promising. Each candidate complexity addition has to clear two bars:

1. **Trigger:** some specific diagnostic must fire on the current canonical baseline.
2. **Acceptance:** the new variant has to fix the trigger AND pass the calibration bundle AND not regress >2 fat-tail anchors.

The v4 experiments (B1 Platzer local-linear, A2.1 corrwindow matcher, B5 joint) are the canonical cautionary tale: all three improved fat-tail anchors materially, but each regressed elsewhere; **none promoted**. v2.4 Cell-D-s30 remains canonical. See [`V4_RESULTS.md`](V4_RESULTS.md).

## What this module is *not*

Not a trading system.

- No PnL, position sizing, transaction costs, signal generation, strategy logic.
- No backtests of trading rules.
- No data sourcing layer (single local CSV; a multi-source loader is a deferred separate module).
- No alpha extraction.

`analog_mc` is the **probabilistic-forecast primitive** that some downstream sizing/strategy module will eventually consume. Its job is to hand that consumer a forecast distribution it can actually trust the shape of.

## Eventual deployment shape

Currently exposed as: CLI (`python -m analog_mc walk-forward`), Streamlit dashboard, ad-hoc scripts. These reflect the **research-and-validation phase**.

Intended end state: an **agent-callable tool/skill** that runs a forecast on demand for a given asset / origin date / horizon. `analog_mc.simulate.forecast` is the entry point at the right granularity for that wrapping — preserve it. A tool-shaped output will likely want:

- The path distribution (or summary quantiles).
- The chosen analog dates and weights — interpretability is a feature of analog methods, not a nice-to-have.
- Which diagnostic gates the forecast passes / fails on the calibration bundle.

When designing new APIs, prefer shapes that wrap cleanly as a tool call later: clean function signatures, JSON-serializable outputs, no hidden CLI state.

## How to apply this when working on the module

- When evaluating any modeling change, compute the calibration bundle AND the 15-anchor fat-tail panel. Don't ship a candidate that improves mean CRPS but regresses on >2 anchors. See `FAT_TAIL_EVAL.md` for the precise mandatory deliverable.
- When tempted to add complexity, check the decision-rule architecture first. If no diagnostic is firing for the upgrade, don't build it.
- When asked "should we make this change?" — frame the answer around what the diagnostic bundle and 15-anchor panel say, not around aggregate CRPS alone.
- The 6 critical correctness constraints (C1–C6 in `IMPLEMENTATION_PLAN.md`) are non-negotiable. Causality (C1) is the single most important property and the canonical silent-failure mode for resampling pipelines.
- Don't silently change architectural decisions documented in `IMPLEMENTATION_PLAN.md`. Surface the deviation and ask first.
