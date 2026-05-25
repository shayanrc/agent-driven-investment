---
name: project-overview
description: Repo hosts multiple forecasting/analytics modules — analog_mc, data_pipelines, forecasters, gbdt. Each has its own docs/<module>/goal.md. State as of 2026-05-25.
metadata:
  type: project
---

This repo hosts multiple forecasting/analytics modules. Four are in flight as of 2026-05-25:

- **analog_mc** *(shipped to main; latest experimental work: V5.A.2 ensemble cleanup on `v5-experiments`, not yet merged)* — analog Monte Carlo probabilistic price-path forecasting. First module.
- **data_pipelines** *(shipped to main via PR #1; v1.7 NSE added; `data-seed-nifty-total` adapter-patch branch open and seed-in-flight)* — generic time-series ingestion framework. Two domains shipped: `us_equities` (S&P 500 + indices via stooq/tiingo/yfinance) and `nse_equities` (NIFTY 50 + index family via jugaad/nselib/yfinance; jugaad vendored at `vendor/jugaad-data`). Public `fetch(identifier, start, end)` dispatches by identifier prefix (e.g. `NYSE:AAPL`, `NSE:RELIANCE`, `INDEX:^SPX`, `NIFTY:NIFTY500`). Cache at `data/processed.db` (SQLite).
- **forecasters** *(PR #2 open on `v1-skills`)* — agent-callable forecasting surface. Preset = saved-model artifact (`backend + hyperparameters + fitted_on + validation_metrics`). 5 skills ship in the same PR: `/forecast`, `/tune-preset`, `/list-presets` (forecasters-owned) + `/fetch-data`, `/data-health` (`data_pipelines`-owned, bundled). analog_mc wired as backend #1. Wire-format contract is runtime-validated JSON; typed dataclasses / `Forecaster` ABC / registry deferred to backend #2. NIFTY 500 acceptance demo PASS.
- **gbdt** *(PR open on `gbdt-v1`)* — categorical-outcome GBDT classifiers for probability of `{±10/20/50%}` move in `{10/20/50}` trading days = 18-cell lattice per asset. v1 stages 1-9 still pending; the branch currently carries scaffolding + V0_INVESTIGATION_PLAN (v0.1 NIFTY 50 opportunity scan, v0.2 full direction × threshold × horizon grid, v0.3 drawdown-filtered). Headline = calibration, not accuracy or AUC.

**For module-specific specs:** read `docs/<module>/goal.md` (the why) and `docs/<module>/{IMPLEMENTATION_PLAN,V<N>_PLAN}.md` (the how). CLAUDE.md at the repo root has the short rules; the long-form why is in those plan docs.

For each module's parked follow-ups: `docs/<module>/V<N+1>_TBD.md` (see `[[feedback-branch-retention]]` for the parking-lot convention). For v0 pre-implementation data exploration: `docs/<module>/V0_INVESTIGATION_PLAN.md` (gbdt example).

**How to apply:**
- When a new module gets added, give it its own `docs/<module>/goal.md` following the same pattern. The CLAUDE.md rule "read `docs/<module>/goal.md` before editing files in that module" applies generically.
- When suggesting where to source historical price data for a new analysis, prefer `data_pipelines.fetch(...)` over ad-hoc CSVs unless explicitly working in `analog_mc` / `gbdt` v1's CSV-first contract (see `[[project-data-source]]`).
- When proposing a new skill, follow the conventions in CLAUDE.md § "Claude Code skills" (one verb per skill, runner script lives under owning module, long-running ones bake in `[[feedback-experiment-agent-loop]]`).
