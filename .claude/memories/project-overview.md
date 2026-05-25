---
name: project-overview
description: Repo hosts multiple forecasting/analytics modules. Modules so far — analog_mc (forecasting) and data_pipelines (ingestion). Each module has its own docs/<module>/goal.md.
metadata:
  type: project
---

This repo hosts multiple forecasting/analytics modules. Currently two have shipped:

- **analog_mc** — analog Monte Carlo probabilistic price-path forecasting (first module).
- **data_pipelines** — generic time-series ingestion framework. Two domains shipped: `us_equities` (S&P 500 + indices via stooq/tiingo/yfinance) and `nse_equities` (NIFTY 50 + index via jugaad/nselib/yfinance, jugaad vendored at `vendor/jugaad-data`). Public `fetch(identifier, start, end)` dispatches by identifier prefix (e.g. `NYSE:AAPL`, `NSE:RELIANCE`, `INDEX:^SPX`). Landed in main via PR #1.

**For module-specific specs:** read `docs/<module>/goal.md` (the why) and `docs/<module>/{IMPLEMENTATION_PLAN,V<N>_IMPLEMENTATION_PLAN}.md` (the how). CLAUDE.md at the repo root has the short rules; the long-form why is in those plan docs.

For each module's parked follow-ups: `docs/<module>/V<N+1>_TBD.md` (see `[[feedback-branch-retention]]` for the parking-lot convention).

**How to apply:** When a new module gets added, give it its own `docs/<module>/goal.md` following the same pattern. The CLAUDE.md rule "read `docs/<module>/goal.md` before editing files in that module" applies generically. When suggesting where to source historical price data for a new analysis, prefer `data_pipelines.fetch(...)` over ad-hoc CSVs unless the user is explicitly working in analog_mc's CSV-first v1 (see `[[project-data-source]]`).
