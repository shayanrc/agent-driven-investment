---
name: project-overview
description: Repo hosts multiple forecasting/analytics modules — analog_mc, data_pipelines, forecasters, gbdt. Each has its own docs/<module>/goal.md. State as of 2026-05-26.
metadata:
  type: project
---

This repo hosts multiple forecasting/analytics modules. Four are in flight as of 2026-05-26:

- **analog_mc** *(shipped to main; latest experimental work: V5.A.2 ensemble cleanup on `v5-experiments`, not yet merged)* — analog Monte Carlo probabilistic price-path forecasting. First module.
- **data_pipelines** *(shipped to main via PR #1; v1.7 NSE added; `data-seed-nifty-total` adapter-patch branch open and seed-in-flight; PR #5 added `--back-extend`; PR #6 added NIFTY 50 deep-history seed bringing 44/50 tickers to ≥2,500 rows)* — generic time-series ingestion framework. Two domains shipped: `us_equities` (S&P 500 + indices via stooq/tiingo/yfinance) and `nse_equities` (NIFTY 50 + index family via jugaad/nselib/yfinance; jugaad vendored at `vendor/jugaad-data`). Public `fetch(identifier, start, end, back_extend=False)` dispatches by identifier prefix (e.g. `NYSE:AAPL`, `NSE:RELIANCE`, `INDEX:^SPX`, `NIFTY:NIFTY500`). Cache at `data/processed.db` (SQLite). Universe self-service quirks: see `[[project-nse-data-quirks]]`.
- **forecasters** *(shipped to main via PR #2)* — agent-callable forecasting surface. Preset = saved-model artifact (`backend + hyperparameters + fitted_on + validation_metrics`). 5 skills ship: `/forecast`, `/tune-preset`, `/list-presets` (forecasters-owned) + `/fetch-data`, `/data-health` (`data_pipelines`-owned, bundled). analog_mc wired as backend #1. Wire-format contract is runtime-validated JSON; typed dataclasses / `Forecaster` ABC / registry deferred to backend #2. NIFTY 500 acceptance demo PASS.
- **gbdt** *(spec-lock shipped via PR #7; implementation Stages 1–9 + pilot acceptance demo shipped via PR #8; v1 pilot experiments in flight 2026-05-26)* — categorical-outcome GBDT classifiers. v0 was 18-cell lattice EDA (shipped on main); v1 ships **experiment-loop infrastructure**, not the lattice. Each experiment = one `(universe, direction, threshold, horizon, max_drawdown?)` tuple driven via the `/gbdt-experiment` skill (single end-to-end orchestrator: data → iter-0 → agent-driven synced FS+HP loop → calibration → artifact). Locked decisions: CatBoost (ordered boosting), NIFTY 50 pooled panel (48/50 tickers post-back-extend), 279-col feature pool across 16 families (18 sub-family rows), 800+400+200+100 walk-forward split, conditional isotonic calibration gated by Spiegelhalter Z, 8-iter loop cap with plateau + degradation inner-stop, universe self-service for new NSE universes (curl from archives.nseindia.com). Pilot ran 488s (~8 min), 92→100 tests, native-passable calibration. Multi-experiment pilots (nifty50/100/midcap150/nifty500 cells with drawdown filters) under task #89–#92. Headline metric = calibration, not accuracy or AUC. References: `docs/gbdt/{goal,V1_PLAN,EXPERIMENT_SPEC,CATBOOST_HP_REFERENCE,V1.1_TBD,V0_INVESTIGATION_PLAN}.md` and `.claude/skills/gbdt-experiment/SKILL.md`.

**For module-specific specs:** read `docs/<module>/goal.md` (the why) and `docs/<module>/{IMPLEMENTATION_PLAN,V<N>_PLAN}.md` (the how). CLAUDE.md at the repo root has the short rules; the long-form why is in those plan docs.

For each module's parked follow-ups: `docs/<module>/V<N+1>_TBD.md` (see `[[feedback-branch-retention]]` for the parking-lot convention). For v0 pre-implementation data exploration: `docs/<module>/V0_INVESTIGATION_PLAN.md` (gbdt example).

**How to apply:**
- When a new module gets added, give it its own `docs/<module>/goal.md` following the same pattern. The CLAUDE.md rule "read `docs/<module>/goal.md` before editing files in that module" applies generically.
- When suggesting where to source historical price data for a new analysis, prefer `data_pipelines.fetch(...)` over ad-hoc CSVs unless explicitly working in `analog_mc` / `gbdt` v1's CSV-first contract (see `[[project-data-source]]`).
- When proposing a new skill, follow the conventions in CLAUDE.md § "Claude Code skills" (one verb per skill, runner script lives under owning module, long-running ones bake in `[[feedback-experiment-agent-loop]]`).
