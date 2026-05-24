# gbdt — V0 Investigation Plan

## Purpose

**v0 = data exploration that PRECEDES v1 implementation.** Establishes base rates, validates assumptions, and motivates the v1 target lattice + universe + walk-forward scheme. v0 is investigation, not implementation. Code under v0 is one-shot scan-quality (clear enough to reproduce, not production-discipline), and v0 deliverables are analyses + reports + headline metric JSONs — never library/module APIs.

Findings from v0 either confirm or revise design choices in `V1_PLAN.md`. If v0 surfaces something that contradicts a v1 plan decision, the plan changes — not the data.

## v0 task inventory

| ID | Task | Status | Deliverables |
|---|---|---|---|
| **v0.1** | Rolling-window opportunity scan — NIFTY 50, +10% in {10, 20, 50, 100} days | shipped (this PR) | `scripts/gbdt/v0_opportunity_scan.py`, `results/gbdt/data/_v0_opportunity_scan_data.json`, `docs/gbdt/_v0_opportunity_scan.md` |

(Add new rows here as v0 tasks are commissioned.)

## What v0 outputs inform

- **v0 base rates → v1 calibration baseline.** The bar Brier scores must beat in `V1_PLAN.md` Stage 9 acceptance is "better than predicting the base rate." v0 measures that base rate empirically.
- **Per-horizon event rarity → v1 lattice validation.** If the v0 scan finds e.g. that "+10% in 10 days" happens 40% of the time across the universe, that's a near-coin-flip target where calibration matters but discrimination is hard; if it happens 2%, the rarer-event problem dominates. Either outcome shapes how Stage 7 diagnostics are presented.
- **Per-stock variance in base rates → v1 universe choice.** If base rates are stable across stocks, single-asset (NASDAQ100) v1 generalizes. If they're wildly stock-specific, cross-sectional pooling becomes a v2 priority sooner.

## File / output conventions

- **Scripts:** `scripts/gbdt/v0_<name>.py` (single-file) or `scripts/gbdt/v0_<name>/` (multi-file).
- **Reports:** `docs/gbdt/_v0_<name>.md`.
- **Headline machine-readable:** `results/gbdt/data/_v0_<name>_data.json` (per `[[project-results-layout]]`).
- v0 scripts can hit the cache directly via SQL — they don't need to go through the `forecasters` skill surface or any other framework dispatch.
- v0 reports do not require formal acceptance criteria (those are for v1 stages); the report's "Findings" section is the deliverable.

## What v0 is NOT

- Not production code. Don't import v0 scripts from `src/gbdt/`; if a v0 utility turns out to be reusable, it gets rewritten in `src/gbdt/` as part of a v1 stage.
- Not feature engineering. v0 looks at raw price events, not at features that would predict them.
- Not model training. v0 is event counting + descriptive stats. v1 trains models.
- Not strategy backtesting. v0 counts opportunities; what trading would do with them is downstream and out of scope here per the project-wide anti-rule (see `goal.md`'s "What this module is *not*").
