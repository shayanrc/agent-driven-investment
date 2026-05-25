# gbdt — V0 Investigation Plan

## Purpose

**v0 = data exploration that PRECEDES v1 implementation.** Establishes base rates, validates assumptions, and motivates the v1 target lattice + universe + walk-forward scheme. v0 is investigation, not implementation. Code under v0 is one-shot scan-quality (clear enough to reproduce, not production-discipline), and v0 deliverables are analyses + reports + headline metric JSONs — never library/module APIs.

Findings from v0 either confirm or revise design choices in `V1_PLAN.md`. If v0 surfaces something that contradicts a v1 plan decision, the plan changes — not the data.

## v0 task inventory

| ID | Task | Status | Deliverables |
|---|---|---|---|
| **v0.1** | Rolling-window opportunity scan — NIFTY 50, +10% in {10, 20, 50, 100} days | shipped | `scripts/gbdt/v0_opportunity_scan.py`, `results/gbdt/data/_v0_opportunity_scan_data.json`, `docs/gbdt/_v0_opportunity_scan.md` |
| **v0.2** | Full grid opportunity scan — NIFTY 50, {up, down} × {5, 10, 20, 30, 50%} × {10, 20, 50, 100} days | shipped | `scripts/gbdt/v0_opportunity_scan_full.py`, `results/gbdt/data/_v0_opportunity_scan_full_data.json`, `docs/gbdt/_v0_opportunity_scan_full.md` |
| **v0.3** | Drawdown-filtered opportunity scan — same v0.2 grid, exclude events whose path took a half-threshold adverse excursion before target | shipped | `scripts/gbdt/v0_opportunity_scan_filtered.py`, `results/gbdt/data/_v0_opportunity_scan_filtered_data.json`, `docs/gbdt/_v0_opportunity_scan_filtered.md` |

(Add new rows here as v0 tasks are commissioned.)

Each v0 commit is a point-in-time snapshot against whatever NSE cache existed when the script ran. v0.1/v0.2 ran against the pre-backfill cache (data through 2025-12-31, ~1,492 rows per stock); v0.3 ran after the cache extended to 2026-05-22 (~1,500–1,600 rows per stock). The three deliverables are individually internally consistent (each matches its own JSON), but they are *not* a coordinated cache snapshot. Cross-report comparisons should account for the slightly wider universe in v0.3.

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
- Not snapshot-stable. v0 scripts re-render their headline JSON in place at `results/gbdt/data/_v0_<name>_data.json`; re-running a v0 script against a changed cache will silently overwrite the committed result. To compare across snapshots, rename or move the prior JSON first (or back it up), and `git status` after a re-run to surface drift.

## v1 outcome

v0 informed v1's target lattice (3 thresholds × 3 horizons × 2 directions, 18 cells total). v1 ships **experiment-loop infrastructure** — not the lattice itself as the deliverable. Each v1 experiment runs a single (universe, direction, threshold, horizon) tuple via the `/gbdt-experiment` skill; see `docs/gbdt/V1_PLAN.md` and `docs/gbdt/EXPERIMENT_SPEC.md`.
