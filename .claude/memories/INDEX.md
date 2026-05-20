# Project Memories — Index

Shared, checked-in project memory. Auto-loading is not guaranteed for this folder (Claude Code's auto-memory reads from `~/.claude/projects/<hash>/memory/`), but `CLAUDE.md` at the repo root references this folder so future sessions know to consult it for project-scoped context.

When adding a new memory, append a one-line pointer below.

## Project facts

- [project-overview.md](project-overview.md) — What this repo is and what analog_mc does
- [project-data-source.md](project-data-source.md) — v1 reads from `data/NASDAQ100.csv`; full loader deferred
- [project-csv-schema.md](project-csv-schema.md) — NASDAQ100.csv schema; configurable column names
- [project-layout.md](project-layout.md) — Per-module namespacing convention
- [project-streamlit.md](project-streamlit.md) — Local dashboard, designed for concurrent runs
- [project-results-layout.md](project-results-layout.md) — Aggregated experiment JSONs live in `results/<module>/data/`
- [project-fat-tail-eval.md](project-fat-tail-eval.md) — Mandatory 8-anchor fat-tail eval for every v4+ forecasting experiment

## Implementation discipline

- [feedback-implementation-discipline.md](feedback-implementation-discipline.md) — Stage order, surfacing deviations
- [feedback-branch-per-plan.md](feedback-branch-per-plan.md) — Each new plan doc gets its own branch; don't commit to main
