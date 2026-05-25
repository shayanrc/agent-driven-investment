# Project Memories — Index

Shared, checked-in project memory. Auto-loading is not guaranteed for this folder (Claude Code's auto-memory reads from `~/.claude/projects/<hash>/memory/`), but `CLAUDE.md` at the repo root references this folder so future sessions know to consult it for project-scoped context.

When adding a new memory, append a one-line pointer below. Before adding, check that the fact isn't already in `CLAUDE.md`, in `docs/<module>/goal.md`, or derivable from the code — those are not memory material.

## Project facts

- [project-overview.md](project-overview.md) — Repo hosts multiple modules; analog_mc (forecasting) + data_pipelines (ingestion) shipped; per-module `docs/<module>/goal.md` for specifics
- [project-data-source.md](project-data-source.md) — analog_mc v1 reads from `data/NASDAQ100.csv`; data_pipelines (the deferred loader) has since landed but analog_mc hasn't switched over
- [project-streamlit.md](project-streamlit.md) — Local dashboard designed for concurrent runs (per-run lockfile, no global mutex)
- [project-results-layout.md](project-results-layout.md) — Aggregated experiment JSONs live in `results/<module>/data/`, separated from `docs/<module>/` narratives
- [project-fat-tail-eval.md](project-fat-tail-eval.md) — Operational how-to for the mandatory 15-anchor fat-tail eval panel (scripts, fig paths, regeneration rules)

## Workflow feedback

- [feedback-branch-retention.md](feedback-branch-retention.md) — Don't propose deleting merged plan/feature branches; history is wanted
