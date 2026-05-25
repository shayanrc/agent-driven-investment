# Project Memories — Index

Shared, checked-in project memory. Auto-loading is not guaranteed for this folder (Claude Code's auto-memory reads from `~/.claude/projects/<hash>/memory/`), but `CLAUDE.md` at the repo root references this folder so future sessions know to consult it for project-scoped context.

When adding a new memory, append a one-line pointer below. Before adding, check that the fact isn't already in `CLAUDE.md`, in `docs/<module>/goal.md`, or derivable from the code — those are not memory material.

## Project facts

- [project-overview.md](project-overview.md) — Four modules in flight: analog_mc + data_pipelines (shipped); forecasters + gbdt (PRs open). Per-module `docs/<module>/goal.md` for specifics
- [project-data-source.md](project-data-source.md) — analog_mc + gbdt v1 read from `data/NASDAQ100.csv`; data_pipelines (the deferred loader) has landed; per-module switchover to `data_pipelines.fetch()` is a separate plan
- [project-streamlit.md](project-streamlit.md) — Local dashboard designed for concurrent runs (per-run lockfile, no global mutex)
- [project-results-layout.md](project-results-layout.md) — Aggregated experiment JSONs live in `results/<module>/data/`, separated from `docs/<module>/` narratives
- [project-fat-tail-eval.md](project-fat-tail-eval.md) — Operational how-to for the mandatory 15-anchor fat-tail eval panel (scripts, fig paths, regeneration rules)

## Workflow feedback

- [feedback-branch-retention.md](feedback-branch-retention.md) — Don't propose deleting merged plan/feature branches; history is wanted
- [feedback-experiment-agent-loop.md](feedback-experiment-agent-loop.md) — Bake loop-pattern guidance (background shell + Monitor filter + ScheduleWakeup + parallel-worker w/ per-item timeout) into sub-agent launch prompts up front; SendMessage isn't exposed
- [feedback-track-agent-tasks.md](feedback-track-agent-tasks.md) — Every sub-agent launch → immediate TaskCreate + status=in_progress; mark completed only when the work is actually closed out
