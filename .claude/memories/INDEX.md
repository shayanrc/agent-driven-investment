# Project Memories — Index

Shared, checked-in project memory. Auto-loading is not guaranteed for this folder (Claude Code's auto-memory reads from `~/.claude/projects/<hash>/memory/`), but `CLAUDE.md` at the repo root references this folder so future sessions know to consult it for project-scoped context.

When adding a new memory, append a one-line pointer below. Before adding, check that the fact isn't already in `CLAUDE.md`, in `docs/<module>/goal.md`, or derivable from the code — those are not memory material.

## Project facts

- [project-overview.md](project-overview.md) — Repo hosts multiple forecasting modules; pointer to per-module `docs/<module>/goal.md` for module specifics
- [project-data-source.md](project-data-source.md) — v1 reads from `data/NASDAQ100.csv`; full multi-source loader deferred
- [project-streamlit.md](project-streamlit.md) — Local dashboard designed for concurrent runs (per-run lockfile, no global mutex)
- [project-results-layout.md](project-results-layout.md) — Aggregated experiment JSONs live in `results/<module>/data/`, separated from `docs/<module>/` narratives
- [project-fat-tail-eval.md](project-fat-tail-eval.md) — Operational how-to for the mandatory 15-anchor fat-tail eval panel (scripts, fig paths, regeneration rules)
