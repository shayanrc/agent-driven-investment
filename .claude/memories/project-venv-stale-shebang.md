---
name: project-venv-stale-shebang
description: If `uv run pytest` fails with bogus ModuleNotFound errors on this host, the .venv console-script shebangs are stale (mount-path move) — run `uv run python -m pytest`; durable fix `uv sync --reinstall`.
metadata:
  type: reference
---

**Status: FIXED 2026-06-22** via `uv sync --reinstall` (shebangs now point at the current
`/mnt/Workspace/Workspace/...` path; `uv run pytest` works again). **This recurs if the
repo is remounted at a different path or `.venv` is rebuilt under one** — keep the
diagnosis + fix. Migrated from per-user memory 2026-07-15.

`.venv` was originally built when the repo was mounted at
`/mnt/122CEE982CEE765F/Workspace/agent-driven-investment`; it later moved to
`/mnt/Workspace/Workspace/agent-driven-investment`. `.venv/bin/python` is a symlink to
`/opt/miniconda3/bin/python3.13` (path-stable, fine), but the **console-script shebangs**
(`.venv/bin/pytest`, etc.) hardcode the **old absolute path**, which stops resolving.

**Symptom:** `uv run pytest` falls through to a pyenv 3.11 pytest on PATH that lacks the
project deps → a wall of bogus `ModuleNotFoundError: sklearn / holidays / matplotlib /
catboost` collection errors. These are NOT broken tests — `uv run python -m pytest`
(which execs the path-stable `.venv/bin/python`) collects + runs the whole suite clean.

**How to apply:**
- Run tests as **`uv run python -m pytest …`** when in doubt — it always uses the correct
  interpreter + deps. Same caveat applies to any other `.venv/bin/<console-script>`
  (streamlit, etc.): prefer `uv run python -m <module>`.
- **Durable fix** (restores the `uv run pytest` form CLAUDE.md documents): regenerate the
  shebangs — `uv sync --reinstall`, or `rm -rf .venv && uv sync`. Don't do it while a
  test run or the `/daily-predictions` timer is using the env.
- Don't "fix" this by editing CLAUDE.md or adding deps — pyproject/uv.lock are correct;
  it's purely a local, gitignored `.venv` artifact.

Related host fact: [[project-external-data-fetch]].
