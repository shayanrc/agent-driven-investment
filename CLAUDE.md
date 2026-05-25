# Agent-Driven Investment — Project Conventions

This repo hosts multiple forecasting/analytics modules — each independently versioned and namespaced:

- **analog_mc** — probabilistic price-path forecasting (analog Monte Carlo). [`docs/analog_mc/goal.md`](docs/analog_mc/goal.md)
- **data_pipelines** — generic time-series ingestion framework; `us_equities` + `nse_equities` domains shipped. [`docs/data_pipelines/goal.md`](docs/data_pipelines/goal.md)
- **forecasters** — agent-callable forecasting surface; presets as saved-model artifacts; analog_mc wired as backend #1. [`docs/forecasters/goal.md`](docs/forecasters/goal.md)
- **gbdt** — categorical-outcome GBDT classifiers for probability of `{±10/20/50%}` in `{10/20/50}` days. [`docs/gbdt/goal.md`](docs/gbdt/goal.md)

## Module goal docs (read first)

Before editing any file under a module's directory tree (`src/<module>/`, `tests/<module>/`, `configs/<module>/`, `dashboards/<module>/`, `docs/<module>/`, `results/<module>/`, `scripts/<module>/`), read `docs/<module>/goal.md` first. It defines what the module is optimizing for and which trade-offs are unacceptable. `IMPLEMENTATION_PLAN.md` / `V<N>_PLAN.md` describe *how* the module works; `goal.md` defines the success criteria your change must respect.

`docs/<module>/V<N>_TBD.md` is the parking lot for follow-ups discovered in a branch but out of scope for it. `docs/<module>/V0_INVESTIGATION_PLAN.md` (where present) frames pre-v1 data-exploration scans whose outputs inform v1 design (see `gbdt` for an example).

## Module namespacing

Every module-specific directory nests under the module name. The top-level directories are reserved for cross-module concerns (shared launchers, fixtures, etc.).

```
src/<module>/                  # package code
docs/<module>/                 # design docs (V1_PLAN.md, IMPLEMENTATION_PLAN.md) + per-experiment reports (_<id>_<name>.md)
tests/<module>/                # tests
configs/<module>/*.yaml        # YAML configs
scripts/<module>/              # ad-hoc orchestration / aggregation / report-rendering / v0 investigation scripts
runs/<module>/<timestamp>/     # raw per-fold run artifacts (gitignored)
results/<module>/data/         # aggregated experiment JSONs (_<id>_data.json) — checked in
dashboards/<module>/
  ├── app.py                   # module's runnable Streamlit entry point
  └── views/                   # view modules
dashboards/app.py              # thin global launcher
```

When adding new module-scoped files, always nest them under `<top>/<module>/`. Never put module-specific code in the top-level `dashboards/`, `tests/`, `configs/`, `scripts/`, or `runs/`.

**Docs vs results.** `docs/<module>/_<id>_<name>.md` holds the experiment narrative (setup, mechanistic reading, decision-rule verdict). `results/<module>/data/_<id>_data.json` holds the machine-readable headline metrics that back the narrative. New aggregate scripts write to `results/<module>/data/` by default.

## Claude Code skills

Agent-callable verbs live at `.claude/skills/<name>/SKILL.md` and are invoked as `/<name>` from any session.

- **One skill = one verb.** Inference, fitting, listing, fetching, health-checking are separate skills.
- **Skill is module-owned even when bundled in a cross-module PR.** SKILL.md files live in the shared `.claude/skills/` namespace (because that's where Claude Code reads them), but the runner script lives under the owning module (e.g. `/fetch-data` SKILL.md + `scripts/data_pipelines/skill_runner.py`). Ownership tracks the implementation, not the SKILL.md location.
- **Long-running skills (`/tune-preset` and the like) MUST bake in the loop-pattern guidance** per `.claude/memories/feedback-experiment-agent-loop.md` so sub-agents that invoke them set up properly from launch.
- **No backend-internal hyperparameter flags** on the public skill surface — overrides via `--config-overrides path.yaml`, never `--n-eff 50`.

## Data and configs

- **analog_mc / gbdt**: read from a local CSV (default `data/NASDAQ100.csv`, FRED-style: `observation_date`, `NASDAQ100`). The loader takes `date_col` and `close_col` from config — asset-agnostic.
- **`data_pipelines`**: the general-purpose loader has shipped. New modules that need historical price data should use `data_pipelines.fetch(identifier, start, end)`. `analog_mc` and `gbdt` v1 stay on the CSV-first contract per `[[project-data-source]]`; wiring them to `data_pipelines` is a separate per-module plan.
- **Cache layout**: SQLite at `data/processed.db` (per-domain tables) + immutable per-provider raw downloads at `data/raw/<provider>/...`. Single-writer-per-`data_root` is the contract — concurrent seed processes risk SQLite `BUSY` contention; WAL serializes correctness-wise.

## Plans and branches

Each new plan doc (`V<N>_PLAN.md`, `V<N>_EXPERIMENTS_PLAN.md`, `ABLATION_STUDIES_PLAN.md`, etc.) gets its own git branch — don't stream commits to main. Plan + its implementation + reports land together as a PR. Pure refactors of already-merged work may go on main; the rule is per-*plan*, not per-*change*.

Merged plan/feature branches stay around for reference — don't offer to delete them after merge (local or remote). The history is wanted. See `[[feedback-branch-retention]]`.

Follow-ups discovered during a branch that are out of scope for that branch go in `docs/<module>/V<N+1>_TBD.md` (parking lot), not in scratch files or new branches. Promote to a real `V<N>_PLAN.md` when a coherent slice is big enough to be "a project, not a chore."

## Commit and PR attribution

Never attribute commits or PRs to Claude / an AI / a code-generation tool. No `Co-Authored-By: Claude …` footers, no `🤖 Generated with [Claude Code]` lines, no "generated by"/"written by AI" markers anywhere in commit messages or PR titles/bodies. Commit messages and PR text should read as the user's own work.

## Source of truth for analog_mc

`docs/analog_mc/IMPLEMENTATION_PLAN.md` is the spec for the analog_mc pipeline and was the output of a long design conversation. Every decision in it was made for a reason. **Do not silently change architectural decisions.** If implementation reveals a problem with a decision, surface it explicitly and ask before deviating.

Follow the 11-stage build order in that doc strictly — the diagnostic infrastructure (Stages 6, 9) is what makes the pipeline trustworthy, not the optimizer (Stage 7). Don't skip ahead.

The 6 critical correctness constraints (C1–C6 in the plan) are non-negotiable:
- C1: causal rolling features (zero look-ahead)
- C2: n_eff parameterization for distance → probability
- C3: per-analog vol scaling = demean → clip ratio → rescale → add drift
- C4: running EWMA σ for blocks 2+
- C5: strictly forward sampling
- C6: walk-forward boundary discipline

## Environment

- Python ≥3.12 via uv. Venv at `.venv/`, lockfile at `uv.lock`.
- `analog_mc`, `data_pipelines`, `forecasters`, and `gbdt` are all installed as editable packages via hatchling (`pyproject.toml` `[tool.hatch.build.targets.wheel] packages` list) — `import <module>.foo` works from anywhere.
- Run things with `uv run <cmd>` (e.g., `uv run pytest`, `uv run streamlit run dashboards/analog_mc/app.py`, `uv run python -m data_pipelines fetch ...`, `uv run python -m scripts.gbdt.v0_opportunity_scan`).
- **gh CLI is installed and authenticated** as `shayanrc` (per per-user memory `gh-cli-installed`). Use `gh pr create / view / diff / comment / review` directly for GitHub ops; no need to surface compare URLs. Merges still happen via user action (no autonomous `gh pr merge`).
- **Worktree workflow for parallel agents**: when launching long-running sub-agents that need their own working tree, create a sibling worktree (convention: `wt-<scope>/` next to the main checkout) and symlink `data/` + `.env` from the main checkout so the shared cache and secrets carry over. The lockfile is shared via git, so `uv sync --frozen` in the worktree sets up an isolated `.venv/`.

## Memories

Project-shared facts (architecture decisions, layout conventions, workflow rules) live in `.claude/memories/` (indexed in `.claude/memories/INDEX.md`) or as bullets in this file. The detail in `.claude/memories/` includes the *why* and *how-to-apply* per topic; CLAUDE.md is the summary.

Per-user/per-machine items (personal preferences, role context, machine paths) stay at `~/.claude/projects/<hash>/memory/`. Don't duplicate project facts there — refer to the project memories instead.

## What not to do — analog_mc

(Module-specific anti-patterns. Future modules append their own `## What not to do — <module>` sections rather than dumping into a shared list.)

- Don't use scikit-learn's `StandardScaler` — it batch-fits across the array. Implement causal z-scoring directly.
- Don't swap grid search for BayesOpt — the grid was chosen for diagnostic interpretability.
- Don't implement v2 features (trailing-momentum drift, conditional block sampling, tail inflation) in v1. They are gated on specific diagnostic findings; premature implementation contaminates the diagnostics that decide whether v2 is needed.
- Don't report aggregate CRPS as the headline result without PIT and weight-trajectory diagnostics.
- Don't add transaction costs, position sizing, or PnL to this pipeline. Those belong downstream.
