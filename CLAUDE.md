# Agent-Driven Investment — Project Conventions

This repo will host multiple forecasting/analytics modules. The first is **analog_mc** (analog Monte Carlo price-path forecasting). Treat every module as independently versioned and namespaced.

## Module namespacing

Every module-specific directory nests under the module name. The top-level directories are reserved for cross-module concerns (shared launchers, fixtures, etc.).

```
src/<module>/                  # package code
docs/<module>/                 # design docs (e.g., IMPLEMENTATION_PLAN.md) + per-experiment reports (_<id>_<name>.md)
tests/<module>/                # tests
configs/<module>/*.yaml        # YAML configs
runs/<module>/<timestamp>/     # raw per-fold run artifacts (gitignored)
results/<module>/data/         # aggregated experiment JSONs (_<id>_data.json) — checked in
dashboards/<module>/
  ├── app.py                   # module's runnable Streamlit entry point
  └── views/                   # view modules
dashboards/app.py              # thin global launcher
```

When adding new module-scoped files, always nest them under `<top>/<module>/`. Never put module-specific code in the top-level `dashboards/`, `tests/`, `configs/`, or `runs/`.

**Docs vs results.** `docs/<module>/_<id>_<name>.md` holds the experiment narrative (setup, mechanistic reading, decision-rule verdict). `results/<module>/data/_<id>_data.json` holds the machine-readable headline metrics that back the narrative. New aggregate scripts write to `results/<module>/data/` by default.

## Data and configs

- v1 reads from a single local CSV at `data/NASDAQ100.csv` (FRED-style: `observation_date`, `NASDAQ100`). No yfinance / multi-source dispatch — that's a deferred loader module.
- The data loader takes `date_col` and `close_col` from config, not hardcoded — keeps the pipeline asset-agnostic.

## Plans and branches

Each new plan doc (`V<N>_PLAN.md`, `V<N>_EXPERIMENTS_PLAN.md`, `ABLATION_STUDIES_PLAN.md`, etc.) gets its own git branch — don't stream commits to main. Plan + its implementation + reports land together as a PR. Pure refactors of already-merged work may go on main; the rule is per-*plan*, not per-*change*.

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
- `analog_mc` is installed as an editable package via hatchling — `import analog_mc.foo` works from anywhere.
- Run things with `uv run <cmd>` (e.g., `uv run pytest`, `uv run streamlit run dashboards/analog_mc/app.py`) or activate `.venv/`.

## Memories

Project-shared facts (architecture decisions, layout conventions, workflow rules) live in `.claude/memories/` (indexed in `.claude/memories/INDEX.md`) or as bullets in this file. The detail in `.claude/memories/` includes the *why* and *how-to-apply* per topic; CLAUDE.md is the summary.

Per-user/per-machine items (personal preferences, role context, machine paths) stay at `~/.claude/projects/<hash>/memory/`. Don't duplicate project facts there — refer to the project memories instead.

## What not to do (analog_mc)

- Don't use scikit-learn's `StandardScaler` — it batch-fits across the array. Implement causal z-scoring directly.
- Don't swap grid search for BayesOpt — the grid was chosen for diagnostic interpretability.
- Don't implement v2 features (trailing-momentum drift, conditional block sampling, tail inflation) in v1. They are gated on specific diagnostic findings; premature implementation contaminates the diagnostics that decide whether v2 is needed.
- Don't report aggregate CRPS as the headline result without PIT and weight-trajectory diagnostics.
- Don't add transaction costs, position sizing, or PnL to this pipeline. Those belong downstream.
