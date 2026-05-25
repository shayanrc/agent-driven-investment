# forecasters — V1 Implementation Plan

## Build status

- **v1.0 (analog_mc backend):** drafted (this document). Not yet shipped.

For the *why* (what success looks like, anti-goals, deployment intent), see `goal.md`. This document is the *how* — architecture, contract, stages.

## Revision history

- **v1.0** *(2026-05-24, drafted)* — Initial plan: standalone forecasting skill surface with analog_mc wired as backend #1. Three forecasting skills (`/forecast`, `/tune-preset`, `/list-presets`) plus two bundled `data_pipelines`-owned skills (`/fetch-data`, `/data-health`) shipping in the same PR — both modules ship together because the agent-callable surface needs data + forecast skills to be usable end-to-end, but ownership stays per-module (each skill's implementation lives under its owning module). Preset-as-saved-model framing: preset YAML carries `backend`, `hyperparameters`, `fitted_on`, `fitted_at`, `validation_metrics`; `/forecast --preset <name>` dispatches via `preset.backend`. Wire-format contract documented as a JSON shape (not yet a typed dataclass — gated on backend #2). Canonical preset shipped: `v24-default.yaml` (analog_mc v2.4 on NASDAQ100 1990–2024). Branch: `v1-skills`.

---

## Purpose

This is an implementation specification for a **generic forecasting-surface module**. It is the user- and agent-facing entry point for running price-path forecasts. Each forecasting algorithm is a **backend** plugged into the same framework: implementing `forecast(input) → result` (seconds-to-minutes) and `tune(input) → preset` (minutes-to-hours).

**v1 scope:** ship the forecasters framework + three forecasting skills + one backend wired through (analog_mc), bundled with two `data_pipelines`-owned skills (`/fetch-data`, `/data-health`) in the same PR. The framework should not over-crystallize: build the dispatch layer as a thin `if/elif` table; defer the `Forecaster` ABC + `ForecasterRegistry` + typed wire-format dataclasses until backend #2 forces them.

### Bundled scope (cross-module)

This plan covers work in **two modules** that ship in one branch / one PR:

| Module | Why bundled |
|---|---|
| `forecasters` (new) | The primary module — framework, dispatcher, analog_mc backend adapter, preset machinery, three forecasting skills. |
| `data_pipelines` (additions) | Two skills (`/fetch-data`, `/data-health`) lifted onto the agent-callable surface. The skills exist as their own files but the implementation script and tests live under `data_pipelines/` — module ownership is unchanged. |

Bundling rationale: the agent-callable surface needs both halves to be useful end-to-end (you can't `/forecast --identifier NYSE:AAPL` without `data_pipelines` data being reachable, and you can't trust the data without `/data-health`). Shipping them in separate PRs would leave the in-between state half-functional. The forecasters work is the bigger half; data skills are documented inline in Stage 8.

The end-state public surface is five Claude Code skills, all callable via `/<skill>` from any session:

| Skill | Owner | Verb | Cost | Owns |
|---|---|---|---|---|
| `/forecast` | forecasters | Inference | seconds–minutes | Run a preset against data, return canonical result |
| `/tune-preset` | forecasters | Fitting | minutes–hours | Produce a new preset by tuning a backend on data |
| `/list-presets` | forecasters | Discovery | <1s | Enumerate canonical + user-tuned presets with metadata |
| `/fetch-data` | data_pipelines | Data load | seconds (cache hit) to minutes (cold) | Fetch on-demand for a single identifier; thin agent-callable wrapper over `data_pipelines.fetch()` |
| `/data-health` | data_pipelines | Diagnostic | <1s per identifier | Report cache coverage, last-fetch dates, provider-call failures per identifier (or for the whole cache) |

The plan is the output of a design conversation. Every decision documented here was made for a reason. **Do not silently change architectural decisions.** If implementation reveals a problem with a decision, surface it explicitly and ask before deviating.

---

## High-level architecture

Three layers in the design:

- **Skills** (the surface) — Claude Code SKILL.md files at `.claude/skills/<name>/`. Each is a thin wrapper that invokes the dispatcher script with appropriate args.
- **Framework / dispatcher** (`src/forecasters/`, `scripts/forecasters/run.py`) — preset loader, backend dispatch, result-shape validation, output cache, composition with `data_pipelines`.
- **Backends** (`src/<backend>/forecaster.py` — analog_mc in v1) — implement `forecast()` and `tune()` against their own data structures. Self-contained; no knowledge of the framework's call graph.

```
                          ┌────────────────────────────────┐
            user ────────▶│  /forecast --preset v24-default│  ← skill surface
                          │           --identifier NDX     │    (.claude/skills/)
                          │           --start --end        │
                          │           --origin --horizon   │
                          └─────────────┬──────────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │ scripts/           │  ← argparse, env validation
                              │   forecasters/     │
                              │     run.py         │
                              └─────────┬──────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │  preset loader            │  ← read YAML from
                          │  (src/forecasters/        │     configs/forecasters/presets/
                          │     presets.py)           │     OR results/forecasters/presets/
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │  data composition         │  ← if --identifier:
                          │  (src/forecasters/        │     data_pipelines.fetch(...)
                          │     data.py)              │     if --data-path:
                          │                           │     load CSV/parquet
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │  backend dispatch         │  ← read preset.backend
                          │  (src/forecasters/        │     if "analog_mc":
                          │     dispatch.py)          │       from analog_mc.forecaster
                          │  v1: if/elif table        │       import forecast
                          │  v2+: registry            │
                          └─────────────┬─────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  backend.forecast  │  ← analog_mc.forecaster
                              │  (input_dict)      │     .forecast(input_dict)
                              │  → result_dict     │     returns canonical dict
                              └─────────┬──────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │  result validation +      │  ← assert keys present,
                          │  cache write              │     dtypes correct;
                          │  (src/forecasters/        │     write to
                          │     cache.py)             │     results/forecasters/forecasts/
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          summary.json + fan_chart.png + paths.npz
                          (cache_key, backend, warnings, ...)
```

For `/tune-preset`, the same dispatcher routes to `backend.tune(input_dict) → preset_dict`, which is then written to `results/forecasters/presets/<output_name>.yaml`.

---

## Wire-format contract (v1: by convention, not typed)

The framework's stability surface is the input/output shape that every backend must conform to. In v1 this is **documented and enforced by runtime validation in the dispatcher**, not by a typed dataclass. Promotion to `src/forecasters/contract.py` (typed dataclasses + ABC) happens when backend #2 lands.

### `forecast(input_dict) → result_dict`

**Input:**

```python
{
    "data": pd.DataFrame,      # canonical OHLCV: columns per data_pipelines schema
                               # (date, open, high, low, close, adj_close, volume)
                               # for analog_mc, only `date` and `adj_close` are required
    "origin": str,             # ISO date "YYYY-MM-DD"; forecast starts the next session
    "horizon": int,            # number of trading days to forecast
    "hyperparameters": dict,   # backend-specific knob values from preset.hyperparameters
    "seed": int | None,        # for determinism; if None, backend picks
}
```

**Output:**

```python
{
    "paths": np.ndarray,                # shape (N, H), float64; N sample paths, H horizon
    "anchors": {
        "origin_date": str,             # ISO date (echo of input.origin)
        "horizon_dates": list[str],     # length H, ISO dates following origin
    },
    "summary": {
        "median": list[float],          # length H
        "p05": list[float], "p25": list[float],
        "p75": list[float], "p95": list[float],
        "crps": float | None,           # if backend computes it; None otherwise
    },
    "metadata": {
        "backend_name": str,            # "analog_mc"
        "preset_name": str,             # echo
        "preset_hash": str,             # content hash of the preset YAML
        "config_hash": str,             # backend-internal hash of effective hyperparameters
        "n_paths": int,                 # echo of N
        "seed_used": int,
    },
    "warnings": list[str],              # empty list if no warnings; never None
}
```

**Validation enforced by `src/forecasters/dispatch.py`** before the result is returned/cached:
- Required keys present.
- `paths` is a `np.ndarray` with shape `(N, H)` matching `horizon`.
- `anchors.horizon_dates` length matches `horizon`.
- All `summary.{median, p05, p25, p75, p95}` arrays have length `horizon`.
- `warnings` is a list (possibly empty), never `None`.

Failure to validate raises `ResultContractError` with backend name + violated assertion. This is a backend bug, not a user bug.

### `tune(input_dict) → preset_dict`

**Input:**

```python
{
    "data": pd.DataFrame,      # canonical OHLCV
    "identifier": str,         # e.g., "NYSE:AAPL" — for metadata only
    "range": (str, str),       # (start, end) ISO dates — for metadata
    "search_config": dict,     # backend-specific grid/search spec; may be empty for defaults
    "seed": int | None,
}
```

**Output:** a `preset_dict` matching the preset schema below.

---

## Preset artifact schema

Every preset YAML conforms to:

```yaml
name: v24-default                       # filename stem; must match file
backend: analog_mc                      # which backend produced this
schema_version: 1                       # this schema's version, not the backend's

hyperparameters:                        # backend-specific structure (opaque to framework)
  n_eff: 50
  corr_window: 60
  # ... (see src/analog_mc/configs/v2_4.yaml for full shape)

fitted_on:
  identifier: NASDAQ100                 # what the preset was tuned on
  start: '1990-01-01'
  end: '2024-12-31'
  data_hash: sha256:abc123...           # hash of the source data (for drift detection)
  n_observations: 8783

fitted_at: '2026-05-24T18:30:00Z'       # UTC timestamp of tune completion

validation_metrics:                     # what the preset achieved on its fit data
  crps_mean: 0.0234
  coverage_90: 0.89
  # ... other backend-reported metrics

provenance:                             # optional — how was this preset made
  source: tuned                         # tuned | curated | imported
  search_grid_hash: sha256:def456...    # for tuned presets; pinpoints the grid used
  tune_runtime_seconds: 4521
  git_commit: 3f1968e                   # repo state at tune time
```

**Validation enforced by `src/forecasters/presets.py`** on load:
- Required top-level keys present (`name`, `backend`, `schema_version`, `hyperparameters`, `fitted_on`, `fitted_at`, `validation_metrics`).
- `name` matches the filename stem (catches accidental rename without rewriting metadata).
- `schema_version == 1`.
- `backend` is a string the dispatcher knows about (else `UnknownBackendError`).
- `fitted_at` parses as a UTC ISO-8601 timestamp.

Drift detection: when `/forecast` runs, the dispatcher computes the `data_hash` of the forecasting input data and compares against `preset.fitted_on.data_hash`. Mismatch (different identifier, or same identifier but materially different range) → adds a warning to `result.warnings` quantifying the drift (e.g., `"preset fitted on NASDAQ100 1990-2024 (data_hash sha256:abc); forecasting on NYSE:AAPL 2010-2026 (data_hash sha256:xyz). Hyperparameters may be uncalibrated."`).

---

## File layout

```
docs/forecasters/
  goal.md                                 # done (this PR)
  V1_PLAN.md                              # done (this PR)
  README.md                               # short user-facing intro; written in stage 7

src/forecasters/
  __init__.py
  presets.py                              # load, validate, list presets
  data.py                                 # data composition (data_pipelines.fetch + DataFrame in)
  dispatch.py                             # backend dispatch table + result-shape validation
  cache.py                                # forecast-result cache (content-addressed)
  errors.py                               # ResultContractError, UnknownBackendError, PresetSchemaError

src/analog_mc/
  forecaster.py                           # NEW: public forecast() + tune() wrapping existing pipeline
  data.py                                 # MODIFY: add DataFrame input path alongside CSV-first

scripts/forecasters/
  __init__.py
  run.py                                  # the forecasters dispatcher (called by /forecast, /tune-preset, /list-presets)

configs/forecasters/
  presets/
    v24-default.yaml                      # canonical preset; v2.4 hyperparams + NASDAQ100 metadata

results/forecasters/
  presets/                                # gitignored — user-tuned presets land here
    .gitkeep
  forecasts/                              # gitignored — cached forecast outputs
    .gitkeep

scripts/data_pipelines/                   # bundled additions
  skill_runner.py                         # NEW: entry point for /fetch-data + /data-health
                                          # (subcommands: fetch, health)

.claude/skills/                           # five skill files at the project's skill root
  forecast/SKILL.md                       # forecasters-owned; calls scripts/forecasters/run.py forecast ...
  tune-preset/SKILL.md                    # forecasters-owned
  list-presets/SKILL.md                   # forecasters-owned
  fetch-data/SKILL.md                     # data_pipelines-owned; calls scripts/data_pipelines/skill_runner.py fetch ...
  data-health/SKILL.md                    # data_pipelines-owned

tests/forecasters/
  test_presets.py                         # load, validate, list
  test_dispatch.py                        # backend dispatch + result-shape validation
  test_cache.py                           # cache hit/miss, --no-cache, --cache-path
  test_data_composition.py                # data_pipelines integration + DataFrame input
  test_forecast_e2e.py                    # golden NASDAQ100 reproduces V5.A.2 metrics
  test_cross_asset_drift.py               # NYSE:AAPL with v24-default surfaces drift warning

tests/data_pipelines/
  test_skill_runner.py                    # NEW: /fetch-data and /data-health subcommand smoke tests

tests/analog_mc/
  test_forecaster_public_api.py           # NEW: forecast() and tune() conform to wire-format contract
```

---

## Stage breakdown

The build order is strict — each stage is self-contained and ends with a passing test suite. Don't skip ahead.

### Stage 1 — Preset artifact schema + loader

**Goal:** define the preset YAML schema, write the loader + validator, ship the canonical preset.

**Tasks:**
- `src/forecasters/presets.py` — `load_preset(name) → preset_dict`, `list_presets() → list[preset_summary]`, `validate_preset(preset_dict)`.
- `src/forecasters/errors.py` — `PresetSchemaError`, `UnknownPresetError`.
- `configs/forecasters/presets/v24-default.yaml` — write the canonical preset by:
  - Copying analog_mc's tuned v2.4 hyperparameters.
  - Computing the `data_hash` of `data/NASDAQ100.csv` over the full range.
  - Pulling `validation_metrics` from the existing V5 baseline (e.g., `results/analog_mc/data/fat_tail_v4.json`).
  - Setting `fitted_at` to the actual original tune time if recoverable; else current UTC with a note in `provenance.source = "curated"`.
- Tests: `tests/forecasters/test_presets.py` covers load, validate, list, name-mismatch error, schema-version error, missing-key error.

**Done when:** `python -c "from forecasters.presets import load_preset; print(load_preset('v24-default'))"` works and prints the canonical preset.

### Stage 2 — Backend adapter in analog_mc

**Goal:** wrap the existing analog_mc pipeline behind a clean `forecast(input_dict) → result_dict` and `tune(input_dict) → preset_dict`.

**Tasks:**
- `src/analog_mc/data.py` — add a `load_dataframe(df, date_col, close_col) → InternalData` function alongside the existing CSV-first path. The existing CSV path stays as the default per `[[project-data-source]]`.
- `src/analog_mc/forecaster.py` (NEW):
  - `forecast(input_dict) → result_dict` — takes the canonical input, runs the existing single-origin pipeline, returns the canonical result.
  - `tune(input_dict) → preset_dict` — wraps the existing walk-forward + grid search; returns a full preset dict with `backend: analog_mc`, populated `fitted_on`, `fitted_at`, `validation_metrics`.
- Both functions populate `result.warnings` / preset notes for any unusual conditions (short range, missing data points, etc.).
- Tests: `tests/analog_mc/test_forecaster_public_api.py` — call `forecast()` with a known input and assert the result conforms to the wire-format contract (right keys, right shapes, right dtypes).

**Done when:** `analog_mc.forecaster.forecast(input_dict)` runs end-to-end on the NASDAQ100 fixture and produces a contract-conformant result.

### Stage 3 — Dispatcher + result validation + data composition

**Goal:** wire the framework — load preset, fetch data, dispatch to backend, validate result.

**Tasks:**
- `src/forecasters/data.py` — `prepare_data(identifier=None, data_path=None, start, end) → pd.DataFrame`. If `identifier`: calls `data_pipelines.fetch()`. If `data_path`: loads CSV/parquet directly.
- `src/forecasters/dispatch.py` — dispatch table:
  ```python
  _BACKENDS = {
      "analog_mc": lambda: __import__("analog_mc.forecaster", fromlist=["forecast", "tune"]),
  }
  def dispatch_forecast(preset, input_dict) -> dict:
      mod = _BACKENDS[preset["backend"]]()
      result = mod.forecast(input_dict)
      _validate_result_contract(result, expected_horizon=input_dict["horizon"])
      return result
  ```
- Result validation per the wire-format spec above. On failure: `ResultContractError(backend, violated)`.
- Drift detection (preset.data_hash vs current data hash) → injects a warning into `result.warnings`.
- Tests: `test_dispatch.py` covers happy path, unknown backend error, result-shape violations.

**Done when:** can call `dispatch_forecast(preset, input_dict)` from a unit test and get a validated result back.

### Stage 4 — Forecast-result cache

**Goal:** content-addressed cache for `/forecast` outputs.

**Tasks:**
- `src/forecasters/cache.py`:
  - `cache_key(preset_name, preset_content_hash, identifier, start, end, origin, horizon, seed) → str`
  - `read_cached(key, cache_path) → dict | None`
  - `write_cached(key, result, cache_path) → Path` (atomic: temp + rename)
- Default cache path: `results/forecasters/forecasts/`. Overridden by `--cache-path`. Disabled by `--no-cache`.
- Tests: hit/miss, preset edit invalidates, `--no-cache` bypasses, `--cache-path` relocates.

**Done when:** repeat invocation of the same `(preset, args)` hits the cache; editing the preset YAML invalidates.

### Stage 5 — Dispatcher CLI entry point

**Goal:** `scripts/forecasters/run.py` — the actual entry point called by all 4 skills.

**Tasks:**
- Subcommand argparse:
  - `forecast`: `--preset --identifier|--data-path --start --end --origin --horizon [--config-overrides] [--seed] [--no-cache] [--cache-path] [--output-dir]`
  - `tune`: `--backend --identifier|--data-path --start --end --output-preset [--search-config] [--seed]`
  - `fetch`: `--identifier --start --end [--frequency] [--output-path]`
  - `list-presets`: no required args; `--json` for machine-readable output
- Each subcommand exits non-zero on contract failures, with the error type + context echoed to stderr.
- Output written to `--output-dir/<cache_key>/` with `summary.json`, `paths.npz`, `fan_chart.png`, `warnings.json`. Stdout prints the directory path (so the calling skill can echo it).
- Tests: smoke each subcommand end-to-end against the NASDAQ100 fixture.

**Done when:** `uv run python -m scripts.forecasters.run forecast --preset v24-default --data-path data/NASDAQ100.csv --start 2000-01-01 --end 2024-12-31 --origin 2024-01-02 --horizon 60` produces the expected output directory.

### Stage 6 — Forecaster skill files

**Goal:** the three forecasters-owned `.claude/skills/<name>/SKILL.md` files that route to `scripts/forecasters/run.py`. (Data skills are Stage 8 — kept separate so a partial revert is clean if needed.)

**Tasks:**
- `.claude/skills/forecast/SKILL.md`:
  - Describe purpose (one paragraph).
  - List required args + optional args + examples.
  - Document the result shape (link to V1_PLAN.md wire-format section).
  - Specify: skill calls `uv run python -m scripts.forecasters.run forecast <args>`; echoes the output dir.
- `.claude/skills/tune-preset/SKILL.md`:
  - All of the above, plus: **explicit loop-pattern warning** per `[[feedback-experiment-agent-loop]]` since analog_mc tuning is hours of compute. Include the loop-pattern launch-prompt template inline.
- `.claude/skills/list-presets/SKILL.md`:
  - Document the source columns (canonical vs user-tuned), the metadata shown, the `--json` flag.
- All three skills validate against the project's `CLAUDE.md` skill conventions (no AI attribution in any text produced, etc.).

**Done when:** `/forecast`, `/tune-preset`, `/list-presets` invocations work in a fresh Claude Code session.

### Stage 7 — Integration tests + docs

**Goal:** end-to-end coverage + user-facing doc.

**Tasks:**
- `tests/forecasters/test_forecast_e2e.py`:
  - Run `/forecast --preset v24-default` on the NASDAQ100 fixture for a known origin.
  - Assert: median path within ±X% of the V5.A.2 baseline at every horizon step; CRPS within ±Y of the baseline; paths array correct shape.
- `tests/forecasters/test_cross_asset_drift.py`:
  - Run `/forecast --preset v24-default --identifier NYSE:AAPL` (uses data_pipelines).
  - Assert: `result.warnings` contains a drift warning with quantified `data_hash` mismatch.
  - Assert: forecast still produces a valid result (warnings ≠ errors).
- `docs/forecasters/README.md` — short user-facing intro: "what is this, how do I use it, how do I add a preset, where do I find help."

**Done when:** all tests pass; README reads cleanly to someone unfamiliar with the project.

### Stage 8 — Bundled data_pipelines skill additions

**Goal:** ship `/fetch-data` and `/data-health` as agent-callable surfaces over the existing `data_pipelines` module. Same PR, separate stage so a partial revert is clean.

**Tasks:**
- `scripts/data_pipelines/skill_runner.py` (NEW) — two subcommands:
  - `fetch --identifier <id> --start <YYYY-MM-DD> --end <YYYY-MM-DD> [--frequency daily] [--json]`
    - Calls `data_pipelines.fetch(identifier, start, end, frequency)`.
    - Prints (or JSON-emits) the processed-cache path + a one-line summary: identifier, rows, range covered, source mix (which adapters served the data), cache hit/miss.
    - Exits non-zero on `data_pipelines` typed errors (e.g., `MissingAPIKey`, `ProviderError`, `UnknownDomain`) with the error class + message + identifier echoed to stderr.
  - `health [--identifier <id>] [--domain <name>] [--json]`
    - With no args: report cache health across the whole `data/processed.db` — total identifiers, total rows, per-domain breakdown, oldest/newest `last_fetch_utc`.
    - With `--identifier`: report for that one identifier — schema version, row count, range covered, last fetch UTC, source-mix from the meta row's `sources_json`.
    - With `--domain`: report aggregated across all identifiers in that domain.
    - Output is a small table (default) or JSON (`--json`); agent-friendly either way.
    - Reads the meta tables (`<domain>_meta`) via the existing `cache.py` accessors — no new SQL, no schema changes to `data_pipelines`.
- `.claude/skills/fetch-data/SKILL.md`:
  - Describe purpose (one paragraph).
  - List args + examples (NYSE:AAPL, NSE:RELIANCE, INDEX:^NDX, etc.).
  - Document the identifier-prefix convention from `data_pipelines` (point at `docs/data_pipelines/goal.md`).
  - Specify: skill calls `uv run python -m scripts.data_pipelines.skill_runner fetch <args>`; echoes the cache path.
  - Note: composing with `/forecast --identifier <id>` is **not required** — `/forecast` calls `data_pipelines.fetch()` internally. `/fetch-data` exists for users/agents who want to inspect data on its own.
- `.claude/skills/data-health/SKILL.md`:
  - Describe purpose.
  - List the three call modes (no args, `--identifier`, `--domain`).
  - Note the typical agent flow: `/data-health --identifier X` before `/forecast` when in doubt about cache freshness.
- `tests/data_pipelines/test_skill_runner.py`:
  - `fetch` subcommand happy path on a seeded identifier (NASDAQ:AAPL from the existing test cache).
  - `fetch` subcommand surfaces `UnknownDomain` for `FOO:BAR`.
  - `health` no-args produces a non-empty report against the seeded cache.
  - `health --identifier NASDAQ:AAPL` produces a single-row report with the expected columns.
  - `health --json` is parseable.
- Both skills follow the same `CLAUDE.md` no-AI-attribution rules.

**Non-scope for this stage:**
- `/data-seed` (bulk seed a universe) — not in v1; gated on user demand.
- `/list-data` — deferred; `/data-health` no-args covers most of the discovery need.
- `/data-reprocess` — not user-facing; stays as the existing CLI subcommand.

**Done when:** `/fetch-data` and `/data-health` work in a fresh Claude Code session, against the seeded `data/processed.db` from the existing data_pipelines tests.

### Stage 9 — End-to-end acceptance demo (NIFTY 500, PR merge gate)

**Goal:** the concrete bar from `goal.md` §"v1 end-to-end acceptance demo" — an agent autonomously fetches NIFTY 500 history, fine-tunes a preset on it, and test-forecasts the last 60 trading days. **This stage gates the PR merge: stages 1–8 may be green in isolation, but the PR does not open until Stage 9 passes end-to-end.**

**Pre-flight (do before any other Stage 9 work):**
- Verify `NIFTY:NIFTY500` is reachable via the existing NSE adapter chain. Run `/fetch-data --identifier NIFTY:NIFTY500 --start 2020-01-01 --end <today>` against a fresh cache; observe which adapter served, what range was returned. If all three adapters reject it (jugaad-data, nselib, yfinance), file the smallest possible fix as part of this PR — either an identifier-mapping line or a one-adapter patch. If at least one adapter returns the index level, no `data_pipelines` change is needed.

**Tasks:**
- `scripts/forecasters/run_acceptance_demo.py` (NEW) — a thin orchestrator that:
  - Computes `tune_end` = today − 60 NSE trading days (using the NSE calendar from `data_pipelines.domains.nse_equities`).
  - Invokes `/fetch-data` for the full range.
  - Invokes `/tune-preset` for the held-in range, with `--output-preset nifty500-v1`.
  - Invokes `/forecast` for the full range with origin = `tune_end`, horizon = 60.
  - Reads the resulting `summary.json` and the realized last-60-days closes; computes (a) 90-band coverage of the realized path, (b) CRPS of the forecast distribution against the realized point path, (c) CRPS of a naïve random-walk baseline (forecast = `close[tune_end]` flat).
  - Asserts the acceptance criteria from `goal.md`:
    - All four skill calls returned success exit codes.
    - The preset YAML at `results/forecasters/presets/nifty500-v1.yaml` passes `presets.validate_preset`.
    - The forecast result's `warnings` is empty.
    - 0.5 ≤ 90-band coverage ≤ 1.0.
    - Forecast CRPS is finite and strictly less than the random-walk baseline CRPS.
  - Writes a `docs/forecasters/_acceptance_demo.md` report with the dates used, the resolved preset path, the coverage / CRPS numbers, and a pass/fail verdict.
- `tests/forecasters/test_acceptance_demo.py` — a smoke test that invokes `scripts/forecasters/run_acceptance_demo.py` against a frozen NIFTY 500 fixture (a tiny pre-cached parquet of, say, 2024-01-01 → 2024-12-31; the tune→forecast→verify flow runs end-to-end but on a small range so the test is fast). The full-history live run remains a manual / pre-merge step driven by the script above.
- After the acceptance demo passes, commit the resulting preset (`results/forecasters/presets/nifty500-v1.yaml`) and the `_acceptance_demo.md` report on the branch. (Both live in normally-gitignored directories, so commit explicitly.)

**Done when:**
- Pre-flight identifier check resolved (either NIFTY:NIFTY500 fetches cleanly, or the smallest-possible adapter fix has shipped on this branch).
- `python -m scripts.forecasters.run_acceptance_demo` runs to completion against live data, all five acceptance assertions pass, the report is generated.
- The fixture-based `test_acceptance_demo.py` passes in CI.
- The PR description includes the acceptance-demo report (coverage, CRPS, baseline comparison, dates) so reviewers can confirm the bar was met without re-running the live demo.

---

## Skill specifications

### `/forecast`

```
Usage:
  /forecast --preset <preset_name>
            (--identifier <id> | --data-path <path>)
            --start <YYYY-MM-DD> --end <YYYY-MM-DD>
            --origin <YYYY-MM-DD> --horizon <int>
            [--config-overrides <yaml_path>]
            [--seed <int>]
            [--no-cache] [--cache-path <path>]
            [--output-dir <path>]
```

**Output (stdout):** one line — absolute path to the result directory.
**Output directory contents:** `summary.json`, `paths.npz`, `fan_chart.png`, `warnings.json`.

### `/tune-preset`

```
Usage:
  /tune-preset --backend <name>
               (--identifier <id> | --data-path <path>)
               --start <YYYY-MM-DD> --end <YYYY-MM-DD>
               --output-preset <preset_name>
               [--search-config <yaml_path>]
               [--seed <int>]
```

**Output (stdout):** absolute path to the produced preset YAML (under `results/forecasters/presets/`).
**SKILL.md MUST include:** the loop-pattern launch-prompt template per `[[feedback-experiment-agent-loop]]`. Anyone delegating `/tune-preset` to a sub-agent needs to paste this into the launch prompt.

### `/list-presets`

```
Usage:
  /list-presets [--json] [--backend <name>]
```

**Output:** table with columns `name | source (canonical|user-tuned) | backend | fitted_on (identifier+range) | fitted_at | crps`.
**`--json`:** machine-readable; what a sub-agent should call when composing.

### `/fetch-data` *(data_pipelines-owned)*

```
Usage:
  /fetch-data --identifier <id> --start <YYYY-MM-DD> --end <YYYY-MM-DD>
              [--frequency daily] [--json]
```

**Output (stdout):** absolute path to the processed-cache row(s) for this identifier, plus a one-line summary (rows, range covered, source mix, cache hit/miss). `--json` returns the same as a parseable object.
**Entry point:** `scripts/data_pipelines/skill_runner.py fetch ...`.
**Composes with:** `/forecast --identifier <id>` calls `data_pipelines.fetch()` internally — running `/fetch-data` first is not required. The skill exists for users/agents that want to inspect data on its own.

### `/data-health` *(data_pipelines-owned)*

```
Usage:
  /data-health [--identifier <id>] [--domain <name>] [--json]
```

**Output:** small table (default) or JSON (`--json`).
- No args: project-wide cache health — total identifiers, total rows, per-domain breakdown, oldest/newest `last_fetch_utc`.
- `--identifier`: single-row report (schema version, row count, range covered, last fetch UTC, source mix from `<domain>_meta.sources_json`).
- `--domain`: aggregated report across all identifiers in that domain.

**Entry point:** `scripts/data_pipelines/skill_runner.py health ...`.

---

## Anti-goals (code-level)

These are non-negotiable in v1. Each was a deliberate design choice; revisit them only with an explicit deviation discussion.

1. **No `Forecaster` ABC, no `ForecasterRegistry`.** `src/forecasters/dispatch.py` uses a hand-written `if/elif` (or dict-of-lambdas) table with one entry. Backend #2 lands → extract ABC + registry at that point, not before.
2. **No typed `ForecastResult` / `ForecastInput` dataclasses.** v1 enforces the wire-format by runtime validation (assertions in `dispatch.py`). Promotion to `src/forecasters/contract.py` is gated on backend #2.
3. **`/forecast` does not take `--backend`.** Backend is an attribute of the preset. Trying to forecast with the "wrong" backend means producing a preset for that backend (`/tune-preset`).
4. **No walk-forward exposure in `/forecast`.** Single origin, single horizon. Walk-forward stays in `scripts/<backend>/` where experiment drivers live.
5. **No backend-internal hyperparameter flags on `/forecast`.** Override via `--config-overrides path.yaml`, not `--n-eff 50`. Keeps the skill surface stable as backends evolve.
6. **No silent preset/data drift.** Mismatch between `preset.fitted_on.data_hash` and current data hash always produces a warning in `result.warnings`. Never absorbed silently.
7. **No mutation of canonical presets at runtime.** `configs/forecasters/presets/*.yaml` are read-only. `/tune-preset` writes to `results/forecasters/presets/` only.
8. **No AI attribution** in any commit message, PR title/body, SKILL.md text, or generated output (per `CLAUDE.md`).

## Test strategy

| Layer | What gets tested | Where |
|---|---|---|
| Preset loader | Schema validation, name/filename match, missing keys | `tests/forecasters/test_presets.py` |
| Backend adapter | analog_mc.forecast/tune conform to wire-format contract | `tests/analog_mc/test_forecaster_public_api.py` |
| Dispatcher | Backend dispatch, result-shape violations, unknown backend error | `tests/forecasters/test_dispatch.py` |
| Cache | Hit/miss, preset-edit invalidation, `--no-cache`, `--cache-path` | `tests/forecasters/test_cache.py` |
| Data composition | `data_pipelines.fetch()` integration + `--data-path` direct load | `tests/forecasters/test_data_composition.py` |
| End-to-end | Golden NASDAQ100 reproduces V5.A.2 metrics within tolerance | `tests/forecasters/test_forecast_e2e.py` |
| Drift warning | NYSE:AAPL with `v24-default` surfaces drift warning correctly | `tests/forecasters/test_cross_asset_drift.py` |
| Data skill runner | `/fetch-data` happy path + unknown-domain error; `/data-health` no-args, `--identifier`, `--domain`, `--json` | `tests/data_pipelines/test_skill_runner.py` |

**Fixtures:** reuse the NASDAQ100 CSV under `data/`. For cross-asset tests, `data_pipelines` provides NYSE:AAPL via the test cache (already seeded from prior data_pipelines tests).

**Tolerance for the e2e test:** the golden NASDAQ100 forecast must match V5.A.2's median path within ±0.5% at every horizon step and CRPS within ±0.001 absolute. Stricter tolerances may surface non-determinism (RNG-handling regressions); looser tolerances admit silent algorithm drift.

---

## Open questions

None blocking v1 implementation. Items below are gated on later evidence:

1. **When backend #2 lands, promote the wire-format contract to typed dataclasses.** Live in `src/forecasters/contract.py`. The promotion is itself a small refactor — backend #1 (analog_mc) updates its imports; the dispatcher gains type checks; the JSON-shape doc becomes a docstring on the dataclass.
2. **When backend #2 lands, write `docs/forecasters/adding_a_backend.md`.** Same pattern as `data_pipelines/adding_a_domain.md` (written when domain #2 landed). The right time to write this doc is *after* doing it once, not before.
3. **Forecast-result database?** Currently forecasts are cached as filesystem artifacts. If downstream consumers (dashboards, comparison tools) want to query across many forecasts, a SQLite table makes sense — same trajectory as `data_pipelines` v1.5. Defer until a concrete consumer exists.
4. **Preset metadata for non-equity domains.** `fitted_on.identifier` assumes the data_pipelines identifier convention. If a future backend forecasts on data that isn't from data_pipelines (e.g., a synthetic dataset, a CSV without a canonical identifier), the schema needs a `source: free-form` option. Defer until such a backend appears.

---

## Branch + PR plan

- This work lives on the `v1-skills` branch (renamed from `forecasters-v1` after the data-skills bundling decision).
- Touches two modules: `forecasters` (new) and `data_pipelines` (skill-runner additions only — no changes to the existing fetch/cache/dispatch core; possibly a tiny adapter patch if Stage 9's NIFTY:NIFTY500 pre-flight requires it).
- Each stage above is a logical commit; the PR opens against `main` once stages 1–9 are green.
- Stage 8 (data skills) is intentionally placed before Stage 9 so the data-side surface exists when the acceptance demo orchestrator wires the four skills together.
- **Stage 9 is the PR merge gate.** Stages 1–8 being green proves the units; Stage 9 proves the system. The PR description must include the Stage 9 acceptance-demo report (coverage, CRPS, baseline comparison, dates) — reviewers should not have to re-run the live demo themselves.
- Per `[[feedback-branch-retention]]`, the branch stays after merge.
- Per `CLAUDE.md`, no AI attribution in commits or PR.
