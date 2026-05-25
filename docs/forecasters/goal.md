# forecasters — Goal

**This is the generic forecasting-surface module for the project.** It is the user- and agent-facing entry point for running price-path forecasts on any time-series the rest of the project understands. Each forecasting algorithm (analog Monte Carlo today; ARIMA, Prophet, neural sequence models tomorrow) is a **backend** that plugs into the same framework: implementing one inference function and one tuning function, declaring its required input shape, returning a canonical result. The framework provides everything else (skill surface, preset artifact management, data-source composition with `data_pipelines`, output caching, dispatch).

**v1 ships exactly one backend: analog_mc** (the existing analog-Monte-Carlo pipeline, already validated through V5.A.2). Future backends plug in alongside without changing the framework. The right backend ABC / registry crystallizes when backend #2 is built; v1's job is to ship analog_mc cleanly behind a standalone skill surface with the right seams that make adding backend #2 a plug-in rather than a refactor.

This document states what `forecasters` is optimizing for and what trade-offs are unacceptable. Read it before editing any file under `src/forecasters/`, `tests/forecasters/`, `configs/forecasters/`, `dashboards/forecasters/`, or `docs/forecasters/`.

For *how* it works (architecture, stages, dispatch contract), see `V1_PLAN.md`. This file is the *why* and *what success looks like*.

---

## What this module is optimizing for

Provide **a single, backend-agnostic forecasting surface that the user (and any agent) can call without learning the internals of a specific forecasting algorithm**, with one defining rule:

> **The caller picks a preset and an identifier; they should never have to know which forecasting algorithm produced the forecast, and should not have to learn a new API per algorithm.** Hyperparameters, fit data, and validation metrics travel *inside* the preset artifact — like a saved model. Backend selection, data fetching, result caching, and warnings — all hidden behind a single `/forecast` skill call.

The module exists so that any forecasting question — "what's a 60-day fan chart for NYSE:AAPL from 2024-01-01?" — has a single canonical surface that returns a canonical answer, regardless of which algorithm sits behind it today or in six months.

## Saved-model framing (the central design choice)

**A preset is a fitted artifact, not a config blob.** Every preset file carries:

- `backend` — which forecasting module produced it (`analog_mc` for now).
- `hyperparameters` — the tuned knob values that algorithm uses.
- `fitted_on` — the identifier, date range, and data-hash the preset was tuned against.
- `fitted_at` — UTC timestamp.
- `validation_metrics` — what the preset achieved on its fit data (CRPS, coverage, etc.).

Consequences of this framing:

- **`/forecast` doesn't take `--backend`.** It takes `--preset <name>`, reads `preset.backend`, and dispatches.
- **Cross-asset story dissolves.** There is no "uncalibrated warning" path — either a preset exists for your asset (use it) or it doesn't (run `/tune-preset` to produce one). Drift between `fitted_on.data_hash` and the data you're forecasting on is surfaced as a warning in the result, not silently absorbed.
- **Tuning is a first-class operation.** `/tune-preset` is its own skill with its own SKILL.md, because for analog_mc it's hours of compute (walk-forward + grid search) and needs the long-running loop pattern baked into its launch documentation.
- **Canonical presets ship in the repo; user-tuned presets stay local.** `configs/forecasters/presets/` holds curated, checked-in presets (starting with `v24-default.yaml`, the existing analog_mc v2.4 config). `results/forecasters/presets/` holds user-tuned artifacts produced by `/tune-preset`, gitignored.

## What success looks like

A consumer's forecasting code should look like this and nothing more:

```bash
# Inference — use a preset that already exists
/forecast --preset v24-default --identifier NASDAQ100 \
          --start 2010-01-01 --end 2026-05-01 \
          --origin 2024-01-01 --horizon 60

# Fit a new preset for a different asset
/tune-preset --backend analog_mc --identifier NYSE:AAPL \
             --start 2000-01-01 --end 2026-05-01 \
             --output-preset aapl-v1

# Use the new preset
/forecast --preset aapl-v1 --identifier NYSE:AAPL \
          --start 2010-01-01 --end 2026-05-01 \
          --origin 2024-01-01 --horizon 60

# Discoverability
/list-presets

# Data load on its own (data_pipelines-owned; /forecast calls data_pipelines.fetch() internally,
# so running /fetch-data first is optional — use it when you want to inspect data on its own)
/fetch-data --identifier NYSE:AAPL --start 2000-01-01 --end 2026-05-01

# Cache health check (data_pipelines-owned)
/data-health --identifier NYSE:AAPL
```

The caller does not import `analog_mc` directly; the caller does not know what `n_eff` or `corr_window` mean; the caller does not configure a backend dispatch table. Within a call, the following must hold:

- **Backend dispatch is invisible.** `/forecast` reads `preset.backend`, imports the matching module, and calls its public `forecast()` function. Backends are independent packages with no knowledge of the framework's call graph.
- **Output shape is canonical.** Every backend returns a result conforming to a documented JSON shape: `paths` (N×H float array), `anchors` (origin date + horizon dates), `summary` (CRPS, coverage bands, point estimate), `config_hash`, `backend_name`, `fitted_on_ranges`, `warnings`. In v1 this is enforced by convention (the skill validates against the shape); when backend #2 lands the shape becomes a typed dataclass in `src/forecasters/contract.py`.
- **Cache-first, structurally.** The preset artifact *is* the cache for the expensive (tuning) step — once a preset exists, you never re-pay for hyperparameter search. Single-shot forecast results are also cached at `results/forecasters/forecasts/<key_hash>/`, keyed on `(preset_name, preset_content_hash, identifier, range, origin, horizon, seed)`; `--no-cache` bypasses, `--cache-path` relocates.
- **Data composition with `data_pipelines` is automatic.** Pass `--identifier NYSE:AAPL` and the skill calls `data_pipelines.fetch()` internally. No need to pre-fetch.
- **Honest about preset/data drift.** If `preset.fitted_on.data_hash` does not match the data you're forecasting on (different ticker, materially different range), the result's `warnings` list says so explicitly with the drift quantified. The forecast still runs; the warning travels in the result.
- **Determinism.** Same preset + same data + same `--seed` → bit-identical paths. No silent reordering, no nondeterministic backend dispatch.

## v1 end-to-end acceptance demo (the north-star)

The single concrete bar that gates the v1 PR merge: **an agent, working autonomously, fetches NIFTY 500 index history, fine-tunes a preset on it, and test-forecasts the last 60 trading days in that data.** This proves the four-skill surface composes end-to-end against an asset the canonical preset was *not* tuned on, on a domain (NSE / India) other than the one analog_mc was developed against.

**The flow:**

1. **Fetch.** Pull NIFTY 500 index history via the `data_pipelines` NSE domain. Identifier: `NIFTY:NIFTY500`. Range: the deepest history the `jugaad-data → nselib → yfinance` adapter chain reaches.
   ```bash
   /fetch-data --identifier NIFTY:NIFTY500 --start 2005-01-01 --end <today>
   ```

2. **Fine-tune on the held-in segment.** Run `/tune-preset` against `analog_mc` on the data *excluding* the last 60 trading days — that's the held-out test horizon.
   ```bash
   /tune-preset --backend analog_mc --identifier NIFTY:NIFTY500 \
                --start 2005-01-01 --end <today minus ~60 trading days> \
                --output-preset nifty500-v1
   ```
   Tuning is hours of compute → the agent uses the loop pattern from `[[feedback-experiment-agent-loop]]`. Resulting preset lands at `results/forecasters/presets/nifty500-v1.yaml`.

3. **Test-forecast the held-out 60 days using the new preset.**
   ```bash
   /forecast --preset nifty500-v1 --identifier NIFTY:NIFTY500 \
             --start 2005-01-01 --end <today> \
             --origin <today minus ~60 trading days> --horizon 60
   ```

4. **Verify forecast vs realized.** The realized last-60-days are in the data the skill already loaded. The result's `summary.json` reports CRPS against the realized path and the 5/25/50/75/95 coverage bands. An acceptance check confirms the forecast is informative (90-band covers the realized path with a frequency in the [0.5, 1.0] range; CRPS finite and not catastrophic vs a naïve last-price baseline).

**Acceptance criteria:**
- All four skill invocations succeed without manual intervention.
- `/tune-preset` produces a valid preset file (passes `presets.validate_preset`).
- `/forecast` produces a contract-conformant result with `warnings == []` (preset was fit on this exact identifier; no drift expected).
- 90-band coverage of the realized 60-day path is ≥ 0.5 and ≤ 1.0 (informative — not a trivial pass-through or a useless-wide band).
- Forecast CRPS is finite and better than a naïve random-walk baseline on the held-out segment.

**What this demo proves (and why it's the right bar):**
- **Data layer.** The NSE adapter chain reaches NIFTY 500 index level history end-to-end, on a fresh cold-cache fetch.
- **Cross-asset tuning.** A backend-specific tuning procedure (analog_mc's walk-forward grid search) triggers through the skill surface, survives hours of compute via the loop pattern, and produces a saved preset.
- **Round-trip.** The preset round-trips: written by `/tune-preset`, loaded by `/forecast`, dispatched to the right backend, validated, returned.
- **Cross-domain robustness.** The framework works on a non-NASDAQ100 asset, on a non-US domain, without changes to `analog_mc` core or to the framework's dispatch.
- **Skill composition.** The agent composes four skills (or three — `/fetch-data` can be skipped since `/forecast` calls `data_pipelines.fetch()` internally) in sequence, with the held-out-segment discipline implemented in the prompt, not in the code.

**Gating items to verify during Stage 1–2 of `V1_PLAN.md`:**
- `NIFTY:NIFTY500` reachability through the existing NSE adapter chain. The identifier prefix is registered (`docs/data_pipelines/V1_IMPLEMENTATION_PLAN.md` §"Identifier scheme") but the NIFTY 500 index level data is not in the seeded universe. If the adapters reject it, the smallest possible adapter extension is in scope for this PR. If they accept it on-demand fetch, no `data_pipelines` change is needed.

## Backend strategy

The framework supports any forecasting algorithm that implements the **two-function contract**:

| Function | Input | Output | Cost profile |
|---|---|---|---|
| `forecast(input) → result` | OHLCV DataFrame + origin + horizon + hyperparameters dict | Canonical result JSON shape | Seconds to minutes |
| `tune(input) → preset` | OHLCV DataFrame + range + search-grid config | Preset artifact dict | Minutes to hours |

**v1 backend: analog_mc.**

- Already shipped through V5.A.2; v2.4 hyperparameters validated on NASDAQ100 1990–2024.
- `forecast()` wraps the existing path-sampling pipeline.
- `tune()` wraps the existing walk-forward + grid search (hours of compute → needs the loop pattern).
- Canonical preset shipped: `v24-default.yaml`.

**Future backends.** Anything with the two-function contract plugs in. Likely candidates:
- Classical statistical (ARIMA, GARCH, state-space)
- Other ensemble Monte Carlo variants (block bootstrap, copula-based)
- Sequence models (when the project's compute budget supports it)

The framework does not constrain *how* a backend produces its forecast — only the input/output shapes and the cost-profile expectations (fast `forecast`, expensive `tune`).

## What this module is *not*

- **Not backend-restricted in principle, just in current implementation.** The architecture supports any forecasting algorithm. v1 ships only analog_mc; that's a scoping decision, not a structural limit.
- **Not a tuner itself** — delegates tuning to each backend's own procedure. The framework provides the skill surface and preset persistence; the search grid, validation metric, and fit procedure are backend-owned.
- **Not a data fetcher** — composes with `data_pipelines`. If a backend needs OHLCV, it gets a canonical DataFrame from the framework; the framework gets it from `data_pipelines.fetch()`.
- **Not a walk-forward driver** — single origin, single horizon. Walk-forward evaluation is an experiment-driver concern (lives in `scripts/<module>/` per module) and is what `/tune-preset` *uses internally*, not what `/forecast` exposes.
- **Not a backtester** — produces forecasts, doesn't run strategies on them. No position sizing, no PnL, no transaction costs (those belong downstream, per `[[no-backtester-in-analog-mc]]`).
- **Not a pre-extracted framework.** v1 builds analog_mc behind a clean public function and a thin dispatch layer, but does *not* crystallize a `Forecaster` ABC or `ForecasterRegistry` until backend #2 actually needs them. The wire-format contract is documented by convention in v1; it gets promoted to typed dataclasses in `src/forecasters/contract.py` when backend #2 lands and the shape has been pressure-tested by two real backends, not one.
- **Not a Streamlit dashboard.** Output is JSON + a fan-chart PNG. Visualization beyond that is a separate concern.

## Scope for v1

v1 = thin framework + four skills + one backend wired through.

**Framework (generic, backend-agnostic):**
- Skill dispatcher (`scripts/forecasters/run.py`) — parses args, loads preset, dispatches to the right backend, normalizes result, manages cache.
- Preset loader + validator — reads YAML, checks required metadata fields are present, supports the layered `configs/` (canonical) vs `results/` (user-tuned) lookup.
- Forecast-result cache layer — content-addressed by (preset, data, call args, seed).
- Data composition with `data_pipelines.fetch()` for `--identifier` calls.
- Output normalization — every backend's return is validated against the documented JSON shape before being written / returned.

**Backend #1: analog_mc.**
- Public `analog_mc.forecaster.forecast(input_dict) → result_dict` — wraps the existing single-origin pipeline.
- Public `analog_mc.forecaster.tune(input_dict) → preset_dict` — wraps the existing walk-forward + grid search.
- DataFrame-input adapter in `src/analog_mc/data.py` (alongside the existing CSV-first path; CSV-first contract preserved per `[[project-data-source]]`).
- Canonical preset: `configs/forecasters/presets/v24-default.yaml` with full metadata (backend, hyperparameters, fitted_on, fitted_at, validation_metrics).

**Skills (the forecasters user/agent surface — three skills):**
- `/forecast` — inference. `--preset <name> --identifier <id> --start --end --origin --horizon [--config-overrides path] [--no-cache] [--cache-path path] [--seed N]`.
- `/tune-preset` — fitting. `--backend <name> --identifier <id> --start --end --output-preset <name>`. SKILL.md bakes in the loop-pattern guidance per `[[feedback-experiment-agent-loop]]` because analog_mc tuning is hours of compute.
- `/list-presets` — enumerates `configs/forecasters/presets/` and `results/forecasters/presets/` with a `source` column (canonical vs user-tuned) and the key metadata (backend, fitted_on, fitted_at).

The `/fetch-data` and `/data-health` skills the forecasters module composes with are **`data_pipelines`-owned**, not forecasters-owned. They ship in the same v1 PR (per `docs/forecasters/V1_PLAN.md` Stage 8) so the end-to-end agent surface is usable, but the implementation, ownership, and future evolution stay with the `data_pipelines` module. The forecasters dispatcher composes with `data_pipelines.fetch()` directly in Python — it does **not** invoke `/fetch-data` as a skill.

**Tests:**
- Skill-level: golden NASDAQ100 forecast reproduces V5.A.2 metrics within tolerance.
- Cross-asset: NYSE:AAPL with the wrong preset surfaces the drift warning correctly.
- Cache: repeat invocation hits cache; preset edit invalidates; `--no-cache` bypasses.
- Preset round-trip: write → read → forecast produces same output as the inline equivalent.

**Explicitly deferred to later versions / backends:**
- Backend #2 (ARIMA, Prophet, neural, ...) — gated on actual need.
- Typed `ForecastResult` / `ForecastInput` dataclasses in `src/forecasters/contract.py` — promoted when backend #2 forces the abstraction.
- `Forecaster` ABC and `ForecasterRegistry` — same gating as the dataclasses.
- Streamlit dashboard for forecast inspection.
- Multi-origin / walk-forward exposure in the skill surface (stays in scripts/).
- Ensemble-across-backends (e.g., averaging analog_mc + ARIMA forecasts).
- Real-time / streaming forecasting.
- Forecast persistence beyond the cache (e.g., committing forecasts to a database).

## Eventual deployment shape

Currently the planned surface is: three Claude Code skills (this module) + a Python entry point + a preset directory layout, composing with two `data_pipelines`-owned skills (`/fetch-data`, `/data-health`) shipped in the same v1 PR. These reflect the **build-and-validate phase**.

Intended end state: an **agent-callable suite** that any sub-agent can compose to answer forecasting questions end-to-end:
1. `/data-health` *(data_pipelines)* to check whether the cache covers what you need.
2. `/fetch-data` *(data_pipelines)* to backfill if not, or to inspect the underlying data on its own.
3. `/list-presets` to discover which preset to use.
4. `/tune-preset` if no preset matches the asset.
5. `/forecast` to produce the answer.

When designing new APIs in this module, prefer shapes that wrap cleanly as a skill call:
- Clean function signature, all args explicit, no positional ambiguity.
- JSON-serializable return shape (paths array, summary dict, metadata dict).
- Errors raise typed exceptions with backend + preset context, not bare strings.
- No hidden CLI state or environment-dependent behavior beyond what `data_pipelines` already requires (API keys for the underlying data fetch).

## How to apply this when working on the module

- **Preset is the unit of currency.** Whenever you'd be tempted to add a `--hyperparam-x` flag to `/forecast`, ask: should this live in the preset instead? Default answer is yes. The skill surface stays thin.
- **The wire-format JSON shape is the contract.** Any change to what backends return (adding a field, renaming, changing semantics) is a breaking change that affects all current and future backends. In v1 it's enforced by convention; in v2 (when crystallized) it'll be a typed dataclass. Either way, treat it as the most stable surface in the module.
- **`/forecast` never knows about backend internals.** If you find yourself importing `analog_mc.foo` inside `scripts/forecasters/run.py` for anything beyond the dispatch table, that's a contract leak — push the logic into the backend's public function instead.
- **Tuning is long-running.** Any work touching `/tune-preset` (the skill or its SKILL.md) must include the loop-pattern guidance per `[[feedback-experiment-agent-loop]]`. Sub-agents that run `/tune-preset` against `analog_mc` need the loop pattern baked into their launch prompt.
- **Cross-backend differences are surfaced, not hidden.** If backend X's `tune()` takes a different shape of search-grid config than backend Y's, that asymmetry lives in each backend's own docs and config schema — the framework doesn't try to unify them in v1.
- **No AI attribution in commits or PRs** (per the project-wide rule in `CLAUDE.md`).

## Implementation discipline

The how-it-works specification is `docs/forecasters/V1_PLAN.md`. The stages defined there are the build order; the dispatch contract and the wire-format JSON shape are non-negotiable. Don't silently change architectural decisions — surface the deviation and ask first.
