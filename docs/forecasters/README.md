# forecasters

A backend-agnostic forecasting surface. Pick a preset, pick an identifier, get a probabilistic forecast back. Backend selection is invisible to the caller — every preset carries its `backend` field, and the framework dispatches accordingly.

For the *why* see [`goal.md`](goal.md); for the *how* (architecture, contract, stages) see [`V1_PLAN.md`](V1_PLAN.md).

## Skills surface

| Skill | Owner | Purpose |
|---|---|---|
| `/forecast` | forecasters | Run a preset against data; produce a canonical result directory. |
| `/tune-preset` | forecasters | Fit a new preset by tuning a backend on a data range (hours of compute for analog_mc). |
| `/list-presets` | forecasters | Enumerate canonical + user-tuned presets with metadata. |
| `/fetch-data` | data_pipelines | Fetch on-demand for a single identifier. Optional — `/forecast` calls `data_pipelines.fetch()` internally. |
| `/data-health` | data_pipelines | Report cache coverage and last-fetch dates. |

## Quick start

```bash
# Discover presets.
uv run python -m scripts.forecasters.run list-presets

# Forecast on the NASDAQ100 CSV with the canonical preset.
uv run python -m scripts.forecasters.run forecast \
    --preset v24-default \
    --data-path data/NASDAQ100.csv \
    --start 2010-01-01 --end 2024-12-31 \
    --origin 2024-01-02 --horizon 60 --seed 42

# Forecast a NASDAQ ticker by identifier (data_pipelines fetches on demand).
uv run python -m scripts.forecasters.run forecast \
    --preset v24-default --identifier NASDAQ:AAPL \
    --start 2010-01-01 --end 2024-12-31 \
    --origin 2024-01-02 --horizon 60

# Tune a new preset (hours of compute — see /tune-preset SKILL.md for the
# loop-pattern guidance you must paste into any sub-agent launch prompt).
uv run python -m scripts.forecasters.run tune \
    --backend analog_mc \
    --identifier NIFTY:NIFTY500 \
    --start 2005-01-01 --end 2024-12-31 \
    --output-preset nifty500-v1
```

## What lands in the output directory

`/forecast` writes a directory keyed by a hash of all its inputs (preset content, identifier/data path, range, origin, horizon, seed). Default root: `results/forecasters/forecasts/<cache_key>/`. Contents:

| File | Content |
|---|---|
| `summary.json` | Anchors, summary percentiles (`median`, `p05`, `p25`, `p75`, `p95` in PRICE space, plus CRPS if the realized horizon is available), metadata (backend, preset, hashes, n_paths, seed, weights, n_eff), warnings list. |
| `paths.npz` | Sample paths array (`paths` key) — shape `(N, H)`, float64 log returns. |
| `fan_chart.png` | A small fan chart over the horizon. |
| `warnings.json` | The warnings list, mirrored for `grep` convenience. |

## Adding a new preset

You almost always tune one with `/tune-preset` rather than hand-writing it. If you do hand-write one (curated), the schema is in `V1_PLAN.md` §"Preset artifact schema". Required top-level keys:

```yaml
name: my-preset                  # must match the filename stem
backend: analog_mc               # which backend produced this
schema_version: 1
hyperparameters: {...}           # backend-specific
fitted_on:
  identifier: NASDAQ100
  start: '1986-01-02'
  end: '2024-12-31'
  data_hash: sha256:...          # via forecasters.data.data_hash()
  n_observations: 9000
fitted_at: '2026-05-24T18:30:00Z'   # UTC
validation_metrics:
  crps_mean: 0.04
```

Save it under `configs/forecasters/presets/<name>.yaml` (canonical, checked in) or `results/forecasters/presets/<name>.yaml` (user-tuned, gitignored). `/list-presets` will surface both.

## How drift detection works

When `/forecast` runs, the framework hashes the input data (`(date, close)` pairs) and compares against `preset.fitted_on.data_hash`. Mismatch → adds a warning to `result.warnings` quantifying which preset hash you have and what the current data hashes to. The forecast still runs — warnings are not errors. Use the warning to decide whether to `/tune-preset` for the new asset.

## Getting help

- `goal.md` — why this module exists, what success looks like, what trade-offs are unacceptable.
- `V1_PLAN.md` — full architecture, wire-format contract, preset schema, stage breakdown.
- `_acceptance_demo.md` — the NIFTY 500 end-to-end demo report (only present after Stage 9 ships).
- Per-skill `.claude/skills/<name>/SKILL.md` files — invocation docs.

## What this module is NOT

- Not a tuner itself (delegates to each backend's tune procedure).
- Not a data fetcher (composes with `data_pipelines`).
- Not a walk-forward driver (single origin, single horizon per `/forecast` call).
- Not a backtester. No transaction costs, no position sizing, no PnL.

These remain out of scope for the v1 framework. See `goal.md` §"What this module is *not*" for the rationale.
