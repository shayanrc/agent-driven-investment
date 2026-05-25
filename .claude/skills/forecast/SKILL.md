---
name: forecast
description: Run a saved analog Monte Carlo (or other backend) forecast preset against a time series. Loads a preset YAML, fetches data via data_pipelines (or reads a local CSV/parquet), dispatches to the backend named in the preset, and writes a canonical result directory with paths, summary, fan chart, and warnings.
---

# /forecast

Generic, backend-agnostic forecasting surface. The caller picks a preset and an identifier; backend selection is invisible — the preset's `backend` field controls dispatch.

## When to use

- You have an identifier the project's `data_pipelines` knows about (e.g., `NASDAQ:AAPL`, `NIFTY:50`, or a local CSV) and a preset that has been tuned (canonical in `configs/forecasters/presets/` or user-tuned in `results/forecasters/presets/`).
- You want a single-origin, single-horizon probabilistic price-path forecast.

Run `/list-presets` first if you do not know which preset to use; run `/tune-preset` if no preset has been tuned for the asset.

## Usage

```
uv run python -m scripts.forecasters.run forecast \
    --preset <preset_name> \
    (--identifier <id> | --data-path <path>) \
    --start <YYYY-MM-DD> --end <YYYY-MM-DD> \
    --origin <YYYY-MM-DD> --horizon <int> \
    [--config-overrides <yaml_path>] \
    [--seed <int>] \
    [--no-cache] [--cache-path <path>] \
    [--output-dir <path>]
```

### Required

| Flag | Purpose |
|---|---|
| `--preset` | Name of a preset YAML (no `.yaml` suffix). Searched canonical-first (`configs/forecasters/presets/`), then user-tuned (`results/forecasters/presets/`). |
| `--identifier` or `--data-path` | Exactly one. Identifier triggers `data_pipelines.fetch()`; data-path reads a local CSV/parquet. |
| `--start`, `--end` | ISO dates bounding the data range to load. |
| `--origin` | ISO date of the forecast origin (the forecast starts the next trading session). |
| `--horizon` | Number of trading days to forecast (e.g., 60). |

### Optional

| Flag | Purpose |
|---|---|
| `--config-overrides` | Path to a YAML file with hyperparameter overrides (merged on top of the preset's `hyperparameters`). Use sparingly — preset is the unit of currency. |
| `--seed` | Integer for reproducible RNG. Same seed + same inputs → bit-identical paths. |
| `--no-cache` | Skip the cache; write to a tempdir. |
| `--cache-path` | Override the default cache root (`results/forecasters/forecasts/`). |
| `--output-dir` | Write artifacts to an explicit directory instead of the cache. |

### Stdout

A single line — the absolute path to the output directory.

### Output directory contents

| File | Content |
|---|---|
| `summary.json` | Anchors (origin date + horizon dates), summary percentiles (median, p05, p25, p75, p95 in PRICE space, plus CRPS if realized horizon is available), metadata (backend name, preset name + hash, config hash, n_paths, seed used, weights, n_eff), and the warnings list. |
| `paths.npz` | The sample paths (`paths` key) — shape `(N, H)` of float64 log returns. |
| `fan_chart.png` | A small fan chart over the horizon (p05/p25/p50/p75/p95). |
| `warnings.json` | The warnings list, mirrored from `summary.json` for `grep` convenience. |

## Examples

```
# Use the canonical preset on the NASDAQ100 CSV.
uv run python -m scripts.forecasters.run forecast \
    --preset v24-default \
    --data-path data/NASDAQ100.csv \
    --start 2010-01-01 --end 2024-12-31 \
    --origin 2024-01-02 --horizon 60 --seed 42

# Forecast NYSE:AAPL with the same canonical preset (drift warning expected).
uv run python -m scripts.forecasters.run forecast \
    --preset v24-default --identifier NASDAQ:AAPL \
    --start 2010-01-01 --end 2024-12-31 \
    --origin 2024-01-02 --horizon 60
```

## Wire-format contract

The result conforms to `docs/forecasters/V1_PLAN.md` §"Wire-format contract". The framework validates the shape before writing to the cache — a backend bug surfaces as `ResultContractError` on stderr and a non-zero exit code, not as a corrupt cache file.

## Notes

- `/forecast` does NOT take `--backend`. The backend is an attribute of the preset.
- `/forecast` does NOT take backend-internal flags like `--n-eff` or `--weights`. Use `--config-overrides path.yaml` if you really need to override a preset value.
- Drift between `preset.fitted_on.data_hash` and the data you load is always surfaced as a warning in `result.warnings` — never silently absorbed.
- Single origin, single horizon. Walk-forward / multi-origin evaluation is an experiment-driver concern.
