---
name: list-presets
description: Enumerate canonical and user-tuned forecasting presets with their metadata (backend, fitted_on identifier/range, fitted_at, validation CRPS). Use before /forecast to discover which preset to apply.
---

# /list-presets

Enumerate every available forecasting preset. Canonical presets (checked in under `configs/forecasters/presets/`) and user-tuned presets (under `results/forecasters/presets/`) are surfaced in one table with a `source` column distinguishing them.

## When to use

- Before `/forecast` when you do not know which preset to apply.
- Before `/tune-preset` to confirm no usable preset already exists for the asset.
- For diagnostics — surfaces presets whose YAML fails schema validation, with the violation message in the `error` column.

## Usage

```
uv run python -m scripts.forecasters.run list-presets [--json] [--backend <name>]
```

### Options

| Flag | Purpose |
|---|---|
| `--json` | Emit a JSON list (one object per preset) instead of the default table. Use this when composing with another agent / script. |
| `--backend` | Filter to presets whose `backend` field matches the value (e.g., `analog_mc`). |

## Columns

| Column | Source |
|---|---|
| `name` | The preset's `name` field (and its filename stem). |
| `source` | `canonical` (under `configs/forecasters/presets/`) or `user-tuned` (under `results/forecasters/presets/`). Canonical wins when the same name appears in both. |
| `backend` | `preset.backend`. |
| `fitted_on_identifier` | `preset.fitted_on.identifier` (e.g., `NASDAQ100`, `NIFTY:NIFTY500`). |
| `fitted_on_start` / `fitted_on_end` | The data range the preset was tuned on. |
| `fitted_at` | UTC ISO timestamp of tune completion. |
| `crps_mean` | `preset.validation_metrics.crps_mean` if present, else blank. |
| `error` | Schema-validation error message if the YAML fails to load; otherwise blank. |

## Examples

```
# Default table.
uv run python -m scripts.forecasters.run list-presets

# JSON, filtered to analog_mc.
uv run python -m scripts.forecasters.run list-presets --json --backend analog_mc
```

## Notes

- `/list-presets` does not load preset hyperparameters — it just reads the metadata fields. Cheap (<1s typical).
- A preset that fails schema validation still appears in the list with the error surfaced; it is not silently dropped. Investigate `error != null` rows before relying on the preset.
