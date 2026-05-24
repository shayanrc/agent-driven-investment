---
name: tune-preset
description: Fit a new forecasting preset by running a backend's tuning procedure (walk-forward + grid search for analog_mc) over a data range. Produces a saved preset YAML that downstream /forecast calls can use directly. This is hours of compute — must be launched via the long-running loop pattern.
---

# /tune-preset

Backend-specific tuning, surfaced as a single skill. For analog_mc this runs the existing walk-forward + grid search pipeline (~hours of compute on full-history data).

## When to use

- A preset does not yet exist for the asset you want to forecast.
- An existing preset's `fitted_on` range is materially shorter than the data you want to forecast on.
- You want to test a backend's hyperparameter sensitivity to a different domain (e.g., apply analog_mc to a non-equity series).

## Usage

```
uv run python -m scripts.forecasters.run tune \
    --backend <name> \
    (--identifier <id> | --data-path <path>) \
    --start <YYYY-MM-DD> --end <YYYY-MM-DD> \
    --output-preset <preset_name> \
    [--search-config <yaml_path>] \
    [--seed <int>] \
    [--output-root <path>]
```

### Required

| Flag | Purpose |
|---|---|
| `--backend` | Name of the backend whose tune procedure to invoke (`analog_mc` for now). |
| `--identifier` or `--data-path` | Exactly one — same semantics as `/forecast`. |
| `--start`, `--end` | Data range to tune on. The full range is used; the backend internally splits into walk-forward train/val/test windows. |
| `--output-preset` | Name (no `.yaml` suffix) for the produced preset file. Lands under `results/forecasters/presets/<name>.yaml`. |

### Optional

| Flag | Purpose |
|---|---|
| `--search-config` | YAML file overriding the backend's default search grid / search hyperparameters. Backend-specific shape — see the backend's own docs. |
| `--seed` | Integer for reproducible search. |
| `--output-root` | Override `results/forecasters/presets/` as the write directory. |

### Stdout

A single line — the absolute path to the produced preset YAML.

## CRITICAL — long-running compute pattern

`/tune-preset` against `analog_mc` on full-history data (e.g., NASDAQ100 1986–2024, NIFTY 500 multi-year) is **hours of compute**. Do not block a single tool call waiting for it. Paste this guidance into any sub-agent launch prompt that delegates `/tune-preset`:

> For any compute that may take more than ~30 minutes, do not block your single tool call waiting for it. Instead:
>
> 1. Launch the long shell with `Bash run_in_background=true` and a stable log path (e.g. `logs/<exp_id>_<UTC>.log`).
> 2. Use `Monitor` with a filtered `tail -f <log> | grep -E --line-buffered "elapsed|fold=|Error|Traceback|completed"` so progress events stream in without flooding context.
> 3. If the compute will likely outlast your session (rule of thumb: anything ≥2h), use `ScheduleWakeup` to self-pace — schedule a wakeup at 1200–1800s with a self-contained prompt that re-checks progress, decides whether to schedule the next wake, and ends when the job completes.
> 4. Never block one tool call for hours — that wastes the session and prevents you from being interruptible.

Additionally, mention that the parent will check in hourly via `ScheduleWakeup` and that the agent's auto-completion notification reaches the parent directly — so the agent doesn't need to do anything special to "signal done."

See `.claude/memories/feedback-experiment-agent-loop.md` for the source of this guidance.

## Output preset shape

The produced YAML conforms to the preset schema in `docs/forecasters/V1_PLAN.md` §"Preset artifact schema". Key fields:

- `name`: the file's stem (must match `--output-preset`).
- `backend`: echoes `--backend`.
- `hyperparameters`: the tuned knob values, including baked `weights` and `n_eff` (for analog_mc) so subsequent `/forecast` calls skip re-tuning.
- `fitted_on`: identifier, range, data hash, observation count.
- `fitted_at`: UTC ISO timestamp of tune completion.
- `validation_metrics`: walk-forward mean test CRPS and per-fold breakdown.
- `provenance.source = "tuned"`.

## Examples

```
# Tune analog_mc on NIFTY 500 (the v1 acceptance demo target).
uv run python -m scripts.forecasters.run tune \
    --backend analog_mc \
    --identifier NIFTY:NIFTY500 \
    --start 2005-01-01 --end 2024-12-31 \
    --output-preset nifty500-v1
```

## Notes

- Tuning writes to `results/forecasters/presets/`, NEVER to `configs/forecasters/presets/`. Canonical presets are read-only at runtime.
- Tuning is deterministic given `--seed` — the search grid is fixed, the per-(weights, n_eff, origin) RNGs are seeded from `random_seed`.
- Behavior on partial completion: if a tune is interrupted, the run directory under `runs/analog_mc/<timestamp>/` is crash-resumable — point a re-run at it and previously completed folds are skipped.
