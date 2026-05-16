# agent-driven-investment

A modular pipeline for probabilistic price-path forecasting.

The repo is designed to host **multiple forecasting/analytics modules** under a shared layout. **`analog_mc`** — analog Monte Carlo forecasting — is the first module; additional modules can plug in by following the per-module namespacing convention below without touching anything in `analog_mc`.

---

## What `analog_mc` does

`analog_mc` produces calibrated Monte Carlo distributions of forward price paths.

Given a daily price series, for any forecast origin date it:

1. Computes causal multi-horizon z-scores (default: 20/50/200-day trailing-mean-over-trailing-std) and an EWMA volatility estimate.
2. Finds historical dates whose z-score vectors are closest (weighted Euclidean) to today's.
3. Converts those distances into probabilities via an **`n_eff`-parameterized softmax** — the temperature is solved so that `exp(entropy(p)) = n_eff`, giving the search loop a directly interpretable knob for forecast dispersion.
4. Samples forward **10-day blocks** of realised returns from the chosen analogs (6 blocks → 60-day forecast in the default config).
5. **Vol-scales each block** to today's σ regime, with the analog's drift removed against a single shared baseline (`μ_origin` = trailing-200d mean at the forecast origin — see `docs/analog_mc/IMPLEMENTATION_PLAN.md` C3 v1.1).
6. Maintains a **running EWMA σ across blocks** so block 2..6 scaling reflects the simulated path's accumulated vol.

A walk-forward driver searches `(weights, n_eff)` per fold (grid + Nelder-Mead) on a validation block, locks the choice, evaluates on a held-out test block, and rolls forward. The output is a per-fold record of forecasts, realized values, and σ ratios, plus a calibration diagnostic suite that gates whether v2 features (trailing-momentum drift, conditional block sampling, tail inflation) need to be enabled.

The pipeline is **asset-agnostic**: horizons, z-score windows, drift mode, and vol-clip bounds are all config-driven.

For the full design rationale and the 6 critical correctness constraints (causality, n_eff parameterization, μ_origin demean, running σ, forward-only sampling, walk-forward boundary discipline), see `docs/analog_mc/IMPLEMENTATION_PLAN.md`. For the step-by-step math, see `docs/analog_mc/ALGORITHM.MD`.

---

## Repo layout

Every module-specific file is namespaced under `<top>/<module>/`. The top-level directories themselves host **cross-module concerns only** (the global dashboard launcher, shared test fixtures, etc.).

```
src/<module>/                  # package code
docs/<module>/                 # design docs (IMPLEMENTATION_PLAN.md, ALGORITHM.MD)
tests/<module>/                # tests
configs/<module>/*.yaml        # YAML configs
runs/<module>/<timestamp>/     # output artifacts per run (gitignored, .gitkeep tracked)
dashboards/<module>/
  ├── app.py                   # module's Streamlit entry point
  └── views/                   # config_editor, run_experiment, diagnostics

dashboards/app.py              # thin global launcher (auto-discovers modules)
data/                          # raw OHLC CSVs (gitignored, .gitkeep tracked)
.claude/                       # checked-in CLAUDE conventions, agents, skills, memories
```

Currently the only `<module>` is `analog_mc`. New modules plug in by adding directories at each of those paths.

---

## Getting started

The project uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
# install dependencies and the analog_mc package (editable)
uv sync

# run the test suite (164 tests; ~3.5 minutes due to small integration tests)
uv run pytest tests/analog_mc/

# run analog_mc walk-forward on the default NASDAQ100 config
# (full default: ~76 folds, ~2 hours on a laptop; use a faster config for first runs)
uv run python -m analog_mc walk-forward --config configs/analog_mc/default.yaml

# launch the dashboard (auto-discovers all modules; only analog_mc for now)
uv run streamlit run dashboards/app.py
```

`uv run <cmd>` runs `<cmd>` inside the project venv without requiring activation. To activate the venv directly, `source .venv/bin/activate`.

---

## Configuration

Every parameter the pipeline reads lives on a single `Config` dataclass (`src/analog_mc/config.py`) with invariants validated at construction time:

- `forecast_horizon == n_blocks * block_length`
- `len(zscore_horizons) == 3` (the simplex grid is hardcoded for three weights)
- `max(zscore_horizons) < train_initial_size`
- `vol_clip_lower < 1.0 < vol_clip_upper`

Configs live in `configs/<module>/<name>.yaml`. The Config editor view in the dashboard exposes every field with form widgets and validates before saving.

Key `analog_mc` knobs:

| Field | Default | Meaning |
|---|---|---|
| `forecast_horizon` | 60 | Total trading days to forecast forward |
| `block_length` | 10 | Days per sampled analog block |
| `n_blocks` | 6 | Number of blocks; must satisfy `horizon = blocks × length` |
| `n_paths` | 1000 | Monte Carlo paths per forecast origin |
| `zscore_horizons` | `(20, 50, 200)` | Three trailing windows used for analog matching |
| `ewma_halflife` | 20 | Halflife (days) for the trailing EWMA σ |
| `train_initial_size` | 1000 | First fold's train length (~4 years of trading days) |
| `val_size` / `test_size` | 60 / 60 | Per-fold validation / test block size |
| `weight_grid_resolution` | 0.1 | Simplex spacing for the hyperparameter search grid |
| `n_eff_values` | `(15, 30, 50, 80, 150)` | Discrete `n_eff` candidates in the grid |
| `vol_clip_lower` / `vol_clip_upper` | 0.5 / 3.0 | σ-ratio bounds (asymmetric — defaults reflect the equity leverage effect) |
| `drift_mode` | `"zero"` | v1 default. `trailing_momentum` reserved for v2 |
| `data_path` / `date_col` / `close_col` | `data/NASDAQ100.csv` / `observation_date` / `NASDAQ100` | CSV source and column mapping (configurable per-asset) |

See `configs/analog_mc/default.yaml` for the complete reference.

---

## Data

v1 reads a single local CSV. The default config points at `data/NASDAQ100.csv` (FRED-style: `observation_date`, `NASDAQ100`). The loader accepts any column names via `date_col` and `close_col` in the config — drop in a yfinance-style CSV with `Date`/`Close` and update the config, no code changes needed.

`data/` is gitignored except for `.gitkeep`. A proper multi-source data loader (yfinance, bhav files, etc.) is a separate module planned for later — `analog_mc.data` deliberately stays single-source for v1.

---

## Outputs

Each walk-forward run writes to `runs/analog_mc/<UTC-timestamp>/`:

```
config.yaml              frozen config used for the run
meta.json                git commit hash, config hash, started/finished timestamps, wall seconds
lock                     present while the run is in progress (multi-run safe)
summary.parquet          one row per fold: (weights, n_eff, val_crps, test_crps, n_test_origins)
folds/<k>/
  search.parquet         full grid evaluation table (val CRPS at every weight × n_eff)
  forecasts.npz          test paths (n_origins × n_paths × horizon, float32)
                         + pre-clip σ ratios (n_origins × n_paths × n_blocks)
                         + realized (n_origins × horizon)
                         + origin_idx (n_origins,)
  summary.json           locked (weights, n_eff, val_crps, test_crps, train/val/test bounds)
```

Re-running on the same `--run-dir` skips folds whose `summary.json` already exists (crash-resumable).

---

## Dashboard

`streamlit run dashboards/app.py` opens a single Streamlit app with three views (sidebar nav inside the `analog_mc` module):

1. **Diagnostics** — pick a completed run, see aggregate CRPS, the 8 Stage-9 plots (weight trajectory, CRPS surface, global PIT, conditional PIT by vol regime, reliability diagram, squared-return ACF comparison, clip-hit summary), and the 5 v2-trigger decision-rule verdicts.
2. **Run experiment** — pick a config, optionally override the ticker, launch a walk-forward as a subprocess. The view polls `runs/analog_mc/<ts>/folds/` for per-fold summaries to show progress. Multiple concurrent runs are safe — each gets its own timestamped directory with its own lockfile.
3. **Config editor** — load any YAML in `configs/analog_mc/`, edit fields via form widgets, validate against the invariants, save back.

The dashboard is a thin presentation layer. All real computation lives in `src/analog_mc/`; the views only call into it.

---

## Status

**`analog_mc` v1.1** — 11 stages complete, 164 unit tests passing. The v1.1 fix (shared-baseline `μ_origin` demean replacing the original per-block demean) is documented in `docs/analog_mc/IMPLEMENTATION_PLAN.md` under "Revision history".

**v2 features** are gated behind specific diagnostic findings (see `decision_rules()` in `diagnostics.py`):

| v2 feature | Triggered by |
|---|---|
| Trailing-momentum drift (`drift_mode='trailing_momentum'`) | Sloped global PIT |
| Conditional block sampling | Squared-return ACF degradation > 30% at seam lags |
| Tail inflator | U-shape in high-vol-regime PIT |
| Drop per-fold search | Fixed-weight baseline within 1% of tuned CRPS |
| Revisit distance metric | Clip-hit fraction > 15% on either bound |

---

## Conventions

- **Per-module namespacing** — see "Repo layout" above. Top-level dirs are reserved for cross-module concerns; never put module-specific files in `tests/`, `configs/`, `runs/`, or `dashboards/` directly.
- **No hidden behavioural changes** to the design spec. The plan in `docs/analog_mc/IMPLEMENTATION_PLAN.md` is the source of truth; deviations are recorded explicitly under "Revision history".
- **Determinism** — every (weights, n_eff, origin) tuple gets its own seeded RNG, derived from `config.random_seed`. Re-running the same forecast produces bit-identical results; the search objective is noise-free.
- **Causality discipline (C1)** — every rolling feature at index `t` uses only data with index ≤ `t`. The single most important unit test (`tests/analog_mc/test_features.py::test_causal_*_no_lookahead`) verifies this directly for both `causal_ewma_vol` and `causal_zscore` against a regime-switching synthetic series.

---

## License

Not yet specified.
