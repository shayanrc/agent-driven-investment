# gbdt — Experiment YAML Spec

This is the reference for the YAML files at `configs/gbdt/experiments/<name>.yaml`. One spec = one experiment = one `(universe, direction, threshold, horizon)` tuple. Multi-cell sweeps are multiple spec files.

The spec is consumed by `python -m gbdt.experiment <spec_path>` (the CLI atom) and by the `/gbdt-experiment` skill (the agent surface). Both load through the same `experiment.load_spec()` validator.

Read `V1_PLAN.md` for the architecture context (especially Stage 7's split semantics and Stage 6's FS+HP loop semantics). Read `CATBOOST_HP_REFERENCE.md` for the bounded HP ranges the agent operates within. Read `goal.md` for the why.

---

## File location

```
configs/gbdt/experiments/<experiment_name>.yaml
```

`experiment_name` is also the artifact directory name under `results/gbdt/experiments/<experiment_name>/`. Choose names that read well in a file listing: `<universe>_<direction>_<threshold>pct_<horizon>d[_<suffix>].yaml`.

Examples:
- `nifty50_up_10pct_20d_pilot.yaml` — the v1 PR pilot.
- `nifty50_down_20pct_50d.yaml` — a production-shape spec.
- `nifty50_up_5pct_10d_quickloop.yaml` — a fast exploratory cell with a tight FS+HP cap.

---

## Schema (one block per top-level key)

### `target` (required)

The tuple that defines what this experiment predicts. **No defaults — all four fields required.**

| Field | Type | Allowed values |
|---|---|---|
| `universe` | str | One of the universe presets in `configs/gbdt/default.yaml::universes`. v1 = `nifty50` only. |
| `direction` | str | `up` or `down`. |
| `threshold_pct` | float | Any positive number. Common: `5`, `10`, `20`, `30`, `50`. |
| `horizon_days` | int | Any positive integer. Common: `10`, `20`, `50`, `100`. |

Example:
```yaml
target:
  universe: nifty50
  direction: up
  threshold_pct: 10
  horizon_days: 20
```

Semantics (from `V1_PLAN.md` Stage 3):
- `direction: up` → label `1` if `max(high[t+1:t+horizon_days]) ≥ close[t] * (1 + threshold_pct/100)`.
- `direction: down` → label `1` if `min(low[t+1:t+horizon_days]) ≤ close[t] * (1 − threshold_pct/100)`.

### `date_range` (optional)

Bounds on the data fetched per ticker.

| Field | Type | Default |
|---|---|---|
| `start` | str (ISO date) | `null` (use the deepest history each ticker has) |
| `end` | str (ISO date) | `null` (use the latest cached date per ticker) |

Example:
```yaml
date_range:
  start: 2015-01-01
  end: 2026-04-30
```

If unset, the loader takes the maximum range each ticker has in the cache, then applies `split` to carve segments off the tail.

### `split` (optional)

Walk-forward fold scheme. **Global defaults live in `configs/gbdt/default.yaml::split`; the spec overrides only what it cares about.**

| Field | Type | Default (from `default.yaml`) |
|---|---|---|
| `train_rows` | int | `800` |
| `val_rows` | int | `400` |
| `eval_rows` | int | `200` |
| `test_rows` | int | `100` |
| `n_folds` | int | `1` |
| `min_rows_per_ticker` | int | `1600` (= sum of the four segments) |

Example (override eval to a longer segment for a low-base-rate cell):
```yaml
split:
  eval_rows: 400
```

Tickers with fewer than `min_rows_per_ticker` rows are dropped from the panel and listed in the artifact's `metrics.json::data.tickers_excluded`.

### `features` (optional)

Feature pool overrides. **By default the experiment starts with all 273 candidate columns** (the 16 families in `V1_PLAN.md` Stage 2); the agent prunes per-iteration.

| Field | Type | Default |
|---|---|---|
| `candidates` | list[str] or `"all"` | `"all"` (the full 273-col pool) |
| `exclude` | list[str] | `[]` |
| `lookback_windows` | list[int] | from `default.yaml`: `[5, 10, 20, 50, 100, 200]` |

Examples:

Exclude the volume family entirely:
```yaml
features:
  exclude:
    - "volume_ratio_*"
    - "obv_*"
    - "vol_ret_corr_*"
    - "dollar_move_*"
```

Use only a short lookback set (faster but lower ceiling):
```yaml
features:
  lookback_windows: [5, 10, 20]
```

Restrict to a named subset (rare; typically the agent prunes instead):
```yaml
features:
  candidates:
    - stock_return_20
    - realized_vol_20
    - rel_strength_20
    - sma_distance_50
```

### `backend` (optional)

Per-experiment backend tweaks. Defaults live in `configs/gbdt/default.yaml::backend`.

#### `backend.library`

| Field | Type | Default |
|---|---|---|
| `library` | str | `"catboost"` |

v1 accepts `"catboost"` only. Other libraries are `V1.1_TBD.md` § "Per-experiment library override".

#### `backend.calibration_method`

| Field | Type | Default | Semantics |
|---|---|---|---|
| `calibration_method` | str | `"conditional_isotonic"` | See below. |

Allowed values:

- `"native"` — ship CatBoost's raw predicted probabilities; no post-calibration. Use only when prior knowledge / earlier experiments show CatBoost is already calibrated for this cell.
- `"conditional_isotonic"` *(default)* — run Spiegelhalter Z-test on the val segment; if `|z| < 2.0`, ship native; else fit `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` on val. Decision + Z statistic recorded in `metrics.json::calibration`.
- `"isotonic_always"` — fit isotonic on val unconditionally. Use when you want to compare apples-to-apples across experiments without the conditional gate adding variance.
- `"platt"` — Platt scaling via `sklearn.calibration.CalibratedClassifierCV(method="sigmoid", cv="prefit")`. Use when isotonic is overfitting on small val segments.

Override example:
```yaml
backend:
  calibration_method: isotonic_always
```

#### `backend.fs_hp_loop`

The agent-driven feature-selection + hyperparameter loop config.

| Field | Type | Default | Semantics |
|---|---|---|---|
| `max_iterations` | int | `8` | Hard cap on iteration count. After this many iterations the loop emits the best-checkpoint artifact regardless of remaining headroom. |
| `plateau_threshold` | float | `0.005` | Absolute val Brier improvement floor. If the last 2 iterations both improve by less than this, inner-stop fires. |
| `degradation_gate` | float | `0.01` | Multiplicative tolerance over best-seen val Brier. If the current val Brier > `(1 + degradation_gate) * best_val_brier`, inner-stop fires. |

Override example (tighter loop for a quick exploration):
```yaml
backend:
  fs_hp_loop:
    max_iterations: 4
    plateau_threshold: 0.01
```

#### `backend.hp_starting`

Iteration-0 HP overrides. The agent uses these as the iteration-0 HPs; subsequent iterations are bounded by `CATBOOST_HP_REFERENCE.md` ranges per parameter.

| Field | Type | Default |
|---|---|---|
| `hp_starting` | dict | `{}` (use defaults from `default.yaml::backend.hp_defaults`) |

The values you can put here are exactly the tunable HPs from `V1_PLAN.md` Stage 4 ("Tunable HPs"): `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `min_data_in_leaf`, `rsm`, `bootstrap_type`, `bagging_temperature`, `subsample`, `random_strength`, `auto_class_weights` (or `scale_pos_weight`), `boosting_type`, `early_stopping_rounds`.

Pinned (never overridable from a spec): `has_time=True`, `loss_function="Logloss"`, `eval_metric="BrierScore"`, `custom_metric=["Logloss", "BrierScore", "AUC"]`, `random_seed` (set via top-level `random_seed` below).

Override example (start the agent from a known-good HP set for a rare cell):
```yaml
backend:
  hp_starting:
    iterations: 3000
    learning_rate: 0.03
    depth: 4
    l2_leaf_reg: 10
    auto_class_weights: SqrtBalanced
    early_stopping_rounds: 100
```

### `random_seed` (optional, top-level)

| Field | Type | Default |
|---|---|---|
| `random_seed` | int | `42` |

Pinned across CatBoost (`random_seed=<this>`), NumPy / Python `random` if invoked, and any sampling inside the diagnostic bundle. Sweep only for explicit seed-variance experiments (`{0, 1, 2, 3, 4}`).

---

## Artifact directory layout

After the experiment runs, the artifact lives at:

```
results/gbdt/experiments/<experiment_name>/
├── spec.yaml                         # echo of the input spec, fully expanded with defaults applied
├── model.cbm                         # CatBoost binary
├── calibration.pkl                   # pickle of fitted IsotonicRegression / Platt object; None-marker file if method=native
├── features.yaml                     # the final pruned feature list the model uses
├── hp.yaml                           # the final HP dict (best-checkpoint iteration)
├── iterations.jsonl                  # one row per FS+HP iteration; see schema below
├── metrics.json                      # headline metrics on eval segment + calibration decision + data summary
├── predictions/
│   ├── train.csv                     # (date, ticker, p_raw, p_calibrated, y_true)
│   ├── val.csv
│   ├── eval.csv
│   └── test.csv
├── figs/
│   ├── reliability_diagram.png
│   ├── calibration_curve.png
│   ├── learning_curve_iter_*.png     # one per FS+HP iteration
│   ├── feature_importance_final.png
│   └── train_val_gap_history.png
└── report.md                         # human-readable narrative (see V1_PLAN.md Stage 8 for sections)
```

**Every artifact is self-contained.** Re-running the spec against the same data + seed should reproduce the same artifact. Predictions are saved per segment so downstream analyses don't need to re-load the model.

### `iterations.jsonl` row schema

One JSON object per line, one line per FS+HP iteration:

```json
{
  "iter": 0,
  "hp": {"iterations": 1000, "depth": 6, "learning_rate": 0.05, ...},
  "features": ["stock_return_20", "realized_vol_20", ...],
  "n_features": 273,
  "rationale": "iteration 0 — full feature pool, default HPs",
  "train_brier": 0.234,
  "val_brier": 0.241,
  "train_val_gap": 0.007,
  "calibration_method": "conditional_isotonic",
  "calibration_z": 1.42,
  "early_stop_iter": 487,
  "wall_time_sec": 142.3,
  "inner_stop_signal": null
}
```

The final iteration's row carries `inner_stop_signal` set to one of `"plateau"`, `"degradation"`, `"cap"` — explaining why the loop ended.

### `metrics.json` schema

```json
{
  "experiment_name": "nifty50_up_10pct_20d_pilot",
  "spec_hash": "sha256:...",
  "data_hash": "sha256:...",
  "data": {
    "n_tickers_in_universe": 50,
    "n_tickers_used": 48,
    "tickers_excluded": ["JIOFIN", "MAXHEALTH"],
    "n_rows_train": 38400,
    "n_rows_val": 19200,
    "n_rows_eval": 9600,
    "n_rows_test": 4800,
    "positive_prevalence_train": 0.412,
    "positive_prevalence_eval": 0.398
  },
  "loop": {
    "n_iterations_run": 5,
    "best_iteration": 3,
    "inner_stop_signal": "plateau"
  },
  "calibration": {
    "method": "conditional_isotonic",
    "decision": "isotonic",
    "spiegelhalter_z": 2.84,
    "spiegelhalter_p": 0.0045
  },
  "headline_eval": {
    "brier": 0.231,
    "brier_baseline_baserate": 0.239,
    "brier_improvement_vs_baseline": 0.008,
    "log_loss": 0.654,
    "roc_auc": 0.612
  },
  "headline_test": {
    "brier": 0.244,
    "brier_baseline_baserate": 0.241,
    "log_loss": 0.682,
    "roc_auc": 0.598
  },
  "wall_time_total_sec": 1842.7
}
```

---

## Universe presets

v1 ships one universe: `nifty50`. Defined in `configs/gbdt/default.yaml::universes::nifty50`, which points at the ticker list source:

```yaml
universes:
  nifty50:
    source: configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml
    index_ticker: "INDEX:^NSEI"     # for F1, F5, F9, F9b families (uses ^NSEI history)
    annualization_factor: 250        # √250 for vol families; differs from US's √252
```

The loader resolves `source`, reads the ticker list, and prefixes each with `NSE:` (e.g. `RELIANCE` → `NSE:RELIANCE`) before calling `data_pipelines.fetch()` per ticker. `index_ticker` is fetched separately and joined into the panel for the index-relative feature families.

NDX, NIFTY 100, NIFTY total, and other universe presets are all in `V1.1_TBD.md`.

---

## Worked example — the v1 pilot spec

Below is the spec at `configs/gbdt/experiments/nifty50_up_10pct_20d_pilot.yaml`, annotated line-by-line.

```yaml
# The v1 PR pilot experiment.
# Cell choice: up / +10% / 20 trading days on NIFTY 50. Base rate from v0.2 is
# ~35-45% across the universe — common enough that calibration is meaningfully
# testable, not so extreme that class-imbalance machinery dominates.

target:
  universe: nifty50         # one of configs/gbdt/default.yaml::universes
  direction: up
  threshold_pct: 10
  horizon_days: 20

# date_range omitted - take maximum cached range per ticker.

# split omitted - use default 800+400+200+100 from default.yaml.
# The 4 IPO-bounded tickers below 1,600 rows are excluded by min_rows_per_ticker.

# features omitted - start with all 273 candidates; agent prunes per iteration.

backend:
  # Default library catboost; only one supported in v1.
  calibration_method: conditional_isotonic    # default - run Spiegelhalter Z, ship native or isotonic.
  fs_hp_loop:
    max_iterations: 8                         # default - 8-iter hard cap.
    plateau_threshold: 0.005                  # default - 0.5% absolute val Brier improvement floor.
    degradation_gate: 0.01                    # default - 1% degrade from best-seen val Brier triggers stop.

  # hp_starting omitted - use the iteration-0 defaults from default.yaml::backend.hp_defaults.
  # The agent will explore from there bounded by CATBOOST_HP_REFERENCE.md ranges.

random_seed: 42
```

This is intentionally minimal: it relies entirely on the global defaults. A production-shape spec for a rare cell (e.g. `±50% / 10d`) would override `backend.hp_starting` with tighter regularization (`depth: 4, l2_leaf_reg: 10, auto_class_weights: SqrtBalanced`) per `CATBOOST_HP_REFERENCE.md`'s rare-cell guidance.

---

## Validation rules (enforced by `experiment.load_spec()`)

Loading a spec fails fast on:

- Missing required `target` fields.
- `target.universe` not present in `default.yaml::universes`.
- `target.direction` not in `{"up", "down"}`.
- `target.threshold_pct <= 0` or `target.horizon_days <= 0`.
- `backend.library` other than `"catboost"`.
- `backend.calibration_method` not in `{"native", "conditional_isotonic", "isotonic_always", "platt"}`.
- `backend.fs_hp_loop.max_iterations < 1` or `> 16` (hard ceiling above the default 8 to prevent runaway agent loops).
- `backend.hp_starting` keys not in the tunable-HP allowlist, or values outside the ranges in `CATBOOST_HP_REFERENCE.md`.
- `random_seed < 0`.
- `split.train_rows + split.val_rows + split.eval_rows + split.test_rows > split.min_rows_per_ticker`.

Loader errors print the offending field path + the constraint that failed, not bare stack traces.

---

## See also

- `V1_PLAN.md` — the implementation plan; references this spec from Stages 3, 6, 7, 8, 9.
- `CATBOOST_HP_REFERENCE.md` — the per-parameter "when to change" rubrics the agent reads each FS+HP iteration.
- `goal.md` — why the experiment-loop framing exists.
- `V1.1_TBD.md` — parked extensions (additional universes, multi-target heads, alternative libraries, Bayesian HP search).
- `.claude/skills/gbdt-experiment/SKILL.md` — the agent surface that consumes specs and orchestrates iterations.
