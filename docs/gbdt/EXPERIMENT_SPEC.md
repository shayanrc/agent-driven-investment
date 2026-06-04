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

The tuple that defines what this experiment predicts. **The first four fields are required; `max_drawdown` is optional.**

| Field | Type | Allowed values |
|---|---|---|
| `universe` | str | A universe preset in `configs/gbdt/default.yaml::universes`. v1 pre-registers `nifty50`; `nifty100`, `nifty_midcap_150`, `nifty500` (and any other valid NSE index) are resolvable on first use via the agent's universe self-service flow — see § "Universe presets". Inline `tickers:` at the spec top level is also supported for one-off ad-hoc universes. |
| `direction` | str | `up` or `down`. |
| `threshold_pct` | float | Any positive number. Common: `5`, `10`, `20`, `30`, `50`. |
| `horizon_days` | int | Any positive integer. Common: `10`, `20`, `50`, `100`. |
| `max_drawdown` | float (optional) | Fractional path-honesty bound, e.g. `0.05` = "the path must not draw down more than 5% before breaching the threshold." Omit for the simple binary "did breach in horizon" target. |
| `uniqueness_weighting` | bool (optional) | Apply LdP §4.4 sample-uniqueness weights when training + scoring. Default `true`. Set to `false` to reproduce the pre-uniqueness baseline (every row weight=1.0). See § "Sample-uniqueness weighting" below. |

Example (basic binary):
```yaml
target:
  universe: nifty50
  direction: up
  threshold_pct: 10
  horizon_days: 20
```

Example (with the path-honesty filter):
```yaml
target:
  universe: nifty50
  direction: up
  threshold_pct: 10
  horizon_days: 20
  max_drawdown: 0.05        # positives must not draw down >5% before breaching +10%
```

Semantics (from `V1_PLAN.md` Stage 3):

**Without `max_drawdown` (default, simple binary):**
- `direction: up` → label `1` if `max(high[t+1:t+horizon_days]) ≥ close[t] * (1 + threshold_pct/100)`.
- `direction: down` → label `1` if `min(low[t+1:t+horizon_days]) ≤ close[t] * (1 − threshold_pct/100)`.

**With `max_drawdown` set (two-phase path-honesty filter). UP direction:**
1. Find the first index `t_breach` in `(t, t+horizon_days]` where `close[t_breach] ≥ (1 + threshold_pct/100) * close[t]`.
2. If `t_breach` exists **and** `min(close in (t, t_breach]) > (1 − max_drawdown) * close[t]` → label `1` (positive).
3. Otherwise → label `0` (negative). Negatives bucket two distinct failure modes: "never breached the threshold in the horizon" and "breached the threshold but drew down too much before getting there."

**DOWN direction** (symmetric — `max_drawdown` parametrizes the maximum *adverse path excursion*, which for a short position is an upside rally before the down-breach):
1. Find the first index `t_breach` in `(t, t+horizon_days]` where `close[t_breach] ≤ (1 − threshold_pct/100) * close[t]`.
2. If `t_breach` exists **and** `max(close in (t, t_breach]) < (1 + max_drawdown) * close[t]` → label `1` (positive).
3. Otherwise → label `0` (negative). Same two-bucket failure-mode aggregation as UP.

**Breach criterion switches from HIGH/LOW (simple mode) to CLOSE (path-honesty mode), intentionally.** The simple-binary breach uses HIGH (UP) / LOW (DOWN) — the most aggressive intraday move — because traders observe intraday extremes and the question there is "did the threshold ever fire in any interpretation." The path-honesty filter flips *both* legs (breach and drawdown bound) to CLOSE because the operator semantics are mark-to-market: traders allocate against close-price marks, the realistic exit is the close, and the drawdown the position survives is the close-to-close excursion. Consequence: a path-honesty positive can be strictly stricter than a simple-binary positive even with a permissive `max_drawdown`, because the breach now needs CLOSE (not HIGH/LOW) to clear the threshold.

This is the v0.3 path-honesty filter (`docs/gbdt/_v0_path_honesty_eval.md`) generalized: v0.3 used a hard-wired `max_drawdown = threshold_pct / 200` (half the threshold, as a fraction); v1 makes it an explicit per-experiment parameter. The reading of "positive" is now operator-meaningful: it's not just "the threshold fired at some point" but "the threshold fired without first wiping out the position."

#### Sample-uniqueness weighting (LdP §4.4)

Default: **on**. Each (ticker, date) training row is weighted by an approximation of "fraction of this row's forward window that is unique to this row" — for a contiguous ticker with `N >> H` rows, an interior row's weight is `1 / (2H − 1)` and edge rows ramp up to `1.0`. Implementation in `src/gbdt/uniqueness.py`.

Why this matters: with horizon `H` trading days, two adjacent (ticker, date) rows share `H − 1` of their `H` future bars — so the same outcome event labels `O(H)` neighbors identically. Without down-weighting, the loss is dominated by that overlap and prevalence is inflated. Sweep exp #1 (nasdaq100 +10%/100d/dd5%) observed training prevalence 42.4% vs non-overlapping EDA 19.7% — a 2.15× bias that uniqueness weighting corrects.

The weights enter:
- CatBoost `Pool(weight=...)` on train + val (gradient + early-stop signal).
- Weighted Brier / Spiegelhalter Z / AUC on val (via `gbdt.uniqueness.weighted_*`).
- Weighted Brier / weighted base-rate Brier / weighted log-loss / weighted AUC on the `headline_eval` and `headline_test` blocks; unweighted twins kept side-by-side as `*_unweighted`.
- `metrics.json::sample_uniqueness.effective_sample_size_per_fold` reports per-segment `(ess_kish, sum_weights, n_rows, overlap_inflation_ratio)`. `sum_weights` is the natural "number of independent forward events" measure (≈ `N / (2H − 1)`); `ess_kish` is `(Σw)² / Σw²` and is the standard variance-effective sample size (insensitive to uniform scaling).

Opt-out (legacy reproduction):
```yaml
target:
  uniqueness_weighting: false
```

In that mode every row enters with weight `1.0` and the weighted metrics collapse exactly to their unweighted form — useful only for reproducing pre-PR results or measuring the overlap-bias delta directly.

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
| `mode` | `"trailing"` \| `"date_aligned"` | `"trailing"` |
| `train_rows` | int | `800` |
| `val_rows` | int | `400` |
| `eval_rows` | int | `200` |
| `test_rows` | int | `100` |
| `n_folds` | int | `1` |
| `min_rows_per_ticker` | int | `1600` (= sum of the four segments) |
| `train_start` | ISO date | _none_ (required when `mode == "date_aligned"`; runner default `2019-01-01`) |
| `min_train_rows_per_ticker` | int | `200` (= `max(lookback_windows)`) |

**`mode == "trailing"` (default)**: each ticker's last `train_rows + val_rows + eval_rows + test_rows` rows are carved into `[train | val | eval | test]` in time order. Used by every pre-V1.4 spec. **Silently re-defines the cell across cache growth** (the eval/test windows slide forward as new bars arrive) — the V1.4 plan's motivating bug.

**`mode == "date_aligned"` (V1.4 opt-in)**: segment windows are anchored to **universe-level calendar dates** computed from `train_start` and the per-segment durations on the universe's canonical trading calendar (NYSE for US universes, NSE for NSE universes; mapping in `configs/gbdt/default.yaml::universes::<name>::calendar` — defaults inferred from universe name prefix). Per-ticker membership uses `min_train_rows_per_ticker` (≥ 200 valid feature rows) on the train segment and ≥ 1 row on val/eval/test; late-IPO tickers contribute only to whichever segments they have valid features for. The 4-segment row counts are NOT guaranteed to equal `{train,val,eval,test}_rows` per ticker — that's the duration the calendar window spans, not the count of cached bars. **Reproducible across cache growth**: adding new bars past `test_end` leaves segments bit-identical.

When `mode == "date_aligned"`, `train_rows` / `val_rows` / `eval_rows` / `test_rows` are interpreted as **trading-day durations** measured on the universe calendar (not row counts). A `train_start` falling on a non-trading day advances to the next trading day (`searchsorted(side="left")` semantics).

Example (canonical date-aligned cell):
```yaml
split:
  mode: date_aligned
  train_start: 2019-01-01
  # train_rows / val_rows / eval_rows / test_rows fall through (800/400/200/100).
```

Example (override eval to a longer segment for a low-base-rate cell):
```yaml
split:
  eval_rows: 400
```

For trailing-mode rows in the canonical `r_precision_at_k.csv`, the 8 calendar-date columns (`train_start, train_end, val_start, val_end, eval_start, eval_end, test_start, test_end`) carry the **calendar UNION across tickers** — MIN(start) and MAX(end) of `predictions/<seg>.csv['date']` across all tickers in the segment. For date-aligned rows they're the universe-calendar window directly.

Tickers with fewer than `min_rows_per_ticker` rows are dropped from the panel and listed in the artifact's `metrics.json::data.tickers_excluded`. Tickers excluded only from the train segment (via `min_train_rows_per_ticker`) still appear in val/eval/test and are listed in `metrics.json::data.tickers_per_segment`.

### `features` (optional)

Feature pool overrides. **By default the experiment starts with all 279 candidate columns** (the 16 families / 18 sub-family rows in `V1_PLAN.md` Stage 2); the agent prunes per-iteration.

| Field | Type | Default |
|---|---|---|
| `candidates` | list[str] or `"all"` | `"all"` (the full 279-col pool) |
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

The feature-selection + hyperparameter loop config.

| Field | Type | Default | Semantics |
|---|---|---|---|
| `callback_mode` | str | `"default"` | Which per-iteration FS+HP callback drives the loop. `"default"` = the algorithmic prune+nudge fallback (`default_fs_hp_callback`); `"agent_file_protocol"` = the V1.1 agent-driven exit-and-resume loop (see § "Agent-driven FS+HP loop" below). |
| `max_iterations` | int | `8` | Hard cap on iteration count. After this many iterations the loop emits the best-checkpoint artifact regardless of remaining headroom. Validated to `[1, 16]`. |
| `plateau_threshold` | float | `0.005` | Absolute val Brier improvement floor. If the last 2 iterations both improve by less than this, inner-stop fires. |
| `degradation_gate` | float | `0.01` | Multiplicative tolerance over best-seen val Brier. If the current val Brier > `(1 + degradation_gate) * best_val_brier`, inner-stop fires. |
| `tie_band` | float (optional) | `0.005` | Absolute val-Brier tie band for best-checkpoint selection (L1 from `_187`). Configs whose val Brier falls within `[min_val_brier, min_val_brier + tie_band]` are tied with best; among ties the per-cell rule applies (non-anti-AUC: lower gap → smaller `|z|`; anti-AUC: higher eval R-p@1 per V1.3 Option A § 3.5). Decoupled from `plateau_threshold` per bug #223 / memo `_222` (the pre-2026-06-02 default `0.5 × plateau_threshold` collapsed to noise level under the SKILL.md-recommended #204 workaround `plateau_threshold=0.0001`). Set to `0.0` to disable tie-breaking (revert to strict val-Brier argmin). |
| `degenerate_sink_threshold` | float (optional) | `1.05` | V1.3 Option A § 0 D5 — multiplicative threshold for the `degenerate_sink_warning` bundle field. Fires when val_brier ≤ threshold × weighted_base_rate_brier. |
| `search_space` | dict (optional) | unset | Per-spec **narrowing** of the canonical tunable-HP bounds (`TUNABLE_HP_RANGES` / `ENUM_HP_VALUES` in `model.py`). Each key is a tunable HP → `{min, max}` (numeric) or `{values: [...]}` (enum); a decision's `hp_changes` is validated against the intersection of the canonical bounds and this narrowing. Absent (the common case) ⇒ the canonical ranges are authoritative. |

Override example (tighter loop for a quick exploration):
```yaml
backend:
  fs_hp_loop:
    max_iterations: 4
    plateau_threshold: 0.01
```

Agent-driven loop example (the `/gbdt-experiment` agent surface drives the iterations):
```yaml
backend:
  fs_hp_loop:
    callback_mode: agent_file_protocol
    max_iterations: 8
```

> The `callback_mode` value is the spec field (snapshotted into `spec.yaml`, surfaced in `report.md`). It can be overridden at launch with `--callback-mode {default|agent_file_protocol}`; the override is mirrored back into the snapshot so an archived run records the *effective* mode that actually drove it.

##### Agent-driven FS+HP loop (`callback_mode: agent_file_protocol`)

The V1.1 exit-and-resume protocol (authoritative spec: `V1.1_agent_driven_fs_hp_loop_plan.md` § 0). Instead of a fixed prune heuristic, the agent (a Claude Code session, via `/gbdt-experiment`) makes each iteration's FS+HP decision. The runner and the agent talk through files co-located under the artifact dir:

```
results/gbdt/experiments/<experiment_name>/loop/
├── checkpoint.json            # resume checkpoint (full loop state, NO model blob)
├── iter_<N>_request.json      # the per-iteration bundle the agent READS (diagnose.json-shaped)
└── iter_<N>_decision.json     # the agent's decision the runner READS on --resume
```

Mechanic per iteration N: the runner trains iter N, writes `iter_<N>_request.json` + `checkpoint.json`, then **exits cleanly** (no blocking). The agent reads the request, writes `iter_<N>_decision.json`, and relaunches `python -m gbdt experiment <spec> --resume <run_id>`; the resumed run validates + applies the decision and trains iteration N+1 (iterations 0..N are **not** re-trained — they are threaded back from the checkpoint).

**`iter_<N>_request.json`** (the bundle the agent reads). Envelope keys: `schema_version`, `run_id`, `iter`, `max_iterations`, `available_features` (the active feature set a `prune_features` decision is validated against), and `diagnostics` (the `diagnose.json`-shaped payload from `build_diagnose_payload` — `metrics` {train/val Brier, train/val gap}, `overfit` {`no_overfit`, `train_val_gap`}, `prevalence_by_segment` + `prevalence_drift`, `calibration` {Spiegelhalter z/p}, `top_features` + `feature_importance`, `pruned_summary`, per-day `per_day_p_at_k` / `r_precision` (populated only when a prediction frame is threaded — `available: false` in-loop; the `r_precision` JSON key carries the legacy weighted form, retained as the in-loop ranking signal — see `[[project-r-precision-methodology]]` for R-Precision@K, the current 2026-06-01+ cross-cell headline computed post-hoc), `tuning_guidance` lines, and `full_diagnose_available: false` + `artifact_dir` for an on-demand full `/gbdt-diagnose`).

**`iter_<N>_decision.json`** (the agent writes). Schema (validated by `loop_protocol.validate_decision`):

| Field | Type | Semantics |
|---|---|---|
| `prune_features` | list[str] (optional) | Feature names to drop for iter N+1. Each MUST be in the request's `available_features`. |
| `hp_changes` | dict (optional) | HP name → new value. Each key must be a real tunable HP (in `TUNABLE_HP_RANGES` / `ENUM_HP_VALUES`), NOT a pinned HP (`has_time`, `loss_function`, `eval_metric`, `custom_metric`, `calibration_method`), and within the canonical bounds (∩ any `search_space` narrowing). |
| `should_stop` | bool (optional, default `false`) | When `true`, the next `--resume` finalizes the loop WITHOUT training a new iteration (`inner_stop_signal: "agent_should_stop"`); the best checkpoint is selected across the prior history. |
| `rationale` | str (optional) | The decision's lab-notebook entry. Recorded as iter N's `delta_attribution` (surfaced in `iterations.jsonl` + `report.md`). |
| `iter` | int (optional) | The iteration the decision is for (informational; the runner keys off the checkpoint's `iter_idx`). |

Any other fields are ignored. A malformed / out-of-bounds / pinned-HP / unknown-feature decision raises a clear `DecisionError` on `--resume` and does NOT corrupt state — the user fixes the file and relaunches. The full per-iteration reasoning + worked example live in `.claude/skills/gbdt-experiment/SKILL.md` § "Agent-driven FS+HP loop".

#### `backend.hp_starting`

Iteration-0 HP overrides. The agent uses these as the iteration-0 HPs; subsequent iterations are bounded by `CATBOOST_HP_REFERENCE.md` ranges per parameter.

| Field | Type | Default |
|---|---|---|
| `hp_starting` | dict | `{}` (inherit per-key from `default.yaml::backend.hp_starting`) |

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
├── loop/                             # ONLY under callback_mode=agent_file_protocol (V1.1)
│   ├── checkpoint.json               # resume state (no model blob); read by --resume
│   ├── iter_<N>_request.json         # per-iteration bundle the agent reads
│   └── iter_<N>_decision.json        # the agent's decision the runner applies on --resume
└── report.md                         # human-readable narrative (see V1_PLAN.md Stage 8 for sections)
```

The `loop/` subdir is present only for agent-driven runs (`callback_mode: agent_file_protocol`); `default`-mode runs finalize in a single process and write no loop files. See `backend.fs_hp_loop` § "Agent-driven FS+HP loop" above for the protocol.

**Every artifact is self-contained.** Re-running the spec against the same data + seed should reproduce the same artifact. Predictions are saved per segment so downstream analyses don't need to re-load the model.

### `iterations.jsonl` row schema

One JSON object per line, one line per FS+HP iteration:

```json
{
  "iter": 0,
  "hp": {"iterations": 1000, "depth": 6, "learning_rate": 0.05, ...},
  "features": ["stock_return_20", "realized_vol_20", ...],
  "n_features": 279,
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
  "sample_uniqueness": {
    "uniqueness_weighting": true,
    "horizon_days": 20,
    "effective_sample_size_per_fold": {
      "fold_0": {
        "train": {"ess_kish": 37210.2, "sum_weights": 1042.7, "n_rows": 38400, "overlap_inflation_ratio": 36.83},
        "val":   {"ess_kish": 18570.1, "sum_weights": 521.3,  "n_rows": 19200, "overlap_inflation_ratio": 36.83},
        "eval":  {"ess_kish": 9285.4,  "sum_weights": 260.7,  "n_rows": 9600,  "overlap_inflation_ratio": 36.83},
        "test":  {"ess_kish": 4642.7,  "sum_weights": 130.4,  "n_rows": 4800,  "overlap_inflation_ratio": 36.83}
      }
    }
  },
  "headline_eval": {
    "brier": 0.231,
    "brier_baseline_baserate": 0.239,
    "brier_improvement_vs_baseline": 0.008,
    "log_loss": 0.654,
    "roc_auc": 0.612,
    "brier_unweighted": 0.244,
    "brier_baseline_baserate_unweighted": 0.241,
    "brier_improvement_vs_baseline_unweighted": -0.003,
    "effective_sample_size_kish": 9285.4,
    "sum_weights": 260.7,
    "n_rows": 9600,
    "weighted_prevalence": 0.187
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

v1 pre-registers one universe out of the box: `nifty50`. Other valid NSE universes (`nifty100`, `nifty_midcap_150`, `nifty500`, etc.) are **resolvable on first use** via the agent's universe self-service flow — the agent fetches the official constituent list, writes a universe YAML to `configs/data_pipelines/domains/nse_equities/universe_<name>.yaml`, registers it in `configs/gbdt/default.yaml::universes`, and back-fills any missing tickers via `data_pipelines.fetch()`. See `.claude/skills/gbdt-experiment/SKILL.md` § "Pre-flight" for the orchestration.

The pre-registered `nifty50` block in `configs/gbdt/default.yaml::universes::nifty50` is the worked example:

```yaml
universes:
  nifty50:
    source: configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml
    index_ticker: "NIFTY:50"         # for F1, F5, F9, F9b families
    annualization_factor: 250        # √250 for vol families; differs from US's √252
```

The loader resolves `source`, reads the (fully-prefixed) ticker list, and calls `data_pipelines.fetch()` per ticker. `index_ticker` is fetched separately and joined into the panel for the index-relative feature families. See `docs/data_pipelines/universe_yaml_spec.md` for the full schema (both the standalone YAML and the registry-block contract).

**Inline tickers (ad-hoc universes).** A spec can declare a one-off universe inline by giving the universe a fresh name and including a top-level `tickers:` list; the agent's pre-flight registers the inline universe before falling through to the cache check. Example:

```yaml
target:
  universe: my_5_stock_basket
  direction: up
  threshold_pct: 10
  horizon_days: 20

tickers:
  - "NSE:RELIANCE"
  - "NSE:TCS"
  - "NSE:INFY"
  - "NSE:HDFCBANK"
  - "NSE:ICICIBANK"
```

NDX (US universe, different adapter chain) is in `V1.1_TBD.md`. Wider NIFTY universes are v1-supported via self-service (no entry in V1.1_TBD).

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

# features omitted - start with all 279 candidates; agent prunes per iteration.

backend:
  # Default library catboost; only one supported in v1.
  calibration_method: conditional_isotonic    # default - run Spiegelhalter Z, ship native or isotonic.
  fs_hp_loop:
    max_iterations: 8                         # default - 8-iter hard cap.
    plateau_threshold: 0.005                  # default - 0.5% absolute val Brier improvement floor.
    degradation_gate: 0.01                    # default - 1% degrade from best-seen val Brier triggers stop.

  # hp_starting omitted - use the iteration-0 values from default.yaml::backend.hp_starting.
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
- `target.max_drawdown` set but not a float in `(0, 1)` (must be a positive fractional bound less than 1).
- `backend.library` other than `"catboost"`.
- `backend.calibration_method` not in `{"native", "conditional_isotonic", "isotonic_always", "platt"}`.
- `backend.fs_hp_loop.callback_mode` not in `{"default", "agent_file_protocol"}` (the `--callback-mode` CLI override is validated the same way).
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
