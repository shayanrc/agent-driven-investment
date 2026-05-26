# gbdt — V1 Implementation Plan

## Build status

- **v1.0:** scaffolded — module skeleton + v0 EDA shipped on main at `2a62a1c`.
- **v1.0 spec-lock:** this PR. Locks the experiment-loop architecture, the 279-column feature pool, the CatBoost library choice, the synced FS+HP loop, the 800+400+200+100 walk-forward split, and the conditional-isotonic calibration policy. Stages 1–9 below are pending implementation.

For the *why* (what success looks like, anti-goals, deployment intent), see `goal.md`. The YAML spec contract is in `EXPERIMENT_SPEC.md`. The CatBoost HP reference the per-experiment agent consults is `CATBOOST_HP_REFERENCE.md`. Parked v1.1 follow-ups are in `V1.1_TBD.md`.

## Revision history

- **v1.0** *(2026-05-24, scaffolded)* — Initial scaffold: directory tree, stub modules, goal doc, plan skeleton. No implementation. Branch `gbdt-v1` from `main` at `9a03a00`. Stages 1–9 spec'd a per-asset 18-classifier lattice build with LightGBM defaults and several open library / fold / feature questions.
- **v1.0 spec-lock** *(2026-05-26, this PR)* — Long design conversation between the user and the parent agent locked the open questions and shifted the framing from "18-classifier lattice as v1 deliverable" to "experiment-loop infrastructure as v1 deliverable, with one pilot experiment as the merge gate." Library = CatBoost. Universe = NIFTY 50. Feature pool = 279 columns across 16 families (18 sub-family rows in the F-table — F6/F6b and F9/F9b are sibling rows). Split = 800+400+200+100. FS+HP = agent-driven synced loop, 8-iter cap, plateau + degradation inner-stop. Calibration = conditional isotonic gated by Spiegelhalter Z. See § "Decisions log" below for the deltas from v1.0's open questions.

---

## Purpose

This is an implementation specification for a **categorical-outcome forecasting surface** that runs single-tuple experiments end-to-end. Each experiment is `(universe, direction, threshold, horizon)`; the per-experiment artifact is a calibrated CatBoost classifier plus the agent's iteration history. v1's headline per experiment is **calibration**, not accuracy or AUC.

The plan is the output of the design conversations reflected in `goal.md` (v1 framing) and the long spec-lock conversation. Decisions documented here were made for a reason. **Do not silently change architectural decisions.** If implementation reveals a problem with a decision, surface it explicitly and ask before deviating.

---

## High-level architecture

Six layers, each independently testable, composed by `experiment.py`:

```
                          ┌──────────────────────────────────┐
spec.yaml ──▶ experiment.py: orchestrator (CLI + skill)
                          └─────────────────┬────────────────┘
                                            │
       ┌────────────────────────────────────┼────────────────────────────────────┐
       │                                    │                                    │
       ▼                                    ▼                                    ▼
┌──────────────┐                   ┌──────────────┐                    ┌──────────────┐
│  data.py     │                   │  features.py │                    │  targets.py  │
│  fetch       │                   │  16 families │                    │  binary      │
│  universe    │ ─── panel ──▶     │  279 cols    │ ──── X, names ──▶  │  label per   │
│  panel via   │                   │  per-stock + │                    │  spec target │
│  data_pipes  │                   │  xs pipes    │                    │  ───── y ──▶ │
└──────────────┘                   └──────────────┘                    └──────────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │  fs_hp_loop.py   │  ◀── reads CATBOOST_HP_REFERENCE
                                  │  diagnostic      │
                                  │  bundle per      │
                                  │  iteration; the  │
                                  │  agent reads     │
                                  │  the bundle and  │
                                  │  decides prune + │
                                  │  HP change       │
                                  └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │  train.py        │
                                  │  walk-forward    │
                                  │  800+400+200+100 │
                                  │  + model.py      │
                                  │  CatBoost wrap   │
                                  └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │  calibration.py  │
                                  │  Spiegelhalter Z │
                                  │  → native        │
                                  │  or isotonic     │
                                  └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │  report.py       │
                                  │  → report.md     │
                                  │  + figs/         │
                                  └────────┬─────────┘
                                           │
                                           ▼
              results/gbdt/experiments/<experiment_name>/
                ├── spec.yaml             (echo of input)
                ├── model.cbm             (CatBoost binary)
                ├── calibration.pkl       (None or fitted IsotonicRegression)
                ├── features.yaml         (the final pruned feature list)
                ├── hp.yaml               (the final HP set)
                ├── iterations.jsonl      (one row per FS+HP iter)
                ├── metrics.json          (headline metrics on eval segment)
                ├── predictions/          (per-stock per-fold CSV)
                ├── figs/                 (reliability, calibration, learning curves)
                └── report.md             (human-readable narrative)
```

Each layer mirrors `analog_mc`'s separation of concerns: data → features → labels → model → calibration → artifact. The new piece, relative to `analog_mc`, is `fs_hp_loop.py` — the diagnostic-bundle generator that turns the FS+HP search from an algorithmic loop into an **agent-driven loop**. The agent reads the bundle, decides what to prune and what HPs to change, and re-invokes the pipeline. The agent's chain of reasoning is logged into `iterations.jsonl` and narrated in `report.md`.

---

## Stage breakdown

The build order is strict — each stage ends with a passing test suite and a commit. Don't skip ahead. As with `analog_mc`'s plan, the diagnostic infrastructure (Stage 6) is what makes the experiment trustworthy, not the model wrapper (Stage 4).

### Stage 1 — Data loader + look-ahead-leak harness

**Goal:** load a universe panel via `data_pipelines.fetch()`, respecting walk-forward boundary discipline, and stand up the synthetic-data harness that detects causal-feature violations.

**Tasks:**
- `src/gbdt/data.py`:
  - `load_universe(preset_name, start, end) → dict[ticker, pd.DataFrame]` — reads the universe ticker list from the preset YAML (v1 = `configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml`), calls `data_pipelines.fetch()` per ticker, returns the canonical OHLCV panel keyed by ticker. Drops tickers below the spec's `min_rows` (default 1,600).
  - `align_panel(panel) → pd.DataFrame` — returns a long-format frame with `(date, ticker)` MultiIndex and one row per (date, ticker). This is the input to the feature layer; cross-sectional features operate on cross-sections of this frame.
  - Walk-forward discipline: feature computation at row `(t, ticker)` uses only data with `date < t` (per-stock for time-series features, per-cross-section for xs features at exactly `t`).
- `src/gbdt/leakage_harness.py` — synthetic OHLCV generator that plants a known "leak signal" at a future row. Any feature that incorporates the leak achieves perfect AUC on the synthetic data; any causally-correct feature achieves chance AUC.
- `tests/gbdt/test_data_loader.py` — loads NIFTY 50 panel via `data_pipelines.fetch()` (uses the cache), asserts schema, dtypes, monotonic dates per ticker, no duplicate (date, ticker) pairs, and that 48 of 50 tickers meet `min_rows ≥ 1,600` (JIOFIN and MAXHEALTH excluded by row count).
- `tests/gbdt/test_leakage_harness.py` — confirms the harness fires on a known leaky function and stays silent on a known causal function.

**Universe self-service (pre-flight responsibility of the `/gbdt-experiment` skill, not `data.py`).** When a spec names a universe that isn't a registered preset under `configs/gbdt/default.yaml::universes`, the skill is responsible for registering it before the loader runs. The flow:

1. **Detect the gap.** Spec parse fails the universe lookup → don't fall through to `data.load_universe()` yet.
2. **Resolve the ticker list.** Two paths:
   - **Inline tickers** — the spec carries a top-level `tickers:` list; the skill writes `configs/data_pipelines/domains/nse_equities/universe_<name>.yaml` with that list (using the same schema as `universe_nifty50.yaml`).
   - **Well-known NSE index** (`nifty100`, `nifty_midcap_150`, `nifty500`, etc.) — the skill fetches the constituent list via the `data_pipelines` adapter chain (the same chain that already resolves `^NSEI` / `NIFTY:NIFTY100`) and writes the universe YAML.
3. **Register the universe block.** Append a `universes::<name>` entry to `configs/gbdt/default.yaml` pointing at the new YAML, with `index_ticker` and `annualization_factor` inferred from the universe type (NSE indices use a `NIFTY:` index_ticker and `250`; US universes use an `INDEX:` index_ticker and `252`). Schema is documented in `docs/data_pipelines/universe_yaml_spec.md`.
4. **Back-fill the cache.** Run `data_pipelines.fetch()` per ticker (sequential or limited-parallel — respect the SQLite single-writer contract). Skip tickers already in the cache that meet the spec's `min_rows_per_ticker`.
5. **Apply the row gate.** Drop any ticker below `spec.split.min_rows_per_ticker` (default 1,600) with a logged note. The note lands in `metrics.json::data.tickers_excluded` once the experiment runs.

This means a new experiment YAML for `nifty500` can run end-to-end on a fresh checkout without the user editing any infrastructure files first. Cold-pull cost is the only real bound (a NIFTY 500 cold pull is materially longer than NIFTY 50). The skill should surface the cold-pull estimate before committing.

**Done when:** panel loader works on the real NIFTY 50 cache; 48 tickers pass the row-count gate; harness correctly distinguishes leaky vs causal features on synthetic data; the universe self-service flow can register a fresh universe end-to-end (validated against `nifty100` as the first additional universe).

### Stage 2 — Feature implementation (279-column candidate pool)

**Goal:** implement all 16 feature families enumerated below. The pool is fixed; per-experiment pruning is the agent's job in Stage 6.

**Constraint:** all features are strictly causal at row `(t, ticker)`. Annualization on the NIFTY panel uses `√250` (not `√252`). Lookback windows are exposed in config as `[5, 10, 20, 50, 100, 200]`.

**Feature pool (279 columns across 16 families, 18 sub-family rows in the table — F6/F6b and F9/F9b are sibling rows mentally counted as one family each):**

| # | Family | Cols | Notes |
|---|---|---|---|
| F1 | `index_return_N` | 6 | `^NSEI close[t] / close[t−N]` − 1 |
| F2 | `stock_return_N` | 6 | per-stock momentum |
| F3 | `rel_strength_N` | 6 | F2 − F1 |
| F4 | `realized_vol_N` | 6 | std(log returns) × √250 |
| F5 | `index_vol_N` | 6 | ^NSEI realized vol |
| F6 | `drawdown_N` | 6 | (close[t] / max(high[t−N+1:t])) − 1 — uses HIGH not close |
| F6b | `runup_N` | 6 | (close[t] / min(low[t−N+1:t])) − 1 — uses LOW |
| F9 | `index_drawdown_N` | 6 | ^NSEI drawdown using index high |
| F9b | `index_runup_N` | 6 | ^NSEI runup using index low |
| F7 | volume family | 32 | `volume_ratio_N`(6) + `obv_N`(6) + `vol_ret_corr_N`(6) + `dollar_move_zscore_N`(6 rolling) + `dollar_move_rank_N`(6 rolling) + `dollar_move_xs_zscore`(1 cross-sectional) + `dollar_move_xs_rank`(1 cross-sectional). NO raw `volume_N` — replaced by ratio + dollar-move family. |
| F8 | higher moments | 12 | `returns_skew_N`(6) + `returns_kurt_N`(6) |
| F10 | `beta_N` | 6 | rolling OLS coef of stock returns on ^NSEI returns |
| F11 | range vol | 12 | `parkinson_N`(6) + `garman_klass_N`(6); NOT Yang-Zhang |
| F12 | `sma_distance_N` | 6 | (close[t] / mean(close[t−N+1:t])) − 1 |
| F13 | vol regime | 18 | `vol_change_N`(6) + `vol_of_vol_N`(6) + `vol_pct_N`(6) |
| F14 | cross-sectional rank+z | 24 | `return_xs_rank_N`(6) + `return_xs_zscore_N`(6) + `vol_xs_rank_N`(6) + `vol_xs_zscore_N`(6) |
| F15 | calendar | 10 | DOW (sin, cos) + DOM (sin, cos) + MOY (sin, cos) + 4 India binary flags: `fiscal_year_end_week`, `budget_week`, `diwali_week`, `fomc_week` |
| F16 | signed days outside band | 105 | 12 new underlying z-scores (`stock_return_zscore_N`(6) + `realized_vol_zscore_N`(6)) + signed-days-outside-band meta on the z-scored underlying. Sign = sign of current side of mean (current-side convention, "option A"). Value = 0 inside the band; +k = k consecutive days above +Xσ; −k = k consecutive days below −Xσ. Resets to 0 the moment z re-enters the band. Pool size: 12 underlying + 93 meta = 105 base columns in the candidate pool. |

**Total: 279 columns.** (Row-wise sum: F1–F6 + F6b + F9 + F9b = 9 × 6 = 54; +F7 = 86; +F8 = 98; +F10 = 104; +F11 = 116; +F12 = 122; +F13 = 140; +F14 = 164; +F15 = 174; +F16 = 279.)

**Tasks:**
- `src/gbdt/features.py`:
  - Each feature function is a pure callable taking the panel DataFrame and returning a Series (per-stock) or DataFrame (cross-sectional batch) aligned on the input MultiIndex.
  - Two pipeline modes:
    1. **Per-stock rolling** — F1–F13, F15, F16 (rolling on each ticker's time series).
    2. **Cross-sectional point-in-time** — F14 + F7's `_xs_` columns (compute on each date's cross-section).
  - A `build_feature_matrix(panel, spec) → pd.DataFrame` orchestrator assembles the matrix per the candidate list in `spec.features.candidates` (default = all 279).
- Each feature has a unit test on a small fixture (one synthetic ticker, 30 rows) with at least one hand-computed expected value.
- Every feature passes the look-ahead-leak harness from Stage 1.
- `configs/gbdt/default.yaml`'s `lookback_windows` is the source of truth for the windows list.

**Done when:** the 279-column matrix builds end-to-end on the NIFTY 50 panel, every feature has a unit test, the matrix passes the leak harness.

### Stage 3 — Target builder

**Goal:** binary target derivation from a `(direction, threshold_pct, horizon_days)` tuple, with an optional `max_drawdown` path-honesty filter.

**Tasks:**
- `src/gbdt/targets.py`:
  - `compute_target(panel, target_spec) → pd.Series` — returns a `0/1/NaN` series aligned on the `(date, ticker)` MultiIndex.
  - A target at row `(t, ticker)` is `1` if the spec's success criterion is met inside `(t, t+horizon_days]`, else `0`. Rows with insufficient forward data are `NaN` (excluded from train/val/eval/test labels; kept as inference rows so predictions can be emitted).
  - **Simple binary mode (no `max_drawdown`)** — one-phase check:
    - Direction `up`: `1` if `max(high[t+1:t+horizon_days]) ≥ close[t] * (1 + threshold_pct/100)`.
    - Direction `down`: `1` if `min(low[t+1:t+horizon_days]) ≤ close[t] * (1 − threshold_pct/100)`.
    - The threshold check uses HIGH for up / LOW for down (the most conservative interpretation of "did the threshold ever fire").
  - **Path-honesty mode (`target.max_drawdown` set)** — two-phase check. **UP direction:**
    1. **Find the breach index.** Scan `(t, t+horizon_days]` for the first index `t_breach` where `close[t_breach] ≥ (1 + threshold_pct/100) * close[t]`. If no such index exists → label `0`.
    2. **Check the path's max drawdown before breach.** If `min(close in (t, t_breach]) > (1 − max_drawdown) * close[t]` → label `1` (positive). Else → label `0` (the threshold fired, but only after the position would have been wiped out).
  - **Path-honesty mode — DOWN direction** (symmetric; `max_drawdown` denotes the maximum adverse path excursion regardless of direction — for a short, that adverse excursion is an *upside* rally before the down-breach):
    1. **Find the breach index.** Scan `(t, t+horizon_days]` for the first index `t_breach` where `close[t_breach] ≤ (1 − threshold_pct/100) * close[t]`. If no such index exists → label `0`.
    2. **Check the path's max adverse excursion before breach.** If `max(close in (t, t_breach]) < (1 + max_drawdown) * close[t]` → label `1` (positive). Else → label `0` (the threshold fired, but only after the short would have been wiped out by an intervening rally).
    - This generalizes the v0.3 filter (`docs/gbdt/_v0_path_honesty_eval.md`), which hard-wired `max_drawdown = threshold_pct / 200`. v1 makes the bound an explicit per-experiment parameter.
    - Negatives in this mode bucket two distinct failure modes (no-breach + breach-after-drawdown); they are aggregated into the single `0` label intentionally — the model learns "would the position have made it cleanly to the threshold," not "did the threshold fire at all."
    - **Breach criterion switches from HIGH/LOW (simple mode) to CLOSE (path-honesty mode), intentionally.** The simple-binary mode uses HIGH (UP) / LOW (DOWN) — the most aggressive intraday move ever observed — because the question there is "did the threshold ever fire in *any* interpretation." Path-honesty mode flips both legs to CLOSE because the operator semantics are mark-to-market: the breach is the closing price the operator could realistically exit at, and the drawdown filter is the closing-price excursion the position would have weathered. Consequence: a path-honesty positive can be strictly stricter than a simple-binary positive even with a permissive `max_drawdown`, because the breach must clear on CLOSE rather than HIGH/LOW.
    - The drawdown check uses CLOSE (not LOW for UP / not HIGH for DOWN), since the operator-facing semantics are mark-to-market drawdown a position would have experienced day-over-day. If the spec writer wants the more conservative intraday-extremum bound (worst HIGH/LOW excursion), that's a v1.1 follow-up.
- Unit tests on a synthetic price path with known breach + drawdown patterns for each direction × threshold × horizon × max_drawdown combination.
- Edge cases tested: target at the last row (no forward data → `NaN`); breach exactly on day `t+horizon_days` (counts); breach on day `t+1` (counts); no breach in window (`0`); breach after deep drawdown with `max_drawdown` set (counts as `0`); shallow drawdown before breach with `max_drawdown` set (counts as `1`).

**Done when:** target builder produces correct labels against the synthetic fixtures in both modes; the spec drives the tuple; edge cases pass.

### Stage 4 — CatBoost wrapper + model.py

**Goal:** thin CatBoost wrapper exposing a stable interface, with the v1 HP constraints baked in.

**Tasks:**
- `pyproject.toml` — add `catboost` to dependencies.
- `src/gbdt/model.py`:
  - `class GBDTClassifier`:
    - `fit(X_train, y_train, X_val, y_val, hp_dict)` — instantiates `CatBoostClassifier(**hp_dict)`, fits with `eval_set=(X_val, y_val)`, returns `self`.
    - `predict_proba(X) → np.ndarray` — returns calibrated probabilities (post-Stage 5) by default.
    - `raw_predict_proba(X) → np.ndarray` — returns uncalibrated probabilities for diagnostics.
    - `save(path)` / `load(path)` — persists `.cbm` + calibration map together.
    - `feature_importance() → dict[name, float]` — for the diagnostic bundle.
  - **Pinned HPs (never tunable in v1):**
    - `has_time=True` (mandatory; per `CATBOOST_HP_REFERENCE.md` § "Determinism" — walk-forward + ordered boosting require it)
    - `loss_function="Logloss"`
    - `eval_metric="BrierScore"`
    - `custom_metric=["Logloss", "BrierScore", "AUC"]`
    - `random_seed=<spec.random_seed>` (default 42)
  - **Tunable HPs** (bounded by `CATBOOST_HP_REFERENCE.md` per-parameter "Range/values" sections): `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `min_data_in_leaf`, `rsm`, `bootstrap_type`, `bagging_temperature`, `subsample`, `random_strength`, `auto_class_weights` (or `scale_pos_weight`), `boosting_type`, `early_stopping_rounds`.
  - All other CatBoost params are not tuned in v1 (see `CATBOOST_HP_REFERENCE.md` § "Scope" for the explicit exclusion list).
- Unit tests:
  - Trivial-fit smoke: 2-feature synthetic dataset where one feature perfectly predicts the target → near-perfect train accuracy + dominant importance for the predictive feature.
  - `has_time=True` is enforced (wrapper raises if a spec tries to override it).
  - HP values out of the documented ranges raise on `fit()`.

**Done when:** wrapper is import-clean; smoke test green; HP bounds enforced; `has_time=True` is non-overridable.

### Stage 5 — Calibration

**Goal:** conditional isotonic calibration gated by a Spiegelhalter Z-test on val.

**Tasks:**
- `src/gbdt/calibration.py`:
  - `spiegelhalter_z(y_true, p_pred) → (z, p_value)` — standard implementation.
  - `decide_calibration(y_val, p_val_raw, z_threshold=2.0) → "native" | "isotonic"`:
    - Compute `|z|` from the val segment's raw predictions.
    - If `|z| < z_threshold`, calibration passes → return `"native"` (ship raw CatBoost outputs).
    - Else → return `"isotonic"` (fit `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` on val and ship the calibrator alongside the model).
  - `fit_calibrator(y_val, p_val_raw, method)` — returns `None` if `method == "native"`, else the fitted IsotonicRegression.
  - `apply(p_raw, calibrator) → np.ndarray` — passes through if `calibrator is None`, else `calibrator.predict(p_raw)`.
  - The decision + Z statistic are recorded in `metrics.json["calibration"] = {"method": ..., "spiegelhalter_z": ..., "spiegelhalter_p": ...}`.
- Spec overrides via `spec.backend.calibration_method`: `"native"`, `"conditional_isotonic"` (default — runs the test), `"isotonic_always"`, `"platt"` (sklearn `CalibratedClassifierCV(method="sigmoid")`). See `EXPERIMENT_SPEC.md` § "Calibration".
- Unit tests:
  - Deliberately miscalibrated synthetic model → Z-test fires → isotonic improves reliability on held-out chunk.
  - Well-calibrated synthetic model → Z-test passes → no isotonic applied.
  - `calibrator` round-trips through save / load.

**Done when:** Spiegelhalter Z computed correctly; conditional branch tested both ways; decision recorded in artifact.

### Stage 6 — FS+HP loop (the agent's iteration infrastructure)

**Goal:** the diagnostic-bundle generator + inner-stopping logic that the agent uses to drive feature selection and hyperparameter tuning in one synced loop.

**Tasks:**
- `src/gbdt/fs_hp_loop.py`:
  - `class DiagnosticBundle`:
    - `train_brier`, `val_brier`, `eval_brier_provisional` — the headline metric on each split.
    - `train_val_gap` = `val_brier − train_brier`.
    - `learning_curve` — per-iteration train/val loss arrays from CatBoost's training log.
    - `early_stop_iteration` — where CatBoost halted; flag if `iteration_cap_hit` (within 10% of `iterations`).
    - `feature_importance` — top-K importances from the wrapper.
    - `feature_correlation` — pairwise correlation matrix of currently-active features.
    - `calibration_summary` — Spiegelhalter Z + reliability deviation.
    - `class_imbalance_summary` — positive prevalence, recall at threshold 0.5.
    - `hp_history` — list of (iteration, hp_dict, rationale) tuples up to and including this iter.
  - `serialize_bundle(bundle, path)` — writes a JSON the agent can read.
  - `inner_stop_check(history) → (should_stop, reason)`:
    - **plateau**: val Brier improvement < `plateau_threshold` (default 0.005 absolute) over the last 2 iterations → stop.
    - **degradation**: val Brier > `(1 + degradation_gate)` × best-seen val Brier (default `degradation_gate=0.01` → "1% degrade from best") → stop.
    - **cap**: iteration count ≥ `max_iterations` (default 8) → stop.
  - `best_checkpoint(history) → iteration_index` — index of the iteration with the lowest val Brier. The artifact emitted at the end of the loop is **the best-Brier checkpoint, not the last iteration**.
- `iterations.jsonl` schema: one row per iteration with fields `{iter, hp, features, rationale, train_brier, val_brier, train_val_gap, calibration_method, calibration_z, early_stop_iter, wall_time_sec}`.
- Spec overrides via `spec.backend.fs_hp_loop`: `max_iterations`, `plateau_threshold`, `degradation_gate`. See `EXPERIMENT_SPEC.md` § "FS+HP loop".
- Unit tests:
  - Inner-stop fires on a synthetic history that plateaus → returns `("plateau", ...)`.
  - Inner-stop fires on a synthetic history that degrades → returns `("degradation", ...)`.
  - Inner-stop fires at the cap.
  - `best_checkpoint` picks the right iteration on a synthetic history.

**Done when:** diagnostic bundle serialization is stable; inner-stop logic tested on all three branches; best-checkpoint selection tested.

### Stage 7 — Walk-forward driver

**Goal:** the train.py orchestrator that runs the 800+400+200+100 fold scheme across the panel.

**Tasks:**
- **Walk-forward split:** 800 train + 400 val + 200 eval + 100 test = 1,600 rows total per stock (the global default in `configs/gbdt/default.yaml`; spec-overridable).
  - **train**: where the model fits.
  - **val**: where CatBoost early-stops + where calibration is decided (Spiegelhalter Z) + where the FS+HP loop reads its inner-stop signal.
  - **eval**: held-out segment that the agent never sees during FS+HP iteration. This is the segment whose metrics go into `metrics.json` as the headline.
  - **test**: a smaller still-held-out segment kept for a final sanity check after the experiment closes. Mirrors `analog_mc`'s nested split discipline.
- `src/gbdt/train.py`:
  - `train_fold(panel, target, features, hp, split) → FoldResult` — fits CatBoost on (X_train, y_train) with early stopping against (X_val, y_val), calibrates per Stage 5, scores (X_eval), persists predictions.
  - **In v1, a single fold is the default** (one 1,600-row slice anchored on the latest available date per stock). Multi-fold walk-forward is supported by the driver but defaults to `n_folds=1` because the per-experiment compute envelope already includes the FS+HP loop. Specs can override `split.n_folds` for multi-fold runs.
  - Predictions per (date, ticker) saved to `predictions/<segment>.csv` with columns `date, ticker, p_raw, p_calibrated, y_true`.
- A unit test runs a 1-fold mini-walk-forward on the synthetic Stage 1 fixture and confirms artifacts land on disk in the expected layout.

**Done when:** walk-forward driver runs end-to-end on a small synthetic dataset; artifact layout matches `EXPERIMENT_SPEC.md` § "Artifact directory layout".

### Stage 8 — Report renderer + CLI atom

**Goal:** the human-readable `report.md` + the CLI atom that ties Stages 1–7 together.

**Tasks:**
- `src/gbdt/report.py`:
  - Reads `spec.yaml`, `metrics.json`, `iterations.jsonl`, the figs in `figs/`.
  - Renders `report.md` with sections:
    1. **Spec** — echo of the tuple + universe + key spec overrides.
    2. **Data** — N rows train/val/eval/test, positive prevalence per segment.
    3. **Iteration history** — table of the 1–8 iterations: features kept, key HP changes, train Brier, val Brier, train-val gap, agent's rationale, inner-stop signal.
    4. **Final checkpoint** — which iteration was chosen, why (best val Brier).
    5. **Calibration** — Spiegelhalter Z + p-value, decision (native vs isotonic), reliability diagram inline.
    6. **Headline metrics on eval** — Brier vs base-rate baseline, AUC, log-loss, per-stock breakdown (table).
    7. **Per-experiment verdict (agent recommendation)** — the agent's one-paragraph readout. Explicitly NOT an automatic pass/fail.
- `src/gbdt/experiment.py` — top-level orchestrator. The `python -m gbdt.experiment <spec.yaml>` CLI atom invokes this.
  - Phases: load spec → load data → build features → build target → iteration 0 (all features, default HPs) → FS+HP loop (max 8 iter, agent decides each iter) → calibration → emit artifact.
  - In v1 the "agent decides each iter" phase is the **invocation of the agent** — the orchestrator writes the iteration's diagnostic bundle to a temp path, halts, and waits for the agent (via the `/gbdt-experiment` skill) to read the bundle and write back the next iteration's `(prune_list, hp_changes, rationale)` JSON. The CLI atom is a one-shot interactive runner; the skill is the orchestrator wrapper that drives the per-iteration agent reasoning.
- CLI smoke test: invokes `python -m gbdt.experiment configs/gbdt/experiments/_test_tiny.yaml` against the synthetic fixture and confirms expected artifact dir on disk.

**Done when:** CLI atom runs end-to-end on a tiny synthetic fixture; `report.md` renders cleanly; the artifact dir matches the spec.

### Stage 9 — Skill + pilot experiment (PR merge gate)

**Goal:** the `/gbdt-experiment` skill + a green pilot run on the NIFTY 50 panel. **This stage gates the PR merge.**

**Tasks:**
- `.claude/skills/gbdt-experiment/SKILL.md` — the agent-facing skill, ~150 lines. Sections: Purpose / Invocation / Pre-flight / Phases (data build → iteration 0 → FS+HP loop → calibration → artifact emission) / Long-running pattern reference / References to `V1_PLAN.md`, `CATBOOST_HP_REFERENCE.md`, `EXPERIMENT_SPEC.md`.
- `configs/gbdt/experiments/nifty50_up_10pct_20d_pilot.yaml` — the pilot spec.
- Run the pilot end-to-end: `python -m gbdt.experiment configs/gbdt/experiments/nifty50_up_10pct_20d_pilot.yaml` (or via `/gbdt-experiment <spec_path>`).
- Verify the merge-gate condition (from `goal.md` § "v1 PR merge gate"):
  - The artifact dir at `results/gbdt/experiments/nifty50_up_10pct_20d_pilot/` exists and contains every file enumerated in `EXPERIMENT_SPEC.md` § "Artifact directory layout".
  - `report.md` narrates the iteration history with per-iteration rationale.
  - `metrics.json` is well-formed JSON with the headline calibration + Brier metrics.
  - The per-experiment verdict in `report.md` is for the user to read — not an automated gate.
- Commit the artifact (`results/gbdt/experiments/`) and the headline `results/gbdt/data/_v1_<run_id>_data.json` per `[[project-results-layout]]`.

**Done when:** the pilot experiment artifact exists, is complete, and the report is readable. The cell's calibration/Brier outcome itself is information for the user, not a merge blocker.

---

## Decisions log (deltas from the original v1.0 plan)

The v1.0 plan (2026-05-24) left five intentional open questions to be resolved during implementation. The spec-lock conversation (2026-05-26) resolved all five and introduced several framing changes. The deltas:

### D1. Library: LightGBM → CatBoost

**Original (v1.0 OQ #2):** "Default lean: **lightgbm**. ... If lightgbm has packaging / install issues on this Python version, fallback is xgboost."

**Locked:** **CatBoost**.

**Rationale:**
- **Ordered boosting** reduces overfit on small rare-cell training data. Per-cell row counts are ~70k pooled (1.6k rows × 48 stocks); for rare cells (e.g. `±50% / 10d` at <2% prevalence), this is the data-scale where ordered boosting's prediction-shift correction measurably pays off.
- **Native calibration quality.** CatBoost's out-of-the-box probability outputs are better calibrated than LightGBM/XGBoost on similar data shapes. Calibration is the headline metric, so the speed cost (2–3× slower per iteration than LightGBM) is well-spent compute.
- **`has_time=True`** + per-iteration ordered permutations give correct walk-forward behavior with no extra plumbing.

**Cost:** wall-clock per iteration is higher. Mitigated by the FS+HP loop's hard cap of 8 iterations and the inner-stop gates (plateau, degradation).

### D2. Framing: 18-classifier lattice → experiment-loop infrastructure

**Original (v1.0 § "Purpose"):** "v1 is one asset (NASDAQ100) with the 18-target lattice defined in `goal.md`. ... The headline metric is **calibration**."

**Locked:** v1 ships experiment-loop infrastructure. Each experiment is a single `(universe, direction, threshold, horizon)`. The 18-cell lattice was v0 EDA, not the production deliverable.

**Rationale:**
- v0's 18-cell scan was an **exploration grid** to characterize base rates and discover which cells are plausibly predictable. Treating it as the production deliverable conflates "scan we ran" with "models we ship."
- Different downstream consumers will want different cells (an alerting use case wants `±5% / 10d`; a risk-overlay use case wants `±20% / 50d`). A surface that runs any cell on demand is more useful than a fixed bundle of 18 models.
- A per-experiment artifact directory is auditable in a way that "the 18-model run" isn't — each artifact is self-contained and self-narrating.
- Removed the "14/18 acceptance" framing — it was tied to the lattice ship and has no analog under the experiment-loop framing. The PR merge gate is now "the pilot experiment runs end-to-end and emits a complete artifact"; the per-experiment verdict is for the user.

### D3. Dataset: NASDAQ100 single asset → NIFTY 50 pooled panel

**Original (v1.0 § "Anti-goals"):** "**No multi-asset training.** One asset (NASDAQ100). Cross-sectional is v2."

**Locked:** **NIFTY 50 pooled panel** as the v1 universe. NDX deferred to v1.1.

**Rationale:**
- The `data-seed-nifty50-deep` branch (PR #6, merged) brought 44 of 50 NIFTY 50 tickers to ≥2,500 rows of OHLCV history. This is enough for the 800+400+200+100 split scheme on 48 tickers (JIOFIN at 578 rows and MAXHEALTH at 1,322 rows fall below the 1,600 floor and are excluded; the other IPO-bounded tickers — ETERNAL, HDFCLIFE, INDIGO, SBILIFE — all have ≥1,700 rows and stay in).
- A panel enables **cross-sectional features** (F14: rank/z across the universe), which are first-class signals (`this stock is the 3rd-weakest today`) that single-asset training cannot represent.
- Pooled training on 48 tickers gives ~70k rows per experiment vs ~2k for single-asset — enough to fit a real model on rare cells without per-stock overfit.
- The v0 NIFTY 50 opportunity scans (v0.1, v0.2, v0.3) already provide base-rate / regime context for this exact universe.

### D4. FS + HP loop: algorithmic FS + fixed HP grid → agent-driven synced loop

**Original (v1.0 § "Anti-goals"):** "**No HPO framework.** A small fixed hyperparameter grid + early stopping handles tuning."

**Locked:** **Agent-driven synced FS+HP loop**, bounded by `CATBOOST_HP_REFERENCE.md` ranges, capped at 8 iterations with plateau + degradation inner-stop.

**Rationale:**
- Feature selection and HPs **interact**. The right `depth` depends on how many features are in; the right `l2_leaf_reg` depends on how correlated the surviving features are; the right `auto_class_weights` depends on the prevalence after pruning. Running them as separate sequential phases leaves cross-effects on the table.
- An algorithmic FS pass (RFE, importance-threshold) makes local decisions; an agent reading a diagnostic bundle can make decisions that account for calibration, train-val gap, dominance patterns, and correlation simultaneously.
- Bounding the agent to `CATBOOST_HP_REFERENCE.md`'s documented ranges prevents the agent from drifting into pathological regions. The reference doc is per-parameter "when to change" rubrics, not a free-form prompt.
- The hard cap of 8 iterations + plateau + degradation inner-stop bounds compute. The best-Brier checkpoint (not the last iteration) is what ships, so the loop doesn't penalize itself for exploring slightly-worse iterations.
- If the agent loop proves inconsistent across runs in practice, the v1.1 fallback (`V1.1_TBD.md` § "Bayesian HP search alternative") swaps the agent reasoning for an Optuna-style optimizer behind the same diagnostic-bundle contract.

### D5. Calibration: isotonic always → conditional isotonic gated by Spiegelhalter Z

**Original (v1.0 OQ #3):** "Default lean: scikit-learn `IsotonicRegression`. Switch to Platt scaling only if isotonic over-fits on small validation folds."

**Locked:** **Conditional isotonic** — run Spiegelhalter Z-test on val; ship raw CatBoost if it passes (|z| < 2), layer isotonic if it fails. Per-experiment decision recorded in artifact.

**Rationale:**
- CatBoost's native probability outputs are often well-calibrated out of the box. Always-on isotonic adds variance without value when calibration is already good — isotonic on small val segments can over-fit and slightly degrade test-segment calibration.
- The Spiegelhalter Z-test is a cheap statistical test with a clear decision threshold. Gating isotonic behind it gives the best of both: ship the simpler model when it's calibrated, layer the correction when it isn't.
- Recording the decision per experiment (`calibration_method` + Z + p-value in `metrics.json`) makes the calibration choice auditable rather than implicit.
- Specs can override the decision: `spec.backend.calibration_method` accepts `"native"`, `"conditional_isotonic"` (default), `"isotonic_always"`, `"platt"`.

### D6. Walk-forward split: TBD-stage6 → 800 + 400 + 200 + 100

**Original (v1.0 OQ #4):** "Inherit from `analog_mc`'s walk-forward conventions if they apply directly; otherwise pin a v1 scheme in `configs/gbdt/default.yaml` with reasoning in the Stage 6 commit."

**Locked:** **800 train + 400 val + 200 eval + 100 test = 1,600 rows** per stock, single-fold default. Global setting in `configs/gbdt/default.yaml`; spec-overridable via `spec.split`.

**Rationale:**
- 800-row train fits well within the deepest-history slice for 48 of 50 NIFTY 50 tickers (44 have ≥2,500 rows; the other 4 have 1,700–2,400).
- 400-row val gives enough samples for both early-stopping signal and a meaningful Spiegelhalter Z-test on the calibration gate.
- 200-row eval is the segment whose metrics go into `metrics.json` headline. Smaller than val so the Brier numbers are computable without being val-noise-dominated.
- 100-row test sits beyond eval as a final-sanity-check slice for the user (mirrors `analog_mc`'s nested-holdout discipline).
- Single-fold default keeps the per-experiment compute envelope manageable. Multi-fold is supported (`spec.split.n_folds`) but not the default — the FS+HP loop already adds an iteration multiplier.
- JIOFIN (578 rows) and MAXHEALTH (1,322 rows) fall below the 1,600 floor and are excluded by `min_rows=1600`. The Stage 1 loader logs the exclusions.

### D7. Multi-target / shared-tree head — still deferred, now to v1.1

**Original (v1.0 OQ #5):** "v1 trains 18 independent classifiers. ... v2 considers a shared-tree multi-output architecture."

**Locked:** Each experiment is one binary target. Multi-target architectures are deferred to v1.1 (`V1.1_TBD.md` § "Multi-target shared-tree head"). The framing change to "experiment loop" makes this a per-experiment-spec concern rather than a v1 architectural decision.

---

## File layout

```
docs/gbdt/
  goal.md                              # done (this PR)
  V1_PLAN.md                           # done (this PR)
  EXPERIMENT_SPEC.md                   # done (this PR)
  CATBOOST_HP_REFERENCE.md             # done (already in tree, untouched in this PR)
  V0_INVESTIGATION_PLAN.md             # shipped on main; closing note added this PR
  V1.1_TBD.md                          # done (this PR)
  _v1_<experiment>_report.md           # one per experiment that's promoted to docs

src/gbdt/
  __init__.py
  __main__.py                          # invokes experiment.py
  data.py                              # universe loader (Stage 1)
  leakage_harness.py                   # synthetic leak detector (Stage 1)
  features.py                          # 279-col candidate pool (Stage 2)
  targets.py                           # single binary target per spec (Stage 3)
  model.py                             # CatBoost wrapper (Stage 4)
  calibration.py                       # conditional isotonic (Stage 5)
  fs_hp_loop.py                        # diagnostic bundle + inner-stop (Stage 6)
  train.py                             # walk-forward driver (Stage 7)
  predict.py                           # inference against a saved artifact
  experiment.py                        # top-level orchestrator + CLI (Stage 8)
  report.py                            # report.md renderer (Stage 8)

configs/gbdt/
  default.yaml                         # global defaults (lookback windows, split, FS+HP loop, calibration)
  experiments/
    nifty50_up_10pct_20d_pilot.yaml                  # the v1 PR pilot spec (Stage 9)
    nifty50_up_20pct_50d_dd10pct.yaml                # path-honesty filter cell on the pilot universe
    nifty100_up_10pct_20d_dd5pct.yaml                # pilot tuple on the wider NIFTY 100 cohort
    nifty_midcap_150_up_50pct_100d_dd20pct.yaml      # rare cell on a higher-vol midcap cohort
    nifty500_up_30pct_50d_dd15pct.yaml               # medium-rare cell on the broadest NSE cohort

.claude/skills/gbdt-experiment/
  SKILL.md                             # the agent surface (Stage 9)

tests/gbdt/
  __init__.py
  test_data_loader.py                  # Stage 1
  test_leakage_harness.py              # Stage 1
  test_features.py                     # Stage 2
  test_targets.py                      # Stage 3
  test_model.py                        # Stage 4
  test_calibration.py                  # Stage 5
  test_fs_hp_loop.py                   # Stage 6
  test_train.py                        # Stage 7
  test_experiment_cli.py               # Stage 8

results/gbdt/
  data/                                # checked-in headline metric JSONs (per [[project-results-layout]])
  experiments/<experiment_name>/       # checked-in per-experiment artifact dirs (Stage 9)

runs/gbdt/<UTC>/                       # gitignored raw per-iteration tempfiles
```

---

## Anti-goals (code-level)

Non-negotiable in v1. Each was a deliberate spec-lock decision; revisit only through `V1.1_TBD.md`.

1. **No automatic per-experiment PASS/FAIL.** The agent recommends; the user decides.
2. **No multi-target / multi-output single model.** One binary target per spec.
3. **No multi-library backends.** v1 is CatBoost only. LightGBM / XGBoost behind the experiment-loop contract is `V1.1_TBD.md` § "Per-experiment library override".
4. **No algorithmic FS (RFE / importance-threshold) in v1.** The FS step is agent-driven inside the synced loop.
5. **No HP search libraries (Optuna / Hyperopt) in v1.** Same reasoning. `V1.1_TBD.md` § "Bayesian HP search alternative" is the parking-lot entry.
6. **No alternative-data features beyond the 279-col pool.** Macro / sentiment / sector joins live in `V1.1_TBD.md`.
7. **No PnL / position sizing / transaction costs.** Same anti-rule as `analog_mc`.
8. **No `Classifier` ABC or library registry.** v1 uses one library.
9. **No CSV-first data path.** v1 uses `data_pipelines.fetch()` directly — diverges from `analog_mc` v1's CSV-first contract because gbdt v1 needs a panel, not a single asset's CSV.
10. **No silent look-ahead leaks.** The Stage 1 harness gates every new feature in CI.
11. **No AI attribution** in commits / PRs / outputs / docstrings.

---

## Test strategy

| Layer | What gets tested | Where |
|---|---|---|
| Data loader | Panel build via `data_pipelines.fetch()`, ticker exclusion at <1,600 rows | `tests/gbdt/test_data_loader.py` |
| Leakage harness | Detects leaky features, stays silent on causal ones | `tests/gbdt/test_leakage_harness.py` |
| Features | Hand-computed values per family on tiny fixtures + harness-passing | `tests/gbdt/test_features.py` |
| Targets | Binary correctness on synthetic price paths per direction × threshold × horizon | `tests/gbdt/test_targets.py` |
| Model wrapper | Trivial-fit smoke + HP-bound enforcement + `has_time=True` pinned | `tests/gbdt/test_model.py` |
| Calibration | Spiegelhalter Z fires on miscal data; passes on calibrated; isotonic round-trip | `tests/gbdt/test_calibration.py` |
| FS+HP loop | Inner-stop plateau / degradation / cap; best-checkpoint selection | `tests/gbdt/test_fs_hp_loop.py` |
| Walk-forward | 1-fold mini-run produces correct artifact layout | `tests/gbdt/test_train.py` |
| CLI atom | `python -m gbdt.experiment <tiny_spec>` produces artifact dir | `tests/gbdt/test_experiment_cli.py` |
| End-to-end | Pilot experiment via `/gbdt-experiment` against the NIFTY 50 panel | Stage 9 |

---

## Compute envelope

Per-experiment, single-machine CPU baseline:

- **Data:** ~48 tickers × ~1,600 rows = ~76,800 rows. Panel build via `data_pipelines.fetch()` is cache-served (sub-second after first cold pull).
- **Feature matrix:** 279 columns × ~76,800 rows = ~21M cells. Per-stock rolling pipelines vectorize well; cross-sectional batch is one groupby per date. Expect <1 min on this hardware.
- **Per-iteration CatBoost fit:** at default HPs (`iterations=1000`, `depth=6`, `boosting_type=Ordered`), ~2–5 min on this scale per fit. Ordered boosting is 2–3× slower than Plain; budget accordingly.
- **FS+HP loop:** max 8 iterations × ~2–5 min/iter = 16–40 min of pure fit time per experiment. Plus the agent's per-iteration reasoning wall time (interactive, hard to bound).
- **Calibration + report:** negligible (<10 sec).

The original v1.0 plan's `<2hr` per-experiment target was tight even at the smaller feature pool. At 279 columns the FS+HP loop's iterative pruning is the main mitigation — after iteration 2–3 the agent typically prunes to ~40–80 features, at which point per-iteration fit time drops by 3–5×. Empirically, expect end-to-end pilot runs in the 30 min – 2 hr range. Multi-fold runs (`split.n_folds > 1`) multiply linearly.

If wall-clock becomes the binding constraint per experiment, the first lever is `border_count=128` (small quality cost, ~2× faster training; documented in `CATBOOST_HP_REFERENCE.md` § Category 3); the second is `leaf_estimation_iterations=3`; the third is dropping `boosting_type` to `Plain` for cells where Ordered isn't measurably winning. None of these belong in the loop logic — they're agent-level decisions on the cell.

---

## Branch + PR plan

- This work lives on the `gbdt-v1-spec-lock` branch.
- Each stage in the build (1–9) is a logical commit; the implementation PR opens against `main` once Stage 9's pilot is green.
- **This (spec-lock) PR is doc-only.** It does not contain Stages 1–9 implementation; it locks the spec the implementation will build to.
- Per `[[feedback-branch-retention]]`, branches stay after merge.
- Per `CLAUDE.md`, no AI attribution in commits or PR text.

---

## Dependencies (additions expected)

- Stage 4 adds `catboost` to `pyproject.toml`.
- Stage 5 may add `scikit-learn` if not already present (for `IsotonicRegression` + `CalibratedClassifierCV`).
- Stage 1 already depends on `data_pipelines` (in-repo).
