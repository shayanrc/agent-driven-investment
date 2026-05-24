# gbdt — V1 Implementation Plan

## Build status

- **v1.0:** scaffolded (this branch — `gbdt-v1`). Stages 1–9 below are pending implementation.

For the *why* (what success looks like, anti-goals, deployment intent), see `goal.md`. This document is the *how* — architecture, stages, design decisions to make as work proceeds.

## Revision history

- **v1.0** *(2026-05-24, scaffolded)* — Initial scaffold: directory tree, stub modules, goal doc, plan skeleton. No implementation yet. Branch `gbdt-v1` created from `main` at `9a03a00`. Stages 1–9 specify the build order. Several intentional open questions deferred to specific stages (feature set → Stage 2; library choice → Stage 4; calibration method → Stage 5).

---

## Purpose

This is an implementation specification for a **categorical-outcome forecaster** that predicts the probability of fixed-threshold price-move events within fixed horizons. v1 is one asset (NASDAQ100) with the 18-target lattice defined in `goal.md` § "Target structure". The headline metric is **calibration**, not accuracy or AUC.

The plan is the output of the design conversation reflected in `goal.md`. Decisions documented here were made for a reason. **Do not silently change architectural decisions.** If implementation reveals a problem with a decision, surface it explicitly and ask before deviating.

---

## High-level architecture

Four layers in the design, mirroring `analog_mc`'s separation of concerns:

- **Data layer (`data.py`)** — load `data/NASDAQ100.csv`, return a date-indexed DataFrame with canonical columns. Same CSV-first contract as `analog_mc` per `[[project-data-source]]`.
- **Feature layer (`features.py`)** — pure functions that derive feature columns from OHLCV history. Strictly causal: each feature at row `t` uses only data from rows `< t`. The feature set is fixed per a config spec; v1's spec is finalized in Stage 2.
- **Target layer (`targets.py`)** — for each (direction, threshold, horizon) cell, derive a binary column from the close-price path. Targets at row `t` use forward data and are therefore valid only when `t + horizon` exists; rows without sufficient forward data are dropped from training labels (but kept as inference rows).
- **Model layer (`model.py` + `train.py` + `predict.py`)** — one classifier per (direction, threshold, horizon) target. Walk-forward training; per-fold artifact persistence; calibration as a post-processing step on the held-out scores within each fold's training segment.

```
                       ┌──────────────────────────────────┐
       NASDAQ100.csv ──▶│  data.py: load() → DataFrame    │
                       └─────────────────┬────────────────┘
                                         │
                       ┌─────────────────▼────────────────┐
                       │  features.py: causal feature     │
                       │  matrix X (date × features)      │
                       └─────────────────┬────────────────┘
                                         │
                       ┌─────────────────▼────────────────┐
                       │  targets.py: 18 binary target    │
                       │  columns Y (date × targets)      │
                       └─────────────────┬────────────────┘
                                         │
                       ┌─────────────────▼────────────────┐
                       │  train.py: walk-forward loop —   │
                       │  for each fold:                  │
                       │    for each of 18 targets:       │
                       │      fit GBDT on (X_tr, Y_tr)    │
                       │      calibrate on (X_val, Y_val) │
                       │      predict on (X_test)         │
                       │      persist artifact            │
                       └─────────────────┬────────────────┘
                                         │
                       ┌─────────────────▼────────────────┐
                       │  diagnostics: per-target Brier,  │
                       │  AUC, reliability diagram,       │
                       │  base-rate baseline              │
                       └─────────────────┬────────────────┘
                                         ▼
                       results/gbdt/data/_v1_<id>_data.json
                       docs/gbdt/_v1_acceptance_demo.md
```

---

## Stage breakdown

The build order is strict — each stage ends with a passing test suite and a commit. Don't skip ahead. As with `analog_mc`'s plan, the diagnostic infrastructure (Stage 7) is what makes the model trustworthy, not the choice of algorithm (Stage 4).

### Stage 1 — Data loader + look-ahead-leak harness

**Goal:** a working CSV loader and the synthetic-data harness that detects causal-feature violations.

**Tasks:**
- `src/gbdt/data.py` — `load_csv(path, date_col, close_col, ...) → pd.DataFrame` with date index, canonical column names. Takes column-name args from config (asset-agnostic, per the `analog_mc` precedent).
- `src/gbdt/leakage_harness.py` — synthetic OHLCV generator with a known "leak signal" planted at a future row. Any feature that incorporates the leak will achieve perfect AUC on the synthetic data; any causally-correct feature will achieve chance AUC.
- `tests/gbdt/test_data_loader.py` — loads `data/NASDAQ100.csv`, asserts schema + dtypes + monotonic dates + no duplicate dates.
- `tests/gbdt/test_leakage_harness.py` — confirms the harness fires when given a known leaky function and stays silent on a known causal function.

**Done when:** loader works on the real CSV; harness correctly distinguishes leaky vs causal features on synthetic data.

### Stage 2 — Feature set (the v1 spec)

**Goal:** finalize and implement the v1 feature set. **This stage resolves an open question** (see "Open questions" below); document the resolution in the stage's commit message.

**Tasks:**
- Draft the feature spec inline in this plan doc (an "v1 feature set" subsection below this stage). Default candidate set (subject to revision in this stage):
  - Rolling log-returns over multiple windows (5, 10, 20, 50, 100 days).
  - Realized volatility (rolling std of log-returns) over the same windows.
  - Momentum: ratio of close to rolling mean of close over each window.
  - Drawdown: current drawdown from rolling high over each window.
  - Distance from rolling high / low (as fraction of price).
  - Volume-derived: rolling mean volume, current volume / rolling mean.
- Each feature is a pure function in `src/gbdt/features.py` taking the OHLCV DataFrame and returning a single-column Series aligned on the input index.
- A `build_feature_matrix(df, spec) → pd.DataFrame` orchestrator reads the spec from config and assembles the matrix.
- Every feature has a unit test exercising it on a tiny fixture (10–20 rows) with hand-computed expected values for at least one row.
- Every feature passes the look-ahead-leak harness from Stage 1.
- Document the final feature list in `configs/gbdt/default.yaml`'s `features:` section.

**Done when:** the feature matrix builds end-to-end on NASDAQ100, every feature has a passing unit test, the entire matrix passes the look-ahead-leak test.

### Stage 3 — Targets

**Goal:** the 18 binary target columns, computed correctly for every row with sufficient forward data.

**Tasks:**
- `src/gbdt/targets.py` — `compute_targets(df, target_spec) → pd.DataFrame` where the output has 18 columns named like `up_10_h10`, `down_50_h50`, etc. A target at row `t` is `1` if the threshold was breached at any point in `(t, t+horizon]`, else `0`. Rows with insufficient forward data have `NaN` (and are excluded from training labels in Stage 6 but kept in inference rows).
- The target spec is hard-coded as the 18-cell lattice (per `goal.md`). It is NOT user-configurable in v1.
- Unit tests on a tiny synthetic price path with known breach patterns for each of the 18 targets.
- Edge cases tested: target at the last row of the dataset (no forward data → `NaN`); breach exactly on the horizon-end day (counts as breach).

**Done when:** all 18 columns compute correctly against the synthetic fixtures; the spec is hard-coded; the unit tests exercise both breach and no-breach cases per direction.

### Stage 4 — GBDT model wrapper + library choice

**Goal:** wrap the chosen GBDT library behind a stable interface; settle the library choice. **This stage resolves an open question** — document the resolution in the commit.

**Tasks:**
- Pick the library. Default lean: **lightgbm** (fastest training, mature, scikit-learn-compatible API, robust to default hyperparameters). Alternatives evaluated: xgboost (classic, slower), catboost (good defaults, slower on this data scale), sklearn GradientBoostingClassifier (slowest, no built-in early stopping). Document the choice + the rejected alternatives in the commit message.
- Add the dependency to `pyproject.toml`.
- `src/gbdt/model.py` — `GBDTClassifier` wrapper with a fixed minimal interface: `fit(X_train, y_train, X_val, y_val)`, `predict_proba(X)`, `save(path)`, `load(path)`, plus a `feature_importance()` accessor for diagnostics.
- Hyperparameter defaults pinned in `configs/gbdt/default.yaml` under `model.hyperparameters`; intentionally conservative (e.g., `n_estimators` ceiling with early-stopping patience, modest `learning_rate`, small `num_leaves`).
- Unit tests: fit on a trivial synthetic 2-feature dataset where one feature perfectly predicts the target, assert near-perfect train accuracy and the predictable feature has dominant importance.

**Done when:** library chosen and added; wrapper is import-clean; trivial fit smoke test green.

### Stage 5 — Calibration

**Goal:** post-fit probability calibration per target.

**Tasks:**
- Pick the calibration method. Default lean: **scikit-learn's `IsotonicRegression`** fit on the validation-fold scores per target. Alternative: `CalibratedClassifierCV` (sigmoid / Platt scaling). Document the choice + reasoning.
- `src/gbdt/model.py` — extend the wrapper with `calibrate(X_val, y_val)` that fits a calibration map and stores it alongside the model. `predict_proba(X)` returns post-calibration probabilities by default; an `raw_predict_proba(X)` accessor gives uncalibrated scores for diagnostics.
- Persist the calibration map as part of the saved artifact (same `save(path)` / `load(path)` API).
- Unit tests: fit a deliberately mis-calibrated model on synthetic data; assert that after calibration, the reliability diagram on a held-out chunk is closer to the diagonal than before.

**Done when:** calibration round-trips through save/load; reliability-diagram test confirms post-calibration scores are better-calibrated than pre.

### Stage 6 — Walk-forward training loop

**Goal:** the production training driver — for each fold, fit and calibrate all 18 classifiers; persist all artifacts.

**Tasks:**
- `src/gbdt/train.py` — `train_walk_forward(df, config) → list[FoldResult]`.
- Fold scheme is configurable: train window size, validation window size (carved from the end of the train window), test window size, step. Default scheme finalized in this stage and pinned in config.
- For each fold and each of the 18 targets:
  - Build features on the train+val+test rows from data available up to test-start (causal across the fold boundary, mirroring `analog_mc` C6).
  - Drop rows with `NaN` targets in train+val (insufficient forward data).
  - Fit the GBDT on (X_train, y_train) with early stopping against (X_val, y_val).
  - Calibrate on (X_val, y_val).
  - Score (X_test) — but only on test rows that have valid (non-NaN) y_test, so out-of-sample diagnostics are computable.
  - Persist artifacts under `runs/gbdt/<UTC_timestamp>/fold_<NN>/<target_name>/` containing: model binary, calibration map, fold metadata JSON, prediction CSV, feature schema.
- The driver writes a top-level `runs/gbdt/<UTC>/metadata.json` summarizing the run (config hash, fold count, target count, total wall time).
- A unit test runs a 1-fold, 1-target mini-walk-forward on the synthetic fixture and confirms artifacts land on disk in the expected layout.

**Done when:** walk-forward driver runs end-to-end on a small synthetic dataset; artifact layout matches the spec.

### Stage 7 — Diagnostics + reporting

**Goal:** per-target diagnostics, aggregated across folds, written as machine-readable JSON + human-readable Markdown.

**Tasks:**
- `src/gbdt/diagnostics.py`:
  - `brier_score(y_true, p_pred) → float`
  - `reliability_curve(y_true, p_pred, n_bins=10) → DataFrame` (bin_edge, mean_predicted, fraction_observed, count)
  - `base_rate_brier(y_train, y_test) → float` — Brier from predicting `P = mean(y_train)` on every test sample
  - `roc_auc(y_true, p_pred) → float`
  - `log_loss(y_true, p_pred) → float`
- `scripts/gbdt/aggregate_run.py` — read a `runs/gbdt/<UTC>/` directory, compute per-target metrics aggregated across folds, write `results/gbdt/data/_v1_<run_id>_data.json` with the headline metrics (per `[[project-results-layout]]`).
- `scripts/gbdt/render_report.py` — read the aggregated JSON, render `docs/gbdt/_v1_<run_id>_report.md` with a per-target metrics table and per-target reliability diagrams saved as PNGs in `docs/gbdt/figs/<run_id>/`.
- Unit tests: each diagnostic function on a tiny known case (e.g., perfectly-calibrated random predictions → Brier ≈ p(1-p), reliability curve along diagonal).

**Done when:** an aggregated report can be generated from a completed run; all diagnostic functions have unit tests.

### Stage 8 — CLI entry point

**Goal:** `python -m gbdt train` and `python -m gbdt predict` as the run-it surface.

**Tasks:**
- `src/gbdt/__main__.py` with two subcommands:
  - `train --config <path> --output-dir <path>` — runs the walk-forward driver, aggregates, renders the report.
  - `predict --run-dir <path> --features-csv <path>` — loads the latest fold's models, scores the rows in the features CSV, prints / writes the 18-column probability output.
- Each subcommand has a smoke test that invokes the CLI against the synthetic fixture and confirms expected stdout / output files.

**Done when:** both CLI subcommands work end-to-end against the synthetic fixture.

### Stage 9 — Acceptance demo on NASDAQ100 (PR merge gate)

**Goal:** the concrete bar from `goal.md` § "v1 end-to-end acceptance demo" — full walk-forward run on real NASDAQ100, per-target acceptance check, written report. **This stage gates the PR merge.**

**Tasks:**
- Run `python -m gbdt train --config configs/gbdt/default.yaml --output-dir runs/gbdt/nasdaq100_v1` against the full NASDAQ100 history.
- Aggregate via `scripts/gbdt/aggregate_run.py`; render via `scripts/gbdt/render_report.py`.
- Verify the 5 acceptance criteria from `goal.md`:
  1. All 18 classifiers trained without error.
  2. Brier score < base-rate baseline on ≥14/18 targets.
  3. Calibration within ±5pp on ≥14/18 targets across [0.05, 0.95].
  4. ROC-AUC ≥ 0.55 on ≥14/18 targets.
  5. Zero leaks per the look-ahead-leak harness across the full feature matrix.
- Write `docs/gbdt/_v1_acceptance_demo.md` with the per-target table, reliability diagrams, and pass/fail verdict per target. If any target fails its bar, the report's post-mortem section identifies whether the issue is rarity, feature inadequacy, or implementation bug — and the v1 ship/no-ship decision is made on that analysis.
- Commit the aggregated JSON (`results/gbdt/data/`) and the report. Per the standard pattern, raw per-fold `runs/gbdt/<UTC>/` stays gitignored.

**Done when:** acceptance demo passes its criteria (or its failures are characterized and ship-decision-able), report is written, results JSON committed.

---

## Open questions (to resolve during implementation)

These are intentional unknowns that should be settled during the stage indicated, with the resolution documented in the stage's commit message. They are NOT blocking for scaffolding.

1. **Feature set (Stage 2).** The default candidate list above is a starting point. Final set + lookback windows decided when implementing Stage 2 against the synthetic harness and the real data. Constraint: must be small enough that the 18-classifier walk-forward run fits in a single-session compute budget (target: <2 hours wall on this hardware).

2. **GBDT library (Stage 4).** Default lean: lightgbm. Decision documented in Stage 4 commit with rejected alternatives. If lightgbm has packaging / install issues on this Python version, fallback is xgboost.

3. **Calibration method (Stage 5).** Default lean: scikit-learn `IsotonicRegression`. Switch to Platt scaling only if isotonic over-fits on small validation folds.

4. **Walk-forward fold scheme (Stage 6).** Inherit from `analog_mc`'s walk-forward conventions if they apply directly; otherwise pin a v1 scheme in `configs/gbdt/default.yaml` with reasoning in the Stage 6 commit. Constraint: enough fold count for per-target Brier-score stability on the rarer targets (`±50% in 10 days` is rare; needs enough test instances to compute a meaningful Brier).

5. **Multi-target correlation handling (deferred to v2).** v1 trains 18 independent classifiers. If diagnostics reveal that targets are highly correlated AND the independent classifiers have inconsistent calibration on the same underlying event, v2 considers a shared-tree multi-output architecture. Don't pre-engineer for this in v1.

---

## File layout

```
docs/gbdt/
  goal.md                              # done (this PR)
  V1_PLAN.md                           # done (this PR)
  _v1_acceptance_demo.md               # written in Stage 9
  figs/<run_id>/                       # reliability diagrams per run

src/gbdt/
  __init__.py
  __main__.py                          # CLI (Stage 8)
  data.py                              # loader (Stage 1)
  leakage_harness.py                   # synthetic-data leak detector (Stage 1)
  features.py                          # feature functions (Stage 2)
  targets.py                           # binary targets (Stage 3)
  model.py                             # GBDT wrapper + calibration (Stages 4-5)
  train.py                             # walk-forward driver (Stage 6)
  predict.py                           # inference (Stages 6 + 8)
  diagnostics.py                       # Brier / AUC / reliability (Stage 7)

scripts/gbdt/
  aggregate_run.py                     # runs/ → results/ aggregation (Stage 7)
  render_report.py                     # results/ → docs/ report (Stage 7)

configs/gbdt/
  default.yaml                         # feature spec + model hyperparams + fold scheme

tests/gbdt/
  __init__.py
  test_data_loader.py                  # Stage 1
  test_leakage_harness.py              # Stage 1
  test_features.py                     # Stage 2
  test_targets.py                      # Stage 3
  test_model.py                        # Stage 4-5
  test_train.py                        # Stage 6
  test_diagnostics.py                  # Stage 7
  test_cli.py                          # Stage 8

runs/gbdt/<UTC>/                       # gitignored raw per-fold artifacts (Stage 6)
results/gbdt/data/                     # checked-in headline metrics (Stage 7)
```

---

## Anti-goals (code-level)

These are non-negotiable in v1. Each was a deliberate design choice; revisit only with an explicit deviation discussion.

1. **No multi-target / multi-output single model.** 18 independent classifiers. Gated on v1 diagnostic evidence.
2. **No user-configurable target lattice.** The 18-cell lattice is hard-coded.
3. **No multi-asset training.** One asset (NASDAQ100). Cross-sectional is v2.
4. **No HPO framework.** A small fixed hyperparameter grid + early stopping handles tuning.
5. **No alternative-data features.** OHLCV-derived only in v1.
6. **No PnL / position sizing / transaction costs.** Same anti-rule as `analog_mc`.
7. **No `Classifier` ABC or multi-library registry.** v1 uses one library.
8. **No data-source dispatch.** CSV-first per `[[project-data-source]]`. Wiring to `data_pipelines` is a separate plan.
9. **No silent look-ahead leaks.** The Stage 1 harness gates every new feature in CI.
10. **No AI attribution** in commits / PRs / outputs / docstrings.

---

## Test strategy

| Layer | What gets tested | Where |
|---|---|---|
| Data loader | CSV → DataFrame schema, dtypes, monotonic dates | `tests/gbdt/test_data_loader.py` |
| Leakage harness | Detects leaky features, stays silent on causal ones | `tests/gbdt/test_leakage_harness.py` |
| Features | Hand-computed values on tiny fixtures + harness-passing | `tests/gbdt/test_features.py` |
| Targets | Binary correctness on synthetic price paths for all 18 cells | `tests/gbdt/test_targets.py` |
| Model wrapper | Trivial-fit smoke + calibration round-trip + save/load | `tests/gbdt/test_model.py` |
| Walk-forward | 1-fold-1-target mini-run produces correct artifact layout | `tests/gbdt/test_train.py` |
| Diagnostics | Known-value cases for each metric | `tests/gbdt/test_diagnostics.py` |
| CLI | Both subcommands run on synthetic fixture | `tests/gbdt/test_cli.py` |
| End-to-end | NASDAQ100 acceptance demo (manual, pre-merge) | Stage 9 |

---

## Branch + PR plan

- This work lives on the `gbdt-v1` branch.
- Each stage is a logical commit; the PR opens against `main` once Stage 9's acceptance demo is green.
- **Stage 9 is the PR merge gate.** Stages 1–8 prove the units; Stage 9 proves the system.
- Per `[[feedback-branch-retention]]`, the branch stays after merge.
- Per `CLAUDE.md`, no AI attribution in commits or PR text.

---

## Dependencies (additions expected)

Stage 4 adds the GBDT library (default: `lightgbm`). Stage 5 may add `scikit-learn` if not already present (it's not currently in `pyproject.toml`). These additions go to `pyproject.toml` in the stage that introduces them, with the version constraint pinned to match what's available on Python ≥3.12.
