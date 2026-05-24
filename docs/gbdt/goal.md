# gbdt — Goal

**This is the categorical-outcome forecasting module.** Given a price time series, it predicts the **probability that the price will move by ±10%, ±20%, or ±50% within the next 10, 20, or 50 trading periods** — six directions × three horizons = **18 probability outputs** per asset, per origin. The model is a gradient-boosted decision tree classifier, trained per (direction, horizon) target, with calibrated probabilities.

This is a **sibling module to `analog_mc`**, not a successor or extension. `analog_mc` produces continuous price-path samples (a distributional forecast over price levels); `gbdt` produces categorical event probabilities (the chance a specific threshold will be breached in a fixed window). The two answer different questions and use different downstream evaluation metrics; both will eventually be consumable through the `forecasters` skill surface, but that requires extending the wire-format contract to express categorical outputs — which is deferred until after `gbdt v1` ships.

This document states what `gbdt` is optimizing for and what trade-offs are unacceptable. Read it before editing any file under `src/gbdt/`, `tests/gbdt/`, `configs/gbdt/`, `dashboards/gbdt/`, or `docs/gbdt/`.

For *how* it works (architecture, stages, library choice, validation), see `V1_PLAN.md`. This file is the *why* and *what success looks like*.

---

## What this module is optimizing for

Produce **well-calibrated probabilities** for a small fixed set of practically-meaningful price-move events on a single asset's history, with one defining rule:

> **A probability of 0.30 from this model should mean the event happens 30% of the time on out-of-sample data.** Calibration on held-out folds is the headline metric, not raw accuracy or AUC. A high-AUC but poorly-calibrated model is a v1 failure; a well-calibrated model with modest AUC is a v1 success.

The module exists because:
- `analog_mc`'s distributional forecast answers "what is the *shape* of the next 60 days of price moves?" but not directly "what's the chance of a ≥20% drop in the next 50 days?" — the latter is a marginal probability over a specific event that's expensive to extract reliably from path samples.
- Many downstream questions (risk-overlay sizing, alerting, decision rules) take categorical event probabilities as input, not price-path distributions.
- A GBDT classifier is the natural fit: it consumes engineered features cheaply, handles non-linear interactions, calibrates well with isotonic post-processing, and trains in minutes per target on this data scale — making walk-forward validation tractable.

## Target structure

For each origin date `t` and asset, the model outputs **18 probabilities**:

| Direction | Threshold | Horizon (trading days) |
|---|---|---|
| Up | +10%, +20%, +50% | 10, 20, 50 |
| Down | −10%, −20%, −50% | 10, 20, 50 |

A target `(direction=up, threshold=+10%, horizon=20)` evaluates as: did the asset's close ever rise by ≥10% above `close[t]` at any point in `(t, t+20]`? Yes/no — a binary label. The model output is `P(yes | features observed at t)`.

**One classifier per (direction, threshold, horizon) target** — 18 classifiers total per asset. This is deliberately not a multi-task or multi-output single-model approach in v1: independent classifiers give clean per-target calibration curves and per-target diagnostics, at the cost of redundant feature computation. A shared-feature-tree, multi-target head is a v2 consideration once the v1 diagnostics tell us whether targets are correlated enough to benefit.

The thresholds and horizons are **fixed in v1** — not user-configurable. They're a chosen lattice of practically-meaningful moves over short, medium, and long windows. If a user wants ±5% / ±25% / ±75% at 30 days, that's a v2 generalization (parameterizable target spec) gated on whether v1's fixed lattice proves insufficient.

## What success looks like

Per-target diagnostics on held-out walk-forward folds for the v1 asset (NASDAQ100):

- **Calibration is the headline.** Each target's reliability diagram on out-of-sample predictions should track the diagonal within ±5pp across the [0.05, 0.95] probability range. The headline metric is **Brier score** (lower is better), benchmarked against the empirical-base-rate baseline (predicting `P = mean(target on training fold)` for every test sample).
- **Discrimination is the secondary metric.** ROC-AUC ≥ 0.55 per target — i.e., the model is meaningfully better than chance at ranking. (AUC alone isn't enough; calibration gates the ship.)
- **Walk-forward determinism.** Same data + same features + same hyperparameters + same seed → bit-identical fold predictions.
- **Causal features only.** Every feature at origin `t` is computed from data strictly before `t` (zero look-ahead). This is the same constraint as `analog_mc` C1; violations are silent failures that look like spectacular AUC. Enforced by tests on synthetic data.
- **Per-target reproducibility.** A fit per (direction, threshold, horizon) is a self-contained artifact on disk (trained model + feature schema + fold split + held-out predictions + calibration curve). The artifact is the unit that goes into reports and that downstream code consumes.
- **No transaction costs, no PnL, no position sizing.** Same anti-rule as `analog_mc` per `[[project-overview]]`. The module produces probabilities; what downstream does with them is downstream's concern.

## v1 end-to-end acceptance demo (the north-star)

The single concrete bar that gates the v1 PR merge: **walk-forward-train all 18 classifiers on NASDAQ100 history, generate held-out predictions for the most recent 60-trading-day window, and produce a reliability + Brier-score report per target.**

**The flow:**

1. Load `data/NASDAQ100.csv` (the same CSV `analog_mc` consumes, same column convention).
2. Compute the v1 feature set (TBD in `V1_PLAN.md` Stage 2 — likely: rolling returns, realized volatility, momentum, drawdown, distance-from-rolling-high, on multiple lookback windows).
3. Compute the 18 binary target columns for each row that has sufficient forward data.
4. Walk-forward train (per-target, per-fold) with a fixed window scheme matching `analog_mc`'s convention.
5. Score the held-out folds: per-target Brier score, calibration curve, ROC-AUC, base-rate-baseline Brier.
6. Write `docs/gbdt/_v1_acceptance_demo.md` with: per-target metric table, side-by-side reliability diagrams, and a pass/fail verdict per target against the acceptance criteria below.

**Acceptance criteria:**

- All 18 classifiers train without error to completion on the walk-forward folds.
- Brier score < base-rate-baseline Brier on at least 14/18 targets (with ties counting as misses — strict improvement required).
- Calibration curves track the diagonal within ±5pp across [0.05, 0.95] on at least 14/18 targets.
- ROC-AUC ≥ 0.55 on at least 14/18 targets.
- Zero causal-feature violations detected by the look-ahead-leak test on synthetic data.

The "14/18" floor (≈78%) leaves room for the most extreme targets (`±50% in 10 days`, very rare events) to be uninformative — that's an expected v1 outcome to *measure*, not a failure mode to engineer around. If fewer than 14/18 pass, the post-mortem in the report identifies whether the issue is target rarity, feature inadequacy, or implementation bug, and the v1 ship decision is made on that basis.

## What this module is *not*

- **Not a price-path forecaster.** That's `analog_mc`. Don't add path-sampling methods here.
- **Not a regressor.** Output is `P(binary event)`, not an expected price level or return magnitude.
- **Not a backtester / strategy.** No PnL, no position sizing, no transaction costs (per `[[project-overview]]` anti-rule, same as `analog_mc`).
- **Not a multi-asset universe model in v1.** One asset (NASDAQ100), one model set per asset. Cross-sectional features / pooled training across assets is a v2 generalization gated on v1 calibration outcomes.
- **Not wired into `forecasters` in v1.** The wire-format contract `forecasters` is building around expects continuous path arrays; `gbdt` produces a different output shape. Integrating `gbdt` as a `forecasters` backend requires extending the contract (e.g., a `categorical_predictions` field alongside `paths`) — deferred to a separate plan once `gbdt v1` and `forecasters v1` are both shipped.
- **Not a feature-engineering library.** Features are an *input* to this module. The v1 feature set is chosen to be small, defensible, and computable from the same OHLCV columns `analog_mc` consumes. Aggressive feature engineering (sector / macro / sentiment / alternative-data joins) is out of scope.
- **Not a hyperparameter-search framework.** v1 uses a small fixed hyperparameter grid per target (early-stopping handles most of the tuning); a heavier-weight HPO layer is a v2 concern.
- **Not a model-selection framework across algorithm families.** v1 uses GBDT. Comparing GBDT vs random forest vs neural net is a v2 concern.

## Scope for v1

v1 ships **one module + one asset + 18 calibrated classifiers + the acceptance-demo report**.

**Module code (`src/gbdt/`):**
- `data.py` — CSV loader (same CSV-first contract as `analog_mc` per `[[project-data-source]]`).
- `features.py` — the fixed v1 feature set (rolling returns, realized vol, momentum, drawdown, etc. — final list in `V1_PLAN.md` Stage 2).
- `targets.py` — binary target construction for each (direction, threshold, horizon) cell.
- `model.py` — GBDT classifier wrapper with calibration. Library choice resolved in `V1_PLAN.md` Stage 4 (default lean: lightgbm + isotonic post-processing via sklearn).
- `train.py` — walk-forward training loop, one fit per (target, fold).
- `predict.py` — load a saved model, score new feature rows.
- Config-driven: a single YAML at `configs/gbdt/default.yaml` declares feature spec, target spec, walk-forward scheme, model hyperparameters, calibration method.

**Diagnostics:**
- Per-target reliability diagrams.
- Per-target Brier / AUC / log-loss vs base-rate baseline.
- Per-fold prediction CSV for downstream inspection.
- A causal-feature look-ahead-leak test on synthetic data (gates every PR).

**Asset scope:** NASDAQ100 (the existing CSV under `data/`). Other assets and the eventual `data_pipelines.fetch()` wiring are v1.1+ work.

**Explicitly deferred to later versions:**
- Multi-asset training / cross-sectional features.
- Parameterizable target lattice (user-configurable thresholds and horizons).
- Multi-task / multi-output single-model architectures.
- Hyperparameter optimization beyond the v1 fixed grid.
- Wiring into the `forecasters` skill surface (requires contract extension).
- Streamlit dashboard for inspection (the existing `dashboards/analog_mc/` pattern is the template if/when it's wanted).
- Comparing GBDT vs other algorithm families.
- Alternative-data features (sector, macro, sentiment).
- Transaction costs, position sizing, PnL — same anti-rule as `analog_mc`.

## Eventual deployment shape

Currently the planned surface is: a Python module + a CLI entry point (`python -m gbdt train`, `python -m gbdt predict`) + a results layout under `results/gbdt/data/`. These reflect the **build-and-validate phase**.

Intended end state: **agent-callable inference** — given a saved model artifact and a feature row (or DataFrame), return calibrated probabilities per target. The natural surface is either a `forecasters` backend (after the contract extension) or a standalone `/gbdt-predict` skill if the contract extension is delayed.

When designing new APIs in this module, prefer shapes that wrap cleanly later:
- Clean function signatures, all args explicit, no positional ambiguity.
- JSON-serializable return shapes: dicts of `(target_name, probability)` rather than tuples.
- Errors raise typed exceptions with target + fold context, not bare strings.
- Models persist as self-contained artifacts (model binary + feature schema + calibration map + fit metadata in one directory).

## How to apply this when working on the module

- **Calibration before accuracy.** When tempted to add a feature, ask: does it improve calibration on the targets it's relevant to, or just AUC? Calibration is the headline. AUC without calibration is a v1 failure.
- **Causal features only.** Same C1-equivalent discipline as `analog_mc`: every feature at origin `t` uses only data from before `t`. Adding a feature that uses `t+1` data is a silent failure mode — the synthetic look-ahead-leak test must remain green after every change.
- **One artifact per (target, fold).** The unit of saved work is the per-target-per-fold artifact. Don't share state across targets at training time; don't average predictions across targets at inference time. Each target stands alone.
- **No transaction costs, PnL, or position sizing.** This module produces probabilities. Downstream concerns belong downstream.
- **Don't crystallize a `Classifier` ABC or multi-algorithm-registry.** v1 is one library (the GBDT choice resolved in `V1_PLAN.md`). Adding a second algorithm family is gated on v1 results — same "wait for use-case #2" discipline as `data_pipelines` and `forecasters`.
- **Don't add data sources beyond the CSV-first contract.** The same constraint that holds `analog_mc` to local CSVs holds here in v1. Wiring to `data_pipelines.fetch()` is a separate plan (would unify with `analog_mc`'s wiring effort).
- **No AI attribution in commits / PRs / outputs** (per the project-wide rule in `CLAUDE.md`).

## Implementation discipline

The how-it-works specification is `docs/gbdt/V1_PLAN.md`. The stages defined there are the build order. **Do not silently change architectural decisions** — surface the deviation and ask first. The library choice, the feature set, the target lattice, and the walk-forward scheme are decisions to be made in stages 1–4 of the plan, not silently shifted later.

See also:
- `[[project-overview]]` — multi-module structure.
- `[[project-data-source]]` — CSV-first contract for v1.
- `[[project-results-layout]]` — where machine-readable headline metrics live (`results/gbdt/data/`).
- `analog_mc`'s `goal.md` — sibling module's why; useful for understanding the parallel design discipline.
