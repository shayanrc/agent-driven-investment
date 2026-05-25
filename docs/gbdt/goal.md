# gbdt — Goal

**This is the categorical-outcome forecasting module.** Given a price time series, it predicts the **probability that a defined price-move event occurs within a defined horizon** — e.g. "P(close rises ≥10% above today within the next 20 trading days)". The model is a gradient-boosted decision tree classifier, trained per (direction, threshold, horizon) target, with calibrated probabilities.

This is a **sibling module to `analog_mc`**, not a successor or extension. `analog_mc` produces continuous price-path samples (a distributional forecast over price levels); `gbdt` produces categorical event probabilities (the chance a specific threshold will be breached in a fixed window). The two answer different questions and use different downstream evaluation metrics; both will eventually be consumable through the `forecasters` skill surface, but that requires extending the wire-format contract to express categorical outputs — which is deferred until after `gbdt v1` ships.

This document states what `gbdt` is optimizing for and what trade-offs are unacceptable. Read it before editing any file under `src/gbdt/`, `tests/gbdt/`, `configs/gbdt/`, `dashboards/gbdt/`, or `docs/gbdt/`.

For *how* it works (architecture, stages, library choice, validation), see `V1_PLAN.md`. The YAML spec contract is in `EXPERIMENT_SPEC.md`. The CatBoost hyperparameter reference the per-experiment agent consults is `CATBOOST_HP_REFERENCE.md`. This file is the *why* and *what success looks like*.

---

## What v1 actually ships (re-framed)

v0 (see `V0_INVESTIGATION_PLAN.md`, already shipped on main) explored an **18-cell lattice** — 2 directions × 3 thresholds × 3 horizons across NIFTY 50 — to characterize event base rates and discover which cells are even plausibly predictable. The lattice was the **exploration grid**, not a production deliverable. It informed v1's scope but is not itself what v1 builds.

**v1 ships experiment-loop infrastructure**, not a fixed lattice of models. The unit of work is a single **experiment**: one tuple of `(universe, direction, threshold, horizon)`. An agent receives a spec, builds the labeled dataset, trains a CatBoost classifier with synced feature-selection + HP iteration, calibrates, and emits a self-contained artifact. Each experiment is a separate invocation; multi-cell sweeps are multiple invocations against multiple spec files.

The shift in framing matters. The lattice framing implied "train all 18, ship if 14/18 pass acceptance bars." The experiment-loop framing says "ship the surface that lets us run *any* cell on demand, with the per-experiment quality bar (calibration) being for the user to judge from the agent's report." Two consequences:

- **No automatic PASS/FAIL on individual experiments.** The agent writes a report; the user reads it and decides whether the cell is shippable downstream.
- **The agent is the iteration loop.** Feature selection and hyperparameter tuning happen inside one synchronized loop, with the agent reading a per-iteration diagnostic bundle and deciding both the prune list and the HP changes for the next iteration. The agent's chain of reasoning is part of the artifact.

## What this module is optimizing for

Produce **well-calibrated probabilities** for an arbitrary price-move event on an arbitrary universe, with one defining rule:

> **A probability of 0.30 from this model should mean the event happens 30% of the time on out-of-sample data.** Calibration on the held-out eval segment is the headline metric per experiment, not raw accuracy or AUC. A high-AUC but poorly-calibrated model is a v1 failure; a well-calibrated model with modest AUC is a v1 success.

The module exists because:
- `analog_mc`'s distributional forecast answers "what is the *shape* of the next 60 days of price moves?" but not directly "what's the chance of a ≥20% drop in the next 50 days?" — the latter is a marginal probability over a specific event that's expensive to extract reliably from path samples.
- Many downstream questions (risk-overlay sizing, alerting, decision rules) take categorical event probabilities as input, not price-path distributions.
- A GBDT classifier is the natural fit: it consumes engineered features cheaply, handles non-linear interactions, calibrates well with isotonic post-processing, and trains in minutes per target on this data scale — making walk-forward validation tractable.

## What an experiment is

A v1 experiment is a single YAML spec at `configs/gbdt/experiments/<name>.yaml` whose `target` block fixes one tuple:

```yaml
target:
  universe: nifty50
  direction: up        # up | down
  threshold_pct: 10    # 5, 10, 20, 30, 50, ...
  horizon_days: 20     # any positive integer
```

The full schema is in `EXPERIMENT_SPEC.md`. The agent invocation that runs the experiment is the `/gbdt-experiment` skill, which orchestrates: data build → iteration 0 (all features, default HPs) → FS+HP loop → calibration → artifact emission. Each experiment produces one artifact directory under `results/gbdt/experiments/<experiment_name>/` containing the model, calibration map, feature schema, HP history, per-iteration diagnostic bundles, metrics JSON, predictions, figures, and a human-readable report.

Targets train **pooled across the universe** (one classifier sees all 48 NIFTY 50 stocks' rows). Predictions are emitted per stock. The cross-sectional features (F14: rank/z-score across the universe at each point in time) are what make pooled training preferable to per-stock training — the model learns "this stock is unusually weak today vs the cohort" rather than just "this stock's recent returns are X."

## What success looks like per experiment

Per-experiment diagnostics on the held-out eval segment:

- **Calibration is the headline.** The reliability diagram on out-of-sample predictions should track the diagonal across the [0.05, 0.95] probability range. The headline metric is **Brier score** (lower is better), benchmarked against the empirical-base-rate baseline. The calibration test is the Spiegelhalter Z-test on the val segment, which gates whether the artifact ships raw CatBoost outputs or layers an isotonic regression on top.
- **Discrimination is the secondary metric.** ROC-AUC reported but not gated. AUC without calibration is a v1 failure.
- **Walk-forward determinism.** Same spec + same data + same seed → bit-identical predictions. `random_seed` is pinned; `has_time=True` is mandatory; see `CATBOOST_HP_REFERENCE.md` § "Determinism".
- **Causal features only.** Every feature at origin `t` is computed from data strictly before `t` (zero look-ahead). Same constraint as `analog_mc` C1; violations are silent failures that look like spectacular AUC. Enforced by tests on synthetic data via the Stage 1 leakage harness.
- **Per-experiment reproducibility.** The artifact dir is self-contained: spec + model + calibration map + features used + final HP set + iteration history + metrics + predictions + figures + report. Re-running the spec against the same data + seed reproduces the same artifact.
- **No transaction costs, no PnL, no position sizing.** Same anti-rule as `analog_mc`. The module produces probabilities; what downstream does with them is downstream's concern.

## v1 PR merge gate

The single concrete bar that gates the v1 PR merge:

> **The `/gbdt-experiment` skill runs the pilot experiment end-to-end and produces a complete artifact directory.**
>
> Spec: `configs/gbdt/experiments/nifty50_up_10pct_20d_pilot.yaml`
> Artifact: `results/gbdt/experiments/nifty50_up_10pct_20d_pilot/`

The artifact must contain every file enumerated in `EXPERIMENT_SPEC.md` § "Artifact directory layout" — `spec.yaml`, `model.cbm`, `calibration.pkl`, `features.yaml`, `hp.yaml`, `report.md`, `metrics.json`, `iterations.jsonl`, `figs/`, `predictions/` — and the report must narrate the FS+HP iteration history with per-iteration rationale and conclude with a per-experiment verdict (the agent's recommendation; the user reads it and decides).

**The PASS/FAIL verdict on the experiment itself is for the user to read from the report**, not for an automated gate. A miscalibrated pilot is *information*, not a merge blocker — it tells us either the cell is genuinely hard or the loop needs another iteration. What gates the merge is that the infrastructure ran end-to-end, the artifact is complete, and the report is readable.

The pilot cell (`up / +10% / 20 trading days` on NIFTY 50) was chosen because v0.2 / v0.3 measured its base rate at roughly 35–45% across the universe — common enough that calibration is testable, not so extreme that the class-imbalance machinery dominates the run.

## What this module is *not*

- **Not a price-path forecaster.** That's `analog_mc`. Don't add path-sampling methods here.
- **Not a regressor.** Output is `P(binary event)`, not an expected price level or return magnitude.
- **Not a backtester / strategy.** No PnL, no position sizing, no transaction costs (per `[[project-overview]]` anti-rule, same as `analog_mc`).
- **Not a fixed-lattice ship.** The 18-cell lattice was v0 EDA. v1 ships the experiment surface; which cells get run, when, in what order, is per-invocation.
- **Not a per-stock-only model in v1.** v1 pools training across the NIFTY 50 universe and emits per-stock predictions. Cross-sectional features (rank/z-score across the universe) are first-class.
- **Not wired into `forecasters` in v1.** The wire-format contract `forecasters` is building around expects continuous path arrays; `gbdt` produces a different output shape. Integrating `gbdt` as a `forecasters` backend requires extending the contract (e.g., a `categorical_predictions` field alongside `paths`) — deferred to a separate plan once `gbdt v1` and `forecasters v1` are both shipped.
- **Not a feature-engineering library.** Features are an *input* to this module. v1 ships a fixed candidate pool of 273 columns across 16 families (see `V1_PLAN.md` Stage 2 and `EXPERIMENT_SPEC.md` § "Feature pool"). Adding families beyond that pool is a v2 decision; pruning the pool per-experiment is the agent's job inside the FS+HP loop.
- **Not an HPO framework.** v1 uses an agent-driven synced FS+HP loop bounded by the ranges in `CATBOOST_HP_REFERENCE.md` and capped at 8 iterations. A Bayesian-optimization library (e.g. Optuna) is a v1.1 alternative if the agent loop proves inconsistent — see `V1.1_TBD.md`.
- **Not a model-selection framework across algorithm families.** v1 uses CatBoost. The library was chosen for ordered boosting (helps small-data, rare-cell training) and native calibration quality. Comparing CatBoost vs LightGBM vs XGBoost is v1.1+ work behind the same experiment-loop contract.

## Scope for v1

v1 ships **the experiment-loop infrastructure + one pilot experiment + one universe preset + one library backend**.

**Module code (`src/gbdt/`):**
- `data.py` — loads the universe panel via `data_pipelines.fetch()` per the preset's ticker list (NIFTY 50 v1).
- `features.py` — the 273-column candidate pool (16 families). Per-stock rolling pipeline + cross-sectional point-in-time pipeline.
- `targets.py` — binary target construction from a single `(direction, threshold_pct, horizon_days)` tuple.
- `model.py` — thin CatBoost wrapper. `has_time=True` pinned; HPs bounded by `CATBOOST_HP_REFERENCE.md`.
- `calibration.py` — conditional isotonic with Spiegelhalter Z-test on val; per-experiment decision recorded in artifact.
- `fs_hp_loop.py` — diagnostic bundle generator that emits the inputs the agent reads each iteration. Inner-stop = plateau + degradation gate; hard cap 8 iterations; best-Brier checkpoint = final artifact.
- `train.py` — walk-forward driver (800 train + 400 val + 200 eval + 100 test = 1,600 rows per stock).
- `experiment.py` — top-level orchestrator. CLI atom: `python -m gbdt.experiment <spec.yaml>`.
- `report.py` — renders `report.md` from `iterations.jsonl` + `metrics.json` + `figs/`.
- Config-driven via a YAML spec at `configs/gbdt/experiments/<name>.yaml`; defaults live in `configs/gbdt/default.yaml`.

**Skill surface:**
- `/gbdt-experiment <spec_path>` at `.claude/skills/gbdt-experiment/SKILL.md` — single end-to-end orchestrator. No separate `/gbdt-prepare-data` or `/gbdt-render-report` skills; the orchestrator handles everything inside one invocation.

**Diagnostics (per experiment):**
- Reliability diagram, calibration curve.
- Brier score / log-loss / AUC vs base-rate baseline.
- Per-iteration train-vs-val gap, learning curve, feature importance ranking, correlation matrix.
- Per-iteration HP history and prune/keep decisions with the agent's rationale.

**Universe scope:** `nifty50` is the only pre-registered preset in v1; other NSE universes (`nifty100`, `nifty_midcap_150`, `nifty500`, and any other valid NSE index, plus inline ad-hoc baskets) are resolvable on first use via the `/gbdt-experiment` skill's universe self-service flow. NDX (US universe, different adapter chain), macro features, and other extensions are in `V1.1_TBD.md`.

**Explicitly deferred to v1.1+:**
- NDX universe preset (US adapter chain, different calendar / annualization).
- Multi-target shared-tree head (deferred from V1_PLAN's original open question #5).
- Macro features (USD/INR, 10Y yield, oil).
- Lagged-target features (risky; gated on the Stage 1 leakage harness proving robust on the simpler feature set first).
- Per-experiment library override (LightGBM / XGBoost behind the same experiment-loop contract).
- Multi-spec sweep runner.
- Bayesian-HP-search alternative (Optuna behind the same diagnostic-bundle contract).
- Wiring into the `forecasters` skill surface (requires categorical contract extension).
- Streamlit dashboard for inspection.
- Transaction costs, position sizing, PnL — same anti-rule as `analog_mc`.

## Eventual deployment shape

Currently the planned surface is: a Python module + a CLI atom (`python -m gbdt.experiment <spec.yaml>`) + an agent skill (`/gbdt-experiment <spec_path>`) + a per-experiment artifact layout under `results/gbdt/experiments/<name>/`. These reflect the **build-and-validate phase**.

Intended end state: **agent-callable inference** — given a saved experiment artifact and a feature row (or DataFrame), return calibrated probabilities. The natural surface is either a `forecasters` backend (after the wire-format extension) or a standalone `/gbdt-predict` skill if the contract extension is delayed.

When designing new APIs in this module, prefer shapes that wrap cleanly later:
- Clean function signatures, all args explicit, no positional ambiguity.
- JSON-serializable return shapes: dicts of `(stock_id, probability)` rather than tuples.
- Errors raise typed exceptions with experiment + iteration context, not bare strings.
- Experiments persist as self-contained artifact directories (one directory = everything needed to reproduce + audit + consume).

## How to apply this when working on the module

- **Calibration before accuracy.** When tempted to add a feature, ask: does it improve calibration on the experiments it's relevant to, or just AUC? Calibration is the headline. AUC without calibration is a v1 failure.
- **Causal features only.** Same C1-equivalent discipline as `analog_mc`: every feature at origin `t` uses only data from before `t`. Adding a feature that uses `t+1` data is a silent failure mode — the synthetic look-ahead-leak test must remain green after every change.
- **One artifact per experiment.** The unit of saved work is the per-experiment artifact directory. Don't share state across experiments at training time; don't average predictions across experiments at inference time. Each experiment stands alone.
- **The agent owns the iteration loop.** Feature selection and HP tuning are not separate algorithmic phases — they're one synced loop driven by an agent reading per-iteration diagnostic bundles. Don't replace the agent loop with an algorithmic FS pass (e.g. RFE) and an algorithmic HP search (e.g. random/grid) without going through `V1.1_TBD.md`'s Bayesian-alternative entry first.
- **No transaction costs, PnL, or position sizing.** This module produces probabilities. Downstream concerns belong downstream.
- **Don't crystallize a `Classifier` ABC or multi-algorithm-registry.** v1 is one library (CatBoost). Adding a second library is the v1.1 per-experiment-library-override entry — gated on a real second-library experiment.
- **Don't add data sources beyond what `data_pipelines.fetch()` provides.** Macro / sentiment / alternative data are in `V1.1_TBD.md`. Adding them requires new `data_pipelines` adapters.
- **No AI attribution in commits / PRs / outputs** (per the project-wide rule in `CLAUDE.md`).

## Implementation discipline

The how-it-works specification is `docs/gbdt/V1_PLAN.md`. The YAML schema is `docs/gbdt/EXPERIMENT_SPEC.md`. The HP reference is `docs/gbdt/CATBOOST_HP_REFERENCE.md`. Parked v1.1 work is in `docs/gbdt/V1.1_TBD.md`. v0 background is in `docs/gbdt/V0_INVESTIGATION_PLAN.md`.

The stages in `V1_PLAN.md` are the build order. **Do not silently change architectural decisions** — surface the deviation and ask first. The library choice (CatBoost), the feature pool (273 columns × 16 families), the split scheme (800+400+200+100), the calibration policy (conditional isotonic gated by Spiegelhalter Z), and the loop budget (8 iter, plateau + degradation inner-stop) are all locked decisions whose rationale is documented in `V1_PLAN.md` § "Decisions log".

See also:
- `[[project-overview]]` — multi-module structure.
- `[[project-data-source]]` — data-source conventions (gbdt v1 uses `data_pipelines.fetch()`, not local CSVs).
- `[[project-results-layout]]` — where machine-readable headline metrics live.
- `[[feedback-experiment-agent-loop]]` — long-running pattern for agent-driven experiments.
- `analog_mc`'s `goal.md` — sibling module's why; useful for understanding the parallel design discipline.
