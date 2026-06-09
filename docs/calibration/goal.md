# calibration — Goal

This document states what `calibration` is optimizing for and what trade-offs are unacceptable. Read it before editing any file under `src/calibration/`, `tests/calibration/`, or `docs/calibration/`.

For *how* it works (component APIs, fit/transform lifecycle), see `V1_PLAN.md`. This file is the *why* and *what success looks like*.

---

## What this module is optimizing for

Provide a **backend-agnostic probability-calibration toolkit** that turns raw classifier or probabilistic-forecast outputs into well-calibrated probabilities with explicit uncertainty, with one defining rule:

> **The calibrator knows nothing about the predictor.** Inputs are NumPy arrays of `(p_raw, y_true)` for fit and `p_raw` for transform; outputs are NumPy arrays of `(p_mean, p_low, p_high)`. Any change that couples a calibrator to a specific predictor (gbdt, analog_mc, XGBoost, etc.) is not acceptable.

Why: the same Bayesian recalibrator must work on outputs from gbdt classifiers today, analog_mc fan-quantile probabilities tomorrow, and any future probabilistic backend. Coupling kills re-use.

## What success looks like

- **`Calibrator` is a Protocol, not an ABC.** Anyone can implement `fit(p_raw, y_true) → Calibrator` and `transform(p_raw) → CalibrationOutput` without inheriting; the type checker enforces the contract.
- **Bayesian calibrators report posterior moments.** `(p_mean, p_low, p_high)` is the standard return — point estimate plus 95% credible interval. Non-Bayesian calibrators may return `None` for `p_low` / `p_high`.
- **Diagnostics are shipped, not optional.** ECE computation and reliability-diagram plotting live in this module; every calibrator gets a default reliability figure for free.
- **Save / load is round-trippable.** A fitted calibrator is a pickle (or YAML for human-readable ones) that downstream code can load and apply without re-running fit.
- **No data-fetching, no model code, no strategy code.** Pure NumPy in, NumPy out.

## What this module is *not*

- **Not a model trainer.** Calibrators consume already-trained model outputs; they don't fit base predictors.
- **Not a metric library.** ECE and reliability are here because they're calibration-specific. Sharpe, drawdown, R-Precision — those live elsewhere.
- **Not where gbdt's current isotonic lives** (yet). gbdt has its own `conditional_isotonic` calibration inside `src/gbdt/`; v1 of this module does NOT migrate that code. Migration is a follow-up branch.
- **Not bound to any particular prior or binning scheme.** v1 ships `BetaBinomialBucketed` with `Beta(1, 1)` prior + M=10 quantile bins, but the protocol admits hierarchical priors, Bayesian Platt, conformal prediction, etc. in v1.1+.

## How to apply this when working on the module

- **No predictor imports.** `src/calibration/` should never `import gbdt` / `import analog_mc`. If you find yourself needing predictor-specific knowledge, the abstraction is broken.
- **`(p_raw, y_true)` is the only fit signature in v1.** Future calibrators may take additional inputs (e.g., features for conditional calibration), but the v1 protocol is the minimal one.
- **Tests live in `tests/calibration/`.** No `pytest.importorskip("gbdt")`-style coupling to other modules in test setup.
- **Diagnostics are part of the module charter.** Don't push ECE / reliability into per-caller scripts; they live here so every calibrator gets them.
- **Save format compatibility.** Prefer pickle for binary state (numpy arrays, scipy distributions); reserve YAML for human-readable metadata. A loaded calibrator must produce identical outputs to the original on the same inputs.

## What not to do

- **Don't add data sourcing.** No `data_pipelines` imports here. Callers fetch arrays; this module operates on them.
- **Don't add backend-specific shortcuts.** If gbdt and analog_mc both need calibration, the calibrator code doesn't change — the *caller* code adapts.
- **Don't conflate calibration with selection.** The calibrator transforms probabilities; it doesn't pick thresholds, doesn't compute precision/recall. Selection logic belongs in `trading_strategies/`.
- **Don't add ML metrics.** AUC, Brier, log-loss — useful, but not here. This module is calibration-specific.
- **Don't depend on `forecasters` or `backtesting`.** Both consume calibration outputs (potentially); calibration doesn't consume them.
