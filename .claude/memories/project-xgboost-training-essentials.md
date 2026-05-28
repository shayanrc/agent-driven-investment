---
name: project-xgboost-training-essentials
description: XGBoost training/implementation essentials for the gbdt backend — the load-bearing DETERMINISM contract for the exit-and-resume loop, HP-name mapping vs CatBoost, built-ins (regularization / missing values / constraints), and portability. Grounds V1.2 (#165).
metadata:
  type: project
---

What the gbdt XGBoost backend (#165, `docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`, PR #56) must respect when training. Grounded in xgblog.ai "XGBoost is all you need" Parts 4–5 + XGBoost docs + the V1.1 Phase-2 exit-and-resume analysis.

**1. THE DETERMINISM CONTRACT (load-bearing — do not skip).**
- The FS+HP loop's finalization step **retrains a prior `(features, hp)` config** (`src/gbdt/train.py::_fit_one`) and assumes it reproduces the in-loop model **bit-identically** (the checkpoint stores no model blob). If the retrain drifts, `predictions/*.csv` silently disagree with the val-Brier the checkpoint was selected on.
- XGBoost reproducibility requires ALL of: fixed `seed`/`random_state`; `tree_method="exact"` (`hist` is deterministic only with controlled threading); **`n_jobs=1`** (multi-thread float-reduction order varies run-to-run); `device="cpu"` (GPU/`gpu_hist` is non-deterministic). 
- **Pin these as XGBoost's never-override set** (`PINNED_HPS_XGB`), analogous to how CatBoost pins `has_time`; `_validate_hp_xgb` must **hard-fail** on any override (V1.2 plan D5).
- We deliberately **forgo XGBoost's GPU/Dask training speedups** (blog Part 5: multi-GPU + Dask, 30–60s/trial on 8×H100) inside the loop — performance is not the goal; the bit-identical retrain is. (GPU is fine for the read-only interaction *diagnostic* — see [[project-xgboost-interaction-analysis]] #4.)

**2. HP names differ from CatBoost → backend-conditional sibling tables (V1.2 plan D3).**
- `max_depth` (↔ CatBoost `depth`); `eta`/`learning_rate` (↔ `learning_rate`); `lambda`/`reg_lambda` (L2 ↔ `l2_leaf_reg`); `alpha`/`reg_alpha` (L1); `min_child_weight`; `gamma`/`min_split_loss`; `subsample`; `colsample_bytree`/`bylevel`/`bynode`; `max_bin`.
- **No `has_time` analogue** — XGBoost has no ordered-boosting equivalent, so walk-forward C6 correctness rests on the **split discipline alone** (already backend-agnostic in `train.py`), guarded by the backend-parametrized synthetic-leakage test. Reject a canonical cross-backend HP vocab as lossy; keep `*_XGB` tables + a `hp_tables_for(backend)` resolver, and thread a `backend` param into `loop_protocol.validate_decision`.

**3. Built-ins.**
- L1 (`alpha`) + L2 (`lambda`) regularization built in. Automatic **missing-value direction learning** (sparsity-aware split finding) — no imputation needed. Objective `binary:logistic` for the binary target; raw output is margin → probability via sigmoid. **Calibration is unchanged** — `src/gbdt/calibration.py` operates purely on `(y_val, p_raw)` arrays and never touches the model object, so XGBoost's raw probabilities flow through the identical Spiegelhalter-gated isotonic/Platt path.
- XGBoost supports `interaction_constraints` + `monotone_constraints` (**CatBoost does NOT** — see [[project-gbdt-tuning-playbook]] #4). `interaction_constraints` is a Phase-4 **causal-ablation** tool (forbid the top SHAP pairs, retrain, confirm Brier/R-prec degradation tracks the SHAP magnitude) — never on the hot FS+HP loop.

**4. Portability.** Models export as JSON / UBJ (`.ubj` recommended binary); the `backend.library: xgboost` spec field already exists (default.yaml + EXPERIMENT_SPEC.md) and currently rejects non-catboost — V1.2 lifts that. `/gbdt-diagnose`'s model loader needs backend dispatch (currently hard-codes `CatBoostClassifier().load_model("model.cbm")`).

Apply sample-uniqueness weights (LdP §4.4) for XGBoost too — see [[project-gbdt-uniqueness-weights]].
