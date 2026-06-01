# XGBoost `interaction_constraints` — capability, effect, and causal-ablation on the D6 cell

> **Methodology note (2026-06-01)**: Numbers in this memo's body use the legacy "weighted R-precision" metric (per-day variable K = R(d), micro-aggregated). The project headline metric was renamed 2026-06-01 to **R-Precision@K** (per-day fixed K, macro-aggregated via `(1/Q)·Σ r_q/min(K,R_q)`). See the "R-Precision@K (current methodology)" section at the bottom of this memo for the cells in this memo recomputed under the new metric, plus `.claude/memories/project-r-precision-methodology.md` for the full definition + relationship.

**Cell**: nifty50 UP +10% within 25 trading days, max_drawdown 5% (the `_147` answer-key / D6 acceptance cell).
**Date**: 2026-05-29.
**Branch**: `gbdt-v12-phase8-interactions`.
**Plan**: `docs/gbdt/V1.2_xgboost_feature_interactions_plan.md` § 8 Phase 8 (D6) — the final V1.2 deliverable, closing goal #165.
**Backend**: XGBoost 3.2.0, `tree_method=exact / n_jobs=1 / device=cpu`, seed 42, depth-8 / eta-0.05 / lambda-3.0 / 279-feature config (the invB `_149` / invA `_174` end-state).

> **n=1 caveat.** This is ONE cell and ONE forbidden pair. The capability finding generalizes (it is a property of the pipeline, not the cell); the *metric effect* of forbidding an interaction is a single data point and must not be over-read. **The verdict below is the user's to read — there is no automated PASS/FAIL** (goal.md § "What v1 actually ships").

---

## THE HEADLINE — can we run `interaction_constraints` in XGBoost through our pipeline today?

**YES — today, with no new plumbing.** XGBoost supports `interaction_constraints` (restrict which feature groups may co-split within a tree path); CatBoost does not. Exercising that capability is a core reason V1.2 moved to XGBoost, and the pipeline already plumbs it. Concretely:

1. **It is a *passable* HP through the construction / spec path.** `interaction_constraints` is **not** in `TUNABLE_HP_RANGES_XGB`, `ENUM_HP_VALUES_XGB`, or `PINNED_HPS_XGB`. That matters because `model.py::_validate_hp_xgb` only **range/enum-checks the HP keys it knows** — an *unknown structured key passes straight through* (it is not whitelisted-out). So an `XGBoostModel` constructed with `interaction_constraints` in its HP dict carries it untouched, and `XGBoostModel.fit` forwards the entire validated HP dict into `xgb.XGBClassifier(**model_hp)` (model.py line ~824), so the booster receives it. I confirmed empirically that a fitted booster's `save_config()` then carries `interaction_constraints: [[0],[1],...]`, i.e. XGBoost stored and honored it.

2. **The sanctioned surface is `gbdt.interactions.ablate_interactions`** (shipped in V1.2 Phase 4). It builds the correct XGBoost constraint spec from a list of *forbidden* pairs (the allowed-edge-list construction — one 2-feature group per allowed pair, integer-index JSON so it resolves on the name-less booster matrix), injects it post-construction (`fresh._hp["interaction_constraints"] = ...`), and refits. This is the "out-of-band intervention" path the plan intends (§ 3.2 / § 8 Phase 4).

3. **The agent-loop *decision* path REJECTS it — by design.** `loop_protocol.validate_decision` raises `DecisionError: references unknown HP 'interaction_constraints'` for an `hp_changes: {interaction_constraints: ...}` proposal, because the decision path *does* whitelist (it only accepts keys in the `*_XGB` tunable/enum tables). This is **exactly symmetric to `monotone_constraints` on CatBoost** (see `_174` § "Iteration 5 … the resume-decision path rejects `monotone_constraints`"), and is documented intent: `XGBOOST_HP_REFERENCE.md` line 19 lists `interaction_constraints` as out of the agent's decision schema — a Phase-4 read-only causal-ablation tool, never a per-iteration FS+HP knob. So the asymmetry is: **runnable via the ablation surface / spec `hp_starting`, blocked from the per-iteration loop schema.**

**Plumbing I added this branch: NONE.** The capability was already present (V1.2 Phases 1–5, merged to main). `git diff main` on `src/gbdt/` is empty — this branch added only the two finalize specs, the verification runner (`scripts/gbdt/phase8_interaction_constraints_verify.py`), and this analysis. Flagged so the reviewer knows no backend behavior changed.

---

## DOES THE CONSTRAINT TAKE EFFECT? — verified, honored, not silently ignored

I fit the D6 XGBoost cell unconstrained, ranked pairwise interactions via native TreeSHAP `pred_interactions` (the dense reference over a 200-row seeded sub-sample of train+val — see the cost note below), identified the **top interaction pair**, then refit with that pair forbidden via `interaction_constraints` and re-measured.

- **Top SHAP-interaction pair**: `index_vol_50 × index_vol_200`, mean-|interaction| **0.0801**. (The full top-15 is dominated by long-window vol-estimator × index-regime pairs — `index_vol × garman_klass_200`, `index_runup_200 × garman_klass_200`, … — the same *"interaction-driven cell"* structure `_147`/`_174`/`_149` found: vol×regime conditional, not flat marginals.)
- **After forbidding the pair, the constraint was HONORED on two independent measures:**

| measure | unconstrained | constrained | reading |
|---|---:|---:|---|
| TreeSHAP `pred_interactions` (mean \|interaction\|, the pair) | 0.0801 | **0.0** | collapsed to zero — the model no longer attributes any pairwise interaction to this pair |
| native split co-occurrence (gain-weighted tree-path) | 58.94 | **0.0** | the pair **never co-splits** on any root→leaf path in the constrained model |

`all_honored = True`. This is the SHAP→causal-contribution link closed end-to-end: a measured interaction, forbidden, verifiably removed from both the Shapley decomposition and the tree structure. (The XOR unit fixture `tests/gbdt/test_interactions.py::test_ablation_zeroes_interaction_and_degrades_brier` proves the same mechanism on controlled synthetic data — 12/12 green.)

**Determinism held throughout**: the unconstrained config refit bit-identically twice, and the refit reproduced the on-disk `model.ubj` eval predictions — the § 5.1 finalization-retrain contract is intact under the determinism pins.

---

## WHAT'S THE EFFECT ON THE METRICS? — constrained vs unconstrained

Brier + weighted R-precision (per-day variable-K = R(d), the standard panel-invariant cross-cell metric; `scripts/gbdt/compute_r_precision.py` semantics, computed via `diagnose_core.per_day_r_precision`). Both models calibrated with a fresh isotonic fit on val (the conditional-isotonic Z-test rejects native on this cell).

| segment | base rate | Brier (unconstr.) | Brier (constr.) | wtd R-prec (unconstr.) | wtd R-prec (constr.) |
|---|---:|---:|---:|---:|---:|
| eval | 0.1374 | 0.11668 | 0.11617 | 0.29794 | 0.30288 |
| test | 0.1965 | 0.14572 | 0.14041 | 0.38511 | 0.38350 |

**Forbidding the top SHAP-interaction pair did *not* degrade the model — it slightly *improved* Brier** (eval −0.0005, test −0.0053) and left weighted R-precision essentially flat (eval +0.005, test −0.002). The unconstrained weighted R-precision lift over base rate is eval 2.17× / test 1.96× (constrained 2.20× / 1.95×) — i.e. the ranking signal is unchanged.

**Reading.** The plan's expectation — *"forbidding genuinely-interacting features should degrade metrics in proportion to the interaction magnitude"* — holds *conditionally*: the mechanism fired (the interaction verifiably collapsed), but on this cell the top interaction was **not net-positive for predictability**, so forbidding it was benign-to-mildly-helpful rather than harmful. That is consistent, not contradictory, with the cell's well-established regime (`_147`/`_174`/`_149`): **no-overfit, where feature-selection and heavy regularization are neutral-to-harmful**. The forbidden pair `index_vol_50 × index_vol_200` is two correlated lookbacks of the *same* index-volatility signal — their co-split is largely redundant, so forbidding it acts as mild regularization (which on a sub-noise HP-ceiling cell nudges Brier the right way). A *load-bearing* interaction would have degraded the metrics; this one didn't, which tells us something about the cell, not about the tooling. **n=1: one cell, one pair — do not generalize the sign of the metric move.**

### Cost note (why 200 rows, not 5,000)

Native TreeSHAP `pred_interactions` at the full **F=279** active set is **~0.8 s/row** on this single-thread `exact` CPU booster (a 200-row pass = 175.6 s; 3,000 rows timed out at 600 s). This is the documented § 3.3 R2 risk: `pred_interactions` materializes a 280×280 per-row tensor and the work scales badly at this feature width. The model keeps all 279 features (the no-overfit gate never pruned — `_147` lesson), so the plan's primary mitigation (*"compute on the pruned active set"*) doesn't shrink F here, and a post-hoc column subset can't be fed to a 279-feature booster (shape mismatch). I therefore used the **200-row seeded sub-sample** (plan § 3.3 mitigation 2 — "a few thousand rows is enough to rank pairs stably"; 200 was sufficient: the SHAP top pairs match the near-free co-occurrence ranking) with the existing public `shap_interaction_dense_reference`, plus co-occurrence (the § 3.2 / mitigation-5 near-free cross-check) for the path check. The tooling is **correct, just O(F²)-expensive at F=279** — not broken; this is the plan's anticipated wall-clock ceiling, not a defect.

---

## Supporting context — the CatBoost contrast (the *point* of XGBoost here)

CatBoost **can measure** interactions (native split-pair co-occurrence, `get_feature_importance(type="Interaction")`) — its top pairs on the matched D6 CatBoost finalize (depth-8 / 279-feat, `_174` config) are the same vol×index-regime family (`index_return_100 × index_vol_200` 1.14, `index_vol_200 × stock_return_zscore_50_outside_band_1` 0.96, `parkinson_200 × garman_klass_200` 0.88). **But CatBoost exposes no `interaction_constraints` knob** — `interaction_constraints` is not in its parameter surface. So the constrained-vs-unconstrained ablation above is *impossible on CatBoost*. That asymmetry — both backends measure, only XGBoost can *intervene* — is the reason V1.2 moved to XGBoost for the interaction analysis. CatBoost gives you the *what* (which pairs interact); only XGBoost gives you the *causal contribution* (forbid the pair, measure the cost).

The calibration story matches across backends (both isotonic, eval Brier ≈ base rate under the prevalence-drift ceiling), so the contrast is genuinely about the *interaction-intervention* capability, not a model-quality gap.

---

## User-facing verdict (for the user to read)

- **Can we run `interaction_constraints` in XGBoost through our pipeline today?** **YES**, with no new plumbing: via the `ablate_interactions` surface / spec `hp_starting` (the construction path passes the structured value through and the booster honors it). It is deliberately **not** a per-iteration agent-loop knob (rejected by `validate_decision`, symmetric to CatBoost's `monotone_constraints`).
- **Does the constraint take effect?** **Yes, verified** — the forbidden pair's TreeSHAP interaction collapsed 0.0801 → 0.0 and its tree-path co-occurrence dropped 58.94 → 0; determinism + artifact-reproduction held.
- **What's the effect?** On this cell, forbidding the top interaction was **neutral-to-mildly-beneficial** (Brier improved slightly, ranking flat) — the top interaction was a redundant vol-lookback co-split, not load-bearing. Consistent with the cell's no-overfit / regularization-neutral regime. **The metric-sign result is n=1 and not to be generalized.**

---

## Reproducibility

- **Fits**: `uv run python -m gbdt experiment configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_xgb_phase8.yaml` (+ `..._catboost_phase8.yaml`). Artifacts at `results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_{xgb,catboost}_phase8/` (model binaries are gitignored per repo policy; specs + metrics + reports are tracked).
- **Verification + ablation**: `uv run python -m scripts.gbdt.phase8_interaction_constraints_verify` → `results/gbdt/data/_175_xgboost_interaction_constraints_capability.json` (raw evidence). A full-row alternative variant is `scripts/gbdt/interaction_constraints_capability_check.py` (left in place; slower at full SHAP rows — same three-part design).
- **Headline JSON**: `results/gbdt/data/_175_xgboost_interaction_constraints_h25_data.json`.

## R-Precision@K (current methodology — added 2026-06-01)

Per `.claude/memories/project-r-precision-methodology.md`, R-Precision@K is the post-2026-06-01 headline cross-cell metric for gbdt — defined as `R-Precision@K = (1/Q) · Σ_q r_q / min(K, R_q)` over the Q days where R_q > 0 (R_q = positives on day q; r_q = positives caught in top-K picks on day q; macro-averaged, equal weight per day; K fixed). Recomputed from each cell's `predictions/test.csv`:

| cell | rows | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|---|---|
| nifty50_up_10pct_25d_dd5pct_xgb_phase8 | 3450 | 17.9% | 0.639 | 0.314 | 0.329 | 0.266 | 0.257 | 0.536 |
| nifty50_up_10pct_25d_dd5pct_catboost_phase8 | 3450 | 17.9% | 0.729 | 0.171 | 0.205 | 0.213 | 0.328 | 0.559 |

The canonical CSV carries the Phase-8 unconstrained finalize artifacts; the in-memo constrained vs unconstrained ablation tables stay anchored to the legacy weighted R-precision metric in the body.

Cross-links: `docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md` (CatBoost answer key), `_174` (invA CatBoost agent loop — `monotone_constraints` rejection precedent), `_149` (invB XGBoost agent loop — same D6 cell), `docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`, `docs/gbdt/XGBOOST_HP_REFERENCE.md` (line 19 — interaction_constraints out of the loop schema), `[[project-r-precision-methodology]]`, `[[project-xgboost-interaction-analysis]]`.
