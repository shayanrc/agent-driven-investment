---
name: project-xgboost-interaction-analysis
description: XGBoost feature-interaction analysis via NATIVE TreeSHAP (pred_contribs / pred_interactions) — the method, costs, interpretation rules, and how it feeds the gbdt agent-loop pruning. Grounds the V1.2 XGBoost work (#165).
metadata:
  type: project
---

Reference for the gbdt XGBoost backend's headline capability: measuring how feature **interactions** drive the categorical-outcome models. Grounded in xgblog.ai "XGBoost is all you need" Part 6 (Shapley values) + XGBoost docs, and the V1.2 plan (`docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`, PR #56). CatBoost lacks the per-row interaction tooling — this is the *reason* the backend exists.

**1. Use XGBoost's NATIVE TreeSHAP, not the external `shap` package.**
- `booster.predict(dmatrix, pred_contribs=True)` → per-row `(n_features + 1)` SHAP contributions (last column = bias/base value; row sum = model margin output).
- `booster.predict(dmatrix, pred_interactions=True)` → per-row `(n_features + 1) × (n_features + 1)` interaction matrix: **diagonal = main effects, off-diagonal = pairwise interaction attributions**; each row sums to the margin.
- Built into XGBoost ≥ 1.3, GPU-accelerated. The blog's hard number: the standalone `shap` library took "days" on subsampled data vs the native method's "minutes" (~340× on interaction values on a V100; ~1.2M rows/s on 8 GPUs). **Always prefer native** for the heavy compute.

**2. Cost is O(rows × features²) — never compute on the full pool.**
- The dense interaction tensor at the full ~279-feature pool is ≈ 31 GB — infeasible. Mitigations (baked into the V1.2 plan): compute on the **pruned active feature set** (the fitted artifact's `features.yaml`, ~30–80 features, not the iter-0 pool), **subsample rows** (default cap ~5000), and **stream-aggregate** pair magnitudes — never materialize the dense tensor. Native split co-occurrence frequency is a near-free cross-check.

**3. Interpretation rules (the agent-loop's pruning logic).**
- **Drop a feature only if BOTH its main-effect |SHAP| is near-zero AND its total interaction load is low.** A feature with ≈0 marginal importance but high Σ|off-diagonal| is load-bearing through interactions — keep it. This is the **third pruning axis** for the agent loop (V1.2 plan D7), and it externally grounds the gbdt rule that `importance≈0 ≠ unrelated` (see [[project-gbdt-tuning-playbook]] #2).
- Zero/negative pairwise interaction → redundant pair. High pairwise interaction → synergistic.
- Feature-selection impact is biased unless HP is re-optimized after pruning (Part 6's own caveat) — this validates the gbdt loop's **joint FS+HP** design, not FS-then-HP.

**4. Determinism split — diagnostics MAY use GPU even though training must not.**
- The interaction computation is a **read-only diagnostic on an already-fit model**, so it can opt into GPU without touching reproducibility. **Training** stays deterministic-CPU (the load-bearing finalization-retrain assumption — see [[project-xgboost-training-essentials]]). Do NOT conflate the two backends.

**5. Interaction-guided feature ENGINEERING is OUT of scope for V1.2 (parked V1.3+).**
- Part 6 shows synergistic pairs → product/combined features improved scores. But the gbdt feature family is a fixed **asset-agnostic causal** set and the loop does selection + HP, not synthesis (CLAUDE.md § gbdt). Synthesis would expand the loop's mandate — park it; don't pull it into V1.2. Caveat if ever pursued: "a simple multiplication may not work, especially when at least one feature is categorical/signed" (relevant to the F16 signed family).

Report interaction strength with the no-lift-column + raw-numbers conventions (CLAUDE.md § Reporting). See [[project-r-precision-methodology]] for the cross-cell metric the CatBoost-vs-XGBoost comparison memo uses.
