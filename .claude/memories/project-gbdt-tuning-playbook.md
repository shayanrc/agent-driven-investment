---
name: project-gbdt-tuning-playbook
description: Diagnostic-first FS+HP tuning rules for gbdt cells, learned from the nifty50 H=25 hand-driven loop — check train/val gap before pruning; detect HP ceilings early; monotone constraints are neutral-to-harmful on interaction-driven cells; importance≈0 means redundant not unrelated.
metadata:
  type: project
---

Hard-won tuning rules from the nifty50 H=25 hand-driven FS+HP loop (8 configs, `docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md`). These are the agent's playbook when *manually* driving the loop (or steering the V1.1 callback) — they encode what the fixed `default_fs_hp_callback` cannot reason about.

**1. Read the train/val gap BEFORE pruning anything.**
- **Why**: the fallback callback prunes on sight, assuming overfit. On nifty50 H=25 the gap was *negative* (val brier < train brier) — **no overfit**. Pruning then only removed capacity → the fallback's blind cut raised val brier (0.1642 → 0.1663). The fallback's instinct was actively wrong.
- **How to apply**: `train_val_gap = val_brier − train_brier`; POSITIVE = val worse than train = overfit (src/gbdt/train.py acts at gap > 0.02). If `train_val_gap ≤ 0.02` (val not meaningfully worse than train) → the model is NOT overfitting; do NOT prune for regularization (at best neutral, usually harmful). **Early-stopping firing is NOT an overfit signal** — it's the healthy mechanism that picks the tree count; a model can early-stop at tree 67 with a deeply negative gap (the nifty50 case). Base the no-overfit call on the gap alone, not on whether early-stop fired.

**2. importance≈0 ≠ unrelated — it usually means REDUNDANT.**
- **Why**: 191/279 features had <0.01 importance, yet 26 of them had a clean monotone relationship with the target; 15 of those were 0.7–0.99 collinear with a *kept* feature. The model needs one vol estimator per window-band, not garman_klass+parkinson+realized_vol (≈0.96–1.00 correlated).
- **How to apply**: don't expect feature selection to *improve* a non-overfit model — removing redundant features is ~neutral on val (nifty50: all-279 0.1642 vs lean-88 0.1650). FS buys leanness/calibration, not accuracy. Verify "is this feature redundant?" via correlation with kept features before concluding it's noise.

**3. HP-ceiling detection — scan depth + lr, then stop.**
- **Why**: across depth {4,6,8} × lr {0.02,0.05}, val brier stayed in [0.1641,0.1661]; R-Precision@10 pinned at ~2.1× throughout (the metric quoted in the original playbook as "R-precision ~2.1×" was the legacy weighted R-precision form — under R-Precision@K the nifty50 H=25 cell lands at 0.368/0.179 = 2.06×, ~2.1× — either way the lesson holds: HP search wasn't moving the needle). No HP config beat the untuned baseline. Burning 8 iterations would have been waste.
- **How to apply**: map the depth response curve (expect an inverted-U; the minimum is the sweet spot — nifty50 was depth 6) and test lr once at the best depth. If the val-brier band across these diverse configs is tiny (≈≤0.002), declare the **HP ceiling** and stop. A clean negative ("no HP headroom") is a valid, valuable result.

**4. Monotone constraints (`monotone_constraints` via `hp_starting`) are neutral-to-harmful on interaction-driven cells — and the marginal-correlation check is a trap.**
- **Why**: nifty50 H=25's signal lives in vol×regime *conditional* interactions. Constraining vol estimators `+1` (chosen by clean *marginal* Spearman) HURT (val 0.1642→0.1664, test AUC 0.733→0.677). Three diagnostics: (a) interaction mass barely moved; (b) 2D PDP showed no sign-flip to destroy; (c) 1D-PDP audit showed 13/17 already monotone in the model, but 4 long-window vol estimators had a learned **inverted-U** (too much vol kills the breakout) the `+1` flattened. Restricting to the 13 already-monotone features STILL lost vs baseline (0.1657) — CatBoost enforces monotonicity at the *tree-structure* level, degrading conditional interactions even when the *net* PDP is monotone. Constraining unused/pruned features = no-op (iter7 ≈ iter6).
- **Two-component harm (iter 8/9 ablation)**: splitting the 17 estimators by interaction involvement and constraining each half alone showed (a) a **fixed ~0.0012 "constraints-on" cost** present even for zero-interaction features (turning on `monotone_constraints` switches CatBoost to a more restrictive tree builder globally), plus (b) a **dominant interaction-specific cost** — constraining just the 8 high-interaction estimators reproduced the *entire* 17-constraint val_brier harm, while removing the low-interaction constraints recovered 0%. There is **no safe subset** to constrain on an interaction-driven cell.
- **How to apply**: (i) marginal monotonicity (Spearman/decile) is NOT sufficient justification — check the **unconstrained model's 1D partial dependence** instead, and even that is necessary-but-not-sufficient. (ii) For an interaction-driven cell (good AUC/R-Precision@K, signal in feature *combinations*), don't bother — constraints range from no-op to harmful, with no safe subset. (iii) `monotone_constraints` does pass through the runner (unknown HP keys flow through `_validate_hp` → `CatBoostClassifier`; named-dict format resolves via Pool feature_names). CatBoost has **no** interaction constraints (XGBoost/LightGBM-only); v1 pins catboost.

**5. The lever for a brier-capped cell with good ranking is OUT of the FS/HP loop.**
- **Why**: nifty50 H=25 ranks well (R-Precision@10 ≈ 2.1× base) but brier improvement is weak and capped by a train→eval **prevalence non-stationarity** (28%→14% positive rate). No FS/HP/constraint touches this — the loop only sees val.
- **How to apply**: when val/eval brier is stuck but AUC/R-Precision@K are strong, suspect a prevalence/regime shift (compare `positive_prevalence` across train/val/eval/test). The fix is recency weighting / regime-conditional calibration (a methodology change), not tuning. Flag it; don't grind iterations against it.

**6. One attributable change per iteration; let results pick the next question.**
- **Why**: each nifty50 iteration held everything fixed except the variable under test, so 8 runs produced a coherent response surface instead of 8 disconnected points. The next experiment was always *generated by* the last result (fast overfit → test shallower; depth settled → test lr; etc.) — the opposite of a fixed heuristic.

See `docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md` (full narrative + figures), `[[project-r-precision-methodology]]` (the ranking metric), and `docs/gbdt/CATBOOST_HP_REFERENCE.md` (per-parameter rubrics). The analysis scripts (`scripts/gbdt/{monotonic_feature_analysis,monotone_1d_audit,interaction_before_after,pdp_and_corr,pruned_feature_investigation}.py`) are reusable diagnostics for any cell.
