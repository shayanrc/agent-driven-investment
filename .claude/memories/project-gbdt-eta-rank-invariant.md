---
name: project-gbdt-eta-rank-invariant
description: xgboost eta/learning_rate IS applied but is a monotonic rescaling — it preserves prediction rank, so isotonic calibration + all rank-based metrics (R-Precision@K, AUC) are invariant to it. Sweeping eta cannot move R-p@K. NOT a bug. Tree-structure knobs (n_estimators, subsample, min_child_weight, colsample, gamma, max_depth) DO change rank and are real levers.
metadata:
  type: project
---

**Sweeping the learning rate (`eta` / `learning_rate`) on a gbdt cell does NOT
move R-Precision@K or AUC — and this is a property, not a bug.**

Mechanism (proven on nifty500 30/100d fund cell, 2026-07-11):
- eta **is** applied — the raw model output changes (lower eta compresses the
  logit range: eta0.05 raw ∈ [0.248, 0.267] vs eta0.3 ∈ [0.213, 0.324]).
- But eta is a **monotonic rescaling** at a fixed tree set → it preserves the
  **exact prediction ranking** (Spearman = 1.000000, 0 / 63,524 rank inversions).
- The pipeline's calibration is **isotonic** (`conditional_isotonic`), which is
  **rank-based** → identical ranks map to **identical `p_calibrated`** (Δ = 0.0).
- **R-Precision@K and AUC are rank-based** → identical across eta. The scored
  metric is structurally blind to a pure learning-rate change.

**So do NOT sweep eta hoping to improve the top-K book — it cannot, by
construction.** (I initially misdiagnosed the flat eta sweep as "eta silently
dropped — a runner bug." It is NOT a bug: `XGBoostModel` applies eta, the trained
booster LR is 0.05 when asked, and `p_raw` differs. Two red herrings that fooled
me: (1) the `.ubj` `learning_rate` readout **resets to 0.3 on load** — unreliable
for diagnosis; (2) checking `p_calibrated` (identical) without checking `p_raw`
(differs). Always compare `p_raw` + Spearman rank, not the calibrated output, when
asking "did this HP change the model?")

**How to apply:**
- Levers that change the **tree structure** — and therefore the rank — ARE real:
  `n_estimators` (adds splits; see [[project-gbdt-xgboost-tree-count]]),
  `subsample`, `min_child_weight`, `colsample_bytree`, `gamma`, `max_depth`. Sweep
  those to move R-p@K.
- `eta` only matters indirectly IF it changes which trees early-stopping selects
  (i.e. a different tree *set*, not just rescaled weights) — on cells where ES
  stops at the same count regardless (the 30/100d case: 51 trees at both 0.05 and
  0.3), it is a strict no-op for rank metrics.
- Corollary: any HP whose only effect is a monotone transform of the score is
  invisible under isotonic calibration + rank metrics. Verify a candidate HP
  actually perturbs the *rank* before sweeping it.
