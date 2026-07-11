---
name: project-gbdt-xgboost-tree-count
description: xgboost gbdt cells use n_estimators=100 (the XGBClassifier default) unless the spec sets it explicitly — the default.yaml `hp_starting.iterations: 1000` is a CatBoost-only key and does NOT map to xgboost. So the original xgboost A/B sweeps + single-fits are 100-tree models.
metadata:
  type: project
---

**xgboost gbdt specs train exactly 100 trees by default** — the `XGBClassifier`
default `n_estimators=100`. Confirmed empirically (every fitted xgboost
`model.ubj` reports `num_boosted_rounds() == 100`; the cell's `hp.yaml` is `{}`),
verified on the nifty500 F18 A/B sweep + the 30/100d manual depth/subsample
sweeps (2026-07-10).

**The `iterations: 1000` in `configs/gbdt/default.yaml::backend.hp_starting` is a
CatBoost key** (its comment points at `CATBOOST_HP_REFERENCE.md`). It is NOT
translated to xgboost's `n_estimators` — xgboost specs with an empty/`max_depth`-
only `hp_starting` silently get 100 trees. `src/gbdt/model.py` builds
`xgb.XGBClassifier(early_stopping_rounds=…, **model_hp)`, and `model_hp` carries
`n_estimators` only if the spec set it.

**Why it matters:** the whole original xgboost A/B sweep (base vs +F18, all
cells) and any xgboost single-fit are **100-tree** models — modest, and plausibly
under-fit for shallow depths (a depth-2 weak learner wants *many* more trees).
The `[[project-gbdt-tuning-playbook]]` rule-12 aside about the agent loop's
"n_estimators=large" refers to the **CatBoost** loop path (iterations=1000), NOT
the xgboost single-fit path — don't conflate them.

**How to apply:**
- To use more trees on an xgboost cell, set `backend.hp_starting.n_estimators`
  explicitly (fixed count) OR add `early_stopping_rounds` (needs a val eval_set,
  which the pipeline supplies — early stopping is on VAL) so the count self-tunes.
- CAVEAT: at the xgboost default `eta=0.3`, `n_estimators=1000 + early_stopping`
  early-stops FAST (~50 trees observed) — the 1000 budget is barely used. To
  actually leverage many trees, lower `eta` (~0.05–0.1) alongside the large
  budget + early stopping.
- When reading/reporting an xgboost cell's capacity, state the realized tree
  count (`Booster.num_boosted_rounds()`), don't assume the config's 1000.
