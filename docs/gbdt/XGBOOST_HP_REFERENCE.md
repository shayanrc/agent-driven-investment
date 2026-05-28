# XGBoost Tunable Parameters — gbdt V1.2 reference

## Scope

This is the per-iteration tuning reference the gbdt agent consults when deciding which **XGBoost** hyperparameters to change between FS+HP iterations, when a cell's spec sets `backend.library: xgboost`. It is the sibling of [`CATBOOST_HP_REFERENCE.md`](CATBOOST_HP_REFERENCE.md) and is scoped to the identical use case:

- **Task:** binary classification — one binary target per cell (e.g. "did stock close ≥10% above entry within 25 trading days").
- **Headline metric:** **calibration** (Brier score, reliability curves), *not* AUC / accuracy / F1.
- **Data shape:** ~1,500 rows/stock × ~48 stocks pooled per cell → ~70k rows. Small-data regime for boosting.
- **Class imbalance:** several cells are rare (±50% in 10 days fires <2% of rows). Class weighting matters.
- **Splits:** strictly **walk-forward** — never shuffled CV. XGBoost has **no `has_time` analogue** (see § "C6 and the missing `has_time`").
- **Features:** no categoricals in v1 (day-of-week / month-of-year are sin/cos float pairs). No text, no embeddings. XGBoost learns missing-value direction natively (sparsity-aware split finding), so no imputation is needed — but our v1 features have no missing values by construction.
- **Hardware:** single-machine CPU is the baseline, **pinned** (`tree_method=exact`, `n_jobs=1`, `device=cpu`) for determinism (see § "Determinism"). GPU is allowed only for the read-only interaction *diagnostic*, never for training.

**Why XGBoost in V1.2:** the interaction tooling (native `pred_interactions` TreeSHAP, `interaction_constraints`) is the driver — see `docs/gbdt/V1.2_xgboost_feature_interactions_plan.md`. This doc covers only the *training* knobs the FS+HP loop tunes; the interaction analysis lands in a later phase.

**What is *not* in this doc (and why):**

- `monotone_constraints`, `interaction_constraints` — V1.2 *measures* interactions and (later) runs `interaction_constraints` as a Phase-4 **causal-ablation experiment**, never as a per-iteration FS+HP knob. They are out of the agent's decision schema.
- `tree_method`, `n_jobs`, `device`, `seed` — these are the **pinned** determinism set (analogous to CatBoost's `has_time`), not quality knobs. See § "Determinism".
- `num_parallel_tree` (random-forest mode), `dart`-family (`rate_drop`, `skip_drop`, `sample_type`, `normalize_type`) — niche; not in the v1 search space.
- `gpu_id`, `predictor`, `nthread` aliases, `validate_parameters`, `verbosity` — infra / diagnostics, not tuning.
- `base_score`, `booster` (we use the default `gbtree`) — pick once, not per-iteration.
- `max_cat_to_onehot`, `enable_categorical` — no categorical features in v1.

If an XGBoost parameter doesn't appear in this doc, the agent should **not** touch it.

---

## CatBoost ↔ XGBoost name map (read this first)

The agent tuning an `xgboost` cell requests **XGBoost** names; the agent tuning a `catboost` cell requests **CatBoost** names. The two vocabularies are validated against different tables (`hp_tables_for(backend)` in `src/gbdt/model.py`; V1.2 plan D3). The mapping for the knobs that have a counterpart:

| CatBoost name | XGBoost name | Mechanism |
|---|---|---|
| `iterations` | `n_estimators` | max boosting rounds |
| `learning_rate` | `eta` | shrinkage |
| `depth` | `max_depth` | tree depth |
| `l2_leaf_reg` | `lambda` (`reg_lambda`) | L2 regularization |
| *(no analog)* | `alpha` (`reg_alpha`) | L1 regularization |
| *(no analog)* | `gamma` (`min_split_loss`) | minimum loss reduction to split |
| `min_data_in_leaf` | `min_child_weight` | min leaf occupancy (Hessian-weighted) |
| `rsm` | `colsample_bytree` | feature subsampling per tree |
| *(per-level rsm)* | `colsample_bylevel` | feature subsampling per level |
| *(none)* | `colsample_bynode` | feature subsampling per split |
| `subsample` (Bernoulli) | `subsample` | row subsampling |
| `border_count` | `max_bin` | numeric quantization bins (hist only) |
| `max_leaves` (Lossguide) | `max_leaves` | leaf cap (lossguide) |
| `grow_policy` (Symmetric/Depthwise/Lossguide) | `grow_policy` (depthwise/lossguide) | tree growth |
| `scale_pos_weight` | `scale_pos_weight` | positive-class up-weight |
| `early_stopping_rounds` | `early_stopping_rounds` | early stop patience |
| `has_time=True` | **— (no analogue)** | walk-forward C6 (see below) |
| `loss_function=Logloss` | `objective=binary:logistic` | binary objective |
| `eval_metric=BrierScore` | `eval_metric=logloss` | early-stop metric (Brier computed in the bundle) |

XGBoost has **no `SymmetricTree`** (CatBoost's oblivious-tree default) and **no ordered boosting** — it is plain gradient boosting, which is exactly why the C6 guarantee differs (below).

---

## C6 and the missing `has_time`

CatBoost pins `has_time=True` so its *ordered-boosting permutation* respects the time order (a second-order leakage guard). **XGBoost has no ordered-boosting concept**, so it has no `has_time` knob.

For XGBoost the walk-forward C6 guarantee rests **entirely on the split discipline**, which is upstream of the model and identical for both backends: `train.py::carve_single_fold` + `_gather_segment` build each segment as a strictly-forward trailing slice per ticker (no shuffling; train precedes val precedes eval precedes test in time). The model never sees a future row in training. This is a deliberately weaker guarantee than CatBoost's (split-discipline + ordered-boosting) and is documented as acceptable in V1.2 plan § 5.3, because the split discipline is the primary C6 mechanism and the synthetic-leakage harness tests both backends end-to-end.

So: there is **no time-axis knob to tune** on XGBoost. The "always pinned" list (§ "Determinism") replaces `has_time`'s role as "the pin you can never override."

---

## Category 1: Core boosting controls

The first knobs to reach for — capacity, learning speed, complexity.

### `n_estimators` (↔ CatBoost `iterations`)

- **Description:** Maximum number of boosting rounds (trees). Early stopping can cut this short.
- **Range/values:** Integer `[1, ∞)`. Practical: `100–10000`.
- **Default (v1):** `1000`.
- **When to change:**
  - **Raise** if early stopping fires close to `n_estimators` (best iter within ~10% of the cap) — the model wanted more trees.
  - **Raise** if the validation learning curve is still trending down at the final iteration.
  - **Lower** only if wall-clock matters and you have quality headroom; otherwise let early stopping handle it.
- **Example:** `n_estimators=3000` once a previous run early-stopped at ~950 of 1000 — give it more rope before lowering `eta`.

### `eta` (↔ CatBoost `learning_rate`)

- **Description:** Shrinkage applied to each tree's leaf updates. Smaller = more, smaller corrections per tree.
- **Range/values:** Float `(0, 1]`. Practical: `0.01–0.3`. Treat as **log-scale**.
- **Default (v1):** `0.05`.
- **When to change:**
  - **Lower** (e.g. halve) if validation Brier oscillates iteration-to-iteration, or a large train/val gap opens early.
  - **Lower** if you raised `n_estimators` and want the model to use the extra trees.
  - **Raise** if every run terminates by hitting the `n_estimators` cap (not overfitting) **and** the val curve has plateaued.
  - Always change `eta` and `n_estimators` **together**: halving `eta` should roughly double `n_estimators`.
- **Example:** `eta=0.03, n_estimators=4000` for a rare cell with a noisy val Brier curve at the default rate.

### `max_depth` (↔ CatBoost `depth`)

- **Description:** Maximum depth of each tree. Deeper = more feature interactions modelled per tree.
- **Range/values:** Integer `[1, 16]`. Practical: `4–10`. `0` means no limit (use `lossguide` + `max_leaves` for unbounded-depth leaf-wise growth).
- **Default (v1):** `6`.
- **When to change:**
  - **Lower** (`4–5`) on clear overfit (`train Brier << val Brier`) — shallower trees model fewer interactions.
  - **Lower** on a rare cell — shallow trees generalize better when positives are sparse per leaf.
  - **Raise** (`7–8`) if train and val Brier are both high and close (underfit) with plenty of rows per leaf.
  - Each `+1` roughly doubles tree complexity — search `{4, 5, 6, 7, 8}`, not finer.
- **Example:** `max_depth=4` for the `±50% / 10d` cell where only ~1.5% of rows are positive.

### `lambda` (`reg_lambda`; ↔ CatBoost `l2_leaf_reg`)

- **Description:** L2 regularization on leaf weights. Larger = leaves pulled toward zero, smoother predictions.
- **Range/values:** Float `[0, ∞)`. Practical: `1–30`. Treat as **log-scale**.
- **Default (v1):** `3.0`.
- **When to change:**
  - **Raise** (`5, 10, 20`) on clear overfit when you don't want to drop `max_depth` yet.
  - **Raise** especially on rare cells — fewer positives per leaf means noisier raw leaf weights; L2 pulls them toward the prior.
  - **Lower** (toward `1`) only if both train and val Brier are flat-high and adding capacity hasn't helped.
- **Example:** `lambda=10` when a rare-cell run overfits despite early-stopping early.

---

## Category 2: Regularization (L1 + structural)

XGBoost's regularization levers beyond L2. Try `lambda` first (Category 1); these are the second line.

### `alpha` (`reg_alpha`; L1 — no clean CatBoost analog)

- **Description:** L1 regularization on leaf weights. Drives small leaf weights to exactly zero — a sparsity-inducing regularizer.
- **Range/values:** Float `[0, ∞)`. Practical: `0–10`. Treat as **log-scale**; `0` disables.
- **Default (v1):** `0.0`.
- **When to change:**
  - **Raise** (`0.1, 1, 5`) on overfit when L2 alone hasn't fixed it — L1 prunes weak leaf contributions entirely.
  - Useful when the active feature set is large and you suspect many features contribute near-zero signal.
  - **Lower** to `0` (the default) when underfitting.
- **Example:** `alpha=1.0, lambda=10` to combine L1+L2 on a stubbornly overfit common cell.

### `gamma` (`min_split_loss`; min loss reduction to split)

- **Description:** Minimum loss reduction required to make a further partition on a leaf node. Larger = more conservative (fewer splits), a structural regularizer.
- **Range/values:** Float `[0, ∞)`. Practical: `0–10`.
- **Default (v1):** `0.0`.
- **When to change:**
  - **Raise** (`0.5, 1, 5`) on overfit to forbid low-gain splits — complements `max_depth` by pruning at the split level rather than the depth level.
  - **Lower** to `0` when underfitting (allow all gainful splits).
- **Example:** `gamma=1.0` on a rare cell where deep trees keep making marginal splits that don't generalize.

### `min_child_weight` (↔ CatBoost `min_data_in_leaf`, loosely)

- **Description:** Minimum sum of instance Hessian (≈ weighted sample count) required in a child. Larger = forced averaging over more rows = smoother predictions. (For `binary:logistic` the per-row Hessian is `p(1-p) ≤ 0.25`, so this is *not* a raw row count — it's Hessian-weighted; on confident regions the effective row count per unit is higher.)
- **Range/values:** Float `[0, ∞)`. Practical: `1–100`.
- **Default (v1):** `1.0`.
- **When to change:**
  - **Raise** (`5, 20, 50`) on rare cells — prevents leaves dominated by one or two positives.
  - **Raise** when overfit persists and `max_depth`/`lambda` haven't fixed it.
  - **Lower** toward `0–1` when underfitting on a common cell.
- **Example:** `min_child_weight=20` on the `±50% / 10d` cell.

---

## Category 3: Tree growth

### `grow_policy`

- **Description:** How each tree is grown.
- **Range/values:** `depthwise` | `lossguide`.
  - `depthwise`: split nodes closest to the root first (level-wise) — the default, well-regularized.
  - `lossguide`: split the leaf with the highest loss reduction first (LightGBM-style leaf-wise) — most flexible, easiest to overfit; requires `max_leaves` to bound it.
- **Default (v1):** `depthwise`.
- **When to change:**
  - Try **`lossguide`** only with strong regularization (`max_leaves` capped, `min_child_weight` raised, `lambda`/`gamma` up) and only on common cells with plenty of positives. Avoid on rare cells.
  - Default is best most of the time — switching is a deliberate experiment, not a routine knob.
- **Note:** values are **lowercase** (`depthwise`/`lossguide`) — unlike CatBoost's CamelCase. The agent must use the XGBoost casing.
- **Example:** `grow_policy=lossguide, max_leaves=16, min_child_weight=50` for a regularized leaf-wise experiment.

### `max_leaves`

- **Description:** Maximum number of leaves. `0` = no limit. Most meaningful with `grow_policy=lossguide`.
- **Range/values:** Integer `[0, 64]` practical (`0` = unbounded).
- **Default (v1):** `0` (no explicit cap; `max_depth` bounds the tree under `depthwise`).
- **When to change:**
  - **Set** (`8–16`) when using `lossguide` and seeing overfit; **raise** (`40–64`) when underfitting under `lossguide`.
  - Leave at `0` under `depthwise`.
- **Example:** `grow_policy=lossguide, max_leaves=16`.

### `max_bin` (↔ CatBoost `border_count`)

- **Description:** Number of quantization bins per numerical feature (only the `hist` family uses binning). Higher = finer splits, slower, marginally higher quality ceiling.
- **Range/values:** Integer `[2, 65535]`. Practical: `64–256`.
- **Default (v1):** `256`.
- **When to change:**
  - **Note:** with the **pinned `tree_method=exact`** (§ Determinism), `max_bin` has **no effect** — `exact` does not bin. It is kept tunable for forward-compatibility (if a future phase relaxes the `tree_method` pin) but is a no-op under the current determinism pins. Don't tune it in v1.2.
- **Example:** *(no-op under `tree_method=exact`)*.

---

## Category 4: Sampling

Stochastic regularization via row/column subsampling. Under the pinned single-thread `exact` recipe these are seeded by `seed` and therefore reproducible (§ Determinism).

### `subsample` (↔ CatBoost `subsample` under Bernoulli)

- **Description:** Fraction of training rows sampled (without replacement) for each tree.
- **Range/values:** Float `(0, 1]`. Practical: `0.5–0.95`.
- **Default (v1):** `0.8`.
- **When to change:**
  - **Lower** (`0.5–0.7`) when overfitting — classic bagging-style regularization.
  - **Raise** toward `0.9–0.95` when underfitting.
- **Example:** `subsample=0.7` for an overfit-prone rare cell.

### `colsample_bytree` (↔ CatBoost `rsm`)

- **Description:** Fraction of features sampled per tree.
- **Range/values:** Float `(0, 1]`. Practical: `0.5–1.0`.
- **Default (v1):** `1.0`.
- **When to change:**
  - **Lower** (`0.7–0.9`) when 1–2 features dominate every tree — forces tree diversity.
  - **Lower** when overfit persists despite L2/depth changes, or when the feature count is large (>50).
  - **Raise** to `1.0` if underfitting or FS has already cut the set tight (<10 features).
- **Example:** `colsample_bytree=0.7` once FS settles on ~30 features.

### `colsample_bylevel` / `colsample_bynode`

- **Description:** Feature subsampling per tree *level* / per *split node* respectively. Multiplicative with `colsample_bytree`.
- **Range/values:** Float `(0, 1]`. Practical: `0.5–1.0`.
- **Default (v1):** `1.0` each.
- **When to change:**
  - Use `colsample_bylevel` as a finer-grained alternative to `colsample_bytree` for tree diversity; `colsample_bynode` is the most aggressive (resamples at every split).
  - Tune at most one of the three at a time — they compound.
- **Example:** `colsample_bylevel=0.8` when `colsample_bytree` alone hasn't decorrelated dominant features enough.

### `sampling_method`

- **Description:** Row sampling scheme.
- **Range/values:** `uniform` | `gradient_based`.
  - `uniform`: each row equally likely (requires `subsample ≥ 0.5` for stability).
  - `gradient_based`: rows sampled ∝ gradient magnitude — allows much lower `subsample` without quality loss (GPU `hist` only in practice).
- **Default (v1):** `uniform`.
- **When to change:**
  - Keep `uniform` under the pinned CPU `exact` recipe; `gradient_based` is a GPU-`hist` feature and is not used while training is pinned to CPU/`exact`.
- **Example:** *(leave at `uniform` in v1.2)*.

---

## Category 5: Class balancing

For rare cells. XGBoost's binary convenience knob is `scale_pos_weight` (it has no `auto_class_weights` equivalent).

### `scale_pos_weight`

- **Description:** Multiplier on the positive-class gradient — the binary analog of up-weighting positives.
- **Range/values:** Float `(0, ∞)`. Practical: `1–50`.
- **Default (v1):** `1.0`.
- **When to change:**
  - **Set** when a rare cell shows near-zero positive recall early ("predict 0 forever").
  - Start at `(1 - p) / p` (inverse-prevalence) and search log-spaced around it (`{0.5x, 1x, 2x}`).
  - **Calibration warning:** any positive up-weighting biases probabilities. Calibration is the headline metric, so **expect to re-calibrate** (the conditional-isotonic / Platt path runs post-training regardless and is backend-neutral).
- **Example:** `scale_pos_weight=12.0` for a ~7.5% positive-rate cell (inverse-prevalence ≈ 12.3); sweep `{6, 12, 24}`.

---

## Category 6: Early stopping

Don't run to the `n_estimators` cap blindly — let validation tell you when to stop.

### `early_stopping_rounds`

- **Description:** Stop training if `eval_metric` (pinned `logloss`) hasn't improved on the validation set for this many rounds. Returns the best iteration (`best_iteration`).
- **Range/values:** Integer `[1, ∞)`. Practical: `20–200`.
- **Default (v1):** `75`.
- **When to change:**
  - **Always set** in v1.2. `50–100` is reasonable for our cell sizes.
  - **Raise** (`100–200`) if you've also lowered `eta` — slower learning means longer plateaus before improvement.
  - **Lower** (`20–30`) only for cheap exploration sweeps.
- **Example:** `early_stopping_rounds=75` paired with `n_estimators=3000, eta=0.03`.

---

## Determinism (PINNED — never tunable)

Walk-forward + agent-driven iteration means **runs must be bit-reproducible**: the loop's finalization step *retrains* the best `(features, hp)` config and assumes byte-identical reproduction of the in-loop fit (the checkpoint stores no model blob). XGBoost reproducibility requires the **whole** recipe below; these are pinned in `PINNED_HPS_XGB` (the XGBoost analog of CatBoost's `has_time` pin) and are **never overridable** from a spec or an agent decision:

| Pin | Value | Why |
|---|---|---|
| `objective` | `binary:logistic` | the binary objective (↔ CatBoost `Logloss`) |
| `eval_metric` | `logloss` | early-stop metric; Brier is computed in the diagnostic bundle regardless |
| `tree_method` | `exact` | `hist` is order-dependent across threads |
| `n_jobs` | `1` | multi-thread float-reduction order varies run-to-run |
| `device` | `cpu` | GPU floating-point reductions are non-deterministic |
| `seed` | run's `random_seed` (default 42) | set at construction, like CatBoost's `random_seed` — all internal randomness (subsampling, column sampling) is seeded by it |

We deliberately **forgo XGBoost's GPU / multi-thread `hist` / Dask training speedups** inside the loop — correctness (the bit-identical retrain) beats wall-clock for a v1.2 analysis backend. (GPU is fine for the read-only interaction *diagnostic*, which never retrains — a separate code path.)

> **Phasing note (V1.2):** Phase 2 ships the pin *table data* (so decision validation rejects an agent request to change `tree_method`/`n_jobs`/`device`/`objective`). The **construction-time hard-fail** that raises when a non-deterministic override is passed to the XGBoost model (`_validate_hp_xgb`) lands in **Phase 3** along with the determinism CI test. Until then there is no XGBoost model adapter to construct.

---

## Loss and evaluation

- **`objective` (pinned `binary:logistic`):** the right choice for hard-labeled `{0,1}` binary classification; raw margin → probability via sigmoid. Don't change.
- **`eval_metric` (pinned `logloss`):** XGBoost's built-in metric list (`logloss`, `error`, `auc`, `rmse`) has **no native "BrierScore"** (`rmse` on probabilities = √Brier for binary, but isn't pinned). We pin `logloss` for early stopping (calibration-aligned, smooth) and compute **Brier in the diagnostic bundle** backend-neutrally, exactly as the CatBoost path does. So the headline calibration metric is unchanged across backends; only the early-stop signal differs (`logloss` vs CatBoost's `BrierScore`), which is immaterial at the cell sizes here.

---

## Suggested per-iteration agent rubric

Mirrors the CatBoost rubric, with XGBoost names. Apply after reading the iteration's diagnostic bundle (train/val Brier curves, reliability plot, calibration error, positive recall, top-feature importance):

1. **Read the bundle.** train Brier, val Brier, the gap, reliability deviation, val-curve shape, early-stop iteration, top-5 importances, positive recall at 0.5.
2. **Overfit** (`train Brier << val Brier`, over-confident tails):
   - First: raise `lambda` (`2x`).
   - If insufficient: raise `min_child_weight`, lower `max_depth`, raise `gamma`, or set `colsample_bytree=0.7`.
   - If still: lower `subsample` to `0.7`, or add `alpha` (L1).
   - Last resort: drop features in FS.
3. **Underfit** (`train ≈ val`, both high; flat curve):
   - First: raise `max_depth` (`+1`, max `8`).
   - If insufficient: raise `n_estimators` and lower `eta` proportionally.
   - If still: add features back, or try `grow_policy=lossguide` + `max_leaves`.
4. **Rare cell + low recall** (positives <5% AND recall ~0):
   - Set `scale_pos_weight=(1-p)/p`; search `{0.5x, 1x, 2x}`.
   - **Always re-check calibration** after weighting.
5. **Learning curve still descending at the cap** (best iter within 10% of `n_estimators`):
   - Raise `n_estimators` (`2x`) **and** `early_stopping_rounds` proportionally, or lower `eta` and raise `n_estimators` together.
6. **One-or-two-feature dominance** (top importance >50%, val plateaued):
   - Set `colsample_bytree=0.7` (or `colsample_bylevel`) to decorrelate trees.
   - Consider whether the dominant feature is leaking.
7. **Calibration drift in tails:**
   - Inspect `scale_pos_weight` — up-weighting often causes this; reduce it or post-hoc isotonic/Platt-calibrate.
   - Lower `max_depth` (smoother probabilities).
8. **Always pinned, never tuned:** `objective=binary:logistic`, `eval_metric=logloss`, `tree_method=exact`, `n_jobs=1`, `device=cpu`, `seed=42` (production). There is **no `has_time`** — C6 rests on the split discipline (§ "C6 and the missing `has_time`").
9. **Record rationale per change.** Every HP change must be logged with the diagnostic signal that triggered it.

---

## Notes on defaults

- `xgboost` was **not** added as a dependency in V1.2 Phase 2 (it lands with the model adapter in a later phase). The defaults above are sourced from the XGBoost parameter docs (https://xgboost.readthedocs.io/en/stable/parameter.html) and `project-xgboost-training-essentials`; re-verify against the installed version's resolved params on the first production run, as the CatBoost doc cautions.
- `eta` / `learning_rate`, `lambda` / `reg_lambda`, `alpha` / `reg_alpha`, `gamma` / `min_split_loss` are aliases in XGBoost; the gbdt tables use the short forms (`eta`, `lambda`, `alpha`, `gamma`). The agent should use the short forms the validation table recognizes.
- The **iteration-0 starting HPs** for an xgboost cell: `n_estimators=1000, eta=0.05, max_depth=6, lambda=3.0, subsample=0.8, colsample_bytree=1.0, grow_policy=depthwise, early_stopping_rounds=75` (V1.2 plan § 6.3).

## Sources

- XGBoost parameters reference: https://xgboost.readthedocs.io/en/stable/parameter.html
- XGBoost Python API (`XGBClassifier` / native `Booster`): https://xgboost.readthedocs.io/en/stable/python/python_api.html
- `docs/gbdt/V1.2_xgboost_feature_interactions_plan.md` — D3 (sibling HP tables), § 5.1 (determinism), § 5.3 (C6 without `has_time`), § 6.3 (iteration-0 HPs).
- `.claude/memories/project-xgboost-training-essentials.md` — the determinism contract + the CatBoost↔XGBoost name map.
- `docs/gbdt/CATBOOST_HP_REFERENCE.md` — the sibling doc this mirrors.

Fetched 2026-05. XGBoost was not installed in the project's venv at time of writing (the dep lands in the model-adapter phase), so defaults were sourced from the canonical docs above.
