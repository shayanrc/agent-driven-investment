# CatBoost Tunable Parameters — gbdt v1 reference

## Scope

This is the per-iteration tuning reference the gbdt v1 agent will consult when deciding which CatBoost hyperparameters to change between FS+HP iterations. It is scoped tightly to **our use case**:

- **Task:** binary classification — one binary target per cell (e.g. "did stock close ≥10% above entry within 20 trading days").
- **Headline metric:** **calibration** (Brier score, reliability curves), *not* AUC / accuracy / F1.
- **Data shape:** ~1,500 rows/stock × ~48 stocks pooled per cell → ~70k rows. Small-data regime for boosting.
- **Class imbalance:** several cells are rare (±50% in 10 days fires <2% of rows). Class weighting matters.
- **Splits:** strictly **walk-forward** — never shuffled CV. `has_time=True` is mandatory.
- **Features:** no categoricals in v1 (day-of-week / month-of-year are sin/cos float pairs). No text, no embeddings.
- **Hardware:** single-machine CPU is the baseline. GPU is optional.

**What is in this doc:** every CatBoost parameter the agent is allowed to tune, grouped by mechanism, with a per-parameter "when to change" decision rule keyed off the kinds of diagnostic signals the v1 pipeline produces.

**What is *not* in this doc (and why):**

- `cat_features`, `one_hot_max_size`, `simple_ctr`, `combinations_ctr`, `ctr_target_border_count`, `ctr_complexity`, `counter_calc_method` — we have **no categorical features** in v1. If v2 introduces them, add a new category here.
- `text_features`, `text_processing`, `tokenizers`, `dictionaries`, `feature_calcers` — no text features.
- `embedding_features` — no embedding features.
- `task_type`, `devices`, `gpu_ram_part`, `pinned_memory_size`, `gpu_cat_features_storage` — GPU is optional and infra, not a "tune for quality" knob. Pick once per machine.
- `thread_count`, `used_ram_limit` — infra throttles, not quality knobs.
- `verbose`, `logging_level`, `metric_period`, `train_dir`, `name`, `json_log`, `profile_file` — diagnostics / IO, not tuning.
- `snapshot_file`, `snapshot_interval`, `save_snapshot`, `allow_writing_files` — checkpointing infra.
- `approx_on_full_history`, `ensemble_size`, `model_shrink_rate`, `model_shrink_mode`, `posterior_sampling`, `langevin`, `diffusion_temperature` — niche / experimental; not in v1 search space.
- `fold_permutation_block`, `fold_len_multiplier` — Ordered-boosting permutation internals. Don't touch unless you have a specific reason.
- `nan_mode` — we don't have missing values in v1 features by construction; defaults are fine.

If a CatBoost parameter doesn't appear in this doc, the agent should **not** touch it.

---

## Category 1: Core boosting controls

These four parameters determine the capacity, learning speed, and complexity of every tree. They are the first knobs the agent should reach for.

### `iterations`

- **Description:** Maximum number of boosting rounds (trees) the model can build. Early stopping can cut this short.
- **Range/values:** Integer `[1, ∞)`. Practical: `100–10000`.
- **Default:** `1000`.
- **When to change:**
  - **Raise** if early-stopping fires close to `iterations` (e.g. best iter is within 10% of the cap) — the model wanted more trees than you allowed.
  - **Raise** if the validation learning curve is still trending down at the final iteration.
  - **Lower** only if wall-clock matters and you have headroom on quality; otherwise just let early stopping handle it.
- **Example:** `iterations=3000` once a previous run early-stopped at ~950 of 1000 — give the optimizer more rope before lowering `learning_rate`.

### `learning_rate`

- **Description:** Shrinkage factor applied to each tree's leaf updates. Smaller values force more, smaller corrections per tree.
- **Range/values:** Float `(0, 1]`. Practical: `0.01–0.3`. Treat as **log-scale** when searching.
- **Default:** Auto-derived from `iterations`, dataset size, and `loss_function` (typically lands around `0.03–0.1` for our cell size).
- **When to change:**
  - **Lower** (e.g. halve) if validation Brier oscillates iteration-to-iteration, or if you see large gap open between train Brier and val Brier early on.
  - **Lower** if you raised `iterations` and want the model to use them.
  - **Raise** if every run terminates by reaching the `iterations` cap (not by overfitting) **and** the val curve has plateaued — you're stepping too small.
  - Always change `learning_rate` and `iterations` **together**: halving `learning_rate` should roughly double `iterations` to keep total capacity comparable.
- **Example:** `learning_rate=0.03, iterations=4000` when a rare cell (`±50% / 10d`) shows a noisy val Brier curve at the default rate.

### `depth`

- **Description:** Depth of each oblivious tree (`SymmetricTree` policy). For `Lossguide` / `Depthwise` it caps tree depth differently — see `grow_policy`.
- **Range/values:** Integer `[1, 16]` on CPU. Practical: `4–10`.
- **Default:** `6` (or `16` if `grow_policy=Lossguide`).
- **When to change:**
  - **Lower** (e.g. `4–5`) if you see clear overfit (`train Brier << val Brier`) — shallower trees model fewer feature interactions.
  - **Lower** when training on a rare cell — shallow trees generalize better when positives are sparse per leaf.
  - **Raise** (e.g. `7–8`) if train Brier and val Brier are both high and close together (underfit) and you have plenty of rows per leaf.
  - Each `+1` roughly doubles tree complexity — search the grid `{4, 5, 6, 7, 8}`, not finer.
- **Example:** `depth=4` for the `±50% / 10d` cell where only ~1.5% of rows are positive, so deep trees memorize positive patterns.

### `l2_leaf_reg`

- **Description:** L2 regularization coefficient on leaf values. Larger = leaves are pulled toward zero, smoother predictions.
- **Range/values:** Float `[0, ∞)`. Practical: `1–30`. Treat as **log-scale**.
- **Default:** `3.0`.
- **When to change:**
  - **Raise** (`5, 10, 20`) when overfit signal is clear (train Brier << val Brier) and you don't want to drop `depth` yet.
  - **Raise** especially in rare cells — fewer positives per leaf means raw leaf values are noisier; L2 pulls them toward the prior.
  - **Lower** (toward `1`) only if both train and val Brier are flat-high and adding capacity (`depth`, `iterations`) hasn't helped — the model may be over-smoothed.
- **Example:** `l2_leaf_reg=10` when rare-cell experiment overfits despite hitting `od_wait` early-stop cap.

---

## Category 2: Regularization (stochastic + structural)

Anything that injects noise or shrinks structure to fight overfitting. Try `l2_leaf_reg` first (Category 1); these are the second line.

### `random_strength`

- **Description:** Standard deviation of Gaussian noise added to split score when scoring candidate splits. Larger = more random split selection.
- **Range/values:** Float `[0, ∞)`. Practical: `0.5–10`.
- **Default:** `1`.
- **When to change:**
  - **Raise** (`2–5`) if you see persistent overfit and L2 alone hasn't fixed it — randomizing split choice decorrelates trees.
  - **Raise** when feature-importance plots show one or two features dominate every tree (the model latched onto them); randomization gives weaker features a chance.
  - **Lower** toward `0.5` when underfitting; less noise = sharper split decisions.
- **Example:** `random_strength=3` when the same `lag5_return` feature wins the top split in >80% of trees and val Brier is plateaued.

### `bagging_temperature`

- **Description:** Controls Bayesian bootstrap weight variance. Active only when `bootstrap_type=Bayesian` (the default for classification).
- **Range/values:** Float `[0, ∞)`. `0` disables (all weights = 1). Practical: `0–10`.
- **Default:** `1`.
- **When to change:**
  - **Raise** (`3–10`) for stronger row reweighting → more bagging-style regularization. Use when you see overfit.
  - **Lower** to `0` to disable Bayesian bootstrap (equivalent to no row sampling). Use as a diagnostic to isolate whether bagging is helping.
  - Don't tune simultaneously with `subsample` — switch `bootstrap_type` first.
- **Example:** `bagging_temperature=5` (still `bootstrap_type=Bayesian`) when you want more row noise without switching to Bernoulli.

### `subsample`

- **Description:** Fraction of rows sampled per tree (or per level — see `sampling_frequency`). **Only active when `bootstrap_type` is `Bernoulli`, `Poisson`, or `MVS`** — ignored under `Bayesian`.
- **Range/values:** Float `(0, 1]`. Practical: `0.5–0.95`.
- **Default:** `1.0` for datasets < 100 rows; for ≥100 rows: `0.66` (Bernoulli/Poisson) or `0.8` (MVS).
- **When to change:**
  - **Lower** (`0.5–0.7`) when overfitting and you've switched to `Bernoulli` or `MVS` bootstrap.
  - **Raise** toward `0.9–0.95` when underfitting — every tree sees more of the data.
  - For our ~70k row pooled datasets the default `0.8` (MVS) is a sensible starting point.
- **Example:** `bootstrap_type=Bernoulli, subsample=0.7` for an overfit-prone rare cell — together they emulate classic bagging.

### `bootstrap_type`

- **Description:** Row sampling scheme used at each iteration.
- **Range/values:** `Bayesian` | `Bernoulli` | `MVS` | `Poisson` (GPU only) | `No`.
- **Default:** `Bayesian` for classification on CPU.
- **When to change:**
  - **`Bernoulli`** when you want hard subsampling controlled by `subsample` — most interpretable; classical bagging.
  - **`MVS`** when you have a lot of rows and want gradient-aware sampling (often modestly better at fixed compute).
  - **`No`** only as a diagnostic — disables all row sampling, useful to isolate the contribution of bagging to your val Brier.
  - **`Bayesian`** is the safe default; tune `bagging_temperature` instead of switching.
- **Example:** `bootstrap_type=MVS, subsample=0.8` for a pooled common-cell run where you want a quality bump over Bayesian without losing speed.

### `sampling_frequency`

- **Description:** When the row subsample is redrawn.
- **Range/values:** `PerTree` | `PerTreeLevel`.
- **Default:** `PerTreeLevel`.
- **When to change:**
  - **`PerTree`** for a slightly faster, slightly less-regularized run (one resample per tree). Try this if total runtime is hurting and you have spare regularization budget elsewhere.
  - Otherwise leave at `PerTreeLevel`.
- **Example:** `sampling_frequency=PerTree` only if you're profiling a slow run and want to shave time without losing meaningful quality.

### `mvs_reg`

- **Description:** Regularization for `MVS` (Minimal Variance Sampling) bootstrap. Active only when `bootstrap_type=MVS`. Controls how aggressively low-gradient rows are downsampled.
- **Range/values:** Float `[0, ∞)`.
- **Default:** Auto-set per iteration from the gradient distribution.
- **When to change:**
  - Almost never. Override only if you have a specific reproducibility need; otherwise CatBoost's auto-value is well-tuned.
  - If you do override, try `0.01–1.0` log-scale.
- **Example:** `mvs_reg=0.1` only when pinning a result for an ablation that requires deterministic sampling behaviour.

### `model_size_reg`

- **Description:** Penalty on model size (mostly meaningful when categorical-feature combinations balloon the model). Effectively a no-op for our v1 (no categoricals) but cheap to leave at default.
- **Range/values:** Float `[0, ∞)`.
- **Default:** Varies by task (often `0.5`); has minimal effect without categorical CTRs.
- **When to change:**
  - Don't tune in v1. Note in v2 if categorical features are introduced.
- **Example:** *(skip — not tuned in v1)*

---

## Category 3: Tree growth

Structure of each individual tree. Default policy is `SymmetricTree` (oblivious trees), which is what makes CatBoost CatBoost — fast inference and built-in regularization.

### `grow_policy`

- **Description:** How each tree is grown.
- **Range/values:** `SymmetricTree` | `Depthwise` | `Lossguide`.
  - `SymmetricTree`: every node at a given depth uses the same split — oblivious trees. Fast, regularized, default.
  - `Depthwise`: like classical decision trees; splits levels uniformly but per-node splits can differ.
  - `Lossguide`: leaf-wise growth (LightGBM-style); grows the leaf that most reduces loss. Most flexible, easiest to overfit.
- **Default:** `SymmetricTree`.
- **When to change:**
  - Try **`Depthwise`** when symmetric trees underfit on a feature interaction you know exists (e.g. you added a non-obvious cross-feature and val Brier didn't move).
  - Try **`Lossguide`** only with strong regularization (`min_data_in_leaf`, `l2_leaf_reg`, capped `max_leaves`) and only on common cells with plenty of positives. Avoid on rare cells.
  - Default is best 80% of the time — change is a deliberate experiment, not a routine knob.
- **Example:** `grow_policy=Depthwise, depth=6, l2_leaf_reg=5` when ablating whether symmetry is costing you on a cell with known interaction structure.

### `max_leaves`

- **Description:** Maximum number of leaves per tree. **Only active when `grow_policy=Lossguide`.**
- **Range/values:** Integer `[2, 64]` practical; up to `2^depth`.
- **Default:** `31`.
- **When to change:**
  - **Lower** (`8–16`) when using Lossguide and seeing overfit.
  - **Raise** (`40–64`) when using Lossguide and underfitting.
  - Ignore entirely if `grow_policy=SymmetricTree`.
- **Example:** `grow_policy=Lossguide, max_leaves=16, min_data_in_leaf=50` for a regularized leaf-wise experiment on a common cell.

### `min_data_in_leaf`

- **Description:** Minimum number of training samples required in a leaf. Larger = forced averaging over more rows = smoother predictions.
- **Range/values:** Integer `[1, ∞)`. Practical: `1–500`.
- **Default:** `1`.
- **When to change:**
  - **Raise** (`20, 50, 100`) for rare cells — prevents leaves containing one or two positives that overfit.
  - **Raise** when overfit signal persists and `depth`/`l2_leaf_reg` haven't fixed it.
  - **Lower** toward `1` only when underfitting on a common cell and you want maximum flexibility.
  - For Lossguide policy, tune this together with `max_leaves`.
- **Example:** `min_data_in_leaf=50` on the `±50% / 10d` cell — forces each leaf to average over enough rows that one positive doesn't dominate.

### `rsm` (a.k.a. `colsample_bylevel`)

- **Description:** Random feature subsampling — fraction of features considered at each split. Equivalent to scikit-learn's `colsample_bylevel`.
- **Range/values:** Float `(0, 1]`. Practical: `0.5–1.0`.
- **Default:** `1.0` (all features considered).
- **When to change:**
  - **Lower** (`0.7–0.9`) when feature importance is dominated by 1–2 features — forces tree diversity.
  - **Lower** when overfitting persists despite L2 / depth changes.
  - **Lower** automatically when feature count grows large (>50) — gives weaker features a chance to shine.
  - **Raise** to `1.0` if underfitting or if FS has already cut the feature set tight (<10 features).
- **Example:** `rsm=0.7` once FS has settled on ~30 features and dominant features are crowding out useful weak ones.

### `border_count` (a.k.a. `max_bin`)

- **Description:** Number of quantization bins per numerical feature. Higher = finer splits, slower, slightly higher quality ceiling.
- **Range/values:** Integer `[1, 65535]`. Practical: `64–254`.
- **Default:** `254` on CPU.
- **When to change:**
  - **Lower** (`64–128`) only if you're CPU-bound and want runtime back. Quality cost is usually small.
  - **Keep at `254`** for our use case — features are small in count and quantization is cheap.
  - **Raise** above 254 only for GPU runs where you have headroom and need extra resolution; rarely worth it.
- **Example:** `border_count=128` only when runtime is the blocker (e.g. sweeping across 48 stocks × many cells).

---

## Category 4: Sampling (see also Category 2)

Sampling knobs that already affect regularization are listed in Category 2 (`bagging_temperature`, `subsample`, `bootstrap_type`, `sampling_frequency`, `mvs_reg`). This section is intentionally cross-referenced — sampling and regularization are the same lever in practice. Pick one mechanism, tune it, then move on.

---

## Category 5: Class balancing

Critical for rare cells (e.g. `±50% / 10d` firing <2%). At most one of `class_weights`, `auto_class_weights`, or `scale_pos_weight` should be active per run.

### `class_weights`

- **Description:** Per-class weights applied to the loss. Indexed by class label.
- **Range/values:** `list`, `dict`, or `OrderedDict`. For binary: `[1.0, k]` where `k` is the up-weight on positives.
- **Default:** `None` (all classes weight `1`).
- **When to change:**
  - **Set** when a rare cell shows zero or near-zero positives in the early validation iterations — model is "predict 0 forever."
  - For a cell with positive prevalence `p`, a reasonable starting weight on positives is `(1-p)/p` (i.e. inverse-prevalence). Equivalent to `auto_class_weights=Balanced`.
  - Use `class_weights` explicitly when you want a *milder* up-weight than `Balanced` (e.g. `(1-p)/p / 2`) — full inverse-prevalence can over-correct and destroy calibration.
- **Example:** `class_weights=[1.0, 10.0]` for a cell with ~10% positives where Balanced (which would set `9.0`) is hurting calibration.

### `auto_class_weights`

- **Description:** Automatic class weighting scheme.
- **Range/values:** `None` | `Balanced` | `SqrtBalanced`.
  - `Balanced`: weight ∝ `1/class_freq` — full inverse-prevalence.
  - `SqrtBalanced`: weight ∝ `1/sqrt(class_freq)` — milder.
- **Default:** `None`.
- **When to change:**
  - **`SqrtBalanced`** is the safe default for rare cells — gives the rare class a boost without the calibration damage of full inverse-prevalence.
  - **`Balanced`** when even SqrtBalanced isn't enough (e.g. cell has <1% positives and recall is ~0).
  - **`None`** for common cells (e.g. `±5% / 20d` with 40%+ prevalence). Don't apply class weights to roughly balanced data.
  - **Calibration warning:** any class weighting biases probabilities. If calibration is the headline metric, **expect to re-calibrate** (Platt / isotonic) after training, or compare weighted vs unweighted with calibration explicitly in the eval.
- **Example:** `auto_class_weights=SqrtBalanced` on the `±20% / 20d` cell where positives are ~8%.

### `scale_pos_weight`

- **Description:** Multiplier on the positive-class weight. Binary-only convenience equivalent to `class_weights=[1, scale_pos_weight]`.
- **Range/values:** Float `(0, ∞)`. Practical: `1–50`.
- **Default:** `1.0`.
- **When to change:**
  - Prefer over `class_weights` when you just want one tunable scalar in the grid search.
  - Start at `(1 - p) / p` and search log-spaced around it (e.g. `{0.5x, 1x, 2x}`).
- **Example:** `scale_pos_weight=12.0` for a 7.5% positive rate cell (inverse-prevalence ≈ 12.3); grid would then sweep `{6, 12, 24}`.

---

## Category 6: Early stopping

Don't run boosting to the `iterations` cap blindly — let validation tell you when to stop. Pick one of `od_type=Iter` (simple, recommended) or `od_type=IncToDec`. Don't mix both.

### `early_stopping_rounds`

- **Description:** Convenience knob in `fit()` that sets `od_type=Iter` and `od_wait=early_stopping_rounds`. Stops training if `eval_metric` hasn't improved on the validation set for this many rounds.
- **Range/values:** Integer `[1, ∞)`. Practical: `20–200`.
- **Default:** `False` (disabled).
- **When to change:**
  - **Always set** in v1. A reasonable default is `50–100` for our cell sizes.
  - **Raise** (`100–200`) if you've also lowered `learning_rate` — slower learning means longer plateaus before improvement.
  - **Lower** (`20–30`) only for cheap exploration sweeps where compute matters more than the last 0.001 of Brier.
- **Example:** `early_stopping_rounds=75` paired with `iterations=3000, learning_rate=0.03`.

### `od_type`

- **Description:** Overfitting-detector algorithm.
- **Range/values:** `IncToDec` | `Iter`.
  - `Iter`: stop if `eval_metric` hasn't improved for `od_wait` rounds. Simple, deterministic.
  - `IncToDec`: probabilistic — stops when p-value of "no improvement" falls below `od_pval`. More aggressive.
- **Default:** `IncToDec` when overfitting detection is active; auto-switched to `Iter` if you set `early_stopping_rounds`.
- **When to change:**
  - **`Iter`** is recommended for our use case — its behaviour is predictable across cells, which matters for an agent doing systematic comparisons.
  - **`IncToDec`** only if you want CatBoost's adaptive stop. Don't combine with `early_stopping_rounds`.
- **Example:** `od_type=Iter, od_wait=75` (or just `early_stopping_rounds=75`).

### `od_pval`

- **Description:** p-value threshold for `IncToDec`. Smaller = stricter (waits longer before stopping).
- **Range/values:** Float `[0, 1]`. Recommended `1e-10` to `1e-2`.
- **Default:** `0` (overfitting detector off — must set explicitly to enable `IncToDec`).
- **When to change:**
  - Only relevant when `od_type=IncToDec`. Start at `1e-3`. Lower to `1e-5` if you want it to wait longer.
- **Example:** `od_type=IncToDec, od_pval=1e-3` for a comparison run against `od_type=Iter`.

### `od_wait`

- **Description:** With `od_type=Iter`, the number of iterations without improvement before stopping. With `IncToDec`, the minimum number of iterations before the detector activates.
- **Range/values:** Integer `[1, ∞)`. Practical: `20–200`.
- **Default:** `20`.
- **When to change:**
  - Same logic as `early_stopping_rounds` (they're aliases when `od_type=Iter`).
- **Example:** `od_type=Iter, od_wait=75`.

---

## Category 7: Determinism

Walk-forward + agent-driven iteration means **runs must be reproducible**. Fix the seed; fix the time ordering.

### `random_seed`

- **Description:** Seed for all internal randomness (subsampling, split scoring noise, permutations).
- **Range/values:** Integer. Any non-negative int.
- **Default:** `None` in Python (effectively `0`).
- **When to change:**
  - **Always set explicitly** for v1. Recommend `random_seed=42` (or any fixed int) across all production runs of a cell.
  - **Sweep** (e.g. `{0, 1, 2, 3, 4}`) only when you need a *seed-variance estimate* — i.e., reporting Brier ± std across seeds. Don't tune the seed.
- **Example:** `random_seed=42` for all production runs; `random_seed in {0,...,4}` for variance bars.

### `has_time`

- **Description:** Tells CatBoost to use the natural row order as the time axis. Disables row permutation in Ordered boosting and CTR computation.
- **Range/values:** `True` | `False`.
- **Default:** `False`.
- **When to change:**
  - **MUST be `True`** for our pipeline. Walk-forward demands that no row at time `t+k` ever influences a feature value or split decision at time `t`. With `has_time=False`, CatBoost shuffles rows for Ordered boosting and for target-statistic CTRs.
  - Treat as a hard correctness constraint, **not a tunable**.
- **Example:** `has_time=True` — non-negotiable.

---

## Category 8: Loss and evaluation metrics

Loss is what the model optimizes; eval metric is what early stopping and best-model selection use. **For our calibration-headlined use case, both should be calibration-aware.**

### `loss_function`

- **Description:** Objective the boosting algorithm minimizes.
- **Range/values:** For binary classification: `Logloss` | `CrossEntropy`.
  - `Logloss`: target must be `{0, 1}`. Standard binary cross-entropy.
  - `CrossEntropy`: target may be a probability in `[0, 1]` — used for soft labels. We have hard `{0, 1}` labels, so `Logloss` is correct.
- **Default:** `Logloss` (for `CatBoostClassifier` with binary target).
- **When to change:**
  - **Don't.** `Logloss` is the right choice for hard-labeled binary classification and is well-aligned with calibration.
  - Switch to `CrossEntropy` only if v2 introduces soft labels (e.g. probability-of-fire from a meta-labeler).
- **Example:** `loss_function="Logloss"` — pinned.

### `eval_metric`

- **Description:** Metric computed on validation data; used by early stopping and `use_best_model`. Can differ from `loss_function`.
- **Range/values:** Many supported. For our use case the candidates are:
  - `Logloss`: same as the loss; smooth, calibration-aligned.
  - `BrierScore`: direct calibration metric. **Recommended as headline.**
  - `AUC`: ranking; **not calibration-aligned** — avoid as the headline.
  - `NormalizedGini`: `2*AUC - 1`; same caveat.
  - `Accuracy`, `Precision`, `Recall`, `F1`, `BalancedAccuracy`: threshold-dependent; not relevant for probability-quality optimization.
- **Default:** Same as `loss_function`.
- **When to change:**
  - **Set `eval_metric="BrierScore"`** when the experiment's headline is calibration — this aligns early stopping and best-model selection with what we actually grade on.
  - Set to `"Logloss"` if you want early stopping to mirror the optimization signal directly (also calibration-friendly; differs from Brier mainly at the extremes of probability).
  - **Don't** use `AUC` here — early stopping on AUC will pick a model with good ranking but possibly miscalibrated probabilities.
- **Example:** `eval_metric="BrierScore"` for production runs; `eval_metric="Logloss"` only when comparing apples-to-apples with the loss.

### `custom_metric`

- **Description:** Additional metrics computed and logged during training. Not used for optimization or early stopping — purely informational.
- **Range/values:** List of metric strings. Same set as `eval_metric`.
- **Default:** `None`.
- **When to change:**
  - **Always log a small diagnostic bundle** so the agent has signal. Recommend: `custom_metric=["Logloss", "BrierScore", "AUC"]`.
  - This way you can see calibration (Brier, Logloss) and ranking (AUC) simultaneously without changing what early stopping optimizes.
- **Example:** `custom_metric=["Logloss", "BrierScore", "AUC"]` for every run.

---

## Category 9: Boosting algorithm

CatBoost's hallmark is *ordered boosting*, which fixes the prediction-shift bias that plain GBDTs suffer from on small data. With ~70k rows per cell we are squarely in the regime where this can matter.

### `boosting_type`

- **Description:** Boosting scheme — Ordered vs Plain.
  - `Ordered`: CatBoost's prediction-shift-corrected boosting. Higher quality on small data; ~2-3x slower per iteration.
  - `Plain`: standard gradient boosting (like XGBoost/LightGBM). Faster, scales better.
- **Range/values:** `Ordered` | `Plain`.
- **Default:** `Plain` on CPU. (On GPU: `Ordered` for ≤50k rows; `Plain` otherwise.)
- **When to change:**
  - **Try `Ordered`** for v1 — our per-cell row counts (~70k) are exactly where Ordered's prediction-shift correction pays off, *and* with our walk-forward setup the small extra cost is worth it.
  - **Stick with `Plain`** if Ordered's runtime makes the sweep budget infeasible, or if Ordered's val Brier is within seed-noise of Plain (i.e., the prediction-shift correction isn't buying anything for that cell).
  - **Always combined with `has_time=True`** so Ordered's permutations respect the time axis.
- **Example:** `boosting_type="Ordered", has_time=True` as the v1 default; benchmark `Plain` once per cell to confirm Ordered's win.

---

## Category 10: Leaf-value estimation

How leaf values are computed once a tree's structure is fixed. Defaults are well-tuned; touch only when you have a reason.

### `leaf_estimation_method`

- **Description:** Numerical method for computing leaf values.
- **Range/values:** `Newton` | `Gradient` | `Exact`.
  - `Newton`: second-order. Default for classification. Most accurate, most expensive.
  - `Gradient`: first-order. Cheaper.
  - `Exact`: closed-form (only for some objectives, e.g. Quantile/MAE; not used for Logloss).
- **Default:** `Newton` for classification (10 iterations).
- **When to change:**
  - **Don't.** Newton is the right choice for Logloss / CrossEntropy.
  - Switch to `Gradient` only as a runtime optimization in compute-bound sweeps; expect a small quality hit.
- **Example:** `leaf_estimation_method="Newton"` — pinned default.

### `leaf_estimation_iterations`

- **Description:** Number of Newton (or Gradient) iterations used to refine each leaf value.
- **Range/values:** Integer `[1, ∞)`. Practical: `1–10`.
- **Default:** Auto — `10` for classification.
- **When to change:**
  - **Lower** to `1–3` when compute-bound and you accept a small quality hit.
  - **Raise** rarely useful — `10` is already near the marginal-returns plateau.
- **Example:** `leaf_estimation_iterations=3` in cheap exploration sweeps.

---

## Category 11: Feature controls

Surgical knobs to inject prior knowledge or strip features. Useful when FS produces a known-bad feature set or when domain knowledge dictates monotonicity.

### `feature_weights`

- **Description:** Per-feature multiplier applied to the split score during split selection. Higher = feature is more likely to be chosen.
- **Range/values:** `list` of floats or `dict[name, weight]`. Length = number of features. Default weight per feature is `1`.
- **Default:** All `1`.
- **When to change:**
  - **Down-weight** (e.g. `0.3`) a feature that FS keeps trying to add but whose contribution looks spurious in the diagnostic bundle.
  - **Up-weight** a feature that domain knowledge says should matter but the model isn't picking up (rare; usually FS is right).
  - **Don't** use this as a substitute for FS — if a feature is bad, drop it via `ignored_features`.
- **Example:** `feature_weights={"day_of_week_sin": 0.5, "day_of_week_cos": 0.5}` when the seasonal sin/cos pair is being chosen too aggressively relative to lagged returns.

### `monotone_constraints`

- **Description:** Force monotonic relationship between a feature and the prediction. `+1` = monotone increasing, `-1` = decreasing, `0` = unconstrained.
- **Range/values:** `dict[feature_name, ±1|0]` or list/string. Default `0` everywhere.
- **Default:** `None` (no constraints).
- **When to change:**
  - **Set** when domain knowledge says a feature has a guaranteed monotonic relationship — e.g. trailing volatility → probability of large move should be monotone increasing.
  - Improves calibration when prior is correct; **hurts** Brier if you impose a constraint that doesn't actually hold in the data.
  - **Don't** apply broadly — pick 1–3 features with strong economic priors.
- **Example:** `monotone_constraints={"trailing_vol_20d": 1}` for an `up_X%_in_Yd` target where higher recent vol should monotonically raise the probability of any large move.

### `ignored_features`

- **Description:** Feature names (or indices) to drop from training. Equivalent to removing them upstream but lets you A/B test cheaply.
- **Range/values:** `list[str]` or `list[int]`.
- **Default:** `None`.
- **When to change:**
  - **Use as an FS diagnostic** — temporarily drop a suspect feature to see if val Brier improves. Promote the drop to the actual FS step if confirmed.
  - **Don't** use as the FS mechanism itself; FS lives in our pipeline, not in this kwarg.
- **Example:** `ignored_features=["lag1_volume_zscore"]` to confirm the suspicion that this feature is leaking and hurting OOS Brier.

---

## Notes on parameters with ambiguous or version-dependent defaults

- **`learning_rate`** default is auto-derived and varies with `iterations`, dataset size, and loss. The docs document the auto-derivation but do not pin a single number. Always log the resolved value from the fitted model (`model.get_all_params()["learning_rate"]`).
- **`bootstrap_type`** default depends on `task_type` (CPU vs GPU), `objective`, `bagging_temperature`, and `sampling_unit`. We have pinned the CPU classification default (`Bayesian`) above; reverify if GPU is used.
- **`subsample`** default varies with dataset size **and** `bootstrap_type` — documented above per-case.
- **`leaf_estimation_method`** default branches on objective; documented above for classification (`Newton`, 10 iterations).
- **`od_pval` / `od_wait`** — the canonical defaults page describes both but the "overfitting detector" page does not pin specific numerical defaults. Using `early_stopping_rounds` is the safe path: it switches `od_type` to `Iter` and sets `od_wait = early_stopping_rounds` in one shot, sidestepping the ambiguity.
- **`model_size_reg`** has minimal impact without categorical features; left at default in v1.
- **`border_count`** default not surfaced in the doc pages we fetched (the dedicated `border-count__desc` page returned 404), but the parameter-tuning page confirms `254` is the recommended CPU value, which matches the long-standing default.
- **`random_seed`** default behaves differently between Python (`None` → effectively `0`) and other bindings (`0` explicit). Always set explicitly to avoid surprises.

---

## Suggested per-iteration agent prompt

This is the decision rubric the skill prompt should reference. Apply on each FS+HP iteration after reading the diagnostic bundle (train/val Brier curves, reliability plot, calibration error, positive-class recall, top-feature importance):

1. **Read the iteration's diagnostic bundle.** Look at: train Brier, val Brier, the gap between them, reliability-plot deviation from diagonal, val Brier curve shape (still descending? plateau? oscillating?), early-stopping iteration, top-5 feature importances, positive-class recall at threshold 0.5.

2. **Overfit signal** (`train Brier << val Brier`, or reliability curve over-confident at the extremes):
   - First: raise `l2_leaf_reg` (try `2x`).
   - If insufficient: raise `min_data_in_leaf`, lower `depth`, or set `rsm=0.7`.
   - If still overfitting: switch to `bootstrap_type=MVS` (or `Bernoulli`, `subsample=0.7`).
   - Last resort: drop features in FS.

3. **Underfit signal** (`train Brier ≈ val Brier`, both high; learning curve flat from early on):
   - First: raise `depth` (`+1`, max `8`).
   - If insufficient: raise `iterations` and lower `learning_rate` proportionally (e.g. `iterations *= 2, learning_rate /= 2`).
   - If still underfitting: add features back, or try `grow_policy=Depthwise`.

4. **Rare cell + low positive-class recall** (positives <5% of rows AND recall near zero):
   - Set `auto_class_weights="SqrtBalanced"` first.
   - If still under-recalling: switch to `auto_class_weights="Balanced"` or set `scale_pos_weight=(1-p)/p`.
   - **Always re-check calibration** after weighting — weights bias probabilities and may need post-hoc recalibration.

5. **Learning curve still descending at the cap** (best iter within 10% of `iterations`):
   - Raise `iterations` (`2x`) **and** raise `early_stopping_rounds` proportionally.
   - Alternatively, lower `learning_rate` and raise `iterations` together.

6. **One-or-two-feature dominance** (top feature importance >50% of total, val Brier plateaued):
   - Raise `random_strength` (`2–3`) and/or set `rsm=0.7` to decorrelate trees.
   - Consider whether the dominant feature is leaking (use `ignored_features` to A/B test).

7. **Calibration drift in tails** (reliability curve over-confident above 0.8 or below 0.2):
   - First inspect `class_weights` — full inverse-prevalence often causes this. Try `SqrtBalanced` or drop weights and post-hoc Platt/isotonic-calibrate.
   - Lower `depth` (smoother probabilities).

8. **Boosting algorithm sanity check** (once per cell, not per iteration):
   - Benchmark `boosting_type="Ordered"` vs `"Plain"`. Keep Ordered if val Brier improves >1 seed-noise SD; otherwise revert to Plain for the runtime savings.

9. **Always pinned, never tuned:**
   - `has_time=True`
   - `random_seed=42` (production); `{0..4}` only for explicit seed-variance bars
   - `loss_function="Logloss"`
   - `eval_metric="BrierScore"` (or `"Logloss"`)
   - `custom_metric=["Logloss", "BrierScore", "AUC"]`

10. **Record rationale per change.** Every HP change must be logged with the diagnostic signal that triggered it. The agent's value-add is the chain of reasoning, not the parameter values themselves — without a rationale trail, future iterations cannot tell which changes were exploratory vs which were responses to signal.

---

## Sources

- CatBoost Python reference — `CatBoostClassifier`: https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
- Training parameters reference (common): https://catboost.ai/docs/references/training-parameters/
- Parameter tuning guide: https://catboost.ai/docs/concepts/parameter-tuning.html
- Overfitting detector: https://catboost.ai/docs/concepts/overfitting-detector.html
- Binary classification loss functions and metrics: https://catboost.ai/docs/concepts/loss-functions-classification.html
- `fit()` method reference (for `early_stopping_rounds`): https://catboost.ai/docs/concepts/python-reference_catboost_fit.html

Fetched 2026-05. CatBoost was not installed in the project's venv at time of writing, so defaults were sourced from the canonical docs above rather than from `inspect.signature(CatBoostClassifier.__init__)`. Re-verify defaults via `model.get_all_params()` on the first production run.
