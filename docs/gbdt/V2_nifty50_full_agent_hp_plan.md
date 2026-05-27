# V2 — Full agent-driven HP run on nifty50 H=25 (+10% / 25d / dd5%)

Status: PLAN. Branch: `plan-nifty50-full-agent-hp`. Owner: user / executor agent.

This is the design doc for a production-quality, full-budget rerun of the
**nifty50 +10% / 25-trading-day / max-drawdown-5%** cell (memo #138's
"Cell C") under the canonical 8-iteration FS+HP loop. The 3-iter screening
run already on disk
(`results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct/`, the version
amended by PR #44) confirmed that signal exists; this run extracts the
**best model the current v1 feature pool + HP space can produce** before any
decision on V2 per-ticker features.

Read alongside:

- [`docs/gbdt/goal.md`](goal.md) — module success criteria; calibration is the headline.
- [`docs/gbdt/EXPERIMENT_SPEC.md`](EXPERIMENT_SPEC.md) — YAML schema, validation rules, artifact layout.
- [`docs/gbdt/CATBOOST_HP_REFERENCE.md`](CATBOOST_HP_REFERENCE.md) — per-parameter "when to change" rubric.
- [`docs/gbdt/V1_PLAN.md`](V1_PLAN.md) — Stage 6/7 spec for the FS+HP loop semantics.
- [`docs/gbdt/_138_h25_cross_market_combined.md`](_138_h25_cross_market_combined.md) — the 4-cell context this run extends.
- [`.claude/skills/gbdt-experiment/SKILL.md`](../../.claude/skills/gbdt-experiment/SKILL.md) — orchestration surface.
- [`.claude/memories/project-r-precision-methodology.md`](../../.claude/memories/project-r-precision-methodology.md) — cross-cell metric framing.
- [`.claude/memories/project-gbdt-uniqueness-weights.md`](../../.claude/memories/project-gbdt-uniqueness-weights.md) — LdP §4.4 sample-uniqueness weights (default on).
- [`.claude/memories/feedback-experiment-agent-loop.md`](../../.claude/memories/feedback-experiment-agent-loop.md) — long-running compute pattern.
- [`.claude/memories/feedback-agent-pkill-antipattern.md`](../../.claude/memories/feedback-agent-pkill-antipattern.md) — process-coordination rule.

---

## § 1 — Why this run, and what's different from screening

### What the screening run showed

The 3-iter screening run (sweep-mode spec at
`configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct.yaml`,
`fs_hp_loop.max_iterations=3`) produced the following on the held-out
**test** segment (post-isotonic, with uniqueness weighting on):

| Metric | Value |
|---|---:|
| Weighted Brier (test) | 0.1383 |
| Weighted base-rate Brier (test) | 0.1470 |
| Brier improvement vs base-rate | **+0.0088** ✓ |
| Weighted base-rate (test prevalence) | 0.179 |
| ROC AUC (test) | **0.7327** |
| Weighted R-precision (test) | **0.416** |
| Weighted R-precision lift | **2.12×** |
| Spiegelhalter z (val) | **+5.93** (well outside ±2 band; isotonic fired) |
| Best iteration | **0** (i.e. iter-0 with all 279 features won) |
| Inner-stop signal | `degradation` (val Brier worsened from iter 0→1→2) |
| Wall time | 382 s |

Source: `wt-exp-nifty50-up10-25d/results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct/metrics.json` and `iterations.jsonl` (commit pre-merge; the version that landed in PR #44 is the same artifact).

The val and eval surfaces also showed strong P@k per quarter (2025Q1 +2.3×,
2025Q2 +2.2×) and the by-now-familiar HDFCBANK/WIPRO/COALINDIA over-pick
cohort with ~0% hit rate in eval (see memo #138).

### What the screening run did NOT do

Three things the screening intentionally skipped, all of which the full
run targets:

1. **HP search was OFF.** The screening ran `max_iterations=3` (< the
   `_HP_SEARCH_ITER_THRESHOLD = 5` gate in `src/gbdt/__main__.py`),
   so the metric block records `loop.hp_search_active=false`. All three
   iterations therefore ran with identical default HPs (`iterations=1000,
   learning_rate=0.05, depth=6, l2_leaf_reg=3.0, boosting_type=Ordered,
   early_stopping_rounds=75`).
2. **FS was driven by the algorithmic fallback** (`default_fs_hp_callback`
   in `src/gbdt/train.py`), which prunes anything below 1 % of the top
   feature's importance and floors at 10 features. The aggressive prune
   279 → 62 → 43 monotonically degraded val Brier (0.1642 → 0.1656 → 0.1663)
   and tripped `degradation` after only three iterations. The best
   checkpoint was iter 0 (full 279-feature pool).
3. **No agent reasoning** entered the loop. The screening's `rationale`
   field reads `"iteration 1 from FS+HP callback :: algorithmic
   fallback: kept 43/62 features"` — there is no per-iteration narrative
   the user could read for "why this HP change."

### Why this run is the right next step

- **All four memo #138 cells are landed** (nasdaq100 / sp500 / nifty50 / nifty100). The cross-market generalization question is answered: H=25 +10% / dd5% has real signal everywhere; NSE cells beat US on weighted R-precision (2.06–2.12× vs 1.46–1.54×).
- **The per-ticker over-pick cohort (HDFCBANK, WIPRO, COALINDIA) is documented as the V2-features trigger.** Before promoting per-ticker features (a project of its own), this run answers: *can HP tuning + a real FS pass close the calibration gap on this cohort, using only the asset-agnostic features the v1 pool already has?* If yes → V2 per-ticker features can be deferred. If no → V2 features become the next plan.
- **The Spiegelhalter z = +5.93** means the iter-0 raw probabilities were materially over-confident. Isotonic fixed the bulk of the eval/test surface (test Brier improvement +0.0088). With a real HP search we expect z to land closer to 0 and reduce the dependence on isotonic correction.
- **The screening's best_iteration=0 is suspicious** — it could mean iter 0 was genuinely best, or that the algorithmic FS just over-pruned. A real FS + HP loop with `max_iterations=8` would tell us which.

---

## § 2 — Spec

### File to author

`configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_full.yaml` (NEW; the
existing screening spec stays in place for reproducibility).

### Full spec content

```yaml
# Full agent-driven HP run on the nifty50 +10% / 25d / dd5% cell
# (memo #138's "Cell C"). The screening run
# (configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct.yaml, the 3-iter
# CLI-atom spec) confirmed signal: test AUC 0.7327, weighted R-precision
# 0.416 (2.12x lift), Brier vs base-rate +0.0088, Spiegelhalter z +5.93
# (isotonic fired). Best checkpoint was iter 0 with the full 279-column
# pool; loop terminated on `degradation` after the algorithmic FS
# pruned aggressively (279->62->43).
#
# This spec re-runs the same cell under the canonical 8-iteration budget
# with a tightened iteration-0 HP starting point and a longer
# early_stopping window so the loop has the headroom to discover whether
# HP tuning + a real FS pass can:
#   (a) improve weighted R-precision lift above 2.12x, and
#   (b) bring Spiegelhalter |z| closer to 2 (reducing isotonic dependence).
#
# Schema reference: docs/gbdt/EXPERIMENT_SPEC.md.
# HP rubric: docs/gbdt/CATBOOST_HP_REFERENCE.md.
# Cross-market context: docs/gbdt/_138_h25_cross_market_combined.md.
# Design rationale: docs/gbdt/V2_nifty50_full_agent_hp_plan.md.

target:
  universe: nifty50
  direction: up
  threshold_pct: 10
  horizon_days: 25
  max_drawdown: 0.05
  uniqueness_weighting: true        # default; explicit so post-hoc readers see it

# date_range omitted - take max cached range per ticker (deepest history each).

# split omitted - 800+400+200+100 default; ETERNAL/JIOFIN/MAXHEALTH/SHRIRAMFIN
# excluded by min_rows_per_ticker (4 of 50 in the screening run).

# features omitted - start with all 279 candidates; the loop prunes.

backend:
  calibration_method: conditional_isotonic
  calibration_z_threshold: 2.0      # default; isotonic fires when |z| >= 2

  fs_hp_loop:
    max_iterations: 8               # canonical full budget; gates hp_search_active=true
    plateau_threshold: 0.005        # default
    degradation_gate: 0.01          # default

  # Iteration-0 HP starting point. The screening run used the global
  # defaults verbatim (depth=6, lr=0.05, l2=3.0, iter=1000) and best_iter
  # came in at iter 0; we keep depth/lr/l2 at defaults so the agent has a
  # comparable starting point against the screening, but extend the
  # boosting budget so the cap-hit early-stop heuristic in
  # CATBOOST_HP_REFERENCE.md § "iterations" has room to apply, and switch
  # bootstrap to MVS so the agent's iteration-1+ tuning can use `subsample`
  # directly (Bayesian default ignores `subsample`).
  hp_starting:
    iterations: 2000                # raised from 1000 default; screening's
                                    # early_stop fired well inside this in 26s
                                    # at iter 0, so we have headroom for
                                    # iter 1+ with lower lr / more depth.
    learning_rate: 0.05             # default; paired with iterations=2000
                                    # this is roughly 2x screening's capacity.
    depth: 6                        # default; agent may raise/lower per
                                    # CATBOOST_HP_REFERENCE.md § "depth".
    l2_leaf_reg: 3.0                # default; agent's first overfit lever.
    min_data_in_leaf: 1             # default; agent raises on overfit
                                    # second-line per CATBOOST_HP_REFERENCE.md.
    rsm: 1.0                        # default; agent lowers to 0.7-0.9 when
                                    # one feature dominates importance.
    bootstrap_type: MVS             # switched from Bayesian default.
                                    # MVS exposes `subsample` as a tunable
                                    # (Bayesian ignores it), giving the agent
                                    # more knobs from iter 1 onward.
    subsample: 0.8                  # MVS's recommended starting value.
    random_strength: 1.0            # default; agent raises to 2-5 if one
                                    # feature wins >50% of trees and val
                                    # Brier is plateaued.
    boosting_type: Ordered          # default; ~70k row pooled training is
                                    # exactly where Ordered's prediction-shift
                                    # correction pays off (small-data regime).
    early_stopping_rounds: 150      # raised from 75 default; lower lr or
                                    # noisier per-iter learning curves on
                                    # rare-cell folds benefit from a longer
                                    # patience window.
    # auto_class_weights left unset -- test prevalence 0.179, training 0.280
    # (post-uniqueness-weighting). Both are well above the 5% threshold below
    # which CATBOOST_HP_REFERENCE.md § 5 recommends class weights. If the
    # agent sees recall@0.5 collapse on a later iteration, it may turn on
    # SqrtBalanced and re-check calibration afterwards.

random_seed: 42                     # production seed; matches all other
                                    # H=25 cells for cross-cell comparability.
```

### Validation against `EXPERIMENT_SPEC.md` rules

| Rule | Status |
|---|---|
| Required `target` fields present | ✓ (universe, direction, threshold_pct, horizon_days, max_drawdown) |
| `target.universe` in registry | ✓ (`nifty50` is pre-registered in `configs/gbdt/default.yaml::universes`) |
| `target.direction` ∈ {up, down} | ✓ |
| `target.threshold_pct > 0` | ✓ (10) |
| `target.horizon_days > 0` | ✓ (25) |
| `target.max_drawdown` ∈ (0, 1) | ✓ (0.05) |
| `backend.library` ∈ {catboost} | ✓ (inherited from default.yaml) |
| `backend.calibration_method` ∈ {native, conditional_isotonic, isotonic_always, platt} | ✓ |
| `backend.fs_hp_loop.max_iterations` ∈ [1, 16] | ✓ (8) |
| `backend.hp_starting` keys in tunable-HP allowlist | ✓ (all 11 keys are in `V1_PLAN.md` Stage 4 / `CATBOOST_HP_REFERENCE.md` Categories 1-9) |
| `random_seed ≥ 0` | ✓ |
| `split.sum ≤ min_rows_per_ticker` | ✓ (inherits 800+400+200+100 ≤ 1600 from defaults) |

---

## § 3 — HP search space

The agent operates within the per-parameter bounds in
[`CATBOOST_HP_REFERENCE.md`](CATBOOST_HP_REFERENCE.md). The full list of
tunable HPs is reproduced here with starting values + the search direction
the agent should prefer at each iteration.

| HP | Starting | Bounds (CATBOOST_HP_REFERENCE.md) | Direction priority |
|---|---|---|---|
| `iterations` | 2000 | [100, 10000] | Raise to 4000 if early-stop fires within 10 % of cap; lower only for compute. Pair changes with `learning_rate` (halve lr → double iter). |
| `learning_rate` | 0.05 | (0, 1]; practical [0.01, 0.3] log-scale | Halve when val Brier oscillates iter-to-iter or train-val gap opens early; raise only if every run cap-hits AND val curve plateaued. |
| `depth` | 6 | [1, 16] CPU; practical {4, 5, 6, 7, 8} | Drop to 4-5 on clear overfit (train Brier << val Brier); raise to 7-8 if both Briers flat-high (underfit). |
| `l2_leaf_reg` | 3.0 | [0, 30] log-scale | First-line overfit lever — raise to 5, 10, 20 (log-scale) before touching depth. Lower toward 1 if both Briers flat-high. |
| `min_data_in_leaf` | 1 | [1, 500] | Raise to 20, 50, 100 on rare-cell overfit (positives concentrate in few leaves). For Lossguide tune together with `max_leaves` — not used in v1. |
| `rsm` | 1.0 | (0, 1]; practical [0.5, 1.0] | Lower to 0.7-0.9 when one feature dominates split frequency; raise back to 1.0 if FS has trimmed pool below 10 features. |
| `bootstrap_type` | MVS | {Bayesian, Bernoulli, MVS, No} | Stay on MVS unless `subsample` proves unhelpful; only switch as a diagnostic. Never switch to GPU-only Poisson. |
| `subsample` | 0.8 | (0, 1]; practical [0.5, 0.95]. Active only under MVS/Bernoulli. | Lower to 0.6-0.7 on persistent overfit; raise to 0.9-0.95 if underfitting. |
| `random_strength` | 1.0 | [0.5, 10] | Raise to 2-5 when top feature importance > 50 % of total AND val Brier plateaued; lower toward 0.5 when underfitting. |
| `bagging_temperature` | (n/a under MVS) | [0, 10] | Only meaningful under `bootstrap_type=Bayesian`. If a later iter switches back to Bayesian, raise to 3-10 for stronger row reweighting. |
| `boosting_type` | Ordered | {Ordered, Plain} | Stay on Ordered (~70k rows is the small-data regime where ordered pays off); switch to Plain only as a one-off ablation. |
| `early_stopping_rounds` | 150 | [20, 200] | Raise to 200 when `learning_rate` is lowered (longer plateaus). Lower only if compute is the blocker. |
| `auto_class_weights` | unset | {None, Balanced, SqrtBalanced} | Set `SqrtBalanced` only if recall@0.5 collapses to ~0 AND positives < 5 % of rows. Re-check calibration immediately afterwards. |

**Pinned (NOT tunable, per `CATBOOST_HP_REFERENCE.md` § 9 of the per-iter
rubric and `configs/gbdt/default.yaml::backend.hp_pinned`):**

- `has_time=True` — non-negotiable (C6 walk-forward; CLAUDE.md gbdt § "What not to do").
- `loss_function="Logloss"` — correct for hard-labeled binary classification.
- `eval_metric="BrierScore"` — calibration-aligned; what early stopping optimizes.
- `custom_metric=["Logloss", "BrierScore", "AUC"]` — logged but not optimized.
- `random_seed=42` — pinned for reproducibility; only varied for seed-variance bars.

### Search-space size sanity

A 4-knob × 3-level Cartesian search would be 81 cells. The agent's job is
NOT to grid-search; it's to walk the decision tree in
`CATBOOST_HP_REFERENCE.md` § "Suggested per-iteration agent prompt" steps
2–7 and apply the changes the diagnostic bundle says are warranted. 8
iterations is enough for ~5–6 meaningful HP changes (iter 0 = baseline;
iters 7–8 = stop-condition triggers).

---

## § 4 — FS strategy

### Feature families in the v1 candidate pool

The pool is 279 columns across 16 families (18 sub-family rows; F6/F6b
and F9/F9b are siblings). Per `docs/gbdt/V1_PLAN.md` Stage 2:

| Family | Columns | What it is |
|---|---:|---|
| F1 `index_return_N` | 6 | NIFTY-50 index returns at 6 lookbacks (5/10/20/50/100/200d) |
| F2 `stock_return_N` | 6 | Per-stock momentum |
| F3 `rel_strength_N` | 6 | F2 − F1 (stock-vs-index relative strength) |
| F4 `realized_vol_N` | 6 | Per-stock annualized realized vol |
| F5 `index_vol_N` | 6 | NIFTY-50 index realized vol |
| F6 `drawdown_N` | 6 | Per-stock drawdown from rolling HIGH |
| F6b `runup_N` | 6 | Per-stock runup from rolling LOW |
| F7 volume family | 32 | volume_ratio, OBV, vol-ret-corr, dollar-move zscore/rank (rolling + cross-sectional) |
| F8 higher moments | 12 | returns skew + kurt |
| F9 `index_drawdown_N` | 6 | Index drawdown |
| F9b `index_runup_N` | 6 | Index runup |
| F10 `beta_N` | 6 | Per-stock rolling OLS beta vs index |
| F11 range vol | 12 | parkinson + garman_klass (NOT yang-zhang) |
| F12 `sma_distance_N` | 6 | (close / SMA_N) − 1 |
| F13 vol regime | 18 | vol_change, vol_of_vol, vol_pct |
| F14 cross-sectional rank+z | 24 | return + vol cross-sectional rank/zscore (panel-pooled signal) |
| F15 calendar | 10 | DOW/DOM/MOY sin-cos + 4 India binary flags (fiscal year end, budget week, diwali week, FOMC week) |
| F16 signed days outside band | 105 | 12 z-score underlyings + 93 meta features (signed-days-outside-Xσ) |
| **Total** | **279** | |

### Protected families (never pruned)

The agent must NOT drop entire families from the following protected list,
even if their per-feature importance is low (their absence breaks the
diagnostic interpretation of subsequent iterations):

- **F2 `stock_return_N`** — the basic momentum anchor; cell semantics depend on it.
- **F4 `realized_vol_N`** — basic volatility anchor; the +10 % / 25d threshold is a vol-conditional event.
- **F14 `*_xs_*`** — cross-sectional features are the only way a pooled panel model learns "this stock is unusually weak today vs the cohort"; removing them collapses the design intent (see `docs/gbdt/goal.md` § "What an experiment is").

The agent may down-weight protected features via `feature_weights` per
`CATBOOST_HP_REFERENCE.md` § "feature_weights", but must not pass them in
the prune list.

### Candidate-for-pruning families (in agent's iter-1+ discretion)

Everything else, applied per family rather than per individual column when
the diagnostic bundle suggests a whole family is noise:

- **F15 calendar** — the India binary flags (`fiscal_year_end_week`, `budget_week`, `diwali_week`, `fomc_week`) are the most likely calendar-family noise; sin/cos DOW/MOY pairs sometimes pick up genuine seasonality and should be pruned only after the binary flags.
- **F8 higher moments** — skew/kurt can be unstable on 5-/10-day rolling windows.
- **F11 range vol** — overlaps with F4 (realized_vol); the agent may prune if importance is duplicated and val Brier doesn't move.
- **F16 meta** — 93 derived columns; if a sigma threshold (1σ / 1.5σ / 2σ / 2.5σ / 3σ) shows uniformly low importance across all 6 lookbacks, drop that sigma's whole block.

### Per-iteration prune decision rule

The decision rule the agent applies after reading the iter `N` diagnostic
bundle:

1. **Compute total importance per family** (sum of native importances of
   that family's columns).
2. **Identify families with total importance < 2 % of the largest family's
   total** AND no overlap with a protected family.
3. **Cap one prune per iteration to 1–2 families.** Pruning more in one
   shot makes the val Brier delta uninterpretable.
4. **Never drop below 30 features total** before iteration 4 — the
   screening run's 279 → 43 collapse cost val Brier; we want to avoid
   replicating that mistake. After iteration 4 the floor relaxes to 15.
5. **Always log the rationale** with the family-level importance numbers
   that triggered the drop. Required by `goal.md` ("the agent's chain of
   reasoning is part of the artifact").

### What about column-level pruning?

Within a kept family the agent may also drop individual columns whose
importance is < 1 % of the family-leader's importance. This matches the
screening's `default_fs_hp_callback` heuristic but applied surgically
(within a family) rather than across the whole pool.

---

## § 5 — Stop conditions

Inherited from `src/gbdt/fs_hp_loop.py::inner_stop_check`, with the
defaults from `configs/gbdt/default.yaml`:

| Gate | Threshold | Semantics |
|---|---|---|
| **Plateau** | `plateau_threshold = 0.005` | If `val_brier[N-1] - val_brier[N] < 0.005` AND `val_brier[N-2] - val_brier[N-1] < 0.005` (i.e. the last two iters both improved by less than 0.5 %), stop with signal `plateau`. |
| **Degradation** | `degradation_gate = 0.01` | If `val_brier[N] > 1.01 × min(val_briers[:N+1])`, stop with signal `degradation`. This is the gate the screening run tripped after iter 2. |
| **Cap** | `max_iterations = 8` | Hard ceiling; iteration 8 (zero-indexed: iter 7) is the final allowed run. |

### Early-stop-on-success (NOT a code gate; agent's call)

If at any iteration the agent observes:

- `val_brier < eval_baseline_brier - 0.005` (i.e. weighted val Brier beats
  the segment base-rate by at least 0.005 absolute — comfortably ahead of
  the screening's `+0.0088` test-segment delta), AND
- Spiegelhalter `|z| < 2.0` (calibration is bulk-shippable without
  isotonic), AND
- weighted R-precision lift on the val segment ≥ 2.5×,

the agent should **document the early-success state in the iteration's
`rationale`** and let the loop run to natural plateau / degradation — we
do NOT shortcut the loop, because the artifact's final
`metrics.json::loop.best_iteration` field captures the early-success
checkpoint regardless of when the loop terminates.

### Out-of-budget rule

If the loop hits `cap` (= 8 iterations exhausted) without plateau or
degradation, the best-checkpoint model is still emitted per
`fs_hp_loop.py::best_checkpoint`. This is fine — `cap` is informational,
not failure.

---

## § 6 — Success criteria

Two reading frames: a **floor** (the run isn't worse than screening — i.e.
not a regression) and a **ceiling** (HP+FS extracted meaningfully more
signal — i.e. the v1 feature pool is closer to exhausted, which informs
the V2-features go/no-go).

### Headline metrics — RAW values, no lift columns

Per CLAUDE.md gbdt § Reporting conventions (and consistent with
`docs/gbdt/_138_h25_cross_market_combined.md`):

| Metric | Screening (test) | Floor (must hold) | Ceiling (success) |
|---|---:|---:|---:|
| ROC AUC | 0.7327 | ≥ 0.72 (within 1 SD of screening) | ≥ 0.75 |
| Weighted Brier | 0.1383 | ≤ 0.1400 (≤ +0.002 worse) | ≤ 0.1340 |
| Weighted base-rate Brier | 0.1470 | (constant, ≈ 0.147) | (constant) |
| Brier improvement vs base-rate | +0.0088 | ≥ +0.005 | ≥ +0.014 |
| Weighted R-precision | 0.416 | ≥ 0.40 | ≥ 0.45 |
| Weighted R-precision lift | 2.12× | ≥ 2.0× | ≥ 2.4× |
| Spiegelhalter z (val) | +5.93 | (free; report only) | \|z\| ≤ 3.0 (closer to calibrated raw) |
| Per-day P@1 (test, raw) | 0.1192 | ≥ 0.11 | ≥ 0.18 |
| Per-day P@5 (test, raw) | 0.2062 | ≥ 0.20 | ≥ 0.24 |
| Per-day P@10 (test, raw) | 0.2215 | ≥ 0.22 | ≥ 0.26 |

Lift columns are excluded from the headline tables per CLAUDE.md gbdt §
Reporting conventions; the base-rate row is alongside the raw P@k row so
the reader can compute lift mentally without baking a misleading
"+1.46×"-style framing into the table itself.

### Calibration

The screening shipped isotonic with Spiegelhalter z = +5.93 (well outside
±2). The HP run's success on calibration is one of:

- **Best**: native pass (|z| < 2), no isotonic needed. Indicates HP tuning
  produced bulk-calibrated raw probabilities.
- **Acceptable**: isotonic fires but z ∈ (2, 4) (closer to bulk-calibrated than screening's +5.93).
- **Floor**: isotonic fires, z roughly matches screening (~ +5 to +6).
- **Regression**: |z| > 8 → flag in the report; something in the HP search
  destabilized probability scale.

### Per-ticker cohort verdict (the V2-features pivot)

Compute on test segment:

- For each of {HDFCBANK, WIPRO, COALINDIA, NTPC} (the screening's
  recurring eval over-pick cohort, ext. NTPC for its 0/75 test-segment
  hits), report `n_picks` / `n_positives` / `hit_rate` at the per-day
  top-5 selection.
- Compare to screening (HDFCBANK 17/0/0.000, NTPC 75/0/0.000, COALINDIA
  11/2/0.182).

**The cohort verdict**:

- **HP fix succeeded** → at least 2 of the 4 cohort tickers have hit_rate ≥ 0.10 in test. V2 features can be deferred.
- **HP fix partial** → exactly 1 of 4 has hit_rate ≥ 0.10. V2 features useful but not urgent.
- **HP fix failed** → none have hit_rate ≥ 0.10. V2 features become the next plan (and the memo at handoff explicitly states so).

### PASS / FAIL framing

| Framing | Definition |
|---|---|
| **PASS-ceiling** | All four headline floors hold AND at least 3 ceiling thresholds hit AND cohort verdict ∈ {fix-succeeded, fix-partial}. → V2 features deferred; nifty50 H=25 cell ships as-is. |
| **PASS-floor** | All four headline floors hold AND ≥ 1 ceiling threshold hit. → run is at parity with screening; V2 features warranted iff cohort verdict = fix-failed. |
| **FAIL** | Any headline floor breached, OR cohort verdict = fix-failed AND no ceiling hit. → investigate (likely a HP-search misstep); rerun before deciding on V2. |

The PASS/FAIL framing is for the **user** to read; the agent writes the
metrics into `report.md` and posts the cohort table without inserting an
automated verdict (per `docs/gbdt/goal.md` § "What success looks like per
experiment" and the gbdt § "What not to do" line).

---

## § 7 — Compute budget + scheduling

### Per-iteration ETA

From the screening's `iterations.jsonl` (3 iters, 382 s total, 26 s for
iter 0, 7.3 s for iter 1, 6.8 s for iter 2):

- **Iter 0**: 26 s with the full 279-feature pool + 1000 iterations cap.
  The model's early-stop fired comfortably inside the cap.
- **Iter 1+ (full HP search)**: the screening's iter 1–2 were 7 s each
  with reduced feature counts (62 / 43). With the full run's `iterations=2000`
  cap + `early_stopping_rounds=150` + 100–250 features per iter, expect
  **~40–80 s per iteration** depending on which HP combination the agent
  picks (Ordered boosting + depth=7 + iterations=2000 is roughly 3–5× the
  screening's per-iter cost).

### Total ETA

- **8 iterations × 60 s average** = ~480 s for the loop itself (~8 min).
- **Phase 1 (data load + feature build)**: ~30 s (cache hit; nifty50 is fully cached).
- **Uniqueness weighting**: ~15 s.
- **Phase 4 calibration + artifact emit**: ~30 s.
- **Figures + report**: ~10 s.

**Total: ~10–15 minutes.** This is well under the 30-minute foreground
threshold in
[`.claude/memories/feedback-sub-agent-foreground.md`](../../.claude/memories/feedback-sub-agent-foreground.md),
so the executor should run **foreground with `timeout 1800`**, NOT
background + Monitor.

### Concurrency

- **No other gbdt experiments running simultaneously** per
  `[[feedback-agent-pkill-antipattern]]` AND the SQLite single-writer
  contract from CLAUDE.md § Data and configs.
- Pre-flight: `ps -ef | grep "gbdt.experiment" | grep -v grep` must be
  empty before launch. If non-empty, STOP and report to user — do NOT
  pkill anyone else's run (per the pkill antipattern memory).
- **No concurrent data_pipelines seeds** either (same single-writer SQLite contract).

### Worktree convention

- **Parent (this session)** creates the executor's worktree:
  `wt-exp-nifty50-full-h25/` as a sibling of `wt-plan-nifty50-full-run/`.
- Symlinks `data/` → `/tmp/exp_data` (or wherever the shared cache lives)
  and `.env` from the main checkout per CLAUDE.md § Environment.
- The executor sub-agent runs **only** in `wt-exp-nifty50-full-h25/`;
  this worktree (`wt-plan-nifty50-full-run/`) is plan-only.

### Foreground-vs-background

Per
[`.claude/memories/feedback-sub-agent-foreground.md`](../../.claude/memories/feedback-sub-agent-foreground.md):

> Sub-30-min runs (cached-data nifty50 experiments, smoke tests):
> foreground with `timeout` as a hard cap. Do NOT background+Monitor.

The launch command (executor copies verbatim into its session):

```bash
cd /mnt/122CEE982CEE765F/Workspace/wt-exp-nifty50-full-h25
timeout 1800 uv run python -m gbdt.experiment \
  configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_full.yaml \
  2>&1 | tee logs/nifty50_full_h25_$(date -u +%Y%m%dT%H%M%SZ).log
```

The executor stays in its session through the `timeout` return; on exit it
reads
`results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_full/metrics.json`
+ `iterations.jsonl` + `report.md` and writes the handoff memo per § 9.

### Disk + WAL pre-flight (from SKILL.md § Pre-flight)

Before launch (executor runs):

```bash
df --output=avail $(pwd) | tail -1   # must be >= 10 GB
ps -ef | grep "gbdt.experiment" | grep -v grep   # must be empty
readlink data   # must return an absolute path
sqlite3 data/processed.db 'PRAGMA quick_check'   # must print 'ok'
```

If any check fails, STOP and report to parent. Do NOT proceed.

### Cache freshness pre-flight

The screening's `metrics.json::data.stale_tickers` flagged **NSE:NTPC as
147 days stale**. Today is 2026-05-28; check NTPC freshness in the
shared cache:

```bash
uv run python -c "
import sqlite3
from datetime import date
con = sqlite3.connect('data/processed.db')
cur = con.execute(
  \"SELECT MAX(observation_date) FROM nse_equities WHERE ticker='NSE:NTPC'\"
)
latest = cur.fetchone()[0]
print(f'NSE:NTPC latest cached: {latest}; today is {date.today()}')
"
```

If NTPC is still > 30 days stale, run:

```bash
uv run python -m data_pipelines fetch NSE:NTPC --start 2015-01-01 --back-extend
```

before launching the experiment, so the test segment isn't artificially
truncated for NTPC.

---

## § 8 — Risks + mitigations

### R1 — The "agent-driven" mode is NOT wired end-to-end in the current code

**Severity: HIGH.** This is the most important caveat to surface.

`docs/gbdt/goal.md` and
`.claude/skills/gbdt-experiment/SKILL.md` both describe the FS+HP loop
as "agent reads diagnostic bundle each iteration and decides feature
pruning + HP changes." In practice, `src/gbdt/__main__.py::run_experiment`
hands `walk_forward_train()` no `fs_hp_callback` argument, so the loop
falls back to `default_fs_hp_callback` (the algorithmic prune + L2/cap
nudge). There is currently NO production wiring that lets an agent insert
per-iteration HP/FS decisions into a single `run_experiment` invocation.

**What this means for THIS run**: the spec at § 2 will execute with the
algorithmic-fallback callback under `max_iterations=8`. The HP shifts the
fallback can make are limited to:

- `l2_leaf_reg *= 1.5` when `train_val_gap > 0.02`
- `iterations *= 2, learning_rate /= 2` when the cap is hit

…and FS pruning is the 1 %-of-top heuristic with a 10-feature floor. The
agent does NOT get to apply the
`CATBOOST_HP_REFERENCE.md` § "Suggested per-iteration agent prompt"
decision tree against each iteration's bundle.

**Mitigations**:

1. **What this plan does**: treat the run as "best-effort full-budget
   single-shot under the algorithmic fallback, with a hand-tuned `hp_starting`
   (§ 2) so iter 0 is a better starting point than the global default."
   The fallback's L2/cap rules will get exercised over 8 iterations
   instead of 3, giving the FS heuristic room to converge to something
   stable (the screening's `degradation` after iter 2 was an artifact of
   too-aggressive pruning + only 3 iterations).
2. **Alternative path (NOT in scope of this PR)**: an "outer agent" loop
   that runs the spec with `max_iterations=1`, reads the artifact,
   rewrites the spec's `hp_starting` block, deletes the artifact dir,
   re-runs. Repeat 8 times. This recreates agent-driven semantics on top
   of the current code, at the cost of 8× the data-load overhead (each
   re-run re-loads the panel + rebuilds features + recomputes uniqueness
   weights). At ~45 s of pre-loop overhead per re-run, total cost is
   ~8 × 45 + 8 × ~60 = ~840 s (~14 min) — comparable to a single
   `max_iterations=8` run.
3. **Long-term fix (V1.1)**: expose `fs_hp_callback` via either (a) a
   `backend.fs_hp_loop.driver: "agent" | "fallback" | "optuna"` switch
   in the spec schema that wires through to a callable, or (b) an
   external "controller" SDK skill that drives the loop via subprocess
   IPC. Both are out of scope for this run; the latter aligns with the
   parked V1.1 Bayesian-HP entry in `docs/gbdt/V1.1_TBD.md`.

**Decision for this run**: proceed with single-shot `max_iterations=8`
under the algorithmic fallback + hand-tuned `hp_starting`. The success
criteria in § 6 are evaluated against the artifact this produces. If the
artifact's `loop.best_iteration > 0` and `loop.n_iterations_run ≥ 5`, the
fallback genuinely exercised the HP space. If `best_iteration = 0` again
(as in screening), document that the algorithmic-fallback is converging
to "do nothing" and promote the V1.1 callback-wiring entry to a real
plan.

### R2 — Re-using the H=25 spec means we know signal exists

**Severity: LOW.** Strictly a benefit — this run isn't gambling on a new
cell. The screening confirmed `+0.0088` Brier-vs-baseline on test, AUC
0.73, R-prec 2.12×. The risk is one-directional (worst case = parity
with screening; can't be net-negative against an existing baseline
because we ship the best-checkpoint model).

### R3 — Cohort calibration may not be fixable by HP+FS

**Severity: MEDIUM.** Memo #138's analysis: HDFCBANK/WIPRO/COALINDIA are
structurally range-bound large-caps whose moderate-probability over-picks
are a calibration failure mode that asset-agnostic features
"cannot fix" by design. If HP tuning + a longer FS loop doesn't move
their hit rates above 0.10 in test, that's evidence the V2 per-ticker
features hypothesis is correct.

**Mitigation**: § 6's cohort verdict + § 9's handoff memo capture this
explicitly; the run's outcome on the cohort directly informs whether to
promote V2 features to a real plan.

### R4 — Data cache staleness

**Severity: LOW.** The screening flagged NSE:NTPC at 147 days stale.
NTPC was the test-segment's most-anti-predictive ticker (75/0 picks).
This could be a real feature ("the model picks NTPC on stale data, and
NTPC's recent regime shift made the picks worthless") or a data quality
issue.

**Mitigation**: § 7's pre-flight refreshes NTPC if still stale. If the
test segment then includes fresher NTPC data and the cohort verdict
changes, that's important context for the handoff memo.

### R5 — `default_fs_hp_callback` may converge to "do nothing" under 8 iter

**Severity: MEDIUM.** Compounds R1. The screening with 3 iters tripped
`degradation`; the fallback's prune logic doesn't have an "undo last
prune" capability, so once it over-prunes, val Brier monotonically
worsens and the loop stops. With 8 iters under the fallback, the same
trajectory could play out — just slower.

**Mitigation**: § 4's "never drop below 30 features before iter 4" rule
is a guideline for an agent-driven loop; under the fallback it's
NOT enforced. If the artifact shows iter 1's `n_features < 30`, that's
the fallback over-pruning regardless of this plan's wishes. Surface in
the handoff memo as evidence for the R1 V1.1 wiring entry.

### R6 — Disk wedge / concurrent-experiment kill

**Severity: LOW (pre-flighted).** Per CLAUDE.md § Environment and
`[[feedback-disk-wedge-pattern]]` / `[[feedback-agent-pkill-antipattern]]`,
the executor pre-flights disk + no-other-experiments + WAL integrity
before launch. Belt and suspenders: the launch command uses `timeout(1)`
alone (no `pkill -f`), so an in-session kill targets only the wrapped
subprocess.

---

## § 9 — Comparison + handoff plan

### After the run completes

The executor sub-agent (in `wt-exp-nifty50-full-h25/`) does the following
sequentially, in foreground, in its single session:

1. **Verify artifact** — `metrics.json`, `iterations.jsonl`, `report.md`,
   `predictions/{train,val,eval,test}.csv`, `figs/`, `model.cbm`,
   `calibration.pkl`, `features.yaml`, `hp.yaml` all present.
   `metrics.json::loop.hp_search_active` should be `true` (since
   `max_iterations=8 >= _HP_SEARCH_ITER_THRESHOLD=5`).
2. **Compute R-precision** — run `scripts/gbdt/compute_r_precision.py`
   on `predictions/test.csv` (and `predictions/eval.csv`); add the
   weighted R-prec + lift to the memo. (Per
   `[[project-r-precision-methodology]]`, R-prec is the primary
   cross-cell metric; the runner doesn't bake it in yet.)
3. **Build the comparison table** — full run vs screening, RAW values
   only (no lift columns), with the base-rate row alongside the P@k row:

   ```
   | Metric                    | Screening | Full HP run | Δ |
   |---|---:|---:|---:|
   | AUC (test)                |   0.7327  |     X.XXXX  | X |
   | Weighted Brier (test)     |   0.1383  |     X.XXXX  | X |
   | Base-rate Brier (test)    |   0.1470  |     X.XXXX  | X |
   | Brier vs base-rate (test) |  +0.0088  |    +X.XXXX  | X |
   | Weighted R-precision      |   0.416   |     X.XXX   | X |
   | R-prec base rate (weighted) |  0.179  |     X.XXX   | X |
   | Spiegelhalter z (val)     |  +5.93    |    +X.XX    | X |
   | Per-day P@1 (test)        |   0.119   |     X.XXX   | X |
   | Per-day P@5 (test)        |   0.206   |     X.XXX   | X |
   | Per-day P@10 (test)       |   0.222   |     X.XXX   | X |
   | Test segment base rate    |   0.179   |     X.XXX   | X |
   ```

4. **Build the per-ticker cohort table** — HDFCBANK/WIPRO/COALINDIA/NTPC
   `n_picks / n_positives / hit_rate` from
   `metrics.json::segment_diagnostics.test.per_ticker_hit_rate`,
   compared to screening.
5. **Write the memo** at
   `docs/gbdt/_<task-id>_nifty50_full_h25_run.md` (next task-id; check
   the latest `_<NNN>_` memos in `docs/gbdt/` and pick the next slot).
   Sections (mirror memo #138's structure):
   - Cell / what was run / spec hash.
   - Headline metrics — RAW values, no lift columns, base rate alongside.
   - Per-ticker cohort verdict (the V2-features pivot).
   - Mechanistic reading — what the iteration history says about HP+FS efficacy.
   - **Verdict** (per § 6 PASS-ceiling / PASS-floor / FAIL framing).
   - **V2 features decision** — per § 6 cohort verdict mapping.
6. **Open the memo as its own PR** on a fresh branch (e.g.
   `exp-nifty50-full-h25-memo`), with the artifact dir staged from
   `results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_full/` (only
   the headline metrics + report.md + iterations.jsonl + figures get
   committed; CSVs and the model binary are gitignored per existing
   patterns — verify against `.gitignore`).
7. **Do not auto-merge the memo PR** (this plan PR is the one that opens
   first; the memo PR follows after the executor finishes). Per
   CLAUDE.md § Environment the parent's autonomous review/merge default
   does apply to the memo PR, but the memo PR is not in this plan's
   scope to launch.

### Decision rule fed back into V2 planning

| Cohort verdict | V2 features decision | Next step |
|---|---|---|
| fix-succeeded | Deferred | Update `docs/gbdt/V1.1_TBD.md` with the V2-deferral note (HDFCBANK et al. fixable by HP). |
| fix-partial | Useful but not urgent | Park V2 in `V2_TBD.md` (new file), schedule for after the H ∈ {5, 10, 50} sweep finishes. |
| fix-failed | Promote V2 to a real plan | Author `docs/gbdt/V3_per_ticker_features_plan.md` (V2 slot is taken by this plan), branch `plan-v3-per-ticker-features`, gated on this memo as the trigger. |

### V1.1 callback-wiring follow-up

Independent of the cohort verdict, if R1 plays out (artifact's
`best_iteration ≤ 1` or `n_features` collapses below 30 by iter 2), open
a separate small PR to promote the parked agent-callback wiring from
`V1.1_TBD.md` § "Bayesian HP search alternative" into a concrete
`backend.fs_hp_loop.driver` spec field, with `"fallback"` (current
behaviour) and `"agent_callback"` (new — accepts a callable name
resolvable via importlib) as the first two values. This is the
prerequisite for real agent-driven runs and is the missing piece between
the design docs and the code.

---

## § 10 — Decisions log

| # | Decision | Reasoning |
|---|---|---|
| D1 | New spec file `nifty50_up_10pct_25d_dd5pct_full.yaml`, screening spec left in place. | Don't shadow the screening artifact's reproducibility; both specs are valid checkpoints. |
| D2 | `fs_hp_loop.max_iterations = 8`. | The canonical full budget per `configs/gbdt/default.yaml` and `goal.md` Stage 6. Triggers `hp_search_active=true` (≥5). |
| D3 | `calibration_method = conditional_isotonic` (no change). | Screening's z = +5.93 means isotonic fires; agent's hope is to reduce \|z\| but the conditional gate makes the right decision regardless. |
| D4 | `hp_starting.iterations = 2000` (up from 1000 default). | Gives the agent's potential `learning_rate` halving + `iterations` doubling rule (CATBOOST_HP_REFERENCE.md § iterations) somewhere to grow into. Screening's iter 0 early-stopped well inside 1000 so capacity wasn't binding there; the headroom is for iters 1+. |
| D5 | `hp_starting.early_stopping_rounds = 150` (up from 75 default). | Paired with raised `iterations`; longer patience window absorbs lower-`learning_rate` plateaus. |
| D6 | `hp_starting.bootstrap_type = MVS` (changed from Bayesian default). | MVS exposes `subsample` as a tunable knob (Bayesian ignores it). Gives the agent more regularization levers from iter 1 forward. CATBOOST_HP_REFERENCE.md § bootstrap_type: "MVS when you have a lot of rows" — 36,800 train rows qualifies. |
| D7 | `hp_starting.subsample = 0.8`. | MVS's recommended starting per CATBOOST_HP_REFERENCE.md § subsample. |
| D8 | Other `hp_starting` keys unchanged from defaults (depth=6, lr=0.05, l2=3.0, etc.). | Want a starting point comparable to screening so iter 0's metrics can be directly compared to the screening's iter 0 metrics. Agent's iter 1+ changes are the variable under test. |
| D9 | `auto_class_weights` left unset. | Test prevalence 0.179, training 0.280 (post-uniqueness). Both well above the 5% threshold below which class weights are recommended (CATBOOST_HP_REFERENCE.md § 5). Agent may turn on `SqrtBalanced` mid-loop if recall@0.5 collapses; the spec doesn't pre-commit. |
| D10 | Protected feature families: F2, F4, F14. | F2 (stock momentum) + F4 (realized vol) anchor the cell's basic signal per CATBOOST_HP_REFERENCE.md and goal.md. F14 (cross-sectional) is the design intent of pooled training per goal.md. |
| D11 | "Never drop below 30 features before iter 4" rule. | The screening's collapse 279 → 43 in two iterations cost val Brier. This rule is a guideline for an agent-driven loop; under the algorithmic fallback it is documentation only. |
| D12 | Foreground + `timeout 1800`, not background + Monitor. | Total ETA ~10–15 min per § 7 is well under the 30-min foreground threshold per `[[feedback-sub-agent-foreground.md]]`. |
| D13 | Cache freshness pre-flight on NSE:NTPC. | The screening flagged it at 147 days stale; NTPC was the test-segment's worst over-pick (75/0). Refresh if still >30 days stale before launch. |
| D14 | Memo handoff at `_<task-id>_nifty50_full_h25_run.md`, separate PR. | Mirrors memo #138's structure; keeps the plan PR (this PR) decoupled from the results PR so the user can review the plan independently of waiting on the run. |
| D15 | V-numbering: `V2_PLAN`. | V0 (investigation) + V1 (PLAN) + V1.1 (TBD) are taken in `docs/gbdt/`. V2 slot is free. Memo #138's "promote V2 TBD to a real V2 plan" was about per-ticker features; we're claiming the V2 slot for this HP+FS run because it's the immediate next plan and (per § 9) it's the gate on whether V2-features (now to be slotted as V3) is launched. |
| D16 | Surface R1 (agent-driven callback NOT wired) explicitly. | This is the most important risk and the largest source of uncertainty in the run's value. The plan executes against the current code as it is, but documents the wiring gap so the user can decide whether to fix it before or after this run. |

---

## Appendix A — Quick-reference executor checklist

The executor sub-agent (in `wt-exp-nifty50-full-h25/`) follows this
ordered checklist in a single foreground session:

```
[ ]  1. Pre-flight: df >= 10G, ps -ef | grep gbdt.experiment empty, readlink data, sqlite quick_check ok
[ ]  2. Pre-flight: NSE:NTPC cache freshness check + refresh if > 30 days stale
[ ]  3. mkdir -p logs/
[ ]  4. timeout 1800 uv run python -m gbdt.experiment \
        configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_full.yaml \
        2>&1 | tee logs/nifty50_full_h25_$(date -u +%Y%m%dT%H%M%SZ).log
[ ]  5. Verify artifact dir has all required files (see § 9 step 1)
[ ]  6. Verify metrics.json::loop.hp_search_active == true
[ ]  7. Run compute_r_precision.py on test + eval predictions
[ ]  8. Write memo at docs/gbdt/_<task-id>_nifty50_full_h25_run.md (see § 9)
[ ]  9. Build comparison + cohort tables (see § 9 step 3, 4)
[ ] 10. Author and verify the V2-features decision (see § 9 mapping)
[ ] 11. Stage explicitly: docs/gbdt/_<task-id>_*.md + results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_full/{report.md,metrics.json,iterations.jsonl,spec.yaml,figs/,features.yaml,hp.yaml}
       (CSVs, model.cbm, calibration.pkl gitignored — verify against .gitignore first)
[ ] 12. Commit (HEREDOC, no AI attribution, subject: "gbdt: nifty50 full HP run on H=25 (+10%/25d/dd5%)")
[ ] 13. Push, open PR. Title: "gbdt: full agent-budget HP run on nifty50 H=25 (cell C revisited)".
[ ] 14. Return PR URL + 2-paragraph TLDR to parent.
```

## Appendix B — What changed vs the screening spec

A line-by-line diff between
`configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct.yaml` (screening)
and `configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_full.yaml`
(this plan's new spec):

| Key | Screening | Full | Reason |
|---|---|---|---|
| `target.universe` | nifty50 | nifty50 | same |
| `target.direction` | up | up | same |
| `target.threshold_pct` | 10 | 10 | same |
| `target.horizon_days` | 25 | 25 | same |
| `target.max_drawdown` | 0.05 | 0.05 | same |
| `target.uniqueness_weighting` | (default true) | true | explicit |
| `backend.calibration_method` | conditional_isotonic | conditional_isotonic | same |
| `backend.fs_hp_loop.max_iterations` | 3 | **8** | full budget; triggers `hp_search_active=true` |
| `backend.fs_hp_loop.plateau_threshold` | 0.005 | 0.005 | same (default) |
| `backend.fs_hp_loop.degradation_gate` | 0.01 | 0.01 | same (default) |
| `backend.hp_starting.iterations` | (default 1000) | **2000** | room for lr-halving |
| `backend.hp_starting.learning_rate` | (default 0.05) | 0.05 | unchanged; baseline parity |
| `backend.hp_starting.depth` | (default 6) | 6 | unchanged; baseline parity |
| `backend.hp_starting.l2_leaf_reg` | (default 3.0) | 3.0 | unchanged; baseline parity |
| `backend.hp_starting.min_data_in_leaf` | (default 1) | 1 | unchanged |
| `backend.hp_starting.rsm` | (default 1.0) | 1.0 | unchanged |
| `backend.hp_starting.bootstrap_type` | (default Bayesian) | **MVS** | exposes `subsample` |
| `backend.hp_starting.subsample` | (n/a under Bayesian) | **0.8** | MVS recommended default |
| `backend.hp_starting.random_strength` | (default 1.0) | 1.0 | unchanged |
| `backend.hp_starting.boosting_type` | (default Ordered) | Ordered | unchanged |
| `backend.hp_starting.early_stopping_rounds` | (default 75) | **150** | longer patience |
| `random_seed` | (default 42) | 42 | unchanged |
