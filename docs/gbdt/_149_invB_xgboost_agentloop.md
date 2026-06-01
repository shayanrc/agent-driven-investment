# Investigation B — agent-DRIVEN FS+HP loop on XGBoost (nifty50 H=25)

> **Methodology note (2026-06-01)**: Numbers in this memo's body use the legacy "weighted R-precision" metric (per-day variable K = R(d), micro-aggregated). The project headline metric was renamed 2026-06-01 to **R-Precision@K** (per-day fixed K, macro-aggregated via `(1/Q)·Σ r_q/min(K,R_q)`). See the "R-Precision@K (current methodology)" section at the bottom of this memo for the cells in this memo recomputed under the new metric, plus `.claude/memories/project-r-precision-methodology.md` for the full definition + relationship.

**Cell**: nifty50 UP +10% within 25 trading days, max_drawdown 5%.
**Date**: 2026-05-29.
**Branch**: `invB-nifty50-h25-xgboost-loop`.
**Backend**: XGBoost 3.2.0 (`backend.library: xgboost`), `callback_mode: agent_file_protocol`.
**Spec**: `configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_xgb_acceptance.yaml`.

**Why this exists**: the first full **agent-DRIVEN** XGBoost FS+HP loop (task #175). The sweep-mode A6 replication (`_148`) already PASSed — it confirmed the XGBoost *backend* reproduces the `_147` CatBoost conclusions on a fixed iter-0 baseline. This investigation proves the **agent-driven exit-and-resume loop** works on XGBoost too: the agent (this session) reads the per-iteration diagnostic bundle and makes a real FS + XGBoost-named-HP decision each iteration, exactly as `_147` did by hand for CatBoost. The verdict is whether the agent-driven XGBoost end-state reproduces the `_147` *conclusions within tolerance* — not the CatBoost HP ceiling (XGBoost HPs differ).

## Setup + the fact that shaped everything

- **Panel**: 46 of 50 tickers (4 IPO-shallow excluded: ETERNAL, JIOFIN, MAXHEALTH, SHRIRAMFIN). Single fold, split 800/400/200/100 per ticker. Sample-uniqueness weighting on (H=25, overlap-inflation 49×, ESS≈128.9k).
- **Prevalence is non-stationary and declining**: train **0.280** → val **0.204** → eval **0.133**. The +10%/25d up-move got monotonically rarer over the sample — identical to the `_147` fact ("the single most important fact in the whole study"). It caps calibrated-probability quality while leaving ranking intact.

## Iteration-by-iteration decision narrative

The loop-target is **val_brier** (lower better); ranking is tracked via AUC + weighted R-precision. Each iteration is one foreground exit-and-resume launch.

| iter | lever (XGBoost) | val_brier | train/val gap | early-stop | verdict |
|---:|---|---:|---:|---:|---|
| 0 | baseline: max_depth 6, eta 0.05, lambda 3.0, 1000 trees, es 75 | **0.16402** | −0.0119 | 13 | no-overfit; calibration clean (z=−0.70) |
| 1 | max_depth 8 | **0.16350** | −0.0082 | 15 | best-on-val (−0.00052 vs base, noise) |
| 2 | max_depth 4 | 0.16447 | −0.0176 | 12 | underfits (worst depth) |
| 3 | depth 8, eta 0.02, es 150 | 0.16367 | −0.0134 | 29 | eta on the ceiling (tie) |
| 4 | depth 8, lambda 10 + min_child_weight 20 | **0.16461** | −0.0215 | 20 | heavy reg HURTS (worst overall) |
| 5 | depth 8, colsample_bytree 0.7 | 0.16351 | −0.0123 | 12 | ties best; dominance is real signal |

- **Iter 0 (baseline).** train/val gap = **−0.0119** (val below train) → **NO overfit** (mirrors `_147`'s −0.0048, same sign). The auto-flagged guidance fired rule 1: *do NOT prune for regularization; FS neutral-to-harmful.* Native calibration clean in-loop (Spiegelhalter z=−0.70). XGBoost-specific signal: **early-stop at tree 13 of 1000** — the model plateaus in very few trees. Belief: invert the prune-on-sight instinct; map the HP landscape first.
- **Iters 1–2 (depth scan).** depth 8 → 0.16350 (best), depth 6 → 0.16402, depth 4 → 0.16447. XGBoost's depth ridge is **monotone-deeper-better** here (8 > 6 > 4), unlike `_147`'s CatBoost inverted-U (depth 6 optimal). But the full spread is only **0.00097** — a tight HP-ceiling band. Depth 4 deepened the underfit gap (−0.0176). Both shapes deliver the same conclusion: a sub-noise depth sensitivity.
- **Iter 3 (eta).** eta 0.02 with patience 150 → 0.16367, a +0.00017 regression vs depth-8/eta-0.05 (noise). Lower eta only moved early-stop 15→29 trees (same ~0.6 effective learning units). **eta is on the ceiling.**
- **Iter 4 (regularization).** lambda 3→10 + min_child_weight 1→20 → **0.16461**, the worst HP config. As the no-overfit gate predicted, heavy regularization HURT: it widened the underfit gap to −0.0215 and collapsed the active set (240 features below the importance floor, concentrating on garman_klass_200=12.17). **Reg axis confirmed neutral-to-harmful** — the `_147` lesson, reproduced.
- **Iter 5 (column-subsampling / dominance probe).** garman_klass_200 dominates every config (one-feature-dominance, HP-ref rubric #6). colsample_bytree 0.7 → **0.16351**, virtually tied with best (+0.00001). Column subsampling redistributed importance across the vol-estimator family (garman_klass_200 4.8→3.57, parkinson_200 up) **with val_brier unchanged** — so the dominance is robust real signal, not a tree-correlation artifact. Wrote `should_stop`.

**HP ceiling declared.** Six configs across depth {4,6,8} × eta {0.02,0.05} × reg {light,heavy} × colsample {1.0,0.7} land in val_brier **[0.16350, 0.16461]** — a **0.00111** band. Best beats baseline by only 0.00052 (noise). Every tunable axis exhausted; no config meaningfully beats baseline; FS neutral-to-harmful on every iteration (no-overfit gate fired throughout). Termination: `inner_stop_signal = agent_should_stop`, `best_iteration = 1`.

> **Note on `monotone_constraints`.** `_147` iters 5–9 explored monotone constraints and found them contraindicated. That lever is **structurally out of the XGBoost FS+HP decision schema by V1.2 design** — `hp_tables_for("xgboost")` does not expose `monotone_constraints`/`interaction_constraints` (they are a Phase-4 read-only causal-ablation tool, never a per-iteration loop knob; see `XGBOOST_HP_REFERENCE.md` § "What is not in this doc"). So the `_147` monotone-contraindication arc is **N/A** for the XGBoost agent loop, not skipped by oversight. XGBoost's interaction handling lives in the read-only native-TreeSHAP diagnostic, not in-loop.

## Final config + metrics (best checkpoint = iter 1)

Final HP: `max_depth=8, eta=0.05, lambda=3.0, subsample=0.8, colsample_bytree=1.0, grow_policy=depthwise, n_estimators=1000, early_stopping_rounds=75`. Determinism pins (`tree_method=exact, n_jobs=1, device=cpu, objective=binary:logistic, eval_metric=logloss, seed=42`) held. **Final feature count: 279 (no pruning)** — the no-overfit gate kept the full pool, exactly as `_147` concluded all-279 is near-optimal.

| segment | base rate | Brier | base-rate Brier | improvement | AUC | weighted R-precision |
|---|---:|---:|---:|---:|---:|---:|
| eval | 0.1325 | 0.1145 | 0.1149 | +0.0005 | 0.6576 | 0.3175 |
| test | 0.1791 | 0.1408 | 0.1470 | +0.0063 | 0.6561 | 0.4078 |

Weighted R-precision lift over base rate: eval 2.30×, test 2.08× (per-day variable-K, `min(R(d),k)` denominator; see `[[project-r-precision-methodology]]`). Per-day P@k (eval): P@1 0.2133, P@5 0.2584, P@10 0.3247 (vs base 0.1325 → 1.61×/1.95×/2.45×). Prediction range well-separated (eval std 0.070, test std 0.059; `flag_low_separation=False`).

**Calibration**: Spiegelhalter |Z|=3.89 on the best-checkpoint's val predictions → gate fired → **isotonic** shipped. (In-loop the iter-0/1 raw-pred z was −0.70/−0.56; the best-checkpoint finalization recomputes on the full val fit and lands at −3.89 → isotonic. Isotonic is order-preserving, so it does not change the ranking.)

Determinism was verified incidentally: a first run (default gates) and this run (loosened gates) reproduced iter-0/iter-1 val_brier (0.16402 / 0.16350) bit-for-bit. (The first run auto-finalized at iter 2 via the default plateau gate; I loosened `plateau_threshold` to 1e-4 so the agent's `should_stop` drives termination and the depth+eta+reg+colsample landscape gets mapped — the SKILL.md worked-example pattern.)

## Tolerance verdict vs `_147` — REPRODUCES WITHIN TOLERANCE

Judged on the four A6-style tolerance axes (same signal/null verdict; AUC on the same side of [0.45,0.55]; comparable R-precision lift magnitude+direction; analogous calibration outcome class):

| `_147` conclusion | XGBoost agent-loop result | within tolerance? |
|---|---|---|
| No overfit (gap −0.0048) → FS neutral-to-harmful | gap −0.0119 (same sign); FS not applied, reg confirmed harmful (iter 4) | **YES** |
| HP ceiling: val_brier band 0.1641–0.1664 across depth×lr | XGBoost band **0.16350–0.16461** across depth×eta×reg×colsample | **YES** (tighter band, same shape: tiny sub-noise sensitivity) |
| Depth: inverted-U, depth 6 optimal | depth ridge monotone-deeper (8>6>4), spread 0.001 | **YES on conclusion** (both: depth is a sub-noise lever); *differs on which depth* — expected, XGBoost HPs ≠ CatBoost HPs |
| No meaningful improvement (best win +0.0001 noise) | best win +0.00052 vs baseline (noise) | **YES** |
| Ranking strong + robust, ~2.1× weighted R-precision | eval 2.30×, test 2.08× | **YES** (≥1.5×; comparable magnitude+direction) |
| AUC clean ranking signal (eval 0.646 / test 0.733) | eval 0.658 / test 0.656 (both ≫ 0.55 null band) | **YES** (same side of the band; SIGNAL) |
| Prevalence-drift calibration ceiling (28%→14%, Brier ≈ base) | train 0.280→eval 0.133 (decline 0.147); Brier +0.0005 over base | **YES** (the irreducible cap reproduced) |
| Calibration: cell needs correction; gate responds | Spiegelhalter |Z|=3.89 → isotonic | **YES** (analogous outcome class) |
| Final model keeps a substantial feature set (all-279 / 88) | 279 (no prune) | **YES** |

**Compound-rule verdict**: AUC 0.66 (≫ 0.55, not in null band) **AND** weighted R-precision lift ~2.1× → **SIGNAL (AUC-visible)** — identical to `_147` CatBoost. The agent-driven XGBoost loop reproduces every `_147` conclusion within tolerance; the only differences (depth-8 vs depth-6 optimum; isotonic vs CatBoost's tree-count-sensitive native/isotonic flip) are the expected backend HP-ceiling and native-probability-surface properties, not verdict flips.

## Comparison to the sweep-mode A6 result (`_148`) for the same cell

`_148` ran the XGBoost *default* callback (fixed iter-0 baseline, no agent reasoning) and reported nifty50 H=25: test AUC 0.667, eval AUC 0.592, test R-prec 0.401 (2.04×), eval R-prec 0.272 (1.97×), isotonic.

This agent-driven loop lands at test AUC 0.656, eval AUC 0.658, test R-prec 0.408 (2.08×), eval R-prec 0.318 (2.30×), isotonic. The **eval segment is notably stronger** here (AUC 0.658 vs 0.592; R-prec lift 2.30× vs 1.97×) because the agent moved to **depth 8** (best-on-val), whereas A6's default ran depth 6; the deeper model generalizes a touch better on eval while staying inside the same tight val ceiling. The **test segment matches** A6 closely (R-prec 2.08× vs 2.04×). Both reach the same SIGNAL verdict and the same isotonic calibration outcome — the agent-driven loop confirms the sweep-mode finding and slightly improves the eval-side ranking by exercising the depth lever the fixed callback never touched.

## Reusable observations (XGBoost-specific deltas vs `_147`)

1. **XGBoost early-stops far faster than CatBoost** on this cell — 12–29 trees vs CatBoost's 48–196. The val ceiling is reached in very few boosting rounds; raising n_estimators or lowering eta only moves the stop point, not the floor.
2. **The depth optimum is backend-specific** (XGB depth 8 vs CatBoost depth 6) but the *conclusion* — depth is a sub-noise lever inside a ~0.001 ceiling band — is identical. Don't expect the CatBoost depth answer to carry over; expect the *shape* to.
3. **`monotone_constraints` is not an XGBoost FS+HP lever** by V1.2 design — the `_147` monotone arc is N/A here, so the acceptance-check's `monotone_contraindicated` is structurally not applicable to the XGBoost loop (it is a Phase-4 read-only ablation tool).
4. The agent-driven exit-and-resume mechanics worked cleanly: each `--resume` applied the decision (`hp_changes` merged, 0 features pruned), trained one iteration, and paused; `should_stop` finalized on the best checkpoint with bit-identical determinism. No HP-validation trips (all proposals stayed within the `*_XGB` tunable ranges).

## Reproducibility

- Run dir: `results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_xgb_acceptance/` (model.ubj, calibration.pkl, predictions/, metrics.json, report.md, loop/ control files, iterations.jsonl).
- Loop decisions: `results/gbdt/experiments/.../loop/iter_{0..5}_decision.json` (the lab-notebook entries).
- R-precision recomputed: `uv run python -m scripts.gbdt.compute_r_precision results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_xgb_acceptance/predictions/{eval,test}.csv`.
- Headline numbers: `results/gbdt/data/_149_invB_xgboost_agentloop_data.json`.

## R-Precision@K (current methodology — added 2026-06-01)

Per `.claude/memories/project-r-precision-methodology.md`, R-Precision@K is the post-2026-06-01 headline cross-cell metric for gbdt — defined as `R-Precision@K = (1/Q) · Σ_q r_q / min(K, R_q)` over the Q days where R_q > 0 (R_q = positives on day q; r_q = positives caught in top-K picks on day q; macro-averaged, equal weight per day; K fixed). Recomputed from each cell's `predictions/test.csv`:

| cell | rows | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|---|---|
| nifty50_up_10pct_25d_dd5pct_xgb_acceptance | 3450 | 17.9% | 0.656 | 0.257 | 0.271 | 0.297 | 0.275 | 0.615 |

Cross-links: `docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md` (CatBoost answer key), `docs/gbdt/_148_xgboost_a6_replication.md` (sweep-mode A6), `[[project-r-precision-methodology]]`, `XGBOOST_HP_REFERENCE.md`, `[[project-xgboost-training-essentials]]`, `[[project-xgboost-interaction-analysis]]`.
