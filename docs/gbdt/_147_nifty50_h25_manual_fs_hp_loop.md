# nifty50 H=25 — hand-driven FS+HP loop (agent-as-data-scientist)

> **Methodology note (2026-06-01)**: Numbers in this memo's body use the legacy "weighted R-precision" metric (per-day variable K = R(d), micro-aggregated). The project headline metric was renamed 2026-06-01 to **R-Precision@K** (per-day fixed K, macro-aggregated via `(1/Q)·Σ r_q/min(K,R_q)`). See the "R-Precision@K (current methodology)" section at the bottom of this memo for the cells in this memo recomputed under the new metric, plus `.claude/memories/project-r-precision-methodology.md` for the full definition + relationship.

**Cell**: nifty50 UP +10% within 25 trading days, max_drawdown 5%.
**Date**: 2026-05-28.
**Branch**: `gbdt-nifty50-manual-hp-loop`.
**Why this exists**: a manual rehearsal of the V1.1 agent-driven FS+HP loop (PR #48). The runner's `fs_hp_callback` is not wired (every prior experiment used `default_fs_hp_callback`, a fixed prune-and-nudge heuristic). Here the **agent (this session) plays the data scientist by hand**: each "iteration" is one `max_iterations: 1` run with an explicit `features` + `hp_starting` the agent chose after reading the previous iteration's diagnostics. The point was both (a) to find a better nifty50 model and (b) to prove out the loop a real callback would automate.

## How to read this

Every iteration below is structured as **Observation → Hypothesis → Change → Result → Belief update**, because the decision chain *is* the deliverable. The headline loop-target is **val_brier** (lower better); ranking quality is tracked via **test AUC** and **weighted R-precision** (per-day variable-K; see `[[project-r-precision-methodology]]`). All brier "improvement" figures are vs the per-segment base-rate brier.

## Setup + the fact that shaped everything

- **Panel**: 46 of 50 tickers (4 excluded as IPO-shallow: ETERNAL, JIOFIN, MAXHEALTH, SHRIRAMFIN). Single fold, split 800/400/200/100 (train/val/eval/test) per ticker.
- **Prevalence is non-stationary and declining**: train **0.280** → val **0.204** → eval **0.138** → test 0.196. The +10%/25d up-move got monotonically rarer over the sample. This is the single most important fact in the whole study: it caps how good *calibrated probability* can be, while leaving *ranking* intact.

## Iteration 0 — baseline (inherited from the screening run)

All 279 features, default HP (depth 6, lr 0.05, Bayesian bootstrap, Ordered boosting, 1000 trees, early-stop 75).

| segment | wtd prev | brier | base | improvement | AUC | R-prec (wtd) |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.204 | 0.1604 | 0.1623 | +0.0019 | — | — |
| eval | 0.138 | 0.1173 | 0.1149 | **−0.0023** | 0.646 | 0.300 (2.18×) |
| test | 0.196 | 0.1383 | 0.1470 | +0.0087 | 0.733 | 0.416 (2.12×) |

- **train/val gap = −0.0048** (val brier *below* train brier) → **No overfit.** If anything, capacity headroom. *(Correction: an earlier draft said "early-stop never fired at 1000 trees" — a misread; the artifact records `early_stop_iteration=67`. Early-stopping firing is healthy and orthogonal to overfit; the no-overfit signal is the negative gap. The depth-curve trees line up cleanly — depth 4→71, depth 6→67, depth 8→48: deeper overfits sooner. All val_brier conclusions unaffected.)*
- **Calibration correct on val** (pred mean 0.2038 = prevalence 0.2038); eval improvement is negative purely from the val(20%)→eval(14%) prevalence shift the model can't see.
- **Ranking is strong and real**: R-precision lift ~2.1× on both held-out segments.
- **Importance highly concentrated**: top feature `vol_of_vol_200` = 10.5% of gain; **191/279 features < 0.01 importance.** A "market-regime + volatility-level" model.

**Belief set by iter 0**: the fallback callback's instinct (prune features, assuming overfit) is *wrong here* — there is no overfit. The untested lever is HP (the screening's "HP search" was a no-op at 3 iters). Invert the fallback: test HP on the full feature set first.

## Iteration 1 — capacity test (depth 6→8, l2 3.0→1.5)

- **Observation**: no overfit (negative train/val gap) → maybe under-using capacity; test it directly.
- **Hypothesis**: more capacity extracts more signal. Change a coherent capacity cluster (depth+l2), hold features=all so the val_brier delta is attributable to HP alone.
- **Result**: val_brier **0.1652** (worse), **early-stop crashed to tree 48** (overfits immediately). Test AUC 0.740, eval improvement +0.0011.
- **Belief update**: capacity headroom does **not** exist upward. Signal is **low-complexity** (shallow interactions). The native-calibration flip (z 5.93→0.10) is a side effect of stopping at 48 trees (softer probs), not a depth win.
- *(**Correction 2026-05-29 — l2 confound.** This iteration changed **two** variables at once: depth 6→8 AND l2 3.0→1.5 (see the section header — "a coherent capacity cluster (depth+l2)"). So this 0.1652 is depth-8-**at-l2-1.5**, NOT comparable to the depth-4/6 points, which used l2=3.0. The l2 cut — not the depth — is what crashed early-stop to tree 48 (immediate overfit). The automated agent-driven loops (`_174` CatBoost, `_149` XGBoost), which enforce one-attributable-change-per-iteration, re-ran depth-8 with l2 **held at 3.0** and got **0.1633** — marginally **below** depth-6, gap +0.0044 (still no overfit). So "capacity headroom doesn't exist upward" was an artifact of the confound: depth 6 and 8 are a **plateau within the cell's ~0.001 noise band**, not a 6-peaked inverted-U. `scripts/gbdt/acceptance_check_147.py` was updated to encode the deconfounded depth curve + the plateau check. The downstream conclusions (HP ceiling, no-overfit, FS-neutral, monotone-contraindicated, prevalence ceiling) are unaffected.)*

## Iteration 2 — shallower trees (depth 6→4), maps the depth curve

- **Observation**: deeper overfit; I'd only sampled one side of depth 6.
- **Hypothesis**: for low-complexity signal, shallower (lower-variance) may win. Reset l2 to 3.0 to keep the depth comparison clean; change only depth.
- **Result**: val_brier **0.1661** (worst), train_brier 0.1760 (underfit), early-stop 71. Test AUC 0.709.
- **Belief update**: clean inverted-U on depth — **0.1661 / 0.1642 / 0.1652** for depth 4/6/8. **Depth 6 is the sweet spot**, confirmed by data, not assumed.
- *(**Correction 2026-05-29.** The "inverted-U" is an artifact of the Iteration-1 l2 confound (see that correction): the depth-8 point was at l2=1.5, the depth-4/6 points at l2=3.0. **Deconfounded** (all at l2=3.0), depth 4/6/8 = **0.1661 / 0.1642 / 0.1633** — depth-4 underfits, but depth 6 and 8 are a **within-noise plateau** (Δ ~0.001), not a 6-peak. The "depth 6 is THE sweet spot" claim should read "depth 6–8 plateau; depth-4 underfits.")*

## Iteration 3 — learning_rate (0.05→0.02, 1000→2000 trees)

- **Observation**: depth settled; lr is the orthogonal major HP, untested.
- **Hypothesis**: slower learning sometimes finds a better minimum. Hold depth 6, change only lr + tree budget.
- **Result**: val_brier **0.1641** — a +0.0001 "win" = noise. Early-stop moved to 196 trees (as expected), same minimum. Test AUC 0.732.
- **Belief update**: **HP ceiling confirmed.** Across depth {4,6,8} × lr {0.02,0.05}, val_brier ∈ **[0.1641, 0.1661]**. No HP config meaningfully beats baseline.

## Iteration 4 — targeted feature selection (88 of 279)

- **Observation**: the original FS question. The fallback's blind cut (→62→43) hurt; does a *gentle* cut help?
- **Hypothesis**: drop only the 191 dead-weight features (<0.01 importance), keep the 88 signal-bearing. Baseline HP.
- **Implementation note**: `features.candidates` expects family tokens (F1–F16), not column names; used `features.exclude` with the 191 dead names instead. (Ergonomic gap the V1.1 callback should address — the agent wants column-level FS.)
- **Result**: val_brier **0.1650** — better than the fallback (0.1663) but still not beating all-279 (0.1642). Best *eval* improvement of any run (+0.0014, the only positive one); natively calibrated (z=−1.32). Test R-prec 0.392 (slightly down).
- **Belief update**: even the gentlest FS costs a hair on val. All-279 is optimal on the loop target; the 88-feat model is the better *deployment* artifact (3× leaner, natively calibrated, best eval).

## Iteration 5 — empirically-derived monotonic constraints

This is the deep dive. (CatBoost has `monotone_constraints` (CPU-only) but **no** interaction constraints — XGBoost/LightGBM-only; v1 pins catboost.) The constraint flows through `hp_starting` → `_validate_hp` passes unknown keys → `CatBoostClassifier(**hp)`; the Pool carries `feature_names`, so the named-dict format resolves.

**Analysis → hypothesis.** `scripts/gbdt/monotonic_feature_analysis.py` computed Spearman ρ(feature, target) + decile-consistency on in-sample rows (test window excluded). The **vol-estimator families** (garman_klass / parkinson / realized_vol, all windows) showed clean monotone-increasing relationships (ρ 0.11–0.19, consistency 0.78–1.00; decile-0 posrate ~0.14 → decile-9 ~0.40). `vol_of_vol_200` (model's #1 feature) + `index_vol_*` tested **non-monotone** (consistency 0.44–0.67) and were left unconstrained. Hypothesis H5: `+1` on the 17 clean vol estimators encodes a true structural prior → variance-reducing regularizer.

**Result**: val_brier **0.1664** (worst all-feature run); **test AUC 0.733 → 0.677** (big ranking loss). Net negative.

**Why it failed — three diagnostics, two refuted hypotheses:**

1. *"Constraint destroyed vol×regime interactions."* — `scripts/gbdt/interaction_before_after.py`: pairwise interaction mass involving constrained features **barely moved (11.47→11.40)**. Refuted. CatBoost interaction "strength" = split co-occurrence, not sign-flip freedom.
2. *"Constraint forbade regime-conditional sign-flips."* — `scripts/gbdt/pdp_and_corr.py` 2D PDP of `garman_klass_200 × index_return_50`: **0/12 sign-flip rows in both** before and after; the unconstrained surface was already monotone in vol (constraint only lowered the *level*, peak 0.37→0.34). Refuted.
3. **The actual mechanism** — `scripts/gbdt/monotone_1d_audit.py`: of the 17 constrained features, **13 were already monotone** in the unconstrained model (constraint redundant), but **4 long-window estimators learned a real inverted-U** (downturn at extreme vol): `realized_vol_200` (−100% dip), `garman_klass_100` (−58%), `garman_klass_50` (−56%), `parkinson_200` (−30%). The `+1` constraint **flattened that inverted-U** — *vindicating the original "too much vol kills the breakout" intuition I'd discarded.*

**The subtle methodological error**: candidates were selected by **marginal** monotonicity (Spearman). But ρ measures the overall trend; the inverted-U's downturn sits only in the top decile, so ρ stays +0.16 / consistency 0.78–1.00 even though the model's **internal conditional function turns down**. **Marginal-monotone ≠ model-internally-monotone.** The correct pre-check for a monotone constraint is the unconstrained model's **1D partial dependence**, not the raw feature-target correlation.

### Iterations 6–7 — can the constraint be made beneficial? (no)

Follow-up: does a *surgically correct* constraint help, or is monotone contraindicated outright?

- **iter 6** — `+1` on only the **13 audit-confirmed already-monotone** estimators (drop the 4 inverted-U): val_brier **0.1657**, test AUC 0.696. Recovered only ~⅓ of iter 5's harm — **still worse than baseline (0.1642)**.
- **iter 7** — extend to **18** (safe-13 + 5 pruned-but-monotone `vol_xs_rank_50/100`, `runup_50/100/200`): val_brier **0.1656**, test AUC 0.687. ≈ iter 6 — constraining the 5 *pruned* features was a **no-op**.

| config | # constr | binds on used feats? | val_brier | test AUC |
|---|---:|---|---:|---:|
| baseline | 0 | — | **0.1642** | **0.733** |
| iter 7 | 18 | 13 used + 5 unused | 0.1656 | 0.687 |
| iter 6 | 13 | 13 used | 0.1657 | 0.696 |
| iter 5 | 17 | 17 used (incl 4 inv-U) | 0.1664 | 0.677 |

**Verdict**: harm does *not* scale with raw constraint count — it scales with **how much the model uses the constrained feature**. Constraining unused/pruned features = no-op; constraining used features = harm (worst on the inverted-U four). **No monotone-constraint configuration beats the unconstrained baseline.** Even the 1D-PDP-monotone check is necessary-but-not-sufficient: CatBoost enforces monotonicity at the *tree-structure* level, so it degrades the vol×regime conditional interactions even when the *net* marginal effect was monotone. **Monotone constraints are contraindicated for this cell**, and "adding pruned features back under constraint" yields no benefit (neutral at best).

### Iterations 8–9 — high- vs low-interaction constraint ablation

To localize the harm, the 17 estimators were split by total interaction involvement in the baseline model (bimodal: 8 features with involvement >0, led by `garman_klass_200`=9.67 / `parkinson_100`=5.97; 9 short-window features with involvement **0.00**), and each half constrained alone:

| config | constrained set | val_brier | test AUC |
|---|---|---:|---:|
| baseline | none | **0.1642** | **0.733** |
| iter 8 | 9 **low**-interaction only | 0.1654 | 0.690 |
| iter 9 | 8 **high**-interaction only | 0.1664 | 0.687 |
| iter 5 | all 17 | 0.1664 | 0.677 |

**Two-component harm:**
1. **Fixed "constraints-on" cost (~0.0012)**: present even when constraining the 9 *zero-interaction* features (iter 8, 0.1654 ≠ baseline). Turning on `monotone_constraints` at all switches CatBoost to a more restrictive tree builder, costing global expressiveness regardless of *which* features are constrained.
2. **Interaction-specific harm (dominant)**: constraining just the 8 high-interaction features (iter 9, **0.1664**) reproduces the *entire* 17-constraint val_brier harm; removing the low-interaction constraints recovers 0%, while removing the high-interaction ones (iter 8) recovers ~45%. The val_brier damage concentrates in the high-interaction features; on test AUC both halves hurt and the harm roughly adds (0.690 / 0.687 → 0.677 for all 17).

So there is **no "safe" subset to constrain** — even surgically constraining only non-interacting features carries the fixed cost, and the high-interaction features carry the rest. Confirms monotone constraints are contraindicated for this interaction-driven cell.

## Investigation — were the 191 pruned features signal or redundancy?

`scripts/gbdt/pruned_feature_investigation.py` (on the cached in-sample matrix):

- **26/182 pruned features have a real monotone marginal relationship** with the target — they *do* show up in the data.
- **15 of those 26 are redundant** (corr ≥0.7 with a kept feature). The strongest pruned features are vol estimators that are **0.78–0.99 collinear** with kept vol features (e.g., `parkinson_50` ρ=+0.177 but 0.99 collinear). importance≈0 ≠ unrelated — it means **redundant**.
- **156/182 are genuinely weak/non-monotone.** Notably the F16 "recent-event/outside-band" families (`dollar_move_zscore` ×20, `stock_return_zscore` ×14, `return_xs_zscore` ×19) — which v0 EDA + heuristics flagged as high-signal — show **ρ≈0.005–0.015** here. **The prior was wrong for this cell**: nifty50 H=25 signal is almost entirely volatility-level, not recent-move.

This explains why iter 4 (88 feat) ≈ baseline: pruning removed redundant-or-weak features, never signal the model needed.

## Unified conclusion

| iter | lever | val_brier | verdict |
|---:|---|---:|---|
| 0 | baseline (all 279, depth 6, lr 0.05) | **0.1642** | best-on-val |
| 1 | depth 8 | 0.1652† | overfits (stop @48) — †l2-confounded |
| 2 | depth 4 | 0.1661 | underfits |
| 3 | lr 0.02 | 0.1641 | tie (noise) |
| 4 | targeted FS (88 feat) | 0.1650 | ≈ baseline, leaner |
| 5 | monotone +1 (17 vol est.) | 0.1664 | worst — flattened a real inverted-U |
| 6 | monotone +1 (safe 13 only) | 0.1657 | still < baseline; harm only ⅓ recovered |
| 7 | monotone +1 (18: +5 pruned) | 0.1656 | ≈ iter6 — constraining unused feats = no-op |
| 8 | monotone +1 (9 low-interaction) | 0.1654 | removing high-interaction constraints recovers ~45% |
| 9 | monotone +1 (8 high-interaction) | 0.1664 | = full 17-constraint harm; removing low recovers 0% |

† **iter-1 l2 confound (correction 2026-05-29):** the depth-8 row also cut l2 3.0→1.5, so 0.1652 is not l2-comparable to depth-4/6. Deconfounded (l2 held at 3.0), depth-8 = **0.1633** — depth 6/8 are a within-noise plateau, not a 6-peaked inverted-U. See the Iteration 1/2 corrections.

The cell's signal lives **irreducibly in the conditional interactions** (vol × regime), is **low-complexity**, and is **robust to model configuration** (val_brier band 0.1641–0.1664; R-precision ~2.1× throughout). Every lever that simplifies or constrains either does nothing (HP, FS redundancy) or hurts (monotone). **The untouched baseline is near-optimal.** The brier is capped not by model configuration but by the **train→eval prevalence non-stationarity (28%→14%)** — a distribution shift no FS/HP/constraint lever can touch, because the loop only sees val.

## Deployment recommendation

Ship the **88-feature model (iter 4)**: val_brier within noise of baseline, 3× leaner, natively calibrated, best eval improvement. Use all-279 baseline if test R-precision (0.416 vs 0.392) is weighted over eval generalization. **Do not** apply monotone constraints to this cell.

## The real lever (out of FS/HP-loop scope → V2)

The ranking signal is strong (2.1× R-precision) but the *calibrated probability* is capped by the prevalence drift. The lever is **recency weighting / regime-conditional calibration**, not FS/HP. Parked for a V2 methodology slice (see `docs/gbdt/V1.1_TBD.md`).

## Reusable lessons (→ skill + memory)

1. **Read the train/val gap before pruning.** A negative (or ≤0.02) train/val gap = no overfit → FS will *hurt*, not help. The fallback callback's prune-on-sight is wrong for non-overfit cells. (Early-stopping *firing* is orthogonal — healthy tree-count selection, not an overfit signal; don't conjoin it with the gap.)
2. **HP-ceiling detection**: scan depth + lr first; if val_brier stays in a tiny band across diverse configs, declare the ceiling and stop — don't burn 8 iterations.
3. **Monotone constraints: check the unconstrained model's 1D PDP, not the marginal correlation — and even then expect neutral-to-harmful, never free.** Marginal-monotone ≠ model-internally-monotone (long-window vol learns an inverted-U the Spearman hides). The 1D-PDP check is *necessary but not sufficient*: CatBoost enforces monotonicity at the tree-structure level, degrading conditional interactions even when the net PDP is monotone (iter 6 still lost vs baseline). The harm has **two components** (iter 8/9 ablation): a fixed ~0.0012 "constraints-on" cost present even for zero-interaction features, plus a dominant interaction-specific cost concentrated in high-interaction features (constraining the 8 high-interaction estimators alone reproduces the full 17-constraint harm). There is **no safe subset** to constrain on an interaction-driven cell — don't bother.
4. **importance≈0 ≠ unrelated.** Often it's redundancy (collinearity). FS that removes redundant features is ~neutral on val — it won't *improve* a non-overfit model.
5. **Heuristic feature priors are cell-specific.** F16 recent-event features that look high-signal in EDA can be ρ≈0 in a given cell. Verify per cell.
6. **A clean negative is a result.** "No FS/HP/constraint headroom; ranking signal robust; brier capped by prevalence drift" is the valuable output a fixed heuristic callback cannot produce.

## Reproducibility

- Specs: `configs/gbdt/experiments/nifty50_manualloop_iter{1..9}.yaml`. iter0 = the screening run artifact.
- Analysis scripts: `scripts/gbdt/{monotonic_feature_analysis,interaction_before_after,pdp_and_corr,monotone_1d_audit,pruned_feature_investigation}.py`.
- Figures: `results/gbdt/experiments/nifty50_{interaction_before_after,pdp_before_after,feature_corr_heatmap}.png`.
- Cached in-sample matrix: `results/gbdt/experiments/_nifty50_insample_matrix.parquet`.

## R-Precision@K (current methodology — added 2026-06-01)

Per `.claude/memories/project-r-precision-methodology.md`, R-Precision@K is the post-2026-06-01 headline cross-cell metric for gbdt — defined as `R-Precision@K = (1/Q) · Σ_q r_q / min(K, R_q)` over the Q days where R_q > 0 (R_q = positives on day q; r_q = positives caught in top-K picks on day q; macro-averaged, equal weight per day; K fixed). Recomputed from each cell's `predictions/test.csv`:

| cell | rows | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|---|---|
| nifty50_up_10pct_25d_dd5pct | 3450 | 17.9% | 0.733 | 0.229 | 0.252 | 0.235 | 0.288 | 0.609 |

The canonical CSV row tracks the iter-0 / all-279 baseline (the documented headline) — the iter-1..9 probes were val-only excursions and are not separate cells.

Cross-links: `[[project-r-precision-methodology]]`, `docs/gbdt/_138_h25_cross_market_combined.md` (the cross-market context), PR #48 (the V1.1 agent-loop design this rehearses).
