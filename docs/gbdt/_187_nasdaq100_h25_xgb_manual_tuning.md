# Manual XGBoost FS+HP tuning — nasdaq100 +10% / 25d / dd5% (H=25)

> **Methodology note (2026-06-01)**: Numbers in this memo's body use the legacy "weighted R-precision" metric (per-day variable K = R(d), micro-aggregated). The project headline metric was renamed 2026-06-01 to **R-Precision@K** (per-day fixed K, macro-aggregated via `(1/Q)·Σ r_q/min(K,R_q)`). See the "R-Precision@K (current methodology)" section at the bottom of this memo for the cells in this memo recomputed under the new metric, plus `.claude/memories/project-r-precision-methodology.md` for the full definition + relationship.

**Task #185.** Hand-driven (data-scientist-in-the-loop) XGBoost tuning loop. Purpose: the human XGBoost reference + probe which XGBoost knobs move the metric on our panels, to inform the agent-loop decision schema (esp. #184 interaction_constraints) BEFORE extending the agent loop again. Single-fit-per-iteration (`max_iterations: 1`); analyst edits `hp_starting` between fits; #181 feature-matrix cache makes post-iter-0 fits ~free. Cell signal documented in `_138` (CatBoost). Backend: XGBoost.

Panel: 92 nasdaq100 tickers, 645,189 rows. Splits 800/400/200/100 per ticker. Base rate ~25% (common-event cell). Calibration: conditional_isotonic.

## Iteration log

| iter | max_depth | η | λ | subsample | colsample | nfeat | val Brier | train-val gap | eval AUC | test AUC | eval wR-prec (base 0.252) | test wR-prec (base 0.273) | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 0.05 | 3.0 | 0.8 | 1.0 | 279 | 0.16888 | +0.01304 | 0.634 | 0.495 | 0.491 | 0.378 | baseline = Phase-7 repl config |
| 1 | 4 | 0.05 | 3.0 | 0.8 | 1.0 | 279 | 0.16881 | +0.00402 | 0.627 | 0.494 | 0.488 | 0.396 | depth↓: overfit gap closed 0.013→0.004, val Brier flat, calib Z 10.3→7.47; test AUC still ≈chance |
| 2 | 8 | 0.05 | 3.0 | 0.8 | 1.0 | 279 | 0.16926 | +0.01462 | 0.629 | 0.489 | 0.490 | 0.374 | depth↑: gap widens to 0.015, val Brier slightly worse; test AUC still ≈chance |
| 3 | 4 | 0.05 | 1.5 | 0.8 | 1.0 | 279 | 0.16967 | +0.00664 | 0.637 | 0.502 | 0.501 | 0.399 | λ↓ at d4: val Brier slightly worse, gap wider, calib worse (Z 13.0) |
| 4 | 4 | 0.05 | 6.0 | 0.8 | 1.0 | 279 | 0.16922 | +0.00163 | 0.629 | 0.489 | 0.490 | 0.398 | λ↑ at d4: gap LOWEST (0.0016), calib BEST (Z 6.26), val Brier ~flat |

### Phase A synthesis (capacity × regularization)
- **Tight HP-ceiling on val Brier:** entire grid spans 0.16881–0.16967 (0.5%), val-best = d4/λ3. Same signature as `_147` — no HP meaningfully moves val Brier.
- **Capacity/reg DO control overfit + calibration but NOT val Brier or ranking:** train-val gap ranges 0.0016 (d4/λ6) → 0.0146 (d8/λ3); calib Z 6.3 → 13.0. Best generalization-hygiene = **d4/λ6** (lowest gap, best calibration), val Brier within noise of val-best.
- **Test discrimination is HP-INVARIANT:** eval AUC ~0.63 / test AUC ~0.49 / test wR-prec ~1.45× hold across the WHOLE grid. No capacity/reg setting recovers test generalization → the test-window non-discrimination is a data/regime property, not a tuning problem (corroborates the #93 systemic eval→test decay finding; this H=25 cell is the one that drops into the test null-band).
- **AGENT-LOOP LESSON:** a val-Brier-only loop cannot distinguish d4/λ6 (gap 0.0016, clean) from d8/λ3 (gap 0.0146, overfit) — they have ~equal val Brier. The loop should ALSO weigh train-val gap + calibration (Spiegelhalter Z) to break val-Brier ties toward the better-generalizing config.

**Phase B plan:** XGBoost-distinct knobs the agent loop doesn't currently sweep — stochasticity (colsample_bytree, subsample) + split-conservativeness (min_child_weight, gamma) — on the d4/λ6 hygiene base (subsample 0.8 / colsample 1.0). One change per iter. Expect val-Brier plateau; watching gap/calibration/eval-rank for any movement.

| iter | knob change (from d4/λ6 base) | val Brier | gap | eval AUC | test AUC | calib Z | notes |
|---|---|---|---|---|---|---|---|
| 5 | colsample_bytree 1.0→0.8 | 0.16931 | +0.0054 | 0.632 | 0.497 | 11.45 | val flat; calib worse — no help |
| 6 | subsample 0.8→1.0 | 0.16907 | +0.0068 | 0.622 | 0.492 | 11.63 | val flat; gap up, calib worse — 0.8 (subsampling) is better |
| 7 | min_child_weight 1→10 | **0.16859** | +0.0017 | 0.634 | 0.497 | 9.41 | NEW val-best (marginal, within ceiling); low gap — split-conservativeness helps slightly. NOT in agent-loop knob set |
| 8 | gamma 0→1.0 | 0.16932 | +0.0021 | 0.632 | 0.491 | 7.17 | val flat — no effect |

### Phase B synthesis (XGBoost-distinct knobs the agent loop doesn't sweep)
- **HP-ceiling holds across ALL knobs:** full grid (depth/λ/colsample/subsample/mcw/gamma, ~13 configs) spans val Brier 0.16859–0.16967 (~0.6%). No knob escapes the ceiling.
- **`min_child_weight=10` is the one knob that nudged val Brier to the (marginal) best (0.16859)** with a clean gap (+0.0017) — and it is NOT in the agent loop's current XGBoost knob set. colsample/subsample/gamma were flat-or-worse; subsample 0.8 + colsample 1.0 (defaults) already near-optimal.
- **test AUC ~0.49 invariant across every config** — reaffirmed: test non-discrimination is fundamental to this cell (regime/data), unmovable by any HP.
- **AGENT-LOOP LESSON (2):** add `min_child_weight` to the XGBoost decision knob set — it was the only distinct knob to improve val Brier here. (Lesson 1 from Phase A: weigh train-val gap + Spiegelhalter Z to break val-Brier ties.)

**Final phase plan:** FS (does pruning to top-K help? — the cross-cell question: `_147` neutral-harmful vs nifty500 `_176` 279→39 helpful) + interaction_constraints (the #184 capability demo on this cell). Best config so far: d4 / λ6 / subsample0.8 / colsample1.0 / mcw10.

### Feature selection (algorithmic fs_hp_loop, max_iter 3, on the best HP base)
| iter | nfeat | val Brier | gap |
|---|---|---|---|
| 0 | 279 | 0.16859 | +0.00171 |
| 1 | 97 | 0.16859 | +0.00171 |
| 2 | 96 | 0.16859 | +0.00171 |

**FS is exactly neutral here.** The depth-4 model splits on only **98 / 279** features; pruning the unused 182 (279→97) is bit-for-bit identical val Brier (0.16859) with unchanged eval/test AUC (0.634 / 0.497) and test R-prec (1.39×). **nasdaq H=25 lands on the `_147` side (FS neutral) — NOT the nifty500 `_176` side (279→39 helpful).** FS-helpfulness is cell-dependent; the algorithmic FS is *safe* (a clean no-op when unhelpful), so it costs nothing to keep in the loop.

**Top-30 features by gain** are a **market-regime signature**: volatility estimators (parkinson_*, garman_klass_*, realized_vol_*), index/market context (index_return/runup/drawdown/vol_*), market beta_* at several lookbacks, cross-sectional vol z-scores, and seasonality (moy_sin/cos). Not stock-specific momentum — consistent with the regime-driven eval→test decay.

### interaction_constraints (#184 data point)
Deferred a full nasdaq re-probe: the TreeSHAP `pred_interactions` pass exceeds the diagnose time budget on this 279-feature US slice (timed out at 60 min), and the effect is predictably neutral on an HP-ceiling / test-invariant cell. **Phase 8 (`_175`) already verified the capability** on the sister `_147` cell — forbidding the top pair collapsed its SHAP interaction → 0 and tree co-occurrence 58.94 → 0 (honored), with neutral metric effect. The #184 conclusion stands on that evidence.

## Synthesis — nasdaq100 +10%/25d/dd5%, XGBoost (10 fits)

1. **HP-ceiling everywhere.** val Brier sits in a 0.6% band (0.16859–0.16967) across the *entire* grid — depth {4,6,8} × λ {1.5,3,6} × colsample {0.8,1} × subsample {0.8,1} × min_child_weight {1,10} × gamma {0,1}. No HP escapes it. Same signature as `_147`.
2. **The val-Brier-optimal config is NOT the generalization-best.** Capacity + regularization control the train-val gap (0.0016–0.0146, ~9×) and calibration Z (6.3–13.0) at ~constant val Brier. The cleanest config is **depth-4 / λ6 / mcw10** (lowest gap + best calibration + marginal val-best 0.16859).
3. **Test discrimination is HP-invariant** — eval AUC ~0.63 / test AUC ~0.49 / test R-prec ~1.45× across every config. The eval→test decay is a regime/data property (corroborates #93's systemic eval→test finding; this is the cell that drops into the test null-band), unmovable by tuning.
4. **FS neutral** (model self-prunes to 98/279; pruning the rest is a no-op); **signal is a market-regime signature**, not stock-specific.

### Agent-loop lessons (the run's purpose — feed #184 + future agent-loop work)
- **L1 — break val-Brier ties with train-val gap + Spiegelhalter Z.** A val-Brier-only loop cannot distinguish a clean config (d4/λ6, gap 0.0016) from an overfit one (d8/λ3, gap 0.0146) at equal val Brier; the decision rule should prefer lower gap + better calibration among val-equivalent configs.
- **L2 — add `min_child_weight` to the XGBoost knob set.** It was the only distinct knob to (marginally) improve val Brier here, and it isn't currently swept.
- **L3 — the algorithmic FS is safe** (bit-identical when unhelpful), so keeping it costs nothing; but it's cell-dependent (helpful on nifty500, neutral here / `_147`).
- **Meta — on HP-ceiling cells the loop should recognize the plateau + stop early (it does, via the plateau gate) and NOT chase test generalization via HP** (it's a regime property, not a tuning target).

**Verdict (user's read — no automated PASS/FAIL):** the run achieved its purpose — a clean knob-sensitivity map + 3 actionable agent-loop upgrades (L1–L3). The *cell* is test-marginal (a regime-decay cell, per #93), so not a production candidate, but it was a sound controlled probe. n=1 cell; the agent-loop lessons should be checked against a cell where FS *is* helpful (e.g. nifty500) before hard-coding.


## Belief updates

**Iter 0 (baseline).** Three findings: (1) **mild overfit at depth-6** — train-val gap is +0.013 (train better than val), the *opposite* of `_147`'s uniformly-negative gaps → this cell likely wants less capacity / more reg. (2) **Eval→test generalization decay** — eval has decent signal (R-prec 1.94×, AUC 0.63) that collapses on test (R-prec 1.04× per-day, AUC ≈0.495). (3) **Common-event cell** (~25% base rate), Brier worse than the base-rate constant on both splits (ranking-useful on eval, poorly calibrated absolutely). Central question for the run: does ANY capacity/reg setting move the eval→test gap, or is the test window a regime the features can't price?

**Phase A plan:** probe capacity DOWN first (depth-4), then depth-8 to bracket; then λ {1.5,6} at best depth, depth×λ jointly. Watch the eval→test gap, not just val Brier.

**Iter 1 (depth-4).** Confirms depth-6 mildly overfits: gap 0.013→0.004 **at zero val-Brier cost** (0.16888→0.16881) and better calibration (Z 10.3→7.47). eval AUC/R-prec ~flat; **test AUC stays ≈chance (0.494)** — capacity reduction did NOT recover test generalization, so the test-window non-discrimination looks like a regime/data property, not an overfit artifact. val Brier is on a plateau across depth (echoes `_147`'s HP-ceiling). depth-4 strictly dominates depth-6 here (same val Brier, less overfit, better calibrated) but unlocks no new signal. Next: depth-8 to close the bracket (expect gap widens, val Brier flat).

**Iter 2 (depth-8) → depth bracket complete.** Monotone capacity→overfit: gap 0.0040 (d4) → 0.0130 (d6) → 0.0146 (d8); val Brier 0.16881 / 0.16888 / 0.16926 (d4 marginally best). **depth-4 is the pick** (lowest val Brier + least overfit). This is the OPPOSITE of `_147` (which wanted depth-8) — optimal depth is cell-dependent, so the agent loop must explore depth DOWN as well as up. test AUC flat ~0.49 across all depths → test non-discrimination is capacity-invariant (regime/data property, reaffirmed). Proceeding to λ probes at depth-4 (λ 1.5 / 6.0) — expect another HP-ceiling plateau on val Brier.

## R-Precision@K (current methodology — added 2026-06-01)

Per `.claude/memories/project-r-precision-methodology.md`, R-Precision@K is the post-2026-06-01 headline cross-cell metric for gbdt. Recomputed from each cell's `predictions/test.csv`:

| cell | rows | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|---|---|
| nasdaq100_up_10pct_25d_dd5pct | 6900 | 27.3% | 0.511 | 0.537 | 0.526 | 0.536 | 0.507 | 0.508 |

The canonical CSV carries the canonical nasdaq100 +10%/25d cell artifact; the manual-tuning iter-0..8 probes were val-only excursions and are not separate cells in the CSV.
