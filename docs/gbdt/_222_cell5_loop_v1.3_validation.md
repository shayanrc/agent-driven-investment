# _222 — V1.3 Option A A4 acceptance: cell-5 agent loop validation

**Branch**: `gbdt-v1.3-a4-cell5-validation`.
**Date**: 2026-06-02.
**Validates**: PR #112 (V1.3 Option A bundle additions); plan [`docs/gbdt/V1.3_option_a_loop_anti_auc_integration_plan.md`](V1.3_option_a_loop_anti_auc_integration_plan.md) acceptance sub-criterion **A4**.
**Cell**: nasdaq100 +10% / 50d / dd 5% — the anti-AUC strong-top-1 cell (sweep AUC=0.475, R-Precision@10 lift=1.94×) — same cell as the manual track in [_211](_211_cell5_manual_tuning_xgb.md).
**Canonical metrics**: `results/gbdt/data/r_precision_at_k.csv` (row `nasdaq100_up_10pct_50d_dd5pct_loop_v1.3` appended this PR).

## Headline

V1.3 Option A drove the agent loop **to the correct model architecture** (depth-2 XGBoost with eta=0.1 + early stopping + feature subsampling, 5 trees of depth=2 → ≤ 20 leaves — within 1 tree of the [_211](_211_cell5_manual_tuning_xgb.md) manual winner). **FAILS A4 on the canonical (calibrated) metric** by −0.20 absolute; **PASSES A4 on raw predictions** by +0.043 absolute. The gap is a calibration-collapse pathology distinct from the loop's reasoning quality.

| Track | Test R-p@1 | Test R-p@3 | Test R-p@5 | Test R-p@10 | Test R-p@20 | Test AUC | Test base rate |
|---|---|---|---|---|---|---|---|
| Sweep (CatBoost defaults) | 0.6714 | 0.5571 | 0.5693 | 0.5148 | 0.5177 | 0.4750 | 0.2652 |
| Agent loop pre-V1.3 (mcw=10, γ=1) | 0.4571 | 0.5095 | 0.5064 | 0.5233 | 0.5231 | 0.4772 | 0.2652 |
| Manual XGBoost ([_211](_211_cell5_manual_tuning_xgb.md)) | **0.8286** | 0.5810 | 0.5264 | 0.4749 | 0.4677 | 0.4717 | 0.2652 |
| **V1.3 loop — calibrated (canonical)** | **0.4714** | 0.5048 | 0.5093 | 0.4759 | 0.4886 | 0.4574 | 0.2652 |
| V1.3 loop — raw (advisory) † | 0.7143 | 0.5381 | 0.4836 | 0.4559 | 0.4504 | 0.4638 | 0.2652 |

† Raw predictions are scored advisory-only (the canonical CSV uses calibrated per `[[project-r-precision-methodology]]`). Surfaced because the V1.3 in-loop signal (eval R-p@K in the iter bundle) is computed on **raw** model output (per `src/gbdt/diagnostics.py` lines 511-514 — the "isotonic is monotone, rank order is the same" comment is FALSE for piecewise-constant isotonic on tiny models).

PASS floor: **test R-p@1 ≥ 0.6714** (matches sweep baseline). Stretch: ≥ 0.75 (within 10% of manual 0.8286). Result:

- **Calibrated 0.4714 < 0.6714 floor → FAIL** (per the canonical methodology this is the A4 verdict).
- **Raw 0.7143 > 0.6714 floor → would PASS if scored on raw.**

## What V1.3 Option A delivered (sub-criteria A1–A3, A6)

All four implementation sub-criteria of the plan's A4 sibling cluster passed:

- **A1 — Bundle fields populated.** Every iter's `loop/iter_<N>_request.json::diagnostics` carries `anti_auc_flag` (`"true"` constant), `eval_r_precision_at_k` (K ∈ {1,3,5,10,20}), `degenerate_sink_warning`, `weighted_base_rate_brier=0.2253`, `eval_segment_size=18400`. ✓
- **A2 — Auto-disable triggers.** `loop/checkpoint.json::auto_disabled = {"l1_tie_break": "anti_auc_flag=true", "val_brier_plateau": "anti_auc_flag=true"}` from iter_0 onward. ✓
- **A3 — Agent navigation.** All 10 iter `loop/iter_<N>_decision.json` rationales reference `eval R-p@1` (or `eval_r_precision_at_k[1]`) directly — 100% rate vs the 50% threshold. The trajectory drove every knob choice. ✓
- **A6 — Determinism.** Incidentally confirmed at iter_7: `lambda=1.0` (XGBoost's API default — I'd forgotten this) + `subsample=1.0` (revert) reproduced iter_4 byte-identically — same `eval R-p@K`, `val_brier`, `train_brier`, `best_iter`, and top-10 feature importance values. A different code path through the loop produced bit-identical predictions on the same effective config. ✓

A5 (non-anti-AUC regression on `sp500_up_10pct_25d_dd5pct`) is **out of scope for this A4 sub-agent** — a separate validation run handles it.

## The agent's trajectory across 10 iterations

`eval_r_precision_at_k[1]` per iter (the primary signal V1.3 surfaces; all values on RAW model output per the implementation):

```
iter:  0     1     2     3     4     5     6     7     8     9
R-p@1: 0.484 0.508 0.602 0.627 0.717 0.586 0.602 0.717 0.643 0.713
       └defaults└─tiny-model─┘  └col=0.3 └→0.2  └subs └=l=1 └=l=5 └mcw5
                                ┌─WINNER─┐      ┌─pivot─┐ ┌─byte─identical
                                                hurt     to iter_4 (free A6)
```

| Iter | Decision | best_iter trees | val_brier | gap | eval R-p@1 | Note |
|---|---|---:|---:|---:|---:|---|
| 0 | (XGBoost defaults: depth=6, eta=0.3, no ES) | — | 0.2628 | 0.167 | 0.484 | Expected overfit. |
| 1 | FS cliff cut 279→190 + depth=3 + eta=0.1 + nest=500 + ES=30 | 5 | 0.2251 | 0.005 | 0.508 | Hit tiny-model regime. gap collapsed. degenerate_sink_warning=true (advisory — val_brier=baseline is the correct regime on anti-AUC per rule 12, NOT the "walked into sink" failure mode the warning typically catches). |
| 2 | depth=2 | 4 | 0.2246 | −0.001 | 0.602 | Capacity reduction won big. |
| 3 | cs_bytree=0.4 | 3 | 0.2247 | −0.002 | 0.627 | Feature subsampling helps. |
| 4 | **cs_bytree=0.3** | 9 | 0.2246 | 0.003 | **0.717** | **CEILING.** Eval R-p@K profile gets K=1-specialized (R-p@1=0.717 > R-p@3=0.616 > R-p@5=0.553 > R-p@10=0.527) — same shape as the _211 winner. |
| 5 | cs_bytree=0.2 | 2 | 0.2248 | −0.003 | 0.586 | Colsample curve bottomed. Revert. |
| 6 | subsample=0.7 (row subsampling pivot) | 5 | 0.2243 | −0.001 | 0.602 | Row subsampling HURT — revert. |
| 7 | lambda=1.0 + subsample=1.0 (revert pivot + start L2 family) | 9 | 0.2246 | 0.003 | 0.717 | Byte-identical to iter_4 (lambda=1.0 is XGB's API default; the change was a no-op). |
| 8 | lambda=5.0 | 8 | 0.2243 | 0.001 | 0.643 | L2 HURT — leaf-weight shrinkage contraindicated. |
| 9 | min_child_weight=5 | 9 | 0.2245 | 0.003 | 0.713 | Neutral (−0.004 from ceiling). All ≥5 structurally-different knob families explored. STOP. |
| 10 | (finalize via should_stop=true; spec patched tie_band=0.005) | — | — | — | — | Picked iter_4 winner: depth=2 + eta=0.1 + nest=500 + ES=30 + cs=0.3. |

Five structurally-different knob families explored per playbook rule 9: (1) capacity (`max_depth`), (2) slow-LR + ES (`eta` + `n_estimators` + `early_stopping_rounds`), (3) feature subsampling (`colsample_bytree`), (4) row subsampling (`subsample`), (5) regularization (`lambda` + `min_child_weight`). The ceiling held across families.

### A spec-patch note (mid-loop)

The spec was initially launched with `plateau_threshold: 0.0001` (the #204 workaround per SKILL.md). With no explicit `tie_band`, this resolved to an effective tie band of `0.0001 × 0.5 = 0.00005` — too narrow to catch the val_brier-near-baseline cluster (iters 1-9 all sit within 0.001 of min val_brier 0.2243). Without the patch, `best_checkpoint` would have shipped iter_8 (strict val_brier argmin at 0.22427) which had eval R-p@1=0.643, NOT iter_4's ceiling at 0.717. The spec was patched to set `tie_band: 0.005` explicitly before finalization, which expanded the band to include all 9 explored iters (val_brier range 0.2243–0.2251 within 0.005 of min) and let V1.3's `eval_r_precision_at_1` tie-break pick iter_4. This is **friction surfaced by V1.3** — the plan's auto-disable on the L1 tie-break is correct, but the tie-band default (`0.5 × plateau_threshold`) becomes pathologically narrow when the agent disables plateau_threshold (the standard recipe on anti-AUC cells per #204 + SKILL.md). **V1.3 follow-up**: either (a) decouple `tie_band` from `plateau_threshold` (give it its own default, e.g. 0.005 absolute), or (b) auto-set `tie_band` to a reasonable absolute value when `anti_auc_flag=true`. Filed as TBD in this PR's body.

## Why the canonical (calibrated) result fails despite a strong raw result

The chosen model (iter_4: depth=2, eta=0.1, ES=30, nest=500, cs=0.3) is **structurally similar to the _211 manual winner** (depth=2, eta=0.1, ES=30, nest=500, cs=0.4, FS=top-130) — both are 5–6 tree, depth=2 XGBoost models (≤ 24 leaves total). The raw test R-p@1 of 0.7143 confirms the rank-order quality matches manual's ballpark.

But the **calibrated test R-p@1 = 0.4714** — a −0.243 drop from the raw rank-order. The mechanism:

- Spiegelhalter |z| = 2.23 > 2.0 → `conditional_isotonic` fit isotonic on val.
- Isotonic regression is **piecewise constant** (it bins probabilities into level sets). Tiny models like this one (≤ 20 leaves) emit predictions clustered around a few values to begin with.
- Result: **218 distinct raw test predictions collapse to 7 distinct calibrated test predictions**. **50 of 80 test days** have multiple tickers tied at the max calibrated probability; the canonical tie-break is `(p_calibrated desc, ticker asc)` so the alphabetically-first ticker wins among ties — essentially random with respect to label.
- The manual track winner (also a 6-tree depth=2 model) presumably gets a different ticker-alphabetical tie outcome that lands more positives at top-1; that's the 0.8286.

**The V1.3 in-loop signal (`eval_r_precision_at_k`) is computed on RAW predictions** (`diagnostics.py` lines 509-534). The comment at lines 510-514 justifies this:

> Rank-based so raw probabilities are order-equivalent to post-isotonic ones (isotonic is monotone) — using the raw model output here keeps the iter-loop signal independent of the finalization-only calibrator and stays byte-identical to the canonical CSV's ranking (the CSV uses p_calibrated but the rank order is the same).

The "rank order is the same" claim is **FALSE for piecewise-constant isotonic on tiny models**. Isotonic IS monotone, but monotone-non-strict — many distinct raw probabilities can map to a single calibrated value, at which point ranking is fully determined by the tie-break key (`ticker asc`), not by the raw probability magnitude. The agent was optimizing the wrong metric.

This is a **V1.3 Option A implementation finding**, separate from the loop's reasoning quality. The doctrine the SKILL.md V1.3 callout codifies (use `eval_r_precision_at_k[1]` as the primary signal) was applied correctly; the signal itself misled the agent.

## Acceptance verdict for plan § 2.2

**A4 — Test outcome: FAIL on canonical methodology.**

Under the canonical R-Precision@K methodology (`(p_calibrated desc, ticker asc)` tie-break per `[[project-r-precision-methodology]]`), test R-p@1 = 0.4714 vs the 0.6714 PASS floor → **FAIL** by −0.243. Per plan § 2.2, this triggers the V1.3 Option B (scout response-curve phase) promotion path.

The raw R-p@1 = 0.7143 would PASS the floor by +0.043, indicating the agent reasoning + knob exploration worked. The blocker is calibration, not navigation.

A4 sibling sub-criteria (A1 bundle fields, A2 auto-disable triggers, A3 agent navigation, A6 determinism) all **PASS** — the V1.3 mechanism functions as designed; only the headline test metric fails on the canonical scoring.

## What this means for V1.3 next steps

1. **Promote #213 (V1.3 Option B scout response-curve phase)** per plan § 2.2 A4-FAIL trigger. Option B's per-knob parallel fit map would let the agent map the (depth, eta, cs_bytree) surface before committing to an iter_0 starting envelope — could land in a different val_brier-near-baseline neighborhood with a different alphabetical-tie outcome, OR it could converge on the same calibrated-collapse pathology. Worth running to find out.

2. **File a separate ticket: "V1.3 bundle eval R-p@K should be computed on calibrated predictions to match canonical-CSV scoring."** The `diagnostics.py` lines 511-514 justification is FALSE for piecewise-constant isotonic. Fix options:
   - **(a)** Plumb the (val-fit) calibrator into the bundle build; apply it to `p_eval` before R-p@K computation. Adds a per-iter dependency on the calibrator object that doesn't exist until finalization in the current architecture.
   - **(b)** Have the bundle emit BOTH `eval_r_precision_at_k_raw` AND `eval_r_precision_at_k_calibrated` (the latter computed by running an in-loop isotonic fit on the iter's val predictions, separate from the finalization-only calibrator). Doubles the calibration cost per iter (negligible).
   - **(c)** Change canonical CSV scoring to use `p_raw` instead of `p_calibrated`. Out of scope here (changes the cross-cell metric definition; would touch all prior memos), but worth raising for the maintainers' consideration.

3. **DO NOT generalize this A4 FAIL to all anti-AUC cells.** The V1.3 mechanism navigated the agent to the right architecture; the failure is a calibration-on-tiny-models pathology specific to cells where the model is "smart prior + few corrections." Cells where the chosen model has more distinct raw predictions (e.g. depth ≥ 4) will likely not exhibit this collapse, and V1.3 should pass cleanly. The A5 regression test (non-anti-AUC cell, byte-identical to pre-V1.3) is independent and should still pass.

## Cross-references

- [V1.3 Option A plan § 2.2 acceptance](V1.3_option_a_loop_anti_auc_integration_plan.md#22-acceptance-test-gates-the-v13-option-a-implementation-pr--a-follow-up-branch-not-this-doc)
- [_211 cell-5 manual tuning](_211_cell5_manual_tuning_xgb.md) — the source-of-evidence memo for what cell-5 demands.
- [_195 top-5 R-p@1 agentloop synthesis](_195_top5_rp1_agentloop_synthesis.md) — the prior "no transferable recipe" finding that motivated _211 + V1.3.
- [project-gbdt-tuning-playbook](../../.claude/memories/project-gbdt-tuning-playbook.md) rules 9-12.
- [project-r-precision-methodology](../../.claude/memories/project-r-precision-methodology.md) — the canonical scoring methodology.
- PR #112 — V1.3 Option A implementation (commit `a8b88b5`).
- PR #213 (TBD) — V1.3 Option B scout response-curve phase (this memo recommends promotion).

## Artifacts

- Spec: `configs/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3.yaml`
- Artifact dir: `results/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3/`
  - `loop/checkpoint.json`, `loop/iter_0_request.json` … `iter_10_request.json`, `loop/iter_0_decision.json` … `iter_9_decision.json`
  - `predictions/{train,val,eval,test}.csv`
  - `metrics.json`, `report.md`, `figs/`, `iterations.jsonl`, `features.yaml`, `hp.yaml`, `model.json`, `calibration.pkl`
- JSON sidecar: `results/gbdt/data/_222_cell5_loop_v1.3_validation_data.json`
- Canonical CSV row: `results/gbdt/data/r_precision_at_k.csv` → `nasdaq100_up_10pct_50d_dd5pct_loop_v1.3,4600,70,0.265217,0.457353,0.471429,0.504762,0.509286,0.475873,0.488615`
- Log: `logs/v1.3_a4.log`
