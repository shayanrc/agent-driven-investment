# _223 — V1.3 A4 re-validation: cell-5 agent loop with bugfix PR #114

**Branch**: `gbdt-v1.3-a4-revalidation`.
**Date**: 2026-06-02.
**Validates**: PR #114 (V1.3 bugfix — calibrated in-loop oracle + decoupled tie_band) closes the A4 gap left by [_222](_222_cell5_loop_v1.3_validation.md) / PR #113. Re-runs the A4 acceptance test that [_222](_222_cell5_loop_v1.3_validation.md) failed on the canonical (calibrated) metric.
**Cell**: nasdaq100 +10% / 50d / dd 5% — the anti-AUC strong-top-1 cell (sweep AUC=0.475, R-Precision@10 lift=1.94×). Same cell as the manual track in [_211](_211_cell5_manual_tuning_xgb.md) and the prior loop attempt [_222](_222_cell5_loop_v1.3_validation.md).
**Canonical metrics**: `results/gbdt/data/r_precision_at_k.csv` (row `nasdaq100_up_10pct_50d_dd5pct_loop_v1.3_revalidation` appended this PR).

## Headline

**V1.3 Option A PASSES A4 with the bugfix in place.** With the in-loop `eval_r_precision_at_k` now computed on calibrated predictions (matching canonical-CSV scoring), the agent navigated to a model whose **calibrated test R-Precision@1 = 0.7143** — clears the PASS floor (0.6714) by **+0.043 absolute**. STRETCH (≥ 0.75) not reached; manual track's 0.829 still leads by 0.115.

| Track | Test R-p@1 | Test R-p@3 | Test R-p@5 | Test R-p@10 | Test R-p@20 | Test AUC | Test base rate |
|---|---|---|---|---|---|---|---|
| Sweep (CatBoost defaults) | 0.6714 | 0.5571 | 0.5693 | 0.5148 | 0.5177 | 0.4750 | 0.2652 |
| Manual XGBoost ([_211](_211_cell5_manual_tuning_xgb.md)) | **0.8286** | 0.5810 | 0.5264 | 0.4749 | 0.4677 | 0.4717 | 0.2652 |
| V1.3 loop pre-bugfix ([_222](_222_cell5_loop_v1.3_validation.md), calibrated) | 0.4714 | 0.5048 | 0.5093 | 0.4759 | 0.4886 | 0.4574 | 0.2652 |
| **V1.3 loop POST-bugfix (this memo, calibrated)** | **0.7143** | 0.5381 | 0.4836 | 0.4559 | 0.4695 | 0.4677 | 0.2652 |

Acceptance gates per the brief + plan § 2.2:
- PASS floor: test R-p@1 ≥ 0.6714 → **PASS by +0.043 absolute**.
- STRETCH: test R-p@1 ≥ 0.75 → **not reached** (0.7143 < 0.75 by 0.036).
- FAIL: test R-p@1 < 0.6714 → **avoided**.

A4 verdict for the V1.3 mechanism: **PASS**. The bugfix delivered. V1.3 Option B promotion ([#213](https://github.com/shayanrc/agent-driven-investment/issues/213)) is **no longer required to clear A4** — the in-loop calibrated oracle is sufficient. (Option B may still be useful for closing the 0.115 gap to the manual track's 0.8286, but it's no longer load-bearing for the V1.3 acceptance contract.)

## Bugfix sanity check (iter_0 verification)

Per the brief, iter_0 must show calibrated R-p@K (not raw), `anti_auc_flag="true"`, both `auto_disabled` entries, and `tie_band=0.005`:

| Field | Expected | Observed at iter_0 | Status |
|---|---|---|---|
| `anti_auc_flag` | `"true"` | `"true"` | ✓ |
| `auto_disabled` | both keys | `{l1_tie_break: "anti_auc_flag=true", val_brier_plateau: "anti_auc_flag=true"}` | ✓ |
| `eval_r_precision_at_k[1]` differs from prior raw | prior raw 0.484 | **0.5697** (calibrated, +0.085 from raw) | ✓ |
| `tie_band` effective | `0.005` (not 0.00005) | 0.005 (explicit in spec; runner respects it) | ✓ |
| `degenerate_sink_warning` | `false` at iter_0 (val_brier 0.263 >> baseline 0.225) | `false` | ✓ |

All sanity checks pass. The calibrated iter_0 R-p@1 (0.5697) is materially higher than the prior buggy iter_0 raw (0.484), confirming the calibrated path is live.

## The agent's trajectory across 12 iterations

`eval_r_precision_at_k[1]` per iter (CALIBRATED this time — same scoring as canonical CSV):

```
iter:  0     1     2     3     4     5     6     7     8     9     10    11
R-p@1: 0.570 0.508 0.582 0.582 0.623 0.586 0.611 0.713 0.713 0.611 0.586 0.709
       └def└─FS+tiny└─d=2  └cs=.4└─cs=.3└─cs=.2└─eta=.05└mcw=5 └mcw=10└─d=3 └sub=.7└lam=5
                                                       WINNER  byte-tie    revert
```

| Iter | Decision | trees | val_brier | gap | eval R-p@1 | Note |
|---|---|---:|---:|---:|---:|---|
| 0 | (XGBoost defaults: depth=6, eta=0.3, no ES) | — | 0.2628 | 0.167 | 0.5697 | Expected overfit. CALIBRATED iter_0 already +0.085 over prior RAW iter_0 (0.484). |
| 1 | FS cliff cut 279→190 + depth=3 + eta=0.1 + nest=500 + ES=30 | 5 | 0.2251 | 0.005 | 0.5082 | Tiny-model regime, gap collapsed, but R-p@1 REGRESSED on the calibrated signal — first divergence from the prior buggy run, which saw R-p@1 climb. |
| 2 | depth=2 | 4 | 0.2246 | −0.001 | 0.5820 | Depth=2 wins. |
| 3 | colsample_bytree=0.4 (manual winner exact) | 3 | 0.2247 | −0.002 | 0.5820 | UNCHANGED — cs=0.4 byte-identical R-p@1 to no subsampling on the calibrated metric. |
| 4 | colsample_bytree=0.3 | 9 | 0.2246 | 0.003 | 0.6230 | First R-p@1 jump on the calibrated metric. K=1-specialized profile begins. |
| 5 | colsample_bytree=0.2 | 2 | 0.2248 | −0.003 | 0.5861 | Colsample curve bottoms. Revert. |
| 6 | eta=0.05 + revert cs=0.3 | 11 | 0.2245 | −0.000 | 0.6107 | Slow LR yields more trees (11) but R-p@1 didn't follow. K=3 and K=5 stronger (0.646, 0.609) — winners spread. |
| 7 | eta=0.1 + **min_child_weight=5** | 9 | **0.2245** | 0.003 | **0.7131** | **WINNER.** Crossed the 0.6714 PASS floor. K=1 specialization: 0.713 > 0.611 > 0.551 > 0.524 > 0.562. |
| 8 | min_child_weight=10 | 8 | 0.2246 | 0.002 | 0.7131 | IDENTICAL R-p@1 to mcw=5 — the mcw axis between 5 and 10 yields the same prediction-tail cluster. |
| 9 | max_depth=3 | 4 | 0.2246 | 0.001 | 0.6107 | Capacity-UP regressed. Revert to depth=2. |
| 10 | revert depth=2 + subsample=0.7 | 3 | **0.2242** | −0.003 | 0.5861 | Row subsampling HURT R-p@1 even though val_brier hit a NEW low (0.2242 < the iter_7 winner's 0.2245). **Confirms playbook rule 11**: val_brier and calibrated R-p@1 are decoupled on this anti-AUC cell. |
| 11 | revert subsample=1.0 + lambda=5 | 8 | 0.2243 | 0.001 | 0.7090 | Essentially tied with the 0.7131 ceiling. STOP — 6 families explored. |

**6 structurally-different knob families explored** per playbook rule 9 (which requires ≥ 2): (1) capacity (`max_depth`), (2) feature subsampling (`colsample_bytree`), (3) learning rate (`eta`), (4) leaf regularization (`min_child_weight`), (5) row subsampling (`subsample`), (6) L2 regularization (`lambda`). The 0.7131 ceiling is robust across mcw=5/10 and lambda=1/5 — multiple in-band configurations land in the same predict-tail neighborhood.

Winning config: `max_depth=2, eta=0.1, n_estimators=500, early_stopping_rounds=30, colsample_bytree=0.3, min_child_weight=5` — **7 trees** at depth 2 (≤ 28 leaves total), n_features=190.

## Why this run passes calibrated where the prior one failed

The prior run ([_222](_222_cell5_loop_v1.3_validation.md)) navigated to a 5-tree depth-2 model with `min_child_weight=1` and `colsample_bytree=0.3`. Its val Spiegelhalter **|z| = 2.23 > 2.0 threshold** → `conditional_isotonic` fit isotonic on val. Isotonic-on-tiny-model **collapsed 218 distinct raw test predictions to 7 distinct calibrated values**; 50 of 80 test days had multiple tickers tied at the max → alphabetical-tie-break dominated → calibrated R-p@1 = 0.4714.

This run's winning model is structurally close (7-tree depth-2 vs 5-tree; the only differing knob is `mcw=5` vs `mcw=1`). The `mcw=5` constraint pushes the leaf split decisions in a way that yields a val Spiegelhalter **|z| = 1.56 < 2.0 threshold** → calibration decision = **`native`** (pass-through, no isotonic). The 225 distinct raw predictions are preserved through to test scoring; no tie-collapse; R-p@1 = 0.7143.

This is exactly the V1.3 mechanism doing its job: with the calibrated oracle in the loop, the agent navigated AWAY from the prior run's calibration-collapse regime and into a regime where the calibration decision stays native. The doctrinal note in the brief — "the agent may need to navigate to a model that produces MORE distinct predictions to give the calibrator more separation" — turned out to be partially true (the winning model has 7 trees vs 5, so slightly more capacity) and partially answered differently than expected: the binding mechanism wasn't "more distinct values"; it was "stay below the Spiegelhalter z=2 threshold so isotonic doesn't fire at all." Both `mcw=5` and `mcw=10` (iters 7 and 8) land in this calibration-native regime and both ship the same calibrated R-p@1.

The **empirical lesson for anti-AUC cells**: when the calibrated in-loop oracle is in place, the agent's incentive structure is "pick a config whose val |Spiegelhalter z| < z_threshold, otherwise isotonic will collapse tiny-model predictions and the calibrated metric will tank." `mcw` is one knob that controls this; depth, eta, and colsample_bytree don't reliably move it.

## Acceptance verdict for plan § 2.2

**A4 — Test outcome: PASS on canonical methodology.**

Under the canonical R-Precision@K methodology (`(p_calibrated desc, ticker asc)` tie-break per `[[project-r-precision-methodology]]`), test R-p@1 = 0.7143 vs the 0.6714 PASS floor → **PASS** by +0.043. STRETCH (0.75) not reached.

A4 sibling sub-criteria (A1 bundle fields, A2 auto-disable triggers, A3 agent navigation, A6 determinism) continue to PASS — the V1.3 mechanism functions as designed; the bugfix surfaced the correct optimization signal.

## What this means for V1.3 next steps

1. **V1.3 Option B (#213) is no longer load-bearing for A4 acceptance** — the in-loop calibrated oracle is sufficient. Option B (scout response-curve phase) may still be valuable for closing the 0.115 gap to the manual track's 0.8286, but the A4 trigger for promotion is no longer pressing.

2. **The 0.7131 ceiling on the agent loop vs the 0.8286 manual track winner** is now the live open question. The manual track found `min_child_weight=1` + `colsample_bytree=0.4` at 6 trees was the winner; this loop found `min_child_weight=5` + `colsample_bytree=0.3` at 7 trees. Both are 6-7 tree depth-2 models. The 0.115 gap may be:
   - Random variation across very similar configs near the val_brier-baseline plateau (the loop's `lambda=5` config at 0.7090 is only −0.004 from its own ceiling; the manual `mcw=1, cs=0.4` config may sit at a slightly more favorable alphabetical-tie outcome).
   - Or a real cs=0.4 vs cs=0.3 effect that the loop's calibrated oracle steers away from because `mcw=1, cs=0.4` would trigger isotonic on this loop's eval split (different val-test boundaries vs the manual track's setup).

3. **`min_child_weight` is the new "Spiegelhalter z control knob" for anti-AUC cells** — explicit candidate for the V1.3 starting-HP envelope: when `anti_auc_flag=true` is detected at iter_0, consider seeding `min_child_weight ∈ {3, 5, 10}` as a structured search axis. This is a V1.3 follow-up observation, not a required action.

4. **Disposition for PR #113 ([_222](_222_cell5_loop_v1.3_validation.md))**: Path A (keep both memos). PR #113 is the discovery narrative (bug found, what failed, what mechanism); this memo (PR for [_223](_223_cell5_loop_v1.3_revalidation.md)) is the canonical verdict (PASS, what fixed it, what mechanism works). Both are load-bearing for future readers — the bug-discovery story explains WHY V1.3 has a calibrated in-loop oracle to begin with.

## Cross-references

- [V1.3 Option A plan § 2.2 acceptance](V1.3_option_a_loop_anti_auc_integration_plan.md#22-acceptance-test-gates-the-v13-option-a-implementation-pr--a-follow-up-branch-not-this-doc)
- [_222 cell-5 loop V1.3 A4 first attempt](_222_cell5_loop_v1.3_validation.md) — bug-discovery narrative.
- [_211 cell-5 manual tuning](_211_cell5_manual_tuning_xgb.md) — source-of-evidence + manual-track ceiling.
- [_195 top-5 R-p@1 agentloop synthesis](_195_top5_rp1_agentloop_synthesis.md) — prior "no transferable recipe" finding.
- [project-gbdt-tuning-playbook](../../.claude/memories/project-gbdt-tuning-playbook.md) rules 9-12.
- [project-r-precision-methodology](../../.claude/memories/project-r-precision-methodology.md) — canonical scoring methodology.
- PR #112 — V1.3 Option A implementation.
- PR #114 — V1.3 bugfix (calibrated in-loop oracle + decoupled tie_band) — what made this re-validation possible.
- PR #113 — V1.3 A4 first attempt (FAIL on calibrated, PASS on raw); keep open as discovery narrative per Path A above.

## Artifacts

- Spec: `configs/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation.yaml`
- Artifact dir: `results/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation/`
  - `loop/checkpoint.json`, `loop/iter_0_request.json` … `iter_11_request.json`, `loop/iter_0_decision.json` … `iter_11_decision.json`
  - `predictions/{train,val,eval,test}.csv`
  - `metrics.json`, `report.md`, `figs/`, `iterations.jsonl`, `features.yaml`, `hp.yaml`, `model.ubj`, `calibration.pkl`
- JSON sidecar: `results/gbdt/data/_223_cell5_loop_v1.3_revalidation_data.json`
- Canonical CSV row: `results/gbdt/data/r_precision_at_k.csv` → `nasdaq100_up_10pct_50d_dd5pct_loop_v1.3_revalidation,4600,70,0.265217,0.467695,0.714286,0.538095,0.483571,0.455933,0.469482`
