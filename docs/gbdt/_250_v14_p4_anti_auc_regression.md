# #250 — V1.4 P4 cell-5 anti-AUC regression replay (V1.3 Option A path unchanged)

**Status:** complete 2026-06-08. V1.4 plan Phase P4 acceptance run. Memo + canonical CSV row appended + JSON sidecar shipped.

## Headline

Re-run of cell-5 (`nasdaq100_up_10pct_50d_dd5pct`, the anti-AUC strong-top-1 cell from [_211](_211_cell5_manual_tuning_xgb.md) / [_223](_223_cell5_loop_v1.3_revalidation.md)) on the V1.4 P1-patched codebase, to verify the V1.3 Option A `anti_auc_eval_rp1` finalize path is unchanged by the V1.4 P1 fix on the non-anti-AUC branch. **Mechanism PASSED**: `anti_auc_flag=true` detected at iter_0, both V1.3 auto-disables fired (`l1_tie_break` + `val_brier_plateau`), `loop/checkpoint.json::tiebreak_path = anti_auc_eval_rp1` (the V1.3 Option A branch — NOT the new V1.4 P1 `v14_val_flat_eval_rp1` branch). The V1.4 P1 patch is correctly scoped to the non-anti-AUC branch and does NOT regress the anti-AUC behavior.

**Substantive R-p@1 FAILED vs sweep gate** (test R-p@1 = 0.408 vs sweep 0.671, −39%) — but this matches the original [_223](_223_cell5_loop_v1.3_revalidation.md) re-validation pattern (3-iter agent loop is too shallow to reach the manual-track winner). The substantive shortfall is NOT a V1.4 P1 regression; it is the agent-in-3-iters vs the manual-track-in-30-iters gap that [_211](_211_cell5_manual_tuning_xgb.md) characterized. The original [_223](_223_cell5_loop_v1.3_revalidation.md) needed 12 iters + the `min_child_weight` knob to clear the floor.

## Setup

| field | value |
|---|---|
| Patch under test | V1.4 P1 (PR #135): non-anti-AUC eval R-p@1-best fallback when `val_brier_range < tie_band` |
| Regression target | V1.3 Option A `anti_auc_eval_rp1` finalize path — must remain unchanged |
| Cell | nasdaq100 +10% / 50d / dd 5% — anti-AUC (sweep AUC=0.477 ∈ [0.46, 0.54], R-Precision@10 lift=1.94× > 1.8×) |
| Backend / Mode | XGBoost / `agent_file_protocol` |
| Snapshot pin | `--snapshot-end 2026-05-22` |
| Gate (V1.4 P4) | `tiebreak_path == anti_auc_eval_rp1` (V1.3 path label, NOT V1.4 P1 label); V1.4 P1 must NOT change anti-AUC selection |

## V1.3 Option A bundle signals (iter_0 sanity)

All four signals present and correct per the [_223](_223_cell5_loop_v1.3_revalidation.md) sanity-check template: `anti_auc_flag=true`; `auto_disabled = {l1_tie_break: "anti_auc_flag=true", val_brier_plateau: "anti_auc_flag=true"}`; `tie_band = 0.005` (explicit spec override); and after finalize, `tiebreak_path = anti_auc_eval_rp1` (the V1.3 Option A path) NOT `v14_val_flat_eval_rp1` (the V1.4 P1 path).

## Iteration loop (3 iters)

| iter | hp_overlay | trees | train_val_gap | val_brier | eval R-p@1 | note |
|---|---|---:|---:|---:|---:|---|
| 0 | (XGBoost defaults: md=6 eta=0.3 n_est=100) | — | +0.1650 | 0.2590 | 0.5574 | Expected overfit on defaults. `anti_auc_flag=true` detected — V1.3 Option A path armed. |
| 1 | max_depth=2, eta=0.1, early_stopping_rounds=20 | 4 | −0.0012 | 0.2245 | **0.5820** | Tiny-model regime per playbook rule 12. 4 trees vs manual winner's 6; gap collapsed; R-p@1 climbed. |
| 2 | max_depth=2, eta=0.05, colsample_bytree=0.4, ES=20 | 18 | +0.0026 | 0.2241 | 0.5287 | Slower eta + cs=0.4 randomness — 18 trees but R-p@1 regressed. val_brier moved by 0.0004 < tie_band 0.005 → enters tie set with iter_1. |

`val_briers = [0.2590, 0.2245, 0.2241]`. After iter_0 drops out (gap 0.0345 > tie_band), tie set = `{iter_1, iter_2}` (val_brier delta 0.0004 < 0.005). V1.3 Option A path picked iter_1 (eval R-p@1-best in tie set: 0.582 > 0.529). Agent emitted `should_stop=true` at iter_2 to ship clean.

`loop.best_iteration = 1`. `inner_stop_signal = agent_should_stop`. `tiebreak_path = anti_auc_eval_rp1` ← **V1.3 Option A path fired (the regression test point)**.

## Shipped metrics

**Test segment** (base_rate = 0.267, n_rows = 4600, Q_days = 71):

| metric | sweep ([_223](_223_cell5_loop_v1.3_revalidation.md) row) | v14p4 | base_rate (test) |
|---|---:|---:|---:|
| AUC | 0.475 | 0.463 | — |
| R-p@1 | 0.671 | 0.408 | 0.267 |
| R-p@3 | 0.557 | 0.484 | 0.267 |
| R-p@5 | 0.569 | 0.465 | 0.267 |
| R-p@10 | 0.515 | 0.513 | 0.267 |
| R-p@20 | 0.518 | 0.520 | 0.267 |
| Brier | — | 0.2065 | 0.1957 (base-rate Brier) |

**Eval segment** (base_rate = 0.354, n_rows = 18400, Q_days = 244):

| metric | v14p4 |
|---|---:|
| AUC | 0.554 |
| R-p@1 | 0.582 |
| R-p@3 | 0.497 |
| R-p@5 | 0.491 |
| R-p@10 | 0.542 |
| R-p@20 | 0.494 |

For methodology cross-walk: the original A4 re-validation ([_223](_223_cell5_loop_v1.3_revalidation.md), 12-iter loop with `min_child_weight=5`) shipped test R-p@1 = 0.7143 (passed the 0.6714 floor). This v14p4 replay's 3-iter loop ships test R-p@1 = 0.408 (fails the floor by 0.263). The eval→test drift here is also large: eval R-p@1 = 0.582 → test R-p@1 = 0.408 (−0.174 absolute, −30% relative). Both gaps point at iteration count + knob coverage, not at any V1.4 P1 mechanism issue.

## Mechanistic read

**Path-label invariant.** `tiebreak_path = anti_auc_eval_rp1` (not `v14_val_flat_eval_rp1`) because `anti_auc_flag=true` at iter_0. The V1.3 Option A branch claims finalize routing first; the V1.4 P1 branch is gated on the opposite condition (non-anti-AUC + `val_brier_range < tie_band`). The two are mutually exclusive on this cell — exactly the regression-test invariant P4 is designed to confirm. V1.4 P1 is scoped correctly and does not bleed into the anti-AUC code path.

**Substantive shortfall (0.408 vs [_223](_223_cell5_loop_v1.3_revalidation.md)'s 0.714) has two independent drivers, neither a V1.4 P1 patch concern:**
- **Iteration budget.** [_223](_223_cell5_loop_v1.3_revalidation.md) ran 12 iters across 6 knob families (depth, colsample_bytree, eta, **min_child_weight**, subsample, lambda) and the `min_child_weight=5` knob was decisive — it kept val Spiegelhalter |z| < 2 so calibration stayed native. This v14p4 run (3 iters, 3 knob families: depth, eta, cs) reproduces the calibration-collapse regime [_223](_223_cell5_loop_v1.3_revalidation.md) navigated AWAY from (Spiegelhalter z=-3.34 → isotonic fired → tiny-model predictions collapse → R-p@1 tanks).
- **Eval→test drift.** On the shipped iter_1 winner, eval R-p@1 = 0.582 → test R-p@1 = 0.408 (−0.174 absolute, −30% relative). Test panel is small (Q_days=71) and covers a different calendar regime than eval. Same drift-magnitude family as [_249](_249_v14_p3_replay.md) cell 2.

## Verdict

- **Mechanical (P4 regression-test invariant): PASS.** `tiebreak_path = anti_auc_eval_rp1` confirms the V1.3 Option A finalize path fired unchanged. `anti_auc_flag = true` + both `auto_disabled` keys set as expected. V1.4 P1 patch does NOT touch the anti-AUC branch routing.
- **Substantive R-p@1 (test ≥ sweep floor 0.6714): FAIL.** Test R-p@1 = 0.408 is below the floor by 0.263. But this is a 3-iter run; the original [_223](_223_cell5_loop_v1.3_revalidation.md) needed 12 iters + the `min_child_weight` knob to clear it. The substantive shortfall is the calibration-collapse failure mode [_223](_223_cell5_loop_v1.3_revalidation.md) already characterized, not new evidence about V1.4 P1.
- **P4 acceptance:** the mechanism passing is the load-bearing finding for V1.4 P4. The substantive R-p@1 below floor is documented and non-blocking for the V1.4 plan's P1 ship decision.

## Implication for the V1.4 plan

P4 confirms the V1.4 P1 patch is **scoped correctly** — the anti-AUC branch and the V1.4 P1 non-anti-AUC fallback do not overlap. With P3 (cells [_239](_239_r1k_up40_100d_agent_mode.md) + [_241](_241_r1k_up50_200d_agent_mode.md), shipped in [_249](_249_v14_p3_replay.md)) demonstrating the V1.4 P1 patch fires on the non-anti-AUC branch when designed, and P4 (this memo) demonstrating it does NOT interfere with the anti-AUC branch, the V1.4 P1 patch is mechanically validated end-to-end. P5 (rare-event + L1-original regression replay) and P7 (gated on #243) remain as further validation slices; this P4 result does not affect their plan-conditional triggers.

## Open follow-ups (parking lot)

- Iter-budget-vs-eval-Q_days policy: 3 iters insufficient for anti-AUC + small eval (Q_days < 100). V1.4+ TBD candidate.
- Eval→test drift signature shared with [_249](_249_v14_p3_replay.md) cell 2; small test panel + iter_0/iter_1-ship discipline class.

## Methodology

- **Backend**: XGBoost, library defaults except `tree_method=hist`, `device=cpu`, `n_jobs=8`.
- **Calibration**: `conditional_isotonic` (Spiegelhalter |z|=3.34 > 2.0 → isotonic fired; same calibration-collapse regime [_223](_223_cell5_loop_v1.3_revalidation.md) navigated AWAY from via `min_child_weight=5`).
- **Split**: `trailing` (spec did not opt into V1.4 `date_aligned` because the baseline comparison is [_223](_223_cell5_loop_v1.3_revalidation.md) which used trailing splits; consistency over canonicality for this regression test).
- **R-Precision@K**: computed via `scripts/gbdt/regenerate_r_precision_at_k_csv.py` on `predictions/test.csv` — `min(K, R_q)` denominator, `(p_calibrated desc, ticker asc)` mergesort tie-break, macro over days with R_q > 0.
- **V1.3 Option A path**: anti_auc_flag detection in `fs_hp_loop.py` triggers `auto_disabled` keys + finalize-time `tiebreak_path = anti_auc_eval_rp1` branch (eval R-p@1-best within the val_brier tie set).

## Cross-references

- V1.4 plan: `docs/gbdt/V1.4_l1_tiebreak_fix_plan.md` § P4; PRs #132 (plan), #134 (fixes), #135 (P1 patch under test), #136 (P2 tiebreak_path surface), #138 (P3 replay)
- Companion P3 (non-anti-AUC side): [_249](_249_v14_p3_replay.md)
- V1.3 Option A: `docs/gbdt/V1.3_option_a_loop_anti_auc_integration_plan.md` § 2.2; original A4 re-validation [_223](_223_cell5_loop_v1.3_revalidation.md) (12-iter passing run); manual-track ceiling [_211](_211_cell5_manual_tuning_xgb.md) (R-p@1=0.829)
- Playbook: `.claude/memories/project-gbdt-tuning-playbook.md` rules 10-12; R-Precision: `.claude/memories/project-r-precision-methodology.md`
- Spec: `configs/gbdt/experiments/nasdaq100_up_10pct_50d_dd5pct_agentloop_v14p4.yaml`; artifact dir + canonical CSV row + JSON sidecar at `results/gbdt/data/_250_data.json`
