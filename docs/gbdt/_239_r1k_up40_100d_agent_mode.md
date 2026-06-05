# #239 — russell1000 +40%/100d agent-mode run (V1.3 Option B, XGBoost)

**Status:** complete 2026-06-05 21:01 IST; 1 cell, 8 agent iterations + finalize. Canonical CSV row appended.

## Headline

V1.3 Option B agent-mode run on the strong-top-1 cell `russell1000_up_40pct_100d_dd20pct_aligned` (sweep baseline R-p@1 = 0.550, AUC = 0.821) **underperformed the CatBoost sweep**: agent test R-p@1 = **0.485** vs sweep **0.550** (Δ = −0.065, −12%). Mid-loop diagnostics flashed a provisional eval R-p@1 = 0.665 (+21% over sweep) at iter_0, but the finalize-refit on train+val with the loop's val_brier-best iteration (iter_5, `max_depth=2`) produced a different model whose final calibrated eval R-p@1 collapsed to 0.540 and test R-p@1 to 0.485. Cross-cell verdict: on a healthy-AUC, strong-top-1 cell where the sweep already converges, the agent loop's val_brier-driven best-iter selection picks a structurally smaller model that loses tail discrimination — the in-loop provisional eval picks are not the model that gets shipped.

| metric | sweep (CatBoost) | agent (XGBoost) | Δ |
|---|---:|---:|---:|
| AUC (test) | 0.821 | 0.808 | −0.013 |
| R-p@1 (test) | 0.550 | 0.485 | −0.065 |
| R-p@3 (test) | 0.517 | 0.407 | −0.110 |
| R-p@5 (test) | 0.447 | 0.431 | −0.016 |
| R-p@10 (test) | 0.382 | 0.379 | −0.003 |
| R-p@20 (test) | 0.341 | 0.338 | −0.003 |
| base_rate (test) | 0.080 | 0.080 | — |

(All rows from `results/gbdt/data/r_precision_at_k.csv`. Sweep: `_228_h100_rerun.md` row 37; agent: this run.)

## Scout (Phase A — 35 single-knob configs, 25.8 min)

FS-prefit kept 143 / 279 features at `cliff_pct=0.01`. Top single-knob R-p@1 wins (vs default 0.475):

| knob | best value | R-p@1 | val_brier | train_val_gap |
|---|---|---:|---:|---:|
| gamma | 0.5 | 0.595 | 0.0637 | −0.002 |
| eta | 0.01 | 0.540 | 0.0568 | −0.021 |
| eta | 0.05 | 0.530 | 0.0596 | −0.005 |
| alpha | 0.1 | 0.530 | 0.0730 | +0.023 |
| gamma | 1.0 | 0.515 | 0.0624 | −0.006 |
| max_depth | 3 | 0.490 | 0.0663 | +0.002 |

Steepest knobs: gamma (response curve `0.0 → 0.475`, `0.5 → 0.595`, `1.0 → 0.515` — peaked at 0.5), eta (lower = better, peak at 0.01), max_depth (peaked at 3; depths 6, 8 destroyed R-p@1). High `scale_pos_weight` collapsed both val_brier and top-K. The runner's lex auto-compose proposed `{depth=3, eta=0.01, gamma=0.5, alpha=0.1}` — alpha=0.1's inclusion was questionable because alpha was the worst single-knob val_brier (0.073 > defaults 0.071). Pre-iter_0 the agent tested whether to keep alpha.

## Combine (Phase B — 34 agent-curated mix configs, 25 min)

Agent wrote `scout/combine_decision.json` with 34 configs spanning: lex zeroth (verbatim) + single-knob anchors + pairwise/triple mixes of top knobs + 8 tiny-model probes (`max_depth=2 + eta {0.05, 0.1} + early_stopping`) per playbook rule 12. **Lex winner (R-p@1 > 3 > 5 > 10 > 20):** config 18 `lex_alpha0_subsample0.7` = `{max_depth: 3, eta: 0.01, gamma: 0.5, subsample: 0.7}` with eval R-p@1 = **0.665**, R-p@3 = 0.632, R-p@5 = 0.561, R-p@10 = 0.513, R-p@20 = 0.408. Notable findings:

- **Dropping alpha + adding subsample = 0.7 was synergistic.** Lex-zeroth (config 0, alpha=0.1, subsample=1.0): R-p@1 = 0.535. Same minus alpha plus subsample (config 18): R-p@1 = 0.665. Alpha=0.1's bulk-Brier penalty (worst single-knob val_brier in scout) hurt the prediction tail at the lex composition level.
- **Tiny models held up.** Configs 20-23 (`depth=2 + eta {0.05, 0.1} + early_stopping`) reached R-p@1 = 0.51–0.56 with R-p@3 = 0.61–0.66 — the playbook rule 12 prediction held: depth=2 is competitive on this cell.
- **No degenerate sink.** None of the top R-p@1 configs had `train_val_gap ≈ 0` with `val_brier ≈ weighted_base_rate`. The lex winner had val_brier = 0.0568 vs weighted_base_rate = 0.0614 — beating baseline cleanly.

Agent wrote `scout/iter_0_decision.json` selecting config 18's HP overlay as iter_0 seed (rationale: lex winner, no degenerate-sink signature, alpha=0.1 dropped on the val_brier evidence).

## Agent iteration loop (Phase C — 8 iters, ~30 min)

| iter | knob changed | hp_overlay (delta) | val_brier | train_val_gap | eval R-p@1 | verdict |
|---|---|---|---:|---:|---:|---|
| 0 | (seed) | `depth=3, eta=0.01, gamma=0.5, subsample=0.7` | 0.05678 | −0.026 | 0.665 | seed from combine lex winner |
| 1 | n_estimators 100 → 300 | + `n_est=300` | 0.05657 | −0.018 | **0.445** | val flat; R-p@1 destroyed by extra trees |
| 2 | mcw 1 → 3 (n_est reverted to 100) | + `mcw=3` | 0.05678 | −0.026 | 0.540 | mcw=3 degrades top-1 vs iter_0 default mcw=1 |
| 3 | subsample 0.7 → 0.5 (mcw reverted to 1) | `subsample=0.5` | 0.05680 | −0.026 | 0.540 | subsample lower than 0.7 = flat |
| 4 | gamma 0.5 → 0.3 (subsample reverted) | `gamma=0.3` | 0.05679 | −0.026 | 0.665 | gamma 0.3-0.5 flat-equivalent |
| 5 | max_depth 3 → 2 (gamma reverted) | `depth=2` | 0.05692 | −0.028 | 0.540 | depth=2 underfits tail |
| 6 | csb 1.0 → 0.7 (depth reverted to 3) | `csb=0.7` | 0.05678 | −0.026 | 0.540 | csb 0.7 = same attractor |
| 7 | alpha 0 → 0.05 (csb reverted) | `alpha=0.05` | 0.05675 | −0.026 | 0.525 | alpha pushes pred toward bulk |
| (stop) | should_stop=true | — | — | — | — | 7 knobs probed; iter_0 unchallenged |

7 knobs explored (`n_estimators`, `min_child_weight`, `subsample`, `gamma`, `max_depth`, `colsample_bytree`, `alpha`) vs the 4-knob playbook threshold. **Sticky-attractor pattern:** every single-knob deviation from iter_0's config collapsed eval R-p@1 from 0.665 to the 0.52-0.54 attractor — except iter_4 (gamma=0.3) which held flat at 0.665. val_brier moved within a band of 0.00017 (0.05675–0.05692) across all 8 iters — flat to the eye, with the runner's L1 tie-break picking iter_5 (depth=2, val_brier=0.05692 — actually the WORST val_brier of the 8 iters; this is the lex tie-break interacting with the agent_file_protocol checkpoint) as `best_iteration=5`.

## Why the in-loop oracle disagreed with the shipped model

The in-loop diagnostic eval (R-p@1 = 0.665 at iter_0) used the model fitted on **train only**, evaluated on **eval** segment. The finalized model is re-fit on **train + val** with the val_brier-best iteration's HP, then calibrated, then evaluated on eval + test. Three things changed:

1. **Different HP**: best_iteration = 5 → final HP = `{depth: 2, eta: 0.01, gamma: 0.5, subsample: 0.7, n_est: 100, mcw: 1}`, NOT iter_0's depth=3.
2. **Bigger fit set**: train+val ≈ 1.06M rows vs train-only ≈ 708K rows → tree splits shift.
3. **Calibration**: conditional_isotonic applied; pulled raw scores into the [0.0085, 0.281] band (per `report.md` prediction_range).

Net: agent loop's iter-by-iter R-p@1 trajectory is **not a reliable predictor** of what the finalized model ships. The val_brier-best selection rule + finalize-refit can transmute a strong-top-1 iter_0 (R-p@1=0.665, depth=3) into a different-shape final model (depth=2, eval R-p@1=0.540, test R-p@1=0.485). This is a structural issue with how V1.3 Option B's agent loop interacts with the finalize step on flat-val_brier cells, NOT an agent-decision error.

## Wall-clock

- Phase A (data + features + scout 35 configs): 2068s (34.5 min)
  - Features build: 2043s (cold cache for this cell)
  - Scout: 1551s (35 configs × ~44s mean)
- Phase B (combine 34 configs): 1437s (~24 min)
- Phase C (8 iters + finalize): 1808s (~30 min)
  - 8 agent iterations: 8 × ~4 min = 32 min runner-side; agent decision time ~1-2 min each
  - Finalize: 220s (per `metrics.json::wall_time_total_sec`)
- **Total runner wall-clock: ~5h 11m** (incl. agent decision time on top)
- Cache state: feature cache (6.2 GB parquet) cold-built; subsequent runs on this cell would be ~30-40 min total.

## Recommendations

1. **Negative-result baseline for agent-mode on healthy-AUC strong-top-1 cells.** This cell is the wrong fit for V1.3 Option B: the sweep already plateaus at R-p@1=0.550, and the agent loop's val_brier objective is too flat to discriminate. Future agent-mode work should target the anti-AUC corner (AUC ∈ [0.46, 0.54] + R-p@10 lift > 1.8×) where the loop's auto-disables fire and the agent has structural room to improve.
2. **Open question: val_brier-flat finalize behavior.** The val_brier range across 8 iters here was 0.00017 (smaller than the loop's `tie_band=0.005`). This effectively makes `best_iteration` a coin-flip among the iters within the tie band, and the chosen iter's HP gets shipped. Worth a follow-up to formalize: on cells where `max(val_brier) − min(val_brier) < tie_band`, should the loop prefer the val_brier-best OR fall back to the eval R-p@1-best iteration? Tracked as a V1.4 TBD candidate (not in this PR's scope).
3. **Confirm sweep methodology is the right default for cells with sweep R-p@1 > 0.4.** The empirical evidence here + the existing V1.3 anti-AUC integration suggest a heuristic: **sweep for strong baselines, agent-mode for null/anti-AUC**. Worth a fuller cross-cell test in a follow-up plan.

## Methodology

- **Backend**: XGBoost (per spec), `tree_method=hist`, `n_jobs=8`, `device=cpu` (V1.3 pinned for determinism).
- **Calibration**: `conditional_isotonic` requested; runner shipped `isotonic` (Spiegelhalter |z| = 88.9, p ≈ 0 → calibration mandatory).
- **Split**: V1.4 `date_aligned`, `train_start = 2019-01-01`, same calendar windows as the sweep companion (`_228`):
  - train: 2019-01-02 → 2022-03-04 (708,264 rows)
  - val: 2022-03-07 → 2023-10-06 (355,600 rows)
  - eval: 2023-10-09 → 2024-07-25 (177,800 rows)
  - test: 2024-07-26 → 2025-05-13 (177,800 rows)
- **Snapshot pin**: `--snapshot-end 2026-05-22` on every resume (per V1.3 Option B spec requirement when scout is enabled in agent_file_protocol mode).
- **Loop config**: `plateau_threshold: 0.0001` (effectively disabled per CLAUDE.md V1.3 guidance), `max_iterations: 16`, `degradation_gate: 0.05`, `degenerate_sink_threshold: 1.05`, `tie_band: 0.005`. Agent emitted `should_stop=true` at iter_7 (8 total iterations including iter_0).
- **R-Precision@K** computed via `scripts/gbdt/regenerate_r_precision_at_k_csv.py` on the freshest `predictions/test.csv` per the project methodology (`min(K, R_q)` denominator, `(p_calibrated desc, ticker asc)` mergesort tie-break, macro over days with R_q > 0).

## Cross-references

- **Sweep companion**: `_228_h100_rerun.md` (cell row at line 37) — sweep baseline this run compared against.
- **V1.3 Option B plan**: `docs/gbdt/V1.3_OPTION_B_PLAN.md` (scout + FS-prefit + combine + agent loop).
- **Playbook**: `.claude/memories/project-gbdt-tuning-playbook.md` rules 8, 10-12 (no mcw=10 default, eval R-p@K as holdout oracle on flat-val cells, don't dismiss tiny models on top-1).
- **R-Precision methodology**: `.claude/memories/project-r-precision-methodology.md` (per-day fixed K, `min(K, R_q)` denominator, macro aggregation).
- **Spec**: `configs/gbdt/experiments/russell1000_up_40pct_100d_dd20pct_aligned_agent.yaml`.
- **Artifact**: `results/gbdt/experiments/russell1000_up_40pct_100d_dd20pct_aligned_agent/` (predictions/, scout/, loop/, report.md, metrics.json).
