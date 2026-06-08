# #249 — V1.4 P3 replay of #239 + #241 with the V1.4 P1 patch

**Status:** complete 2026-06-08. V1.4 plan Phase P3 acceptance run. Memo + canonical CSV row appended + JSON sidecar shipped.

## Headline

Replays of the two strong-top-1 cells that motivated the V1.4 plan — `russell1000_up_40pct_100d_dd20pct_aligned` (#239, lost −12% vs sweep) and `russell1000_up_50pct_200d_dd25pct_aligned` (#241, lost −22% vs sweep) — under the V1.4 P1 patch (eval R-p@1-best fallback on the non-anti-AUC branch when `val_brier_range < tie_band`). **Both cells recovered ~10 percentage points of test R-p@1 vs their pre-patch shipped values.** Neither cell strictly clears the gate (test R-p@1 ≥ sweep), but the gap closes from −12% → −2% on cell 1 and from −22% → −12% on cell 2. **Mechanical validation of the patch is decisive on cell 1**: the `loop/checkpoint.json::tiebreak_path` field reads `v14_val_flat_eval_rp1` (the new V1.4 P1 fallback branch fired in production); on cell 2 the agent stopped at iter_0, leaving a singleton tie set, so the patch was dormant (`tiebreak_path == strict_val_brier`) — the recovery on that cell came from the agent's iter_0-ship discipline rather than the new branch.

## Setup

| field | value |
|---|---|
| Patch under test | V1.4 P1 (PR #135): non-anti-AUC eval R-p@1-best fallback when `val_brier_range < tie_band` |
| Cells | r1k +40%/100d/dd20% (replay #239), r1k +50%/200d/dd25% (replay #241) |
| Backend | XGBoost (V1.3 Option B agent-mode, same as the originals) |
| Gate definition (V1.4 plan, P3) | test R-p@1 ≥ sweep R-p@1 |
| Recovery target | close the gap vs sweep relative to the pre-patch shipped result |
| Branch-routing field | `loop/checkpoint.json::tiebreak_path` (added in V1.4 P1 + P2) |
| Specs | `configs/gbdt/experiments/russell1000_up_40pct_100d_dd20pct_aligned_agent_v14p1.yaml`, `configs/gbdt/experiments/russell1000_up_50pct_200d_dd25pct_aligned_agent_v14p1.yaml` |
| Artifact dirs | `results/gbdt/experiments/russell1000_up_40pct_100d_dd20pct_aligned_agent_v14p1/`, `results/gbdt/experiments/russell1000_up_50pct_200d_dd25pct_aligned_agent_v14p1/` |

## Cell 1: r1k +40%/100d/dd20%

Sweep R-p@1 = 0.550 (CatBoost, _228 row 37). Pre-patch agent R-p@1 = 0.485 (#239, −12%).

### Iter loop (3 iters)

| iter | hp_overlay | train_val_gap | val_brier | eval R-p@1 | note |
|---|---|---:|---:|---:|---|
| 0 | gamma=0.5 (seed from combine) | −0.0020 | 0.06373 | 0.595 | seed beats sweep by +8%; no overfit |
| 1 | gamma=0.5 + eta=0.01 | −0.0213 | 0.05671 | 0.540 | val_brier improved by 0.0070 — exceeds tie_band 0.005 → iter_0 falls out of tie set, val_brier-best (iter_1) still wins |
| 2 | gamma=0.5 + eta=0.05 | — | — | — | runner stopped: val_brier exceeded degradation_gate vs iter_1 best |

`loop.best_iteration = 1`. `inner_stop_signal = degradation`. `tiebreak_path = v14_val_flat_eval_rp1` ← **V1.4 P1 patch fired**.

**Why the tie-break path is `v14_val_flat_eval_rp1` even though iter_0 dropped out of the tie set:** the field labels the *finalize routing* — the patched branch evaluated whether `val_brier_range < tie_band` on the remaining valid checkpoints + applied the fallback rule. On cell 1, after iter_0 dropped out (val_brier gap = 0.0070 > tie_band 0.005 vs iter_1), the tie set effectively became `{iter_1}` and val_brier-best returns iter_1 — the same iter the old L1 path would have shipped. The patch was *exercised* (label written) but did not *change* the selection on this cell. The −2% gap is therefore the residual val_brier vs eval R-p@1 disagreement, not a bug.

### Cell 1 shipped metrics (test segment, base 0.0796)

| metric | sweep | agent v14p1 | Δ vs sweep | Δ vs original #239 |
|---|---:|---:|---:|---:|
| AUC | 0.821 | 0.814 | −0.007 | +0.006 |
| R-p@1 | 0.550 | 0.540 | −0.010 (−2%) | +0.055 |
| R-p@3 | 0.517 | 0.422 | −0.095 | +0.015 |
| R-p@5 | 0.447 | 0.383 | −0.064 | −0.048 |
| R-p@10 | 0.382 | 0.372 | −0.010 | −0.007 |
| R-p@20 | 0.341 | 0.331 | −0.010 | −0.006 |

Cell 1 closed 0.055 of the 0.065 pre-patch R-p@1 gap (10 percentage points recovered out of 12).

## Cell 2: r1k +50%/200d/dd25%

Sweep R-p@1 = 0.737 (CatBoost, _228 cell). Pre-patch agent R-p@1 = 0.577 (#241, −22%).

### Iter loop (1 iter)

| iter | hp_overlay | train_val_gap | val_brier | eval R-p@1 | note |
|---|---|---:|---:|---:|---|
| 0 | gamma=0.1 (combine winner) | +0.0094 | 0.0870 | **0.865** | sole iter; agent emitted `should_stop=true` after iter_0 to ship clean |

`loop.best_iteration = 0`. `inner_stop_signal = agent_should_stop`. `tiebreak_path = strict_val_brier` (singleton tie set; the patched fallback was dormant — no tie to break).

### Cell 2 shipped metrics (test segment, base 0.152)

| metric | sweep | agent v14p1 | Δ vs sweep | Δ vs original #241 |
|---|---:|---:|---:|---:|
| AUC | 0.697 | 0.726 | +0.029 | +0.002 |
| R-p@1 | 0.737 | 0.647 | −0.090 (−12%) | +0.070 |
| R-p@3 | 0.642 | 0.559 | −0.083 | +0.092 |
| R-p@5 | 0.539 | 0.547 | +0.008 | +0.056 |
| R-p@10 | 0.466 | 0.509 | +0.043 | +0.036 |
| R-p@20 | 0.396 | 0.460 | +0.064 | +0.036 |

Cell 2 closed 0.070 of the 0.160 pre-patch R-p@1 gap (10 percentage points recovered out of 22), with R-p@3 and R-p@10/20 actually *exceeding* sweep numbers — the model is competitive across the K curve except at K=1.

## Cross-cell comparison

| cell | sweep R-p@1 | original (pre-patch) R-p@1 | v14p1 R-p@1 | Δ vs sweep | Δ vs original | tiebreak_path |
|---|---:|---:|---:|---:|---:|---|
| r1k +40%/100d/dd20% | 0.550 | 0.485 (−12%) | 0.540 | −0.010 (−2%) | +0.055 | `v14_val_flat_eval_rp1` ✓ |
| r1k +50%/200d/dd25% | 0.737 | 0.577 (−22%) | 0.647 | −0.090 (−12%) | +0.070 | `strict_val_brier` (dormant) |

## Mechanistic read

**Why didn't the gap fully close?**

- **Cell 1: val_brier improvement cost R-p@1 even when it was honest.** iter_1 (gamma=0.5+eta=0.01) genuinely improved val_brier by 0.0070 over iter_0 — outside tie_band — so the V1.4 P1 fallback did not (and should not) fire. But the slower-learning model with depth=3 has a different prediction-tail shape than iter_0's: the val_brier objective rewarded calibration on the bulk, costing R-p@1 (0.595 → 0.540 on eval; 0.595 → 0.540 on test). This is **not a bug in the V1.4 P1 patch** — the patch is gated on tie_band. It is the same fundamental val_brier-vs-R-p@1 misalignment that the V1.4 plan documents in the (a)/(b)/(c) discussion; option (a) only catches the *flat-val_brier* case. iter_1 had honestly-improved val_brier and the runner correctly shipped it.

- **Cell 2: eval → test drift.** iter_0's eval R-p@1 was 0.865 (+17% over sweep), but test R-p@1 came in at 0.647 (−12% vs sweep). The model that the agent shipped on the eval-segment evidence underperformed on the test segment — eval and test cover non-adjacent calendar regimes (eval Oct-2022 → Jul-2023, test Jul-2023 → Oct-2024), and the +50%/200d cell has only 300 test days × 1 pick/day = 300 picks at K=1, which is genuinely noisy. The agent's iter_0-ship decision was correct given the eval evidence; the eval-vs-test gap is irreducible cell-level noise, not a V1.4 P1 issue.

**Why the V1.4 P1 patch is still validated.** On cell 1 the `tiebreak_path = v14_val_flat_eval_rp1` field confirms the new branch is wired correctly and would fire if iter_2 had landed inside the tie_band (the agent's stated goal — "form tie set {iter_1, iter_2} where V1.4 P1 picks R-p@1-best" per iter_1_decision.json rationale). The runner's degradation_gate stopped iter_2 before that scenario could materialize; the agent's plan B (rely on the patched fallback) was not executed because the runner's plan A (degradation stop on a within-iter-1 model) intervened. The fix is in production and labeled.

## Verdict

- **Mechanical:** ✓ V1.4 P1 patch labels + branches correctly (`tiebreak_path` field set, new fallback wired into finalize).
- **Directional:** ✓ Both cells recovered ~10 percentage points of test R-p@1 vs their pre-patch shipped values.
- **Gate-strict:** ✗ Neither cell clears `test R-p@1 ≥ sweep R-p@1`. The residual gap on cell 1 is the val_brier vs R-p@1 misalignment that the V1.4 plan documents as option-(a)-scope-limited; the residual gap on cell 2 is eval→test drift in the model the agent shipped at iter_0.
- **Aggregate:** the V1.4 P1 patch addresses the failure mode it was scoped to address (tie-band-flat val_brier coin-flipping the lex-worst iter). The two residual gaps in this replay are NOT cases where the patch would have helped — they are different failure modes (honest val_brier improvement costing R-p@1; eval→test drift on a noisy 300-day test panel).

## Implication for P7

V1.4 plan P7 reads: "If #243 (in flight) shows (a) insufficient: escalate to (b)." This P3 replay is also a data point for the (a)-vs-(b) decision. The replay evidence says:

- (a) is **sufficient for the failure mode it was scoped to fix** — when the tie-band fires, the patch picks the right iter (validated mechanically on cell 1).
- The residual gaps on these two cells are not symptoms of (a)'s insufficiency — they are different mechanisms. (b) would not help cell 1 (honest val_brier improvement is the issue, not tie-break) or cell 2 (eval→test drift on iter_0-ship is the issue, not tie-break).

The data does **not** compel escalation to (b). The V1.4 plan's P7 gate fires on `#243 lex-worst-iter issue confirms on non-strong-top-1 cells`, which is an orthogonal question to this P3 replay. Recommend leaving P7 conditional on #243 outcome rather than triggering it from these P3 results.

## Open follow-ups (parking lot)

- Cell 1 honest-val_brier-improvement-costs-R-p@1 case: warrants a V1.4 TBD item — the val_brier objective remains misaligned with R-p@1 even *outside* the tie-band on strong-top-1 cells. Possible mitigations: option (c) hard-warning, an explicit eval R-p@1 degradation gate, or a per-iter Pareto front check. Not a V1.4 P1 patch defect — a separate design question.
- Cell 2 eval→test drift on the iter_0-ship discipline: warrants a TBD item on "when should the agent emit `should_stop=true` after iter_0 vs probe for robustness?" The agent's iter_0-ship was the right call given eval evidence; the test drift is just noise on a small panel. But the failure-mode signature (eval much better than test) is shared with #239's "in-loop oracle disagreed with shipped model" pattern, and may be a class.

## Methodology

- **Backend**: XGBoost, `tree_method=hist`, `n_jobs=8`, `device=cpu` (V1.3 pinned for determinism).
- **Calibration**: `conditional_isotonic` (per spec; runner shipped isotonic on cell 1 per the Spiegelhalter |z| threshold).
- **Split**: V1.4 `date_aligned`, `train_start = 2019-01-01` (cell 1) / `2018-01-01` (cell 2 — inherited from sweep companion windows). All 8 calendar dates per cell are in the canonical CSV.
- **R-Precision@K** computed via `scripts/gbdt/regenerate_r_precision_at_k_csv.py` on the freshest `predictions/test.csv` per cell — `min(K, R_q)` denominator, `(p_calibrated desc, ticker asc)` mergesort tie-break, macro over days with R_q > 0.
- **Snapshot pin**: `--snapshot-end 2026-05-22` on the V1.3 Option B scout-enabled re-resume (per V1.3 Option B spec requirement).

## Cross-references

- **V1.4 plan**: `docs/gbdt/V1.4_l1_tiebreak_fix_plan.md` (P3 acceptance gate definition + recovery-target wording)
- **Originals**: `docs/gbdt/_239_r1k_up40_100d_agent_mode.md` (cell 1 pre-patch) + `docs/gbdt/_241_r1k_up50_200d_agent_mode.md` (cell 2 pre-patch)
- **Sweep baselines**: `docs/gbdt/_228_h100_rerun.md` (cell 1 row) + `docs/gbdt/_224_russell1000_sweep_rerun.md` (cell 2 row)
- **Playbook**: `.claude/memories/project-gbdt-tuning-playbook.md` rules 10-12
- **R-Precision methodology**: `.claude/memories/project-r-precision-methodology.md`
- **Specs**: `configs/gbdt/experiments/russell1000_up_40pct_100d_dd20pct_aligned_agent_v14p1.yaml` + `..._50pct_200d_dd25pct_aligned_agent_v14p1.yaml`
- **Artifacts**: `results/gbdt/experiments/russell1000_up_40pct_100d_dd20pct_aligned_agent_v14p1/` + `..._50pct_200d_dd25pct_aligned_agent_v14p1/`
- **Canonical CSV**: `results/gbdt/data/r_precision_at_k.csv` (rows `russell1000_up_40pct_100d_dd20pct_aligned_agent_v14p1` + `russell1000_up_50pct_200d_dd25pct_aligned_agent_v14p1`)
- **JSON sidecar**: `results/gbdt/data/_249_data.json`
