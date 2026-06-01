# _174 — Investigation A: agent-driven FS+HP loop (CatBoost) vs the `_147` answer key

> **Methodology note (2026-06-01)**: Numbers in this memo's body use the legacy "weighted R-precision" metric (per-day variable K = R(d), micro-aggregated). The project headline metric was renamed 2026-06-01 to **R-Precision@K** (per-day fixed K, macro-aggregated via `(1/Q)·Σ r_q/min(K,R_q)`). See the "R-Precision@K (current methodology)" section at the bottom of this memo for the cells in this memo recomputed under the new metric, plus `.claude/memories/project-r-precision-methodology.md` for the full definition + relationship.

**Task**: V1.1 Phase-6 acceptance — drive the **automated** `agent_file_protocol`
FS+HP loop (the agent acting as the data scientist each iteration via the real
exit-and-resume CLI) on the nifty50 UP +10% / 25d / dd5% cell, and check whether
it reaches the same end-state conclusions the **hand-driven** loop documented in
`docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md`.

**Run**: `results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_catboost_loop/`
(CatBoost, `agent_file_protocol`, nifty50 H=25, dd5%, seed 42, 46 tickers kept).
Driven over the **loosened-gate** spec variant (`plateau_threshold: 0.0001`,
`degradation_gate: 0.05`) — see § "Why the loosened-gate spec".

**Headline verdict**: **PASS (9/0)** after an answer-key correction. The run first
scored **FAIL (4 pass, 5 fail, 0 skip)** against the *original* `acceptance_check_147.py`.
All 5 fails traced to a single root cause — **not** a data-snapshot drift (depth-4
and depth-6 reproduce `_147` to 4 decimals → identical data) but an **l2 confound
in `_147`'s own depth-8 iteration**: `_147` changed depth 6→8 AND l2 3.0→1.5 in the
same step, so its depth-8 (0.1652, overfit @ tree 48) was measured at a *different*
regularization than depth-4/6. The automated loop, enforcing one-attributable-change
-per-iteration, re-ran depth-8 with l2 **held** at 3.0 and got **0.1633** — marginally
*below* depth-6 (a within-noise plateau, not a 6-peaked inverted-U). The answer key
encoded the confounded `_147` number; it was corrected (deconfounded depth curve +
plateau check + best-unconstrained monotone reference), `_147` annotated with the
confound correction, and invA then **PASSES 9/0**. See § Resolution. This is a
*stronger* result than a bare reproduction: the agent loop's experimental hygiene
corrected a confound in the hand-driven study. (Original FAIL detail retained below
for the audit trail.)

---

## Per-iteration decision narrative (6 configs explored)

| iter | config | nfeat | val_brier | train/val gap | decision rationale |
|---|---|---|---|---|---|
| 0 | depth 6 / lr 0.05 (baseline) | 279 | 0.16417 | −0.00477 | Baseline. gap negative ⇒ **no overfit** (rule 1) ⇒ do NOT prune; top features are long-window vol (vol_of_vol_200, garman_klass_200, parkinson_200) + index regime (index_vol/return/drawdown) ⇒ interaction-driven cell. Start the depth scan: deepen 6→8 (upper arm). |
| 1 | depth 8 / lr 0.05 | 279 | **0.16326** | +0.00436 | Marginally best. gap still ≤0.02 (no overfit). Probe the shallow arm next: depth 4. |
| 2 | depth 4 / lr 0.05 | 279 | 0.16608 | −0.00992 | Worst of {4,6,8} (shallow underfit). Depth curve mapped, full spread 0.00282 ⇒ HP-ceiling territory. Scan lr at the best depth next. |
| 3 | depth 8 / lr 0.02 | 279 | 0.16447 | −0.00215 | Lower lr gave no gain (early-stop moved 69→176 trees, as expected). No lr headroom. Test the FS-neutral hypothesis next. |
| 4 | depth 8 / lr 0.05 / **FS prune** | 141 | 0.16497 | −0.00628 | Pruned 138 below-0.01-importance features. Raised val_brier +0.00171 vs the same unpruned config ⇒ **FS neutral-to-harmful** on this non-overfit cell, exactly the rule-1 prediction. Agent declared `should_stop` (HP+FS ceiling mapped). |
| 5 | depth 8 / lr 0.05 / **monotone +1 (30 vol estimators)** | 279 | 0.16500 | −0.00375 | Monotone ablation. +0.00083 vs baseline / **+0.00174 vs the same unconstrained config (iter 1, 0.16326)** ⇒ **monotone CONTRAINDICATED** (rule 4): the constraint degrades vol×regime conditional interactions even though each constrained feature is marginally monotone. |

Best checkpoint shipped = **iter 1** (depth 8 / lr 0.05 / full 279-feature pool).
Loop termination: `agent_should_stop` after iter 4 (HP+FS ceiling mapped). The
monotone ablation (iter 5) was run separately — see below.

### Iterations 0–4 were driven through the live `agent_file_protocol` loop
Each iteration: the runner trained one config, wrote `loop/iter_<N>_request.json`
(the full diagnose bundle) + `loop/checkpoint.json`, and exited; the agent read
the bundle, wrote `loop/iter_<N>_decision.json` (one attributable change per
iteration, rule 6), and relaunched `--resume`. The decision files + request
bundles are retained in `loop/`.

### Iteration 5 (monotone) was run via a separate probe spec — a protocol limitation
The `agent_file_protocol` **resume-decision path rejects `monotone_constraints`**:
`loop_protocol.validate_decision` only accepts HPs in `model.TUNABLE_HP_RANGES` /
`ENUM_HP_VALUES`, and `monotone_constraints` (a named-dict, not a scalar tunable)
is in neither — so a `hp_changes: {monotone_constraints: …}` decision raises
`DecisionError: references unknown HP`. The documented escape hatch is spec
`backend.hp_starting` (unknown keys flow through `_validate_hp` →
`CatBoostClassifier` untouched). So the monotone config was fit via
`configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_catboost_monotone_probe.yaml`
(best config + `+1` on the 30 clean vol estimators) and its val_brier appended as
iter 5 to the main run's `iterations.jsonl`. This makes the
`monotone_contraindicated` check *evaluable* (otherwise it SKIPs). **The
contraindication reproduced**: every monotone config > baseline.

---

## Final config + metrics (best checkpoint = iter 1)

- **HP**: depth 8, lr 0.05, l2_leaf_reg 3.0, Ordered boosting, 1000 trees /
  early-stop 75, full 279-feature pool (FS reverted — pruning was harmful).
- **Calibration**: conditional isotonic → **isotonic shipped** (best-checkpoint
  val preds Spiegelhalter |Z| = 5.92, p ≈ 3e-9 ⇒ native rejected). (Note: the
  iter-0 depth-6 native preds were well-calibrated at |Z| = 0.85; the deeper
  depth-8 best checkpoint needed isotonic.)
- **Discrimination**: eval ROC-AUC 0.634, test ROC-AUC 0.706.
- **Brier**: eval 0.11568 vs base-rate baseline 0.11494 (improvement −0.0007, i.e.
  ~at base rate — capped by the prevalence drift); test 0.14183.
- **Weighted R-precision** (per-day variable K = R(d),
  `scripts/gbdt/compute_r_precision.py`): **eval 0.2986 vs base 0.1378 → lift
  2.17×**; **test 0.3948 vs base 0.1963 → lift 2.01×**. Matches the `_147` ~2.1×.
- **Prevalence drift**: train 0.280 → eval 0.133 (decline 0.147) — the
  calibration/Brier ceiling, out of the FS/HP loop's scope (rule 5).

---

## acceptance_check_147.py — per-check verdict (ORIGINAL run, pre-correction)

*This table is the run against the **original** answer key, before the l2-confound
correction (see § Resolution for the corrected 9/0 result). Retained for the audit
trail.*

| Check | Verdict | Observed | Reading |
|---|---|---|---|
| `no_overfit_baseline` | **PASS** | iter-0 gap −0.0048 | Reproduced (rule 1). |
| `prevalence_drift_ceiling` | **PASS** | train 0.280 → eval 0.133 (−0.147) | Reproduced (rule 5). |
| `ranking_robust` | **PASS** | weighted R-precision lift 2.01× (test) | Reproduced (~2.1×). |
| `final_features_not_collapsed` | **PASS** | best model keeps 279 features | Reproduced (FS neutral, no hard cut). |
| `hp_ceiling_band` | FAIL | range [0.1633, 0.1661] | Low end dips below the `_147` band [0.1641, 0.1664] — because depth-8 went to 0.16326. |
| `hp_ceiling_spread` | FAIL | 0.0028 > 0.0020 | Spread slightly wider — depth-8 min pulled it down. |
| `depth_optimal` | FAIL | best depth = **8**, curve d4/d6/d8 = 0.1661/0.1642/**0.1633** | **The root divergence**: monotone-decreasing, not `_147`'s inverted-U (d6 < d8). |
| `no_meaningful_improvement` | FAIL | improvement +0.0009 > 0.0005 | depth-8 beat the depth-6 baseline by 0.0009 (vs `_147`'s +0.0001 noise). |
| `monotone_contraindicated` | FAIL | all monotone > baseline ✓, but best-monotone harm +0.0008 < 0.0010 | The harm vs the iter-0 *baseline* (depth 6) is smaller than `_147`'s +0.0012; vs the SAME config (depth 8) it is +0.00174 (would pass). The check compares to the iter-0 baseline, which is higher here. |

**All 5 FAILs trace to one root cause — and it is NOT a data-snapshot drift.**
A follow-up investigation (2026-05-29) discriminated the hypotheses cleanly:

- **depth-4 and depth-6 reproduce `_147` to 4 decimals** (0.16608 ≈ 0.1661;
  0.16417 ≈ 0.1642) → the data + base training are **identical**. Drift ruled out.
  (An earlier draft of this memo guessed "later snapshot / NTPC stale" — wrong.)
- **depth-8 is the only divergent point** (0.16326 vs `_147` 0.1652) because
  **`_147`'s depth-8 iteration confounded two variables**: it changed depth 6→8
  AND l2_leaf_reg 3.0→1.5 in one step (its own header: "a coherent capacity
  cluster (depth+l2)"; iter-2 then "reset l2 to 3.0 to keep the depth comparison
  clean"). So `_147`'s depth-8 was at l2=1.5 — the l2 cut crashed early-stop to 48
  trees (immediate overfit) → 0.1652 — while depth-4/6 used l2=3.0. The automated
  loop, enforcing one-attributable-change-per-iteration, re-ran depth-8 with l2
  **held at 3.0** → **0.16326**, gap +0.0044 (still no overfit).

So depth 6 and 8 are a **within-noise plateau** (Δ ~0.001), not a 6-peaked
inverted-U; depth-4 underfits. invB (XGBoost, `_149`) independently found the same
depth-8 optimum, corroborating. The agent loop's clean experimental design
**corrected a confound in the hand-driven `_147` study** — a stronger outcome than
a bare reproduction.

### Why the original checker FAILed
`acceptance_check_147.py`'s answer key pinned the **confounded** depth-8 number
(0.1652) + a strict depth-6 optimum; reproducing it would have required the loop to
*repeat* the confound. The 5 fails (band/spread/optimum/improvement/monotone-harm)
all cascade from the legitimate, deconfounded depth-8 = 0.1633.

---

## Resolution (2026-05-29) — answer-key correction → PASS 9/0

User-authorized fix (the loop found a real confound; the gate encoded it):
1. **`_147` annotated** with the l2-confound correction (Iteration 1/2 + the
   Unified-conclusion table): deconfounded depth 4/6/8 = 0.1661/0.1642/**0.1633**;
   "depth 6–8 plateau, depth-4 underfits" replaces "depth-6 the sweet spot".
2. **`acceptance_check_147.py` answer key deconfounded**: `depth_curve[8]`
   0.1652→0.1633; `depth_optimal: 6` → a `{6,8}` plateau check + depth-4 as the
   underfit arm; `ceiling_brier_lo` 0.1641→0.1632; `hp_band_width_max`
   0.0020→0.0030; `meaningful_improvement` 0.0005→0.0010 (this cell's run-to-run
   noise is ~0.001); the monotone check now references the **best unconstrained**
   config (not the iter-0 baseline — invA probed monotone at depth-8, so vs-iter-0
   understated the harm). Unit tests updated → 12/12 pass.
3. **Re-run** on this run dir → **OVERALL PASS (9 pass, 0 fail, 0 skip)**.

**Closes the CatBoost half of #161** (invB `_149` closes the XGBoost half).

### What the acceptance demonstrates
The **V1.1 agent-driven loop mechanics work end-to-end** on CatBoost: exit-and-
resume protocol, per-iteration diagnose bundles, decision files, plateau/
degradation/`should_stop` gates, finalization + calibration + artifact emission.
The agent reproduced every qualitative `_147` finding (no overfit, HP ceiling with
a tiny ~0.003 spread, no config meaningfully beats baseline, monotone worse than
the best unconstrained config, prevalence-drift ceiling, FS neutral-to-harmful,
ranking ~2×) AND, by enforcing clean single-variable changes, surfaced + corrected
a confound the hand-driven loop carried. Both backends land on the depth-8 plateau.

---

## Caveats on the artifacts (for reproducibility audit)

- `iterations.jsonl` was **reconstructed** from the per-iteration
  `loop/iter_<N>_request.json` bundles (the exact diagnostics the agent read each
  iteration) + the iter-5 monotone probe. Reason: the `agent_file_protocol`
  finalization writes `iterations.jsonl` only from the *finalizing process's*
  in-memory history; a `should_stop` finalize trains nothing, so it emitted an
  empty file. The full per-config history lives in `loop/checkpoint.json`
  (`val_briers` + `hp_lists`); the reconstruction is faithful to it. (This is a
  known gap between the exit-and-resume design and the checker, which reads
  `iterations.jsonl` for the full history — noted for V1.1 follow-up.)
- `metrics.json` was augmented with an `r_precision` block (weighted, per-day
  variable-K) computed by `scripts/gbdt/compute_r_precision.py` so the checker's
  `ranking_robust` is evaluable (the standard runner metrics.json does not carry
  R-precision).
- `features.yaml` was rewritten from the runner's `{features: [...]}` dict form to
  a **bare list** so the checker's `load_run` (which expects a bare list) can read
  it. Same 279 features either way — a checker/runner format mismatch, not a
  content change. Noted for V1.1 follow-up.
- The monotone-probe run dir
  (`…/nifty50_up_10pct_25d_dd5pct_catboost_monotone_probe/`) holds the iter-5
  config's full artifact.

## R-Precision@K (current methodology — added 2026-06-01)

Per `.claude/memories/project-r-precision-methodology.md`, R-Precision@K is the post-2026-06-01 headline cross-cell metric for gbdt — defined as `R-Precision@K = (1/Q) · Σ_q r_q / min(K, R_q)` over the Q days where R_q > 0 (R_q = positives on day q; r_q = positives caught in top-K picks on day q; macro-averaged, equal weight per day; K fixed). Recomputed from each cell's `predictions/test.csv`:

| cell | rows | base | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|---|---|
| nifty50_up_10pct_25d_dd5pct_catboost_phase8 | 3450 | 17.9% | 0.729 | 0.171 | 0.205 | 0.213 | 0.328 | 0.559 |

The canonical CSV does not carry the `..._catboost_loop` (invA acceptance) artifact directly; the matched-cell + matched-backend CatBoost end-state in the CSV is the `_catboost_phase8` finalize (the same depth-8 / 279-feat config — see `_175`), used here as the closest-available reference row.
