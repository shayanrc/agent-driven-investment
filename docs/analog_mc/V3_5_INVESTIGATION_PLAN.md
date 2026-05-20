# analog_mc v3.5 — pre-V4 diagnostic investigation plan

Bridge between v3 (closed) and v4 (planned in [`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md)). Runs four cheap diagnostic investigations to discriminate **why** the v2.4 model fails on the 5 fat-tail failure anchors, so v4 sequencing is grounded in evidence rather than literature priors.

## Purpose

[`FAT_TAIL_EVAL.md`](FAT_TAIL_EVAL.md) identified 5 failure anchors (2010-04-23 flash crash, 2001-10-02 post-9/11 rally, 2018-10-08 Q4 selloff, 2020-03-16 COVID rally, 2026-02-19 recent rally) where v2.4 misses by a wide margin. All 5 share a "regime-transition into a sharp move" shape.

The fat-tail panel tells us *that* the model misses these. It does **not** tell us *why*. The "why" determines which v4 experiment to prioritize:

| If the cause is… | Then the v4 lever is… |
|---|---|
| Matcher picks wrong analogs (right state → wrong era) | **A2** (max-corr distance) — change *what* gets matched |
| Matcher picks right analogs but the conditional-mean is dragged down by majority | **B1** (Platzer local-linear) — fix the path *conditional* on matches |
| Search converges to myopic short-horizon weights at transitions | **B3** (Dirichlet weight posterior) — inflate dispersion under weight uncertainty |
| Analog primitive is *actively suppressing* dispersion below what σ-based methods give | **A1** (FHS) ahead of B1; A1 may ship as partial fix |
| Candidate pool lacks +30%+ rally instances at all | None — no analog-primitive fix; needs tail inflation (v5+ scope) |

These hypotheses are mutually exclusive in places and overlapping in others. Cheap diagnostics can rule several in or out before committing to v4 implementation work.

## Starting state (entering v3.5)

| Item | State |
|---|---|
| Production default | v2.4 — Cell-D-s30 (`configs/analog_mc/default.yaml`: drift + conditional, `momentum_shrinkage: 0.30`, `block_length: 10`) |
| Canonical run | `runs/analog_mc/20260520T045525Z` — 76 folds, 66×5 weight grid, 1000 paths |
| Fat-tail eval set | 15 anchors in [`results/analog_mc/data/fat_tail_eval_anchors.json`](../../results/analog_mc/data/fat_tail_eval_anchors.json); 5 failures listed in [`FAT_TAIL_EVAL.md`](FAT_TAIL_EVAL.md) §"Pattern 4" |
| v4 plan | [`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md) — currently sequenced A1 → B1 → C1 → A2 → B2/B3 |
| Open question | Does the v4 sequencing survive contact with the diagnostic evidence? v3.5 answers this. |

## Scope and non-scope

**In scope.** Four read-only or short-compute diagnostics that operate on existing canonical-run artifacts (`runs/analog_mc/20260520T045525Z/`) plus ad-hoc per-anchor inference. No new walk-forward runs.

**Out of scope.**
- Implementing any v4 fix (those go in v4 experiments).
- Re-running canonical at different settings.
- Multi-asset analysis (V2 carryover, separate scope).
- Anything that requires writing new sampling primitives — that's B1/B2/B3 territory.

## Failure anchors

The 5 anchors v3.5 will dissect. Origin index is in the canonical run's returns space (i.e., last observed log-return). Fold index is the test fold in `runs/analog_mc/20260520T045525Z/folds/`.

| Anchor | Origin idx | Fold | 60d realized | 50%-band | 90%-band | Notes |
|---|---:|---:|---:|---:|---:|---|
| 2010-04-23 | 6097 | 40 (verify) | −10.4% | 3/60 | 27/60 | Pre-flash-crash; only bull-momentum failure |
| 2001-10-02 | 3974 | 26 (verify) | +38.6% | 6/60 | 44/60 | Post-9/11 V-recovery |
| 2018-10-08 | 8260 | 54 (verify) | −12.7% | 4/60 | 31/60 | Q4-2018 selloff |
| 2020-03-16 | 8612 | 56 (verify) | +43.8% | 6/60 | 38/60 | COVID rally |
| 2026-02-19 | 10111 | 75 | +17.5% | 32/60 | 41/60 | Recent rally; sole anchor with weights known [0.977, 0.0, 0.023] |

(Fold indices marked "verify" — confirm by searching `summary.json` files in `runs/analog_mc/20260520T045525Z/folds/*/`. The 2010-04-23 anchor was hand-snapped; its actual fold needs lookup.)

For comparison, the 5 anchors that **passed** with strong coverage form the control group: 1991-03-26, 2010-11-10, 2012-03-14, 2025-07-02, 2017-06-01.

---

## Investigations

Listed in priority order — lowest cost first, since early diagnostic verdicts may obviate later steps.

### V3.5.1 — Per-failure weight inspection (~15 min)

**Question.** Do the 5 failure folds all converge on similar `(weights, n_eff)` tuning, or is the tuning heterogeneous? If they all favor short-horizon weighting (z₂₀ dominant), search is myopic at regime transitions.

**Method.**
1. For each failure anchor, find its fold and read `runs/analog_mc/20260520T045525Z/folds/<fold>/summary.json`.
2. Record `weights`, `n_eff`, `val_crps`, `test_crps`.
3. Compute the cross-fold median weights across all 76 folds for context.
4. Repeat for the 5 control anchors.
5. Compare distributions.

**Decision rule.**
- **All 5 failures favor short-horizon (w₀ > 0.5 with w₂ < 0.2)** → search is myopic at transitions; **promote B3 (Dirichlet weight posterior) to P0**. Also worth exploring whether the val_set itself is too short to reliably identify the true weight prior.
- **Weights heterogeneous across failures** → not a search problem in isolation; doesn't change v4 priorities. Note as a "non-finding."
- **Some folds at extreme corners (1.0, 0, 0) and others at uniform** → val set unreliability suspected; could motivate a v4 experiment on regularizing the search prior.

**Deliverable.** `docs/analog_mc/v3.5/_v3_5_1_weights.md` with the table + verdict + recommendation.

**Implementation hint.** ~30 lines of Python. Reads JSONs, prints table, writes markdown. No new dependencies.

---

### V3.5.2 — Candidate pool tail-mass audit (~30 min)

**Question.** Does the historical candidate pool actually **contain** +30%+ 60-day rallies, or are they too rare for any sampling-based method to recover?

**Method.**
1. For each failure anchor's fold, identify the eligible candidate pool (use `eligible_candidates()` from `src/analog_mc/simulate.py` with the same `candidate_idx` from `walk_forward.py`, or just take all valid origins < this fold's `train_end`).
2. For each candidate, compute the 60-day-forward log-return sum: `sum(returns[i+1 : i+61])` → percent return = `exp(sum)-1`.
3. Bucket by magnitude: `[-∞, -30%), [-30%, -20%), ..., [+20%, +30%), [+30%, +50%), [+50%, +∞)`.
4. Tabulate count + share per bucket per failure anchor's fold.
5. Specifically: **how many ≥+30% 60-day rallies exist** in the pool for each fold?

**Decision rule.**
- **+30%+ rallies present with ≥20 instances per fold** → matcher *could* sample them but doesn't → **matching problem**; A2 candidate strengthens.
- **+30%+ rallies present with <5 instances** → pool too sparse; analog primitive structurally cannot solve. **Document tail inflation as v5 scope**.
- **+30%+ rallies absent** (only +10-20% moves exist) → analog primitive *cannot* produce a +40% path regardless of weighting. Confirms structural ceiling.

**Deliverable.** `docs/analog_mc/v3.5/_v3_5_2_tail_mass.md` with histogram per failure fold + verdict.

**Implementation hint.** ~50 lines of Python. Uses `load_returns()` from `analog_mc.data`. No matcher inference — just historical return arithmetic.

---

### V3.5.3 — FHS spot-check at the 5 failures (~1 h)

**Question.** Does textbook GARCH-FHS produce **wider** 90% bands than v2.4 at these anchors? If yes, the analog primitive is *actively suppressing* dispersion below what σ-based methods give.

**Method.** For each failure anchor:
1. Fit GARCH(1,1) on causal returns up to that origin. Reuse `src/analog_mc/vol.py::fit_garch()` (already exists from v3 E9 work).
2. Simulate 1000 σ-paths over 60 days. Reuse `simulate_garch_sigma_paths()`.
3. Sample 1000 residual sequences i.i.d. from the standardized historical residual pool (returns / EWMA-σ standardized; pool = all returns up to origin).
4. Construct return paths: `r[t] = σ_path[t] * residual[t]`.
5. Integrate to prices, compute 50% / 90% bands.
6. Tabulate coverage: how many of 60 realized days are inside each band?

**Decision rule.**
- **FHS catches ≥3 of 5 rallies (90%-band coverage ≥45/60)** → analog primitive is suppressing tail dispersion; **promote A1 ahead of B1**. A1 could ship as partial solution alongside the matcher (e.g., as a hybrid: matcher direction + FHS dispersion).
- **FHS also misses ≥3 of 5** → fundamental tail-events-without-precedent issue; B1 unlikely to fix it either. **Strengthens case for tail inflation as v5 scope.**
- **Mixed (~2 of 5 caught)** → FHS partial; document the per-anchor split. A1 may still be worth running formally but isn't a clear winner.

**Deliverable.** `docs/analog_mc/v3.5/_v3_5_3_fhs_spotcheck.md` with per-anchor coverage table comparing v2.4 vs FHS.

**Implementation hint.** ~100 lines of Python. Major reusable components exist in `src/analog_mc/vol.py`. Be careful with the standardized-residual pool: use `(returns[t] - mean(returns[:t+1])) / σ_t` where σ_t is the EWMA-vol estimate **at time t** (causal). Sampling is from this 1-D pool at each step, independently.

---

### V3.5.4 — Analog autopsy (~2-3 h)

**Question.** For each failure, **what** analog blocks did the matcher actually pick, and what was their 60-day-forward return distribution?

**Method.** For each failure anchor's fold:
1. Read `weights` and `n_eff` from the fold's `summary.json`.
2. Recompute the matcher probability distribution over the eligible pool: `distances_to_probs(composite_distance(z_target, z_candidates, weights), target_n_eff)`. Use existing `analog_mc.distances` + `analog_mc.simulate.forecast()` machinery.
3. Identify the top-20 highest-probability analog origins.
4. For each top analog: record the anchor date, the 60-day forward return realized at that historical origin, and a 1-line "regime" description (manually annotated).
5. Compute the expected 60-day-forward return under the matcher's probability distribution and compare to the realized.

**Decision rule.** Three diagnostic patterns to look for:

- **"Wrong-era matches"**: Top analogs are from "wrong state" eras (e.g., 2020-03 picks from 2008 bear continuations instead of 2002 V-recoveries). The matcher is myopic on state.
  → **A2 (max-corr distance) candidate strengthens**. State representation needs reworking.
- **"Right-era matches, no V-recoveries"**: Top analogs are from similar-state eras but historically didn't produce V-recoveries (or the pool just lacks them — overlaps with V3.5.2 verdict).
  → **A2 unlikely to help**. Confirms either evidence problem or conditional-mean problem.
- **"Bimodal analog distribution"**: Top analogs include some V-recoveries but at low probability vs majority "trend continuation" analogs.
  → **B1 (Platzer local-linear) is well-targeted** — its Jacobian correction would re-weight toward the dispersed tail at high-Lyapunov regimes.

**Deliverable.** `docs/analog_mc/v3.5/_v3_5_4_analog_autopsy.md` with per-anchor top-20 analog table + analysis + which V4 candidate the evidence supports.

**Implementation hint.** ~150 lines. Heaviest of the four. Needs to faithfully reproduce the matcher's probability distribution — read `forecast()` in `src/analog_mc/simulate.py` carefully and use its internal pieces. Manual regime annotation may be the slowest part; consider just attaching anchor date + sign of historical 60d-forward return as automated and skipping manual labeling unless patterns are unclear.

---

## Sequencing and decision tree

Execute in order. Each step may reshape v4 priorities — re-evaluate after every result.

```
1. V3.5.1 weights (15 min)
   ├─ If all failures converge to short-horizon:
   │   └─ Promote B3 to P0 in V4_EXPERIMENTS_PLAN.md
   └─ Proceed to step 2 regardless.

2. V3.5.2 tail mass (30 min)
   ├─ If +30% rallies absent or <5 per fold:
   │   ├─ Document tail inflation as v5 scope
   │   ├─ Demote B1 (won't fix structural ceiling alone)
   │   └─ STOP — no point running V3.5.3/4 if conclusion is already "pool is empty"
   └─ Proceed to step 3.

3. V3.5.3 FHS spot-check (1 h)
   ├─ If FHS catches ≥3 of 5:
   │   ├─ Promote A1 ahead of B1 (A1 ships as partial fix)
   │   └─ Note: this is a major v4 reshape; expected to be informative
   └─ Proceed to step 4 regardless.

4. V3.5.4 analog autopsy (2-3 h)
   ├─ "Wrong-era matches": promote A2 to P0
   ├─ "Bimodal distribution": confirms B1 is well-targeted
   └─ "Right-era, no V-recoveries": confirms V3.5.2's conclusion

Synthesis: write V3_5_RESULTS.md with v4 reshape recommendations.
```

**Stop conditions.** If V3.5.2 confirms the candidate pool lacks +30% moves entirely, V3.5.3 and V3.5.4 add only confirmatory value. Skip them and write the synthesis.

## Final synthesis: V3_5_RESULTS.md

After all (or stopped) investigations, produce `docs/analog_mc/V3_5_RESULTS.md` containing:

1. Per-investigation summary (1 paragraph each linking to detailed report)
2. **Recommended v4 reshape** — explicit edits to `V4_EXPERIMENTS_PLAN.md`'s Sequencing section. If no reshape needed, document that.
3. **New experiments surfaced (if any)** — e.g., "v3.5 found X; propose new v4 experiment B4: tail inflation" with cost/decision-rule sketch.
4. **Out-of-V4 carry-overs** — anything that's a v5+ structural concern.

If v3.5 produces no clear reshape, document why and proceed with V4 as currently planned. The investigation has value either way.

## Deliverables manifest

```
docs/analog_mc/V3_5_INVESTIGATION_PLAN.md       # this doc
docs/analog_mc/V3_5_RESULTS.md                   # final synthesis
docs/analog_mc/v3.5/_v3_5_1_weights.md           # weight inspection
docs/analog_mc/v3.5/_v3_5_2_tail_mass.md         # pool tail-mass audit
docs/analog_mc/v3.5/_v3_5_3_fhs_spotcheck.md     # FHS spot-check
docs/analog_mc/v3.5/_v3_5_4_analog_autopsy.md    # analog autopsy

scripts/v3_5/inspect_failure_weights.py          # V3.5.1
scripts/v3_5/audit_pool_tail_mass.py             # V3.5.2
scripts/v3_5/fhs_spotcheck.py                    # V3.5.3
scripts/v3_5/analog_autopsy.py                   # V3.5.4

results/analog_mc/data/v3_5_1_weights.json
results/analog_mc/data/v3_5_2_tail_mass.json
results/analog_mc/data/v3_5_3_fhs_spotcheck.json
results/analog_mc/data/v3_5_4_analog_autopsy.json
```

(Use the `docs/analog_mc/experiments/` convention if v3.5 reports turn into permanent companions, but a v3.5 subfolder is fine for now.)

## Read-first checklist for a fresh session

A future Claude session picking this up cold should read, in order:

1. **This file** (`V3_5_INVESTIGATION_PLAN.md`) — purpose, scope, sequencing.
2. **[`FAT_TAIL_EVAL.md`](FAT_TAIL_EVAL.md)** — the 15-anchor eval set, especially §"Pattern 4" for the 5 failures.
3. **[`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md)** — the v4 inventory v3.5 may reshape.
4. **[`V3_EXPERIMENTS_REPORT.md`](V3_EXPERIMENTS_REPORT.md)** — what v3 actually shipped + the architectural finding (acf is structural; v4 needs to abandon block geometry as a lever).
5. **`runs/analog_mc/20260520T045525Z/config.yaml`** — the canonical run's config, especially `weight_grid_resolution`, `n_eff_values`, `vol_clip_*`, `momentum_shrinkage`, `vol_regime_quantiles`.
6. **`src/analog_mc/vol.py`** and **`src/analog_mc/simulate.py::forecast`** — the canonical inference path. V3.5.3 (FHS) and V3.5.4 (analog autopsy) both reuse pieces of this.

The `CLAUDE.md` project conventions and `.claude/memories/` are auto-loaded; no need to re-read.
