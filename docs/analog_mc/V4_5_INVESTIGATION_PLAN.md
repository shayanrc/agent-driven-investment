# analog_mc v4.5 — pre-V5 diagnostic investigation plan

Bridge between v4 (closed — [`V4_RESULTS.md`](V4_RESULTS.md)) and v5 (unscoped). Runs five short-compute diagnostics on existing canonical run artifacts to discriminate **why** A2.1's wins and losses split the way they do, so v5 experiment selection is grounded in evidence rather than the heuristic recommendations carried out of V4_RESULTS.

## Purpose

V4_RESULTS landed three canonicals (B1, A2.1, B5) and a closing recommendation: *"A2.1 with selective application"* — either gated by `val_crps > 1.5×` cross-fold median, or Tikhonov-mixed with weighted-Euclidean. That recommendation is a literature-prior guess: nothing in the v4 evidence proves the gate signal actually fires at the right anchors, or that temporal-clustering (V3.5.4's diagnosis) is the dominant mechanism behind the 10/15 regressions, or that COVID-style anchors have any analog fix at all.

v4.5 closes those evidence gaps before v5 commits to an implementation. Each diagnostic is read-mostly (operates on existing run dirs), cheap (≤3h compute combined), and produces a JSON + a per-investigation markdown that feeds the v5 plan.

| If diagnostics show… | Then the v5 lever is… |
|---|---|
| `val_crps` gate fires on A2.1 catastrophic folds but not wins | **V5.1: Gated A2.1** with val_crps threshold |
| `val_crps` gate doesn't separate; temporal-cluster index does | **V5.1': Cluster-gated A2.1** with Herfindahl-on-top-K-analog-years |
| Neither gate signal works | **V5.2: Tikhonov-mixed (α-search)** as the fallback |
| B1 control regressions all have leverage-outlier β | **V5.3: Leverage-trimmed B1** — a 30-line WLS tweak |
| COVID-style anchor pool lacks +30%/60d instances | **No-go for analog-only fix**; document tail inflation as v5+ scope (out of v5 sequencing) |
| 2008-10-03 collapse confirmed as "shape-similar wrong-forward" | Tightens the case for cluster-gated A2.1 specifically |

These hypotheses are not mutually exclusive — most v4.5 investigations have additive value. The sequencing is by cost, with explicit early-stop conditions when a single result obviates later work.

## Starting state (entering v4.5)

| Item | State |
|---|---|
| Production default | v2.4 — Cell-D-s30 (`configs/analog_mc/default.yaml`) |
| Canonical baseline run | `runs/analog_mc/20260520T045525Z` (v2.4, 76 folds) |
| v4 canonical runs | B1: `runs/analog_mc/20260520T155220Z`; A2.1v1: `runs/analog_mc/20260521T061730Z`; B5: `runs/analog_mc/20260521T121025Z` |
| Fat-tail diff JSONs | `results/analog_mc/data/fat_tail_{b1_local_linear,a2_corrwindow_L100,b5_joint}_diff.json` |
| Eval anchor set | 15 anchors in [`results/analog_mc/data/fat_tail_eval_anchors.json`](../../results/analog_mc/data/fat_tail_eval_anchors.json) |
| v5 recommendation (unconfirmed) | V4_RESULTS §"Promotion decision" — gated A2.1 OR Tikhonov-mixed A2.1 |
| Open question | Which of the recommendations actually has gate-signal validity? Is there a 3rd path? |

## Scope and non-scope

**In scope.** Read-mostly diagnostics on existing canonical run artifacts plus light per-anchor re-inference. Total compute budget: ~3h. Each diagnostic writes a JSON + markdown report.

**Out of scope.**
- Implementing any v5 fix (those are v5 experiments).
- Re-running canonicals at different settings.
- Anything requiring new walk-forward grids.
- Multi-asset analysis or transaction costs (deferred).
- Re-running V3.5 investigations on v4 canonical artifacts — they were specific to v2.4's failure anchor mechanism.

## v4 regression set

The 10 anchors where A2.1 regresses CRPS >5% relative to v2.4, ordered by severity. Origin index in the canonical run's returns space.

| Anchor | Origin idx | Fold | 60d realized | A2.1 90-band | v2.4 90-band | A2.1 CRPS Δ vs v2.4 |
|---|---:|---:|---:|---:|---:|---:|
| 2008-10-03 | (look up) | (look up) | ? | 7/60 | 52/60 | **+122%** |
| 2018-10-08 | 8260 | 54 | −12.7% | 10/60 | 31/60 | +31% |
| 2020-03-16 | 8612 | 56 | +43.8% | 11/60 | 38/60 | −19% (CRPS) but coverage worse |
| (7 others — populate from `fat_tail_a2_corrwindow_L100_diff.json`) |

The 5 B1 regressions for cross-reference (from `fat_tail_b1_local_linear_diff.json`):

| Anchor | 90-band v2.4 → B1 | Mechanism hypothesis |
|---|---|---|
| 1990-09-24 | 55 → 11 | Unexplained — V4.5.3 target |
| 2020-03-16 | 38 → 19 | Drift too small for +43.8% rally |
| (3 others) | (populate from diff JSON) | |

Both lists get authoritatively populated by the V4.5 scripts; do not hand-edit.

---

## Investigations

Listed in priority order — earliest results may obviate or refocus later steps.

### V4.5.1 — A2.1 gate-signal validation (~1 h)

**Question.** The V4_RESULTS recommendation is a fold-wise gate: substitute weighted-Euclidean for corrwindow when corrwindow's `val_crps > 1.5×` the cross-fold median. Does this gate actually fire on the catastrophic folds and not on the wins?

**Method.**
1. Read A2.1v1 canonical (`runs/analog_mc/20260521T061730Z/folds/*/summary.json`). Extract per-fold `val_crps`, `test_crps`, `n_eff`.
2. Compute cross-fold median val_crps and the proposed 1.5× threshold.
3. List folds whose `val_crps > threshold` — call these the "gated folds".
4. For each fat-tail anchor (15 total), identify its fold and check: would the gate fire?
5. Construct the predicted V5.1 fat-tail panel by hand: for gated folds substitute v2.4's per-anchor CRPS + coverage; for non-gated folds keep A2.1's. Compare to v2.4 baseline AND to A2.1.
6. Compute predicted V5.1 promotion-bar metrics: failures recovered (≥45/60), anchors regressing >5% CRPS.

**Decision rule.**
- **Predicted V5.1 hits the promotion bar (≥3/5 failures recovered AND ≤2/15 regressions)** → V5.1 is the right experiment. Run it as the first v5 canonical.
- **Predicted V5.1 partially helps** (e.g., 2/5 recovered, 4/15 regress) → gate signal is correct in direction but threshold is wrong. Sweep threshold ∈ {1.2, 1.5, 2.0, 3.0}× median in `_v4_5_1` and report best.
- **Gate doesn't separate wins from losses** (gated folds include 2010-04-23, or non-gated folds include 2008-10-03) → val_crps is the wrong signal. Proceed to V4.5.2's temporal-cluster signal as the gate candidate.

**Deliverable.** `docs/analog_mc/v4.5/_v4_5_1_gate_signal.md` with the gated/non-gated fold table, predicted V5.1 panel, and verdict. `results/analog_mc/data/v4_5_1_gate_signal.json` with the numerical payload.

**Implementation hint.** ~80 lines. Pure analysis over existing run dirs — no new inference. The thresholded fat-tail synthesis can be done by mixing the per-anchor cells from `fat_tail_baseline_v24.json` and `fat_tail_a2_corrwindow_L100.json`. Be careful: the v5.1 gate makes per-fold decisions, so an anchor whose fold is gated takes the v2.4 CRPS + coverage exactly. No statistical adjustment needed — it's a hypothetical reconstruction.

---

### V4.5.2 — A2.1 analog autopsy at the 10 regressions (~1.5 h)

**Question.** V4_RESULTS attributes A2.1's regressions to "shape-similar wrong-forward" — corrwindow confidently selects historical windows that look like the target but whose realized forwards diverge. Is this temporal clustering of top-K analogs the actual signature, and would a cluster-based gate work better than `val_crps`?

**Method.** For each of the 10 A2.1 regression anchors (plus the 2 wins 2010-04-23 and 2001-10-02 as positive controls):
1. Read the A2.1v1 fold's `weights`, `n_eff`, `matcher_distance`, `corrwindow_length` from `summary.json`.
2. Re-run the matcher's probability distribution at the anchor origin via `analog_mc.distances_corrwindow.corrwindow_distance()` + `distances_to_probs()`. (The full forecast() doesn't need to re-run — just the distance/probability layer.)
3. Identify the top-20 highest-probability analogs.
4. Compute the **temporal Herfindahl index** on top-K analog years: H = Σ p²_year where p_year is the summed probability mass per calendar year of the analog origin. H ∈ [1/n_distinct_years, 1].
5. For each top-20 analog: record analog date, year, realized 60-day-forward return, contribution to weighted-mean forward.
6. Compare to v2.4's top-20 analog set at the same anchor (recompute distances with weighted-Euclidean).

**Decision rule.** Define a "high-cluster" fold as H > 0.4 (i.e., ≥40% of top-K mass in a single year-equivalent).

- **≥7/10 regressions have H > 0.4 AND ≤1/2 wins do** → temporal clustering is a discriminating signal. **V5.1 gates on temporal cluster, not val_crps**. Spec out the threshold sweep in v5 plan.
- **5–6/10 regressions cluster; mixed signal at wins** → temporal clustering partially discriminates. Document the per-anchor cluster index in v5 plan as a candidate auxiliary feature; keep val_crps as primary gate.
- **<5/10 regressions cluster** → mechanism is not temporal clustering. Look at the per-anchor analog-date histograms for an alternative pattern (e.g., regime-similarity rather than year-similarity; precedent recency bias).

**Deliverable.** `docs/analog_mc/v4.5/_v4_5_2_analog_autopsy.md` with per-anchor top-20 tables, H values, win-vs-regression comparison, and verdict. `results/analog_mc/data/v4_5_2_analog_autopsy.json`.

**Implementation hint.** ~150 lines, heaviest of the five. The matcher's probability layer needs to be reproduced faithfully — read `_compute_block0_distances()` in `src/analog_mc/simulate.py` for the corrwindow routing and reuse it. Year-extraction goes through the returns index (a DatetimeIndex). The 2008-10-03 anchor needs its origin index looked up (it's not in the V3.5 anchor list); use the returns DatetimeIndex search.

---

### V4.5.3 — B1 β autopsy at the 5 regressions (~1 h)

**Question.** B1's most severe regression is at 1990-09-24 (control 90-band 55 → 11) — unexplained in V4_RESULTS. Is the local-linear regression's β driven by a single leverage outlier? If yes, a leverage-trimmed WLS fix (one v5 follow-up candidate) is well-targeted.

**Method.** For each of the 5 B1 regression anchors:
1. Read the B1 fold's `weights`, `n_eff`, `local_linear_correction=True` from `summary.json`.
2. At the anchor origin, recompute: the K candidates' z-vectors, probabilities, and forward 60-day log-return sums.
3. Call `fit_local_linear_correction()` and capture `LocalLinearDiagnostics`: β, matcher_mean, predicted_mean, correction, clamp_hit, leverage_max (add this field if not already in the diagnostics — see implementation hint).
4. Compute the **leverage score** per candidate: h_ii = w_i x_iᵀ (XᵀWX + λI)⁻¹ x_i. Identify max-leverage candidate.
5. Refit β with that candidate dropped; recompute correction.
6. Compare original correction vs leverage-trimmed correction.

**Decision rule.**
- **≥3/5 regressions have max h_ii > 0.5 AND trimmed correction differs from original by >50%** → leverage outliers drive B1 regressions. **V5.3 candidate: leverage-trimmed B1**, expected to be a 30-line fix.
- **Max h_ii reasonable across regressions but correction still mis-predicts** → mechanism is not leverage; β fits the data but the data itself misleads. Document as "B1's regression is structural; v5 B1 candidates are not promising" and deprioritize the B1 family.
- **clamp_hit fires in ≥2 regressions** → extrapolation guard is doing its job; the regression isn't a B1 implementation bug but a fundamental "target far from cluster" issue. Tightens the case for matcher-side v5 work (V5.1), not drift-side.

**Deliverable.** `docs/analog_mc/v4.5/_v4_5_3_b1_beta_autopsy.md` with per-anchor β/leverage/correction table + verdict. `results/analog_mc/data/v4_5_3_b1_beta_autopsy.json`.

**Implementation hint.** ~120 lines. Reuses `src/analog_mc/local_linear.py::fit_local_linear_correction`. May need to extend `LocalLinearDiagnostics` with `leverage_max` and `leverage_argmax` fields (~5 lines in `local_linear.py`). Be careful with the Tikhonov term — it affects β slightly, which affects leverage; this is fine to leave as-is and document.

---

### V4.5.4 — COVID-style anchor pool sufficiency (~30 min)

**Question.** All three v4 experiments fail at 2020-03-16 (best: A2.1 90-band 11/60). V3.5.2 audited the pool's tail-mass under v2.4's eligibility. Does that audit still hold for COVID at 2020-03-16, or did the V3.5.2 result get obsoleted? More precisely: how many `+30%/60d` rallies exist in the candidate pool at the 2020-03-16 fold, and are any of them weighted in A2.1's top-20?

**Method.**
1. Re-run V3.5.2's pool-tail-mass computation specifically for the 2020-03-16 fold's candidate pool. Bucket by 60-day-forward log-return percentile.
2. Count how many candidates have realized 60d return > +30%, > +40%.
3. For each high-return candidate, check its probability under v2.4, B1, A2.1, B5 at the 2020-03-16 anchor (just compute distances).
4. Tabulate: pool size, count of +30% candidates, mean and max probability assigned by each matcher.

**Decision rule.**
- **<3 candidates with +30%/60d realized AND mean A2.1 probability of those candidates < 1× uniform** → pool is structurally tail-poor for COVID. **No analog-only fix exists**; documented as v5+ scope (tail inflation, gated wider conditional band). Removes COVID from v5 promotion-bar.
- **≥3 high-return candidates exist BUT all matchers underweight them** → matcher problem persists across v4 experiments. v5 should test a long-horizon momentum-aware distance (B2 delay-coordinates, deferred from v4).
- **Pool is rich AND A2.1 weights high-return candidates competitively** → COVID failure is in path-construction, not selection. Out of v5 scope; probably needs v6 mechanism.

**Deliverable.** `docs/analog_mc/v4.5/_v4_5_4_covid_pool.md` with the pool histogram + per-matcher probability table. `results/analog_mc/data/v4_5_4_covid_pool.json`.

**Implementation hint.** ~70 lines. Heaviest reuse from V3.5.2 (`scripts/v3_5/audit_pool_tail_mass.py`) — modify to take a single fold rather than all 5 failure folds, and add the per-matcher probability column. The matcher probability layer needs the same composite-distance computation as V4.5.2; if V4.5.2 has been done first, refactor a small helper.

---

### V4.5.5 — Cross-experiment mechanism map (~1 h, depends on V4.5.2/3/4)

**Question.** Across all 15 fat-tail anchors and the 4 models (v2.4, B1, A2.1, B5), classify each (anchor × model) cell by failure mechanism. Produces a single matrix that the v5 plan reads to decide *which* v5 experiment helps *which* anchor — and surfaces any anchors that no v5 candidate addresses.

**Method.**
1. For each cell, evaluate four indicators against thresholds (calibrated against V4.5.2/3/4):
    - **Magnitude undershoot**: |predicted_mean − realized| > 0.5σ_pool, with predicted_mean having the right sign.
    - **Temporal clustering**: H > 0.4 (from V4.5.2).
    - **Shape-similar wrong-forward**: top-3-analog weighted-mean-forward has opposite sign to realized.
    - **Pool exhaustion**: |realized| > 99th percentile of pool-forward-return distribution AND pool count < 5 in the tail bucket.
2. Classify each cell into the primary mechanism (the indicator that fires most strongly). If none fires, label "uncharacterized".
3. Build a 15×4 heatmap-style table.
4. For each mechanism, list the v5 candidate that addresses it.

**Decision rule.** Produces the v5 plan's "addressable failures" estimate:
- For each v5 candidate (V5.1 gated A2.1, V5.2 mixed, V5.3 leverage-trimmed B1, eventual V5.4 tail inflation), count anchors it would plausibly fix based on the mechanism map.
- The v5 candidate that addresses the most anchors without introducing new regressions becomes V5.1's canonical (replacing the V4_RESULTS recommendation if the map disagrees).

**Deliverable.** `docs/analog_mc/v4.5/_v4_5_5_mechanism_map.md` with the matrix + v5-candidate-coverage analysis. `results/analog_mc/data/v4_5_5_mechanism_map.json`.

**Implementation hint.** ~100 lines, synthesis-heavy not compute-heavy. Reads V4.5.2/3/4 JSONs and the four `fat_tail_*` JSONs. The mechanism classification is heuristic — be explicit in the doc about the threshold calibration.

---

## Sequencing and decision tree

Execute in order. Each step may reshape v5 priorities.

```
1. V4.5.1 gate-signal validation (1 h)
   ├─ Predicted V5.1 hits promotion bar:
   │   ├─ V5.1 spec is confirmed — proceed to V4.5.3+ for v5 follow-ups
   │   └─ Lock val_crps threshold in v5 plan
   ├─ Gate partially helps:
   │   └─ Sweep threshold; report best within V4.5.1; continue to V4.5.2
   └─ Gate doesn't separate wins from losses:
       └─ Continue to V4.5.2 — temporal-cluster signal is the alternate

2. V4.5.2 analog autopsy (1.5 h)
   ├─ Temporal clustering discriminates (≥7/10 regressions cluster, ≤1/2 wins do):
   │   └─ V5.1' (cluster-gated A2.1) replaces V5.1 OR augments it
   ├─ Partial discrimination:
   │   └─ Use cluster index as auxiliary in v5 plan; keep val_crps primary
   └─ No discrimination:
       └─ V5.1 needs a non-obvious gate; fall back to V5.2 (Tikhonov mix) as primary

3. V4.5.3 B1 β autopsy (1 h)
   ├─ Leverage outliers drive ≥3/5 regressions:
   │   └─ V5.3 (leverage-trimmed B1) added to v5 plan as a low-cost candidate
   └─ Mechanism is not leverage:
       └─ B1 family deprioritized in v5; document why

4. V4.5.4 COVID pool sufficiency (30 min)
   ├─ Pool is structurally tail-poor:
   │   └─ Remove COVID from v5 promotion bar; document tail inflation as v5+ scope
   ├─ Pool exists but underweighted:
   │   └─ v5 candidate: B2 (long-horizon delay-coordinate distance) added
   └─ Pool is rich and weighted:
       └─ Out of v5 scope; documented as v6+ structural

5. V4.5.5 mechanism map (1 h, depends on 2/3/4)
   └─ Synthesis matrix → ranked list of v5 candidates by anchor coverage
```

**Stop conditions.**
- **V4.5.1 hits the bar cleanly.** V4.5.2 still has value (tighter gate signal), V4.5.3/4 lower-priority but cheap. Consider whether mechanism map is necessary.
- **V4.5.2 shows clustering doesn't discriminate AND V4.5.1 shows val_crps doesn't either.** v5 has no good gate signal — restructure plan around Tikhonov mixing (V5.2) as primary. Skip V4.5.5.
- **V4.5.4 declares COVID structurally tail-poor.** Subtract COVID from the promotion-bar accounting for v5. The 5 V3.5 failures become 4 (COVID excluded); recovery bar drops to ≥3/4.

## Final synthesis: V4_5_RESULTS.md

After all (or stopped) investigations, produce `docs/analog_mc/V4_5_RESULTS.md` containing:

1. **Per-investigation summary** (one paragraph linking to detailed report).
2. **Mechanism inventory** — which anchors fail for which reasons, drawn from V4.5.5.
3. **V5 plan recommendations** — explicit experiment list, ordered by expected-coverage × cost. May include:
    - V5.1 (gated A2.1 — gate signal TBD by V4.5.1/2)
    - V5.2 (Tikhonov-mixed A2.1) if no clean gate signal
    - V5.3 (leverage-trimmed B1) if V4.5.3 confirms
    - V5.4 (long-horizon delay-coordinate distance) if V4.5.4 supports
    - "Defer to v6: tail inflation" if structural pool limits dominate
4. **Stop-list** — what was investigated and ruled out, so v5 doesn't re-litigate.
5. **Read-first checklist** for the v5 plan author.

## Deliverables manifest

```
docs/analog_mc/V4_5_INVESTIGATION_PLAN.md       # this doc
docs/analog_mc/V4_5_RESULTS.md                   # final synthesis
docs/analog_mc/v4.5/_v4_5_1_gate_signal.md       # gate-signal validation
docs/analog_mc/v4.5/_v4_5_2_analog_autopsy.md    # A2.1 analog autopsy
docs/analog_mc/v4.5/_v4_5_3_b1_beta_autopsy.md   # B1 β autopsy
docs/analog_mc/v4.5/_v4_5_4_covid_pool.md        # COVID pool sufficiency
docs/analog_mc/v4.5/_v4_5_5_mechanism_map.md     # cross-experiment mechanism map

scripts/v4_5/validate_gate_signal.py             # V4.5.1
scripts/v4_5/analog_autopsy_a2.py                # V4.5.2
scripts/v4_5/b1_beta_autopsy.py                  # V4.5.3
scripts/v4_5/covid_pool_sufficiency.py           # V4.5.4
scripts/v4_5/mechanism_map.py                    # V4.5.5

results/analog_mc/data/v4_5_1_gate_signal.json
results/analog_mc/data/v4_5_2_analog_autopsy.json
results/analog_mc/data/v4_5_3_b1_beta_autopsy.json
results/analog_mc/data/v4_5_4_covid_pool.json
results/analog_mc/data/v4_5_5_mechanism_map.json
```

## Read-first checklist for a fresh session

A future Claude session picking this up cold should read, in order:

1. **This file** (`V4_5_INVESTIGATION_PLAN.md`) — purpose, scope, sequencing.
2. **[`V4_RESULTS.md`](V4_RESULTS.md)** — the v4 outcomes that motivate every v4.5 question. Especially §"Per-experiment narrative" and §"Open issues for v5+".
3. **[`V3_5_RESULTS.md`](V3_5_RESULTS.md)** — V3.5's diagnostic patterns that V4.5 partly re-applies (temporal clustering H index, pool tail-mass).
4. **[`experiments/_a2_design.md`](experiments/_a2_design.md)** and **[`_b1_design.md`](experiments/_b1_design.md)** — the implementation decisions whose mechanisms V4.5.2/3 inspect.
5. **`src/analog_mc/distances_corrwindow.py`** and **`src/analog_mc/local_linear.py`** — the matcher / corrector that V4.5.2/3 dissect.
6. **`src/analog_mc/simulate.py::_compute_block0_distances`** — the distance-routing hook V4.5.2 reuses.

The `CLAUDE.md` project conventions and `.claude/memories/` are auto-loaded; no need to re-read.
