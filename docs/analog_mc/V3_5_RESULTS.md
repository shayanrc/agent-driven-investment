# analog_mc v3.5 — investigation results and v4 reshape

Synthesis of the four pre-V4 diagnostics specified in
[`V3_5_INVESTIGATION_PLAN.md`](V3_5_INVESTIGATION_PLAN.md). All four were
executed end-to-end on the canonical run
`runs/analog_mc/20260520T045525Z` (76 folds × 5 failure anchors + 5 controls)
without any new walk-forward runs. None of the plan's stop conditions
fired — all four investigations completed.

The headline result reshapes v4 priorities: **B1 (Platzer local-linear) is
promoted to single-P0, A1 (FHS) is demoted from P0 to P1**, and an
incidental finding about val-set unreliability surfaces as a new candidate
experiment (B4, optional).

## Per-investigation summary

### V3.5.1 — per-failure weight inspection
[`v3.5/_v3_5_1_weights.md`](v3.5/_v3_5_1_weights.md) · [`data`](../../results/analog_mc/data/v3_5_1_weights.json)

**Verdict: weights heterogeneous across failures — not a uniform myopia
problem.** Failure weights span the entire grid: `[0.9, 0.1, 0]`,
`[0, 0.1, 0.9]`, `[0.09, 0.68, 0.24]`, `[0, 0, 1.0]`, `[0.98, 0, 0.02]`.
Controls show the same spread. The plan's "all 5 failures favor short-horizon
→ promote B3 to P0" branch does not fire.

**Incidental finding worth flagging:** 4 of 5 failure folds land at extreme
grid corners (max weight > 0.9) and have val_crps materially worse than
test_crps (e.g. 2001-10-02: val 0.152, test 0.108; 2010-04-23: val 0.050,
test 0.038). Controls show the same corner pattern but with much smaller
val/test gaps. This suggests the val set at failure anchors is *itself*
stressful, and the search ends up at corner solutions that happen not to
generalize. **Not promoting B3 on this evidence alone, but flagging as a
motivation for a v4 candidate "B4: regularized weight search" — see "New
experiments" below.**

### V3.5.2 — candidate pool tail-mass audit
[`v3.5/_v3_5_2_tail_mass.md`](v3.5/_v3_5_2_tail_mass.md) · [`data`](../../results/analog_mc/data/v3_5_2_tail_mass.json)

**Verdict: tail moves are abundant — this is a matching problem, not an
evidence problem.** Each failure-fold pool contains **156–194 ≥+30%
rallies** and **120 ≤−30% drops** (pool sizes 3,681–9,801). Even fold 24
(2001-10-02, the smallest train window) has 156 +30% rallies eligible to
sample. The structural-ceiling hypothesis is rejected: the analog primitive
has the data it needs; the matcher just isn't selecting it.

This rules out the "tail inflation as v5 scope" stop condition. The bottleneck
is upstream of the path-sampling primitive.

### V3.5.3 — GARCH-FHS spot-check
[`v3.5/_v3_5_3_fhs_spotcheck.md`](v3.5/_v3_5_3_fhs_spotcheck.md) · [`data`](../../results/analog_mc/data/v3_5_3_fhs_spotcheck.json)

**Verdict: mixed (2/5 catches) — not a clean A1 promotion, but the *shape*
of the result is informative.**

| Anchor | Realized | v2.4 90% in | FHS 90% in | FHS/v24 width |
|---|---:|---:|---:|---:|
| 2010-04-23 (bull→crash) | −10.4% | 27/60 | **9/60** | 1.04× |
| 2001-10-02 (V-recovery) | +38.6% | 44/60 | **24/60** | **0.54×** |
| 2018-10-08 (sideways→drop) | −12.7% | 31/60 | 30/60 | 1.41× |
| 2020-03-16 (COVID rally) | +43.8% | 38/60 | **60/60** | **2.02×** |
| 2026-02-19 (recent rally) | +17.5% | 41/60 | **60/60** | 1.11× |

FHS wins decisively on the two COVID-style V-recoveries (where it inflates
the 90% band to capture +30 to +40% moves), but *underperforms* v2.4 on the
2001 V-recovery (where the long pre-2001 vol regime was calmer and GARCH fits
to that, producing tight bands). It also makes the bull-momentum failures
slightly worse. The plan's "≥3 of 5 caught" threshold for promoting A1 ahead
of B1 is not met (2/5). The conditional read is: **A1 partially solves a
specific sub-pattern (high-vol-bear-bottom → rally), not the failure surface
as a whole.**

### V3.5.4 — analog autopsy
[`v3.5/_v3_5_4_analog_autopsy.md`](v3.5/_v3_5_4_analog_autopsy.md) · [`data`](../../results/analog_mc/data/v3_5_4_analog_autopsy.json)

**Verdict: the matcher's analogs are temporally clustered, and the cluster's
forwards systematically underestimate the realized magnitude. This is "right
era, wrong magnitude" — not wrong-era matches.**

Headline per-anchor table (matcher expected vs realized 60d):

| Anchor | Realized | E[60d∣matcher] | Miss | P(sign match) | P(\|fwd\|≥\|realized\|) | P(rally≥30%) |
|---|---:|---:|---:|---:|---:|---:|
| 2010-04-23 | −10.4% | +4.1% | **−14.6%** | 21% | 24% | 3% |
| 2001-10-02 | +38.6% | +12.3% | **+26.3%** | 89% | 0.3% | 3% |
| 2018-10-08 | −12.7% | +3.0% | **−15.6%** | 29% | 32% | 1% |
| 2020-03-16 | +43.8% | +5.9% | **+37.9%** | 88% | 0% | 1% |
| 2026-02-19 | +17.5% | +5.1% | **+12.3%** | 76% | 8% | 0.1% |

Three of five failures (2001-10-02, 2020-03-16, 2026-02-19) have **sign match
≥76% but magnitude mass ≤8%**: the matcher gets direction right but
systematically picks analog clusters whose forward windows are *milder* than
the realized move. Top-20 listings confirm this:

- **2020-03-16** (Fed-pivot COVID rally, +43.8%): top-20 dominated by
  post-correction calm rallies (2004-07, 2019-05, 2011-08) with +3% to +15%
  forwards. Zero +30%+ rallies appear in top-20. The matcher's weights
  `[0, 0, 1]` (pure z-200) selected on long-horizon state and pulled
  temporally-clustered mild-rally precedents.
- **2001-10-02** (post-9/11 V, +38.6%): top-20 is a 2001-04 cluster (+15
  to +25% forwards) — directionally right but capped at +33.5% max, with
  most mass on +15-20%.
- **2026-02-19**: top-20 is +5-12% post-correction rallies (1991, 1995,
  2015, 2018, 2014). Pool has 194 +30% rallies; matcher places 0.1% mass on
  any of them.

Two failures (2010-04-23, 2018-10-08) are partly bimodal: the matcher places
~25–30% mass on the realized-sign tail but majority on continuation, with
mean expected return on the wrong side of zero.

**Mechanism.** Distance-based matching on hand-tuned z-score features is
strongly *temporally clustered* — once one date matches, its neighbors
match strongly too (close in state-space). When the dominant cluster's
forward windows are mild, the matcher cannot recover the rare-tail moves
that *do* exist elsewhere in the pool. This is exactly the high-Lyapunov
behavior B1 (Platzer local-linear) is designed for: inflate the conditional
variance around the matcher mean when the local Jacobian is large.

## Cross-investigation synthesis

The four investigations together rule out three hypotheses and strengthen
one:

| Hypothesis | Status after v3.5 |
|---|---|
| Search myopia (all failures → short-horizon weights) | **Ruled out** (V3.5.1) |
| Pool empty of tail moves (structural ceiling) | **Ruled out** (V3.5.2) |
| Analog primitive actively suppresses σ-dispersion | **Mixed** (V3.5.3): true for COVID-style rallies, false elsewhere |
| Matcher picks right state but wrong-magnitude analog cluster | **Confirmed** (V3.5.4) |

The fifth hypothesis ("wrong-era matches → A2 candidate strengthens") is
**partially supported**: A2 (max-correlation distance) might break the
temporal-clustering pathology by replacing Euclidean-on-features with a
correlation criterion that doesn't share the same locality. But it's not
the headline finding.

The headline finding is **right-era-wrong-magnitude**, which is the exact
failure mode B1's Jacobian variance inflation targets.

## Recommended v4 reshape

Specific edits to apply to [`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md):

### 1. Promote B1 to sole P0; demote A1 to P1

Rationale: V3.5.4 demonstrates the precise failure pattern B1 targets
(magnitude mass on chosen-cluster forwards is too thin; Jacobian inflation
is the canonical fix). V3.5.3 shows A1's σ-inflation lever only helps a
sub-pattern (2/5 caught) and *hurts* on other failures — A1 is no longer
the highest-info-density experiment for the failure-anchor problem. A1
retains attribution-baseline value (the original motivation) and should
still ship, but not ahead of B1.

**Concrete edit to the inventory table:**

```diff
- | **A1** | Textbook FHS baseline | Attribution | ... | ~2-3 h compute | **P0** |
- | **B1** | Platzer local-linear correction | Structural | ... | ~1.5 d impl + ~3 h run | **P0** |
+ | **B1** | Platzer local-linear correction | Structural | ... | ~1.5 d impl + ~3 h run | **P0** |
+ | A1 | Textbook FHS baseline | Attribution | ... | ~2-3 h compute | P1 |
```

### 2. Promote A2 from P1 to P0 (joint with B1)

Rationale: V3.5.4 shows the matcher's temporal clustering as a *contributing*
mechanism. A2's max-correlation distance is a clean test of whether
breaking the Euclidean-on-features locality recovers tail-magnitude
analogs. A2 and B1 attack different parts of the same diagnosis (matcher
selection vs path inflation) and are independent code paths.

**Concrete edit:**

```diff
- | A2 | OFTER maximal-correlation distance | Attribution | ... | P1 |
+ | **A2** | OFTER maximal-correlation distance | Attribution | ... | **P0** |
```

### 3. Update the Sequencing section

Replace the current sequencing with:

> 1. **Canonical Cell-D-s30** baseline already landed at
>    `runs/analog_mc/20260520T045525Z`; this is the v2.4 reference for all
>    v4 comparisons.
> 2. **B1 (Platzer local-linear)** — single highest-priority experiment;
>    targets the magnitude problem V3.5.4 identified.
> 3. **A2 (OFTER max-corr)** — parallelizable with B1 (independent code
>    path); tests whether breaking Euclidean-on-features locality recovers
>    tail magnitude through better analog selection.
> 4. **A1 (textbook FHS)** — runs after B1/A2 land. Still required as an
>    attribution baseline (does the analog primitive add value over plain
>    FHS?) but no longer expected to close failure anchors on its own;
>    V3.5.3 showed mixed results.
> 5. **C1 (KS GoF diagnostic)** — same as before, runs alongside B1/A2 (no
>    compute conflict). Adopt as decision rule for promotion.
> 6. **B4 (regularized weight search, candidate — see below)** — gated on
>    whether the val/test gap at failure anchors persists after B1/A2 ship.
> 7. **B2 / B3** — scoped, execution deferred. Re-evaluate after the top
>    block lands.

## New experiments surfaced (optional, candidate)

### B4 — regularized weight search (candidate)

**Motivation.** V3.5.1 revealed that failure folds disproportionately end
at extreme grid corners with large val/test CRPS gaps (2001-10-02 val 0.152
vs test 0.108; 2010-04-23 val 0.050 vs test 0.038). The 60-day val window
is short and stressful at regime transitions; the grid search lands at
corners that don't generalize. This is plausibly a small but real
contributor to failure-anchor performance — orthogonal to the matcher /
inflation question.

**Method.** Add a Dirichlet prior on weights centered on the cross-fold
median (`~[0.10, 0.10, 0.15]`) plus a small uniform-prior mass, score
candidates as `val_crps − λ · log p(weights | prior)`. Single hyperparameter
λ. Run with the same 76 folds; compare per-anchor coverage.

**Decision rule.**
- If regularized search resolves ≥2 of the 5 failure anchors (and doesn't
  regress controls): ship it as a v4 minor improvement.
- Otherwise document as a non-finding and close.

**Cost.** ~3 h impl + ~2 h run.

**Status.** Candidate / P2. Gate on B1/A2 results before adding to the
sequencing — if those close the failures, B4 isn't needed.

## Out-of-v4 carry-overs

- **Tail inflation (v5+).** Not raised by V3.5.2 (pool has the moves), so
  this is *not* a v4 add. Re-evaluate at end of v4 if B1+A2 fail to recover
  the 2020-03-16 / 2026-02-19 magnitude.
- **The temporal-clustering pathology itself.** V3.5.4 surfaced that
  distance-based matching pulls top-20 analogs from 1–3 historical clusters.
  If B1+A2 don't address this, v5 questions about cluster-aware analog
  selection (penalize within-cluster duplicates) open up.
- **Conditional-mean correction independent of dispersion.** The matcher's
  *mean* underestimates magnitude in 3/5 failures — B1's variance correction
  may not move the mean far enough. If post-B1 the conditional mean still
  misses by >15% at 2020-03-16, a v5 mean-correction experiment becomes
  necessary.

## Deliverables produced

```
docs/analog_mc/V3_5_RESULTS.md                   # this synthesis
docs/analog_mc/v3.5/_v3_5_1_weights.md
docs/analog_mc/v3.5/_v3_5_2_tail_mass.md
docs/analog_mc/v3.5/_v3_5_3_fhs_spotcheck.md
docs/analog_mc/v3.5/_v3_5_4_analog_autopsy.md

scripts/v3_5/inspect_failure_weights.py
scripts/v3_5/audit_pool_tail_mass.py
scripts/v3_5/fhs_spotcheck.py
scripts/v3_5/analog_autopsy.py

results/analog_mc/data/v3_5_1_weights.json
results/analog_mc/data/v3_5_2_tail_mass.json
results/analog_mc/data/v3_5_3_fhs_spotcheck.json
results/analog_mc/data/v3_5_4_analog_autopsy.json
```

No source code in `src/analog_mc/` was modified; v3.5 is pure read-only
diagnostics on the canonical run.
