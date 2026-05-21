# analog_mc v4 — results and promotion decision

Companion to [`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md). Captures the three canonical experiments that landed (B1, A2.1, B5) and the resulting promotion decision: **none of the three are promotable; v2.4 Cell-D-s30 remains canonical**.

The full session ran 2026-05-20 to 2026-05-21, ~26h of compute split across three sequential walk-forward runs.

## Headline scorecard

Against the V4 promotion bar from [`FAT_TAIL_EVAL.md`](FAT_TAIL_EVAL.md) §"v4 mandatory deliverable":

> An experiment that improves aggregate CRPS but regresses on >2 fat-tail anchors should not be promoted to default without explicit justification. End-of-v4 success criterion: recovery of ≥3 of 5 failure anchors (90%-band ≥45/60 days).

| Experiment | Mean CRPS | Failure CRPS Δ | Control CRPS Δ | Failures @ 90 ≥45 | Anchors regress CRPS >5% | Promotion |
|---|---:|---:|---:|---:|---:|---|
| v2.4 baseline | 0.04755 | — | — | 0/5 | — | (current) |
| B1 (Platzer local-linear) | 0.05021 | **−6.6%** ✅ | **−9.3%** ✅ | 1/5 | 5/15 | ❌ |
| A2.1v1 (corrwindow L=100, n_eff=50) | 0.06141 | **−20.1%** ✅✅ | +10.0% ❌ | 2/5 | 10/15 | ❌ |
| B5 (A2.1+B1 joint) | 0.06417 | −11.0% ✅ | +28.6% ❌ | 2/5 | 10/15 | ❌ |

None of the three pass either the recovery bar (need ≥3/5) or the regression bar (≤2/15 regressions). v2.4 stays.

## Per-anchor CRPS at the 5 V3.5 failure folds

| Anchor | Realized 60d | v2.4 | B1 | A2.1v1 | B5 | Winner |
|---|---:|---:|---:|---:|---:|---|
| 2001-10-02 V-recovery | +38.6% | 0.114 | 0.095 | 0.060 | **0.050** | B5 (−56% vs v2.4) |
| 2010-04-23 flash crash | −10.4% | 0.071 | 0.054 | **0.038** | 0.043 | A2.1 (−47%) |
| 2018-10-08 Q4 selloff | −12.7% | 0.062 | **0.061** | 0.081 | 0.076 | B1 (barely) |
| 2020-03-16 COVID rally | +43.8% | 0.179 | 0.185 | **0.146** | 0.162 | A2.1 (−19%) |
| 2026-02-19 recent rally | +17.5% | 0.048 | **0.047** | 0.054 | 0.057 | B1 (barely) |

Bold = best CRPS for that row.

### 90% band coverage (days inside, of 60)

| Anchor | Realized 60d | v2.4 | B1 | A2.1v1 | B5 |
|---|---:|---:|---:|---:|---:|
| 2001-10-02 V | +38.6% | 44 | **55** | 53 | 53 |
| 2010-04-23 crash | −10.4% | 27 | 37 | **57** | 51 |
| 2018-10-08 Q4 | −12.7% | 31 | **36** | 10 | 16 |
| 2020-03-16 COVID | +43.8% | **38** | 19 | 11 | 8 |
| 2026-02-19 recent | +17.5% | 41 | **43** | 38 | 32 |

### 50% band coverage (days inside, of 60; nominal 30)

| Anchor | Realized 60d | v2.4 | B1 | A2.1v1 | B5 |
|---|---:|---:|---:|---:|---:|
| 2001-10-02 V | +38.6% | 6 | 7 | 32 | **40** |
| 2010-04-23 crash | −10.4% | 3 | 3 | 24 | **27** |
| 2018-10-08 Q4 | −12.7% | 4 | **5** | 0 | 0 |
| 2020-03-16 COVID | +43.8% | **6** | 4 | 3 | 3 |
| 2026-02-19 recent | +17.5% | **32** | **32** | 27 | 14 |

## Per-experiment narrative

### B1 — Platzer local-linear conditional-mean correction
[`_b1_local_linear_fat_tail.md`](experiments/_b1_local_linear_fat_tail.md) · [`fat_tail_b1_local_linear_diff.json`](../../results/analog_mc/data/fat_tail_b1_local_linear_diff.json)

The only experiment with **both failure AND control aggregate improvement** (−6.6% failure, −9.3% control CRPS). Drift correction works exactly as theory predicts at high-Lyapunov regimes: matcher gets direction right but underestimates magnitude → B1 supplies a per-day drift that closes the gap.

But:
- Only 1/5 V3.5 failures recovers the 45/60 bar (2001-10-02, 44→55).
- 5/15 anchors regress >5% CRPS, including the **1990-09-24 control collapse (55→11 90-band)**. This regression is unexplained — worth a v5 follow-up. The fold's matcher chose `[0.50, 0.10, 0.40]` weights; the drift correction may have been over-estimated by the regression's local Jacobian estimate at a regime not similar to the analog cluster.
- 2020-03-16 COVID gets *worse* (38→19) — the bigger lesson is that COVID's +43.8% rally is too far outside the analog pool's selected forwards for any small drift correction to recover.

**Verdict.** Real, partial, broadly positive. Not promotable solo per V4 heuristic, but the cleanest example of "small correction, broad positive effect" — the analog primitive can be drift-corrected without breaking elsewhere.

### A2.1v1 — Correlation-window matcher distance (L=100, n_eff=50)
[`_a2_corrwindow_L100_fat_tail.md`](experiments/_a2_corrwindow_L100_fat_tail.md) · [`fat_tail_a2_corrwindow_L100_diff.json`](../../results/analog_mc/data/fat_tail_a2_corrwindow_L100_diff.json)

**Cleanest failure-anchor signal of any v4 experiment** — failure mean CRPS −20.1%, and the most dramatic single-anchor improvement landed (2010-04-23 flash crash: CRPS −47%, **90-band coverage 27→57**, blowing past the promotion bar).

But also:
- Controls regress aggregate +10% (B1 was −9.3%).
- 10/15 anchors regress CRPS >5%.
- **2008-10-03 catastrophic regression**: CRPS +122%, 90/60 collapses 52→7 days. Mechanism: corrwindow at L=100 pulls 2007-style "bull-momentum-peak" precedents (which there are several of in pre-2008 history), whose forwards diverged sharply once the GFC hit in late 2008. The matcher correctly identifies "this looks like X" — but X's forward distribution is exactly the wrong reference under regime change.
- 2020-03-16 COVID 90-band coverage even worse than B1 (11 vs 19).
- Mid-experiment fix required: A2.1 v0 (with the full v2.4 n_eff_values grid) had to be aborted because the search degenerated to `n_eff=150` everywhere under corrwindow's flat val landscape. v1 pins `n_eff_values=[50]` and works as sanity predicted.

**Verdict.** The cleanest evidence that **matcher distance is the right lever for the failure-anchor problem** — V3.5.4's "temporal clustering" diagnosis is confirmed by A2.1's strong wins where it works. But too unstable at regime-coverage anchors to ship unmoderated. **Strongest candidate for v5 work** (a "gated" or "regularized" variant).

### B5 — A2.1 + B1 joint
[`_b5_joint_fat_tail.md`](experiments/_b5_joint_fat_tail.md) · [`fat_tail_b5_joint_diff.json`](../../results/analog_mc/data/fat_tail_b5_joint_diff.json)

The hypothesis was "additive on wins, attenuated on losses". Reality:
- Additive at 2001-10-02 only: B5 CRPS 0.050 beats both A2.1 (0.060) and B1 (0.095).
- Elsewhere B5 sits *between* A2.1 and B1 — neither dominating nor cleanly attenuating.
- Aggregate failure −11% (worse than A2.1's −20%).
- Aggregate control +29% (much worse than A2.1's +10% — B1's drift inside A2.1's already-tight bands amplifies the band-tightening problem).
- Same 2/5 failure recovery and 10/15 regression count as A2.1.

**Verdict.** Joint is *not* a clean win. B1's drift correction interacts unfavorably with A2.1's distance-change in most regimes — the correction is fit on a different analog set under corrwindow than it was at v2.4, and produces less appropriate shifts. **Solo A2.1 dominates B5 across most failure anchors.** Do not promote.

## Mechanistic synthesis

V3.5.4 hypothesized two contributing failure mechanisms:

1. **Temporal clustering of top-K analogs** → distance change (A2) should help.
2. **Matcher conditional-mean magnitude underestimation** → drift correction (B1) should help.

The canonical evidence:

| Mechanism | Confirmed? | Best fix |
|---|---|---|
| Temporal clustering | **YES** — A2.1's 2001-10-02 and 2010-04-23 wins (−47% each) are direct evidence | A2.1 family (corrwindow + gating) |
| Magnitude underestimation | YES — B1 wins at 2018-10-08 and 2026-02-19 by exactly this mechanism | B1 |
| Sufficient when stacked? | **NO** — B5 is not better than A2.1 alone on failures and worse on controls | Joint is the wrong shape |

Additional mechanism surfaced by the canonical results, not in V3.5.4:

3. **Shape-similar regime mis-matching** — at 2008-10-03 (and to a lesser extent 2018-10-08), corrwindow's "shape-similar window" criterion confidently locks onto historical regimes whose forwards diverge from the realized path under regime change. The matcher is *too* concentrated. **Not addressed by any v4 experiment**; surfaces as the dominant failure mode of A2.1.

## Promotion decision

**v2.4 Cell-D-s30 remains the canonical default.** No v4 experiment passes the promotion bar.

The strongest evidence for the next iteration: **A2.1 with selective application**. The corrwindow distance is a genuine improvement at V-recovery and crash anchors — the failure mode is concentrated at regime-coverage anchors where the distance over-confidently selects bad precedents. Two paths forward:

1. **Gated A2.1** — fall back to weighted-Euclidean when corrwindow's val_crps is anomalously high (>1.5× cross-fold median). Costs nothing at inference; potentially eliminates the catastrophic regressions.
2. **Tikhonov-regularized corrwindow** — mix the two distances `d = (1-α) · d_weighted_euclidean + α · d_corrwindow` and search over α per fold. Gives the matcher a stable fallback.

Both are v5 candidates; surfacing here as the next-experiment proposal.

## Open issues for v5+

- **1990-09-24 B1 regression** (control 90/60: 55 → 11): unexplained mechanism. Worth a per-fold diagnostic on what the local-linear regression's β looked like there.
- **2008-10-03 A2.1 collapse** (52 → 7): the matcher chose shape-similar 2007 precedents. Possible structural feature of any "shape-based" matcher near GFC-style transitions. Either gate it or extend the distance to include a "regime macro indicator" (out of v4 scope).
- **2020-03-16 COVID coverage collapse across all three v4 experiments**: even the best (A2.1 CRPS-wise) ends up with 90-band coverage 11/60. The pool likely has the rally instances but no analog has the COVID-specific dispersion — this is a candidate for tail inflation (v5+ scope per V3.5).
- **B5 control-CRPS catastrophe** (+29%): joint experiments need a way to detect when stacking corrections multiplies the failure surface rather than additively recovering. Currently no diagnostic surfaces this; only the fat-tail panel reveals it.

## Out-of-v4 carry-overs and renaming

The v4 plan also listed:
- **A1 (textbook FHS)** — never run. V3.5.3 spot-check (2/5 failures caught) suggested A1 was already a partial fix. Still worth a formal canonical for attribution; deferred to v5.
- **C1 (block-bootstrap KS PIT GoF)** — never run. Pure diagnostic; defer.
- **B2 (delay-coordinate features), B3 (Dirichlet weight posterior), B4 (regularized weight search)** — all deferred pending the A2.1-gating direction.

## Deliverables manifest

```
docs/analog_mc/V4_RESULTS.md                        # this synthesis
docs/analog_mc/experiments/_b1_design.md
docs/analog_mc/experiments/_a2_design.md
docs/analog_mc/experiments/_b1_local_linear_fat_tail.md
docs/analog_mc/experiments/_a2_corrwindow_L100_fat_tail.md
docs/analog_mc/experiments/_b5_joint_fat_tail.md
docs/analog_mc/experiments/figs/b1_local_linear_fat_tail/  # 15 charts
docs/analog_mc/experiments/figs/a2_corrwindow_L100_fat_tail/  # 15 charts
docs/analog_mc/experiments/figs/b5_joint_fat_tail/  # 15 charts

results/analog_mc/data/fat_tail_baseline_v24.json
results/analog_mc/data/fat_tail_b1_local_linear{,_diff}.json
results/analog_mc/data/fat_tail_a2_corrwindow_L100{,_diff}.json
results/analog_mc/data/fat_tail_b5_joint{,_diff}.json

runs/analog_mc/20260520T155220Z/  # B1 canonical
runs/analog_mc/20260521T061730Z/  # A2.1v1 canonical
runs/analog_mc/20260521T121025Z/  # B5 canonical
runs/analog_mc/_a2_corrwindow_v0_aborted_20260521T033950Z/  # archived
```

## Reading order for a fresh session

1. This file.
2. [`V3_5_RESULTS.md`](V3_5_RESULTS.md) for the pre-v4 mechanism hypotheses.
3. [`experiments/_b1_design.md`](experiments/_b1_design.md), [`_a2_design.md`](experiments/_a2_design.md) for the decisions baked into each experiment.
4. The three `_<exp>_fat_tail.md` reports for per-anchor details.
5. [`V4_EXPERIMENTS_PLAN.md`](V4_EXPERIMENTS_PLAN.md) for what was originally scoped vs what shipped.
