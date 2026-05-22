# B5 — A2.1 + B1 joint

## Status

**Closed (did not promote).** Canonical run complete; v2.4 Cell-D-s30 remains the default. **Dominated by A2.1 alone** on failure-anchor mean CRPS and substantially worse on controls. See [`V4_RESULTS.md`](../V4_RESULTS.md).

## Setup

- Surfaced by the v4 sanity evidence (V4_EXPERIMENTS_PLAN §B5): both B1's drift correction and A2.1's corrwindow distance had material per-anchor wins isolated, and the two knobs are orthogonal code paths. Stacking them was zero new code — a config-only change.
- Config: `configs/analog_mc/ablation_B5_joint.yaml` (v2.4 baseline + `matcher_distance: corrwindow`, `corrwindow_length: 100`, `local_linear_correction: true`, `n_eff_values: [50]`).
- No new tests beyond B1 + A2.1's existing test suites — both knobs already independently tested.
- Canonical run: `runs/analog_mc/20260521T121025Z` — 76 folds, 1000 paths, ~5.2h compute (2026-05-21 17:40 → 22:49).

## Headline numbers

| Metric | v2.4 baseline | B5 canonical | Δ |
|---|---:|---:|---:|
| Mean walk-forward test_crps | 0.04755 | 0.06417 | +35% |
| 15-anchor mean CRPS | 0.06111 | 0.05832 | −4.6% |
| 5 V3.5 failure-anchor mean CRPS | 0.09471 | 0.08423 | **−11.0%** ✅ |
| 5 control-anchor mean CRPS | 0.02051 | 0.02637 | +28.6% ❌❌ |
| V3.5 failures recovered (90-band ≥45/60) | 0/5 | 2/5 | +2 |
| Anchors regressing CRPS >5% | — | 10/15 | far over the ≤2 bar ❌ |

| Same anchor | v2.4 | B1 solo | A2.1 solo | **B5 joint** | Joint dominates? |
|---|---:|---:|---:|---:|---|
| 2001-10-02 V CRPS | 0.114 | 0.095 | 0.060 | **0.050** | ✅ best of all |
| 2010-04-23 crash CRPS | 0.071 | 0.054 | **0.038** | 0.043 | ❌ A2.1 wins |
| 2018-10-08 Q4 CRPS | 0.062 | **0.061** | 0.081 | 0.076 | ❌ B1 wins |
| 2020-03-16 COVID CRPS | 0.179 | 0.185 | **0.146** | 0.162 | ❌ A2.1 wins |
| 2026-02-19 recent CRPS | 0.048 | **0.047** | 0.054 | 0.057 | ❌ B1 wins |

Per-anchor detail in [`_b5_joint_fat_tail.md`](_b5_joint_fat_tail.md).

## Mechanistic reading

The B5 design hypothesis was "additive on wins, attenuated on losses" — each knob's strengths preserved without amplifying weaknesses. Reality: **only one of five failure anchors shows the additive pattern** (2001-10-02 V, where B5 −56% CRPS beats both A2.1's −48% and B1's −16%). The other four anchors place B5 between the two solos:

- 2010-04-23 crash: A2.1 0.038 < B5 0.043 < B1 0.054 — A2.1 better than B5
- 2018-10-08 Q4: B1 0.061 < B5 0.076 < A2.1 0.081 — B1 better than B5
- 2020-03-16 COVID: A2.1 0.146 < B5 0.162 < B1 0.185 — A2.1 better than B5
- 2026-02-19 recent: B1 0.047 < v2.4 0.048 < A2.1 0.054 < B5 0.057 — even v2.4 better than B5

**Why no additive win.** B1's drift correction is computed via WLS on `(z_candidate, y_candidate)` pairs using the matcher's probabilities. Under v2.4's weighted-Euclidean, those probabilities are concentrated on z-similar analogs whose forward returns have small magnitude on average; B1's β estimates a sensible drift adjustment. Under corrwindow, the probabilities are concentrated on *shape-similar* analogs whose forward returns are wider-spread; B1's β picks up a different signal — one that's actively harmful when corrwindow already captures the forward direction correctly. The correction then becomes a *wrong-direction shift* on top of an already-OK forecast.

**Why controls collapse harder under B5 than under A2.1.** A2.1's tight 90-bands plus B1's drift adjustment combine to produce forecasts whose mean shifts but whose 90-band edges don't keep up. The control mean CRPS regression is +28.6% (vs A2.1 alone +10%, B1 alone −9.3%).

## Decision-rule verdict

Per V4 plan §B5 decision rules:
- "**failure CRPS improves more than max(B1_solo, A2.1_solo) by ≥3%**" — A2.1 solo had −20.1%, B5 has −11.0%. **Fails this branch** (B5 is worse than A2.1 alone).
- "**failure CRPS ≈ max(solo)**" — B5 is materially below A2.1 (−11.0% vs −20.1%). So **not "additive at neutral"** either.
- "**failure CRPS materially worse than either solo**" — B5 is between, but closer to dominated than competing. Falls into the **"knobs interact destructively"** decision branch.

Per the V4 mandatory fat-tail criterion: **does not pass** (2/5 recovered; 10/15 regressing).

## Implication for v5 roadmap

B5 closes a clean question: simply stacking matcher-distance changes and drift corrections does not produce additive improvements. Two takeaways:

1. **The interaction between distance and drift correction matters.** B1's WLS fit assumes the matcher's chosen analogs are similar-z-state precedents; under corrwindow they're similar-shape precedents with different forward distributions. A v5 B1-prime could compute the regression on the v2.4-style analog set (a *separate matcher head*) while using corrwindow for path sampling.

2. **Joint experiments need an interaction diagnostic.** v4 had no way to detect that B5 would amplify A2.1's regressions until the canonical results landed. A v5 protocol: sanity-check joint experiments at the 5 V3.5 failure anchors first, abort if the joint sanity numbers exceed both solos.

## Deliverables

- `configs/analog_mc/ablation_B5_joint.yaml` (config-only; no new code)
- `runs/analog_mc/20260521T121025Z/` (canonical artefacts)
- `results/analog_mc/data/fat_tail_b5_joint.json` + `_diff.json`
- `docs/analog_mc/experiments/figs/b5_joint_fat_tail/` (15-anchor panel)
- This narrative + auto-generated [`_b5_joint_fat_tail.md`](_b5_joint_fat_tail.md)
