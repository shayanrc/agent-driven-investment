# A2.1 — Correlation-window matcher distance (L=100, n_eff=50)

## Status

**Closed (did not promote).** Canonical v1 run complete; v2.4 Cell-D-s30 remains the default. See [`V4_RESULTS.md`](../V4_RESULTS.md) for the full v4 synthesis.

This is the v4 A2 implementation. A2.2 (literature-faithful OFTER `arXiv 2304.03877`) remains **deferred** because the paper's exact "modified maximal correlation coefficient" definition was not text-extractable from the available PDF tooling. A2.1 is a defensible approximation that uses Pearson correlation between L-day pre-origin return windows. See [`_a2_design.md`](_a2_design.md).

## Setup

- Spec: [`_a2_design.md`](_a2_design.md) §A2.1.
- Config: `configs/analog_mc/ablation_A2_corrwindow_L100.yaml` (v2.4 baseline + `matcher_distance: corrwindow`, `corrwindow_length: 100`, `n_eff_values: [50]`).
- Implementation: `src/analog_mc/distances_corrwindow.py` (vectorized Pearson-corr-window distance with ε floor + degenerate-window guards). `simulate.forecast()` routes via `_compute_block0_distances()`; corrwindow internally disables conditional block sampling (per design D3 — block-0 probs reused for every block; conditional re-matching would require per-path window assembly, out of v1 scope).
- Tests: `tests/analog_mc/test_corrwindow.py` — 15 tests, all pass.
- Canonical run: `runs/analog_mc/20260521T061730Z` — 76 folds, 1000 paths, weight grid × `n_eff=50` only, ~5.9h compute (2026-05-21 11:47 → 17:39).
- Sanity precursor: `scripts/v4_a2_corrwindow_sanity.py`, results at [`_a2_corrwindow_sanity_v0.md`](_a2_corrwindow_sanity_v0.md). Selected L=100 from sweep over {10, 20, 60, 100}.

### v0 abort

The initial canonical attempt (`runs/analog_mc/_a2_corrwindow_v0_aborted_20260521T033950Z`) used the v2.4 default `n_eff_values: [15, 30, 50, 80, 150]`. The grid search reliably landed at `n_eff=150` because under corrwindow the weight grid is informationally degenerate (single distance per `(target, candidate)`) and val_crps is flat across weights, while broad probability distributions (high n_eff) score better on the val set. But high n_eff washes the shape-similar concentration that makes corrwindow win at failure anchors. At fold 24 (V-recovery), v0 produced test_crps 0.17209 vs sanity's predicted 0.060 — a 3× regression. Killed at fold 25, relaunched as v1 with `n_eff_values: [50]` pinned.

## Headline numbers

| Metric | v2.4 baseline | A2.1v1 canonical | Δ |
|---|---:|---:|---:|
| Mean walk-forward test_crps | 0.04755 | 0.06141 | +29% |
| 15-anchor mean CRPS | 0.06111 | 0.04881 | **−20.1%** ✅✅ |
| 5 V3.5 failure-anchor mean CRPS | 0.09471 | 0.07566 | **−20.1%** ✅✅ |
| 5 control-anchor mean CRPS | 0.02051 | 0.02257 | +10.0% ❌ |
| V3.5 failures recovered (90-band ≥45/60) | 0/5 | 2/5 | +2 (2001-10-02, 2010-04-23) |
| Anchors regressing CRPS >5% | — | 10/15 | far over the ≤2 bar ❌ |

Per-anchor failure detail in [`_a2_corrwindow_L100_fat_tail.md`](_a2_corrwindow_L100_fat_tail.md).

The aggregate walk-forward test_crps of +29% is misleading — that metric averages over all ~4500 forecast origins across 76 folds, and A2.1's wins are concentrated at the 5 failure anchors specifically. The 15-anchor fat-tail aggregate **−20.1% failure CRPS** is the right metric for assessing the matcher-distance hypothesis.

## Mechanistic reading

A2.1 replaces v2.4's weighted-Euclidean-on-z-scores distance with `1 − |Pearson_corr(W_target, W_candidate)|` where W is the L-day causal pre-origin return window. The matcher now selects candidates whose recent past *shape* resembles the target, rather than candidates whose recent z-state values resemble the target.

V3.5.4 had hypothesized that the matcher's temporal clustering (top-K analogs concentrated in 1–3 historical regimes) was a contributing failure mechanism — corrwindow's window-shape criterion breaks the Euclidean-on-features locality that drives this clustering.

**Where A2.1 wins — V-recovery + flash-crash pattern.** The 2001-10-02 post-9/11 V (realized +38.6%) and 2010-04-23 flash crash (realized −10.4%) both share a "regime in transition with distinctive multi-week shape" feature. Corrwindow at L=100 (~5 trading months) captures these shapes; the matched candidates have forward distributions that include large realized moves. CRPS at both anchors −47%; 90-band at 2010-04-23 jumps from 27 → 57 (blows past the promotion bar).

**Where A2.1 loses — shape-similar regimes whose forwards diverged.** 2008-10-03 is the canonical failure: pre-Lehman, the 100-day window contained the August-September 2008 stress but not the October collapse. Corrwindow pulls similar-shaped 2007 bull-momentum-peaks; those forwards were all upward (the 2007 peak rolled over slowly). The matcher confidently picks the wrong precedents. Result: 90-band coverage 52 → 7, CRPS +122%.

The same mechanism operates more mildly at 2018-10-08 Q4 selloff (90-band 31 → 10) and at the 2020-03-16 COVID anchor (38 → 11; though A2.1 has the best CRPS here, the bands are over-tight).

**Why it's not the search.** The aborted v0 run showed that corrwindow + search-over-n_eff is unstable (val_crps optimum at n_eff=150 contradicts test performance at the same n_eff). v1 fixed n_eff=50 directly; the search degeneracy is gone but the structural shape-similar-regime problem is the dominant failure mode.

## Decision-rule verdict

Against the V4 plan §A2 decision rules:
- **mean CRPS ≤ v2.4 within noise**: fails on walk-forward aggregate (+29%); but the 15-anchor fat-tail aggregate is better (−20.1% failure CRPS). The "aggregate" metric used by the original decision rule isn't appropriate for an experiment whose value is concentrated at a specific anchor subset.
- **failure-anchor wins without bull-momentum regression**: fails. Bull-momentum-peak regimes (2010-04-23 won, but 2012-03-14 control +27% CRPS) and 2008-10-03 (+122%) are exactly the regressions.

Against the V4 mandatory fat-tail criterion: **does not pass** (2/5 failures recovered vs ≥3 needed; 10/15 anchors regressing vs ≤2).

## Implication for v5 roadmap

A2.1 has the cleanest failure-anchor signal of any v4 experiment. The failure mode is *not* the corrwindow distance itself — it's the matcher's over-confidence at regime-coverage anchors where the shape-similar window selects pre-divergence precedents.

**Strongest v5 candidate: gated A2.1.** Two designs:

1. **val_crps gate** — fall back to weighted-Euclidean when corrwindow's val_crps for the fold is anomalously high (e.g. >1.5× cross-fold median). This is the cheapest fix; per-fold gating, no per-origin overhead. The 2008-10-03 case would likely trip the gate (val_crps was 0.288 — many σ above median).

2. **Distance blending** — `d = (1 − α) · d_weighted_euclidean + α · d_corrwindow` with α searched per fold or globally. Stabilizes the matcher around the v2.4 distance while letting corrwindow contribute at shape-distinct anchors.

A2.1 also surfaced an unresolved question: corrwindow's disabling of conditional sampling (block-0 probs reused everywhere) is a confound — A2.1 vs v2.4 measures both the distance change and the sampling-regime change simultaneously. A v5 experiment that re-enables conditional sampling under corrwindow (computing per-path windows for blocks 1+) would isolate the distance contribution.

## Deliverables

- `src/analog_mc/distances_corrwindow.py`, `src/analog_mc/simulate.py` (corrwindow hook + conditional-sampling guard), `src/analog_mc/config.py` (`matcher_distance`, `corrwindow_length`)
- `configs/analog_mc/ablation_A2_corrwindow_L100.yaml`
- `tests/analog_mc/test_corrwindow.py` (15 tests)
- `scripts/v4_a2_corrwindow_sanity.py` + `_a2_corrwindow_sanity_v0.md` + `v4_a2_corrwindow_sanity.json`
- `runs/analog_mc/20260521T061730Z/` (canonical v1 artefacts)
- `runs/analog_mc/_a2_corrwindow_v0_aborted_20260521T033950Z/` (archived v0 for diagnostics)
- `results/analog_mc/data/fat_tail_a2_corrwindow_L100.json` + `_diff.json`
- `docs/analog_mc/experiments/figs/a2_corrwindow_L100_fat_tail/` (15-anchor panel)
- This narrative + auto-generated [`_a2_corrwindow_L100_fat_tail.md`](_a2_corrwindow_L100_fat_tail.md)
