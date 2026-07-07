# V4.5.6 — A2.1 path-construction inspection

**Question.** V4.5.2 Mode 3 (8 of A2.1's 10 regressions) identified anchors where the matcher's top-K analog set looks reasonable (diffuse top-K, mass-weighted forward similar to v2.4) yet CRPS still regresses. Hypothesis: A2.1's disabling of conditional block sampling (`simulate.py:306–309`) makes paths under-disperse, narrowing bands inappropriately. Confirmable from existing forecasts — read-only check.

**Method.** For each Mode-3 regression anchor + 3 wins for comparison, read the cached forecast paths from the canonical run's `forecasts.npz`. Compute cross-path **cumulative-log-return std growth ratio** = `std_cum_logret[day 59] / std_cum_logret[day 0]`. Brownian baseline: √60 ≈ 7.75 (since cum std scales as √t for iid returns).

Script: [`scripts/analog_mc/v4_5/path_construction_inspection.py`](../../../scripts/analog_mc/v4_5/path_construction_inspection.py) · Data: [`v4_5_6_path_construction.json`](../../../results/analog_mc/data/v4_5_6_path_construction.json)

## Per-anchor cumulative-std growth

| Anchor | Class | Real % | A2 cum-σ-growth | v24 cum-σ-growth | A2 term σ % | v24 term σ % | Path verdict |
|---|:---:|---:|---:|---:|---:|---:|---|
| 2012-03-14 | REG | −5.5 | 7.17 | 8.03 | 6.4% | 7.3% | A2 ≈ v24 |
| **2022-03-01** | **REG** | **−14.7** | **3.56** | **8.87** | **9.1%** | **15.3%** | **A2 catastrophically tight (40% of v24)** |
| 2026-02-19 | REG | +17.5 | 5.94 | 6.77 | 6.9% | 9.9% | A2 modestly tighter |
| 2010-11-10 | REG | +7.4 | 8.21 | 8.50 | 8.1% | 7.5% | Both fine |
| 2001-04-04 | REG | +33.5 | 7.83 | 6.34 | 30.4% | 26.9% | A2 fine (both over-disperse but OK) |
| 1991-03-26 | REG | −1.6 | 10.37 | 11.47 | 17.0% | 14.8% | Both over-disperse |
| 2017-06-01 | REG | +0.1 | 4.75 | 7.61 | 5.1% | 6.0% | **A2 tighter (62% of v24)** |
| 2010-04-23 | WIN | −10.4 | 11.97 | 9.63 | 13.0% | 8.6% | A2 MORE dispersed |
| 2001-10-02 | WIN | +38.6 | 10.65 | 7.78 | 20.3% | 28.0% | A2 MORE dispersed |
| 2020-03-16 | WIN* | +43.8 | 5.67 | 6.90 | 22.3% | 30.8% | Both tight |

(\* 2020-03-16 is a "win" only by CRPS; coverage still collapses to 11/60.)

## Reading

**The Mode-3 regression mechanism is confirmed at 2022-03-01 and 2017-06-01:** A2.1's path dispersion grows at less than half (2022-03-01) and ~⅔ (2017-06-01) of v24's rate. The matcher's selected analogs produce paths that converge over the horizon — the 90-band squeezes inward.

Specifically at 2022-03-01:
- A2 cum-σ at day 59 = 9.1% of underlying.
- v24 cum-σ at day 59 = 15.3% — almost 70% wider.
- Realized −14.7% sits well inside v24's bands; outside A2.1's narrower bands.
- A2 CRPS regression at this anchor is +33.7% per V4.5.1 — directly attributable to dispersion deficit.

**The WIN anchors show the inverse:** 2010-04-23 and 2001-10-02 have A2 MORE dispersed than v24 (11.97 vs 9.63; 10.65 vs 7.78). These wins owe their CRPS improvement to A2.1's broader bands containing the realized extremes — a feature, not a bug, when the matcher's selected analogs span larger forward variability.

The variable that distinguishes "A2 dispersion helps" from "A2 dispersion hurts" is **top-K diversity** (Mode 1 from V4.5.2). When A2.1 confidently locks on a single analog (top-1 ≥ 0.4), paths converge. When it spreads across multiple analogs, paths disperse.

## What's actually happening — block-0 distance reuse

The mechanism: under `matcher_distance="corrwindow"`, `simulate.py:306–309` forces `use_conditional = False`, so:
- Block 0's distances → probabilities → analog selections are computed at the **real origin**.
- Blocks 1–5 reuse the **same** distances/probabilities at the real origin — not at each path's simulated sub-origin.

Under v2.4 (`conditional_block_sampling=True`), each block recomputes distances using each path's evolving simulated z-vector. This re-mixes the analog set per block per path, diversifying paths.

A2.1 v1's choice was pragmatic — implementing conditional re-matching under corrwindow requires per-path simulated *returns windows*, not just z-vectors, which is a non-trivial pipeline change. V4.5.6 confirms the cost: tighter dispersion at concentration-prone anchors.

## V5 fix — `conditional_corrwindow` (V5.A.3)

Enable conditional re-matching under corrwindow. The implementation needs:
1. Maintain a per-path simulated returns window of length L (sliding append the simulated daily return at each block boundary).
2. At each block, compute corrwindow distance from each path's window to the candidate pool (vectorized — `(n_paths, L) @ (K, L).T`-style GEMM analogous to `composite_distance_batched`).
3. Re-route `simulate.py:306–309` to dispatch a `generate_paths_conditional_corrwindow` variant.

Cost estimate: ~200 LOC in a new module + ~50 in simulate.py + 10 tests. Comparable to the original A2.1 implementation.

This is **V5.A.3** in the v5 plan — a candidate that addresses **the 2 Mode-3 anchors where dispersion is the clear problem** (2022-03-01, 2017-06-01), and possibly other diffuse-top-K cases marginally. Note: it does NOT address Mode-1 (concentrated top-K) regressions — those need the distance-or-feature-augmentation fixes from V4.5.7.

## Open follow-up

- **σ-growth ratio doesn't perfectly discriminate** Mode-3 regressions from non-regressions. 2010-11-10 has growth 8.21 (healthy) yet regresses CRPS by 12.7%. The mechanism there is likely *mean-shift* (A2.1 says +6.6%, v24 says +4.1%, realized +7.4% — A2 is closer to realized!), not dispersion. CRPS may be penalizing fine-grained per-day mismatch despite better point estimate. Sub-investigation deferred.

- **2017-06-01 has both anomalies**: lower dispersion (A2=4.75 vs v24=7.61) AND a small mean shift. Realized is +0.1% — basically flat — so the regression here is more about the noise floor than mechanism. Minor anchor.

## Verdict

**Path-construction confirmed as a contributing mechanism for Mode-3 regressions.** Most severe at 2022-03-01 (A2.1 paths converge at 40% the rate of v24's). Enabling conditional corrwindow re-matching (**V5.A.3**) is a credible fix for 2-3 regressions, complementary to V5.A (ensemble) for the rest.

Sequencing: V5.A.2 (path-level ensemble) remains the cheapest test (no new code). V5.A.3 (conditional corrwindow) is more ambitious but addresses a specific subset of regressions. V5.A.1 (Tikhonov mix) third in priority — it interpolates the matchers but doesn't solve the dispersion deficit.
