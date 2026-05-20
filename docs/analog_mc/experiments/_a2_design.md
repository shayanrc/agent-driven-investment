# A2 (matcher-distance ablation) — design scoping

Companion to [`V4_EXPERIMENTS_PLAN.md` §A2](../V4_EXPERIMENTS_PLAN.md#a2-ofter-maximal-correlation-distance--p0-was-p1-promoted-by-v35), promoted to joint P0 alongside B1 by [`V3_5_RESULTS.md`](../V3_5_RESULTS.md). V3.5.4 identified the v2.4 matcher's **temporal clustering of top-K analogs** as the contributing failure pathology — A2 attacks this by replacing the weighted-Euclidean-on-z-scores distance with a correlation-based criterion that doesn't share the same locality.

## Variants

Splitting A2 into two sub-experiments because OFTER's exact formulation requires the paper PDF, which the current tooling can't reliably extract.

### A2.1 — Correlation-window distance (implementable now)

**Distance.** `1 - |Pearson_corr(W_target, W_candidate)|` where `W_x = returns[x - L + 1 : x + 1]` is the L-day pre-origin causal window for index `x`. Default `L = 20`.

**Rationale.**
- Distance is between *time-series windows*, not z-score snapshots — orthogonal to z-score features' locality, which is V3.5.4's diagnosed pathology.
- Pearson correlation is scale-invariant and shift-invariant, exactly the structural-similarity criterion we want.
- "Modified" max-correlation in OFTER's sense reduces to Pearson under joint Gaussianity, and our log-returns are approximately Gaussian conditional on the local regime — so A2.1 is a defensible first-order approximation of OFTER without claiming faithful reproduction.

**Implementation surface.**
- New module: `src/analog_mc/distances_corrwindow.py` with `compute_corrwindow_distance(returns, origin_idx, candidate_idx, window_length) -> distances`.
- Vectorized across all candidates: precompute per-index window-z-score `W̃_x = (W_x - mean) / std` once, then `corr = W̃_target @ W̃_candidates.T / L`.
- Causality: `W_x` includes `returns[x]` (inclusive); strictly causal at `x`. Eligibility filter (`d + block_length < origin_idx`) is unchanged.
- Wires into `simulate.forecast()` gated on `matcher_distance: str` (default `"weighted_euclidean"`, new option `"corrwindow"`).

**Search interaction.** The `(w₁, w₂, w₃)` weight grid no longer applies — there's a single global distance per `(target, candidate)` pair. The search reduces to optimizing `(window_length, n_eff)`, a 2-D grid. Use the same coarse grid: `L ∈ {10, 20, 60, 100}`, `n_eff ∈ {15, 30, 50, 80, 150}` → 20 combinations vs the current 66×5 = 330. Faster search.

**Config knob.**
```python
matcher_distance: str = "weighted_euclidean"  # or "corrwindow"
corrwindow_length: int = 20
```

**Decision rule.** Same as V4 plan §A2:
- Mean CRPS ≤ Cell-D-s30 within noise: defer the weight grid; investigate joint A2+B1.
- Mean CRPS ≥1% worse: composite-Euclidean is doing real work; close.
- Mean CRPS ≥1% better: structural win; integrate.

Plus the V3.5 fat-tail bar: ≥3 of 5 failure anchors recover 90%-band ≥45/60.

**Cost.** ~0.5 d impl + ~3 h canonical run (fewer search combos than B1).

### A2.2 — OFTER-faithful (deferred pending paper)

When the paper is accessible (PDF text extraction works, or someone summarizes the relevant section), implement OFTER's specific "modified maximal correlation coefficient" with its prescribed estimator (likely ACE-style alternating conditional expectations or a kernel-CCA variant). Compare A2.2 against A2.1 to attribute any gap to the specific OFTER modification vs the general correlation-window principle.

**Status.** Open. WebFetch on `arxiv.org/abs/2304.03877` and `arxiv.org/pdf/2304.03877` returned only the abstract / unreadable binary. Tabled until paper access is solved.

## Open questions

1. **Window length L.** 20 is a reasonable default (covers ~1 trading month), matching `momentum_lookback`. But L=60 (one horizon) or L=100 might give cleaner regime signals. The search includes 4 values.
2. **Correlation strength.** A small fraction of candidate windows could have `|corr| ≈ 1` (perfect anti-correlation) by chance under finite samples. Tikhonov-like guard: floor distances at a small ε > 0 so the n_eff softmax doesn't blow up. Default `ε = 0.05`.
3. **Block-conditional re-match.** v2.4's conditional sampling re-matches in block 1+ using per-path z-scores. Under A2.1 this would require recomputing windows from per-path simulated tails — feasible but expensive. **Decision for A2.1 v1**: re-match in conditional sampling uses the same corrwindow distance, with per-path windows assembled from real tail + simulated block tail (same warm-start strategy as v2.2). Cost: K candidates × n_paths = 9.8M × 1000 dot products per block in the worst case. Vectorizable, but if profiling shows it's the bottleneck, fall back to block-0 fixed probs (search-time only) and accept the conditional regression.
4. **Interaction with B1.** A2.1 and B1 are orthogonal in code paths (different `compute_distance`; B1 hooks the drift). The first A2.1 run should hold `local_linear_correction: false` for clean attribution. A2.1+B1 combination is a **future experiment** ("A2+B1 joint") gated on both shipping individually.

## Sequencing

1. Wait for B1 canonical (in progress, run dir `runs/analog_mc/20260520T155220Z`).
2. Implement A2.1 + tests on `v4-experiments` branch.
3. A2.1 canonical run (~3 h).
4. A2.1 panel + diff vs v2.4 + diff vs B1.
5. Compare verdicts:
   - If A2.1 close to v2.4 mean CRPS: structural feature problem, not a distance problem.
   - If A2.1 beats v2.4 on failures: combine with B1 next.
   - If A2.1 worse than v2.4 across the board: composite-Euclidean is doing more work than V3.5.4 implied.

## Out of scope for A2 v1

- A2.2 (OFTER faithful) — deferred.
- Learned-distance experiments (CRPS-as-loss, learned similarity from V3_PLAN) — v5+.
- A2 + GARCH combination — orthogonal experiment if both ship.
- Multi-asset distances — out of v4 entirely.

## Build order

1. Scaffold `src/analog_mc/distances_corrwindow.py`.
2. Add `matcher_distance` + `corrwindow_length` to Config.
3. Hook into `simulate.forecast()` and `sampling.generate_paths_conditional`.
4. Write `configs/analog_mc/ablation_A2_corrwindow.yaml`.
5. Tests (causality, vectorization, identity-on-self, n_eff bracket).
6. Sanity at 5 failure anchors (like the B1 script).
7. Canonical run.
8. Fat-tail panel + diff.
9. `_a2_corrwindow.md` report.

Steps 1–6 land in one PR-equivalent commit on `v4-experiments`; canonical run + diagnostics in subsequent commits.
