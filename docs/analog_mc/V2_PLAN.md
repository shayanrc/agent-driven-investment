# analog_mc v2 — Implementation Plan

## Purpose

v2 adds the two features that v1's Stage 9 diagnostics specifically flagged as needed for NASDAQ100:

1. **Trailing-momentum drift** — replaces the v1 zero-drift forecast with a shrunk recent-momentum estimate added on top of the `mu_origin` baseline removal.
2. **Vol-aware conditional block sampling** — after each block, re-match analogs against the path's running simulated state, so blocks 2..6 sample from a distribution conditioned on what the path looks like at that point.

A third feature (tail inflator for the high-vol-regime PIT) is *not* in v2 scope — its trigger (`u_shaped_high_vol_pit`) did not fire on the v1 baseline.

Like `IMPLEMENTATION_PLAN.md`, this is a specification, not exploratory notes. Every decision below was made for a stated reason; do not silently change architectural decisions during implementation.

---

## Trigger evidence (from v1 canonical baseline)

Canonical run: `runs/analog_mc/20260516T180000Z/` (`default.yaml`, 76 folds, 273,120 origin × step pairs, mean test CRPS **0.05246**, 3h 47m wall time). The fast proxy (`nasdaq100_fast.yaml`) agrees on aggregate CRPS to ~1% but disagrees on two decision-rule verdicts — only the canonical numbers below should be used to gate v2 scope.

| Decision rule | Fired? | Metric (canonical) | Threshold | v2 action |
|---|---|---|---|---|
| `sloped_global_pit` | **YES** | +0.158 | ±0.1 | Trailing-momentum drift (v2.1) |
| `acf_seam_degradation` | **YES** | −1.071 | −0.30 | Conditional block sampling (v2.2) |
| `u_shaped_high_vol_pit` | no | +2.190 | 2.5 | (deferred — fast proxy fired at +5.08 but canonical de-fired; tail inflator stays out of v2) |
| `fixed_weight_close_to_tuned` | no | +0.178 | 0.01 | (tuned beats ⅓-⅓-⅓ baseline by 17.8%; per-fold search earns its keep) |
| `clip_hit_excessive` | no | +0.100 | 0.15 | (vol clip bounds are fine) |

Per-vol-regime CRPS is the headline calibration concern: low-vol 0.0277 vs high-vol 0.0911 (~3.3× harder). v2 should narrow this gap; if it doesn't, the deferred tail inflator becomes the next candidate.

---

## Versioning of v2

To keep the diagnostic-attribution chain intact, **ship the two features sequentially**, not bundled:

- **v2.1**: trailing-momentum drift only. Re-run baseline, re-evaluate decision rules. **Acceptance (all three required):** (a) `sloped_global_pit` no longer fires on the fast preset, (b) mean aggregate CRPS within +5% of the v1 fast proxy (i.e., ≤ 0.0547 against the v1 fast baseline of 0.0521), (c) per-fold weight trajectory does not collapse to a single corner — at least 3 distinct `(w0, w1, w2)` triples appear across the 21 folds (sanity check that drift didn't make the matcher irrelevant).
- **v2.2**: conditional block sampling added on top of v2.1. Acceptance: `acf_seam_degradation` no longer fires, mean CRPS not worse than v2.1, ACF curve at seam lags within 30% of realized.

If v2.1's acceptance fails, do NOT proceed to v2.2 — debug v2.1 first. Conditional sampling is the bigger compute and design risk; gating it on v2.1 success keeps the failure surface manageable.

---

## Open questions to resolve before any v2 coding

1. **Momentum window** — `momentum_lookback` default is 20 days in the existing `Config`. Is 20 the right default for NASDAQ100, or should we use the longest z-score horizon (200) to stay consistent with `mu_origin`? **Recommendation**: keep 20 for momentum because shorter windows track regime shifts; 200 would just re-state `mu_origin` (which we already subtract).
2. **Momentum shrinkage** — defaults to 0.5. Is half-Kelly a sensible default? **Recommendation**: yes for v1 of momentum. Can be tuned later (per-fold? globally?).
3. **Re-match cadence for conditional sampling** — every block (6 re-matches per path) vs every step (60). **Recommendation**: every block, per the plan's terminology ("conditional *block* sampling"). Every step is a larger design change.
4. **Distance metric for partial-path states** — same composite-Euclidean weighted z-score distance, or something else? **Recommendation**: same. The features at the path's effective new origin are causally computed from (real returns through `origin_idx`) + (simulated returns so far), so distance has the same meaning.
5. **Candidate set re-restriction** — at re-match for block `k+1`, the "effective forecast origin" advances by `block_length`. Should candidates be re-restricted to `d + block_length < origin_idx + k*block_length`, or kept as the original eligible set? **Recommendation**: re-restrict, to keep the C5 forward-only invariant intact for each effective sub-origin.
6. **Performance budget** — naive per-path per-block re-derivation of features and `brentq` solves is **~30× slower** per forecast (see "Cost model" below). What's acceptable? **Recommendation**: vectorized z-score updates across paths + batched root-finder for `n_eff → τ`. Budget: no more than 4× slower than v1 per forecast.
7. **Search vs evaluation split** — should v2.2's conditional sampling apply during hyperparameter search, or only during test evaluation with weights locked under v1 search? **Recommendation**: ablation. Run both and see whether locked-weights agree (this is a cheap diagnostic that tells us whether v2 changes the optimal `(weights, n_eff)`).

Surface any disagreement on these BEFORE coding starts.

---

## New critical correctness constraints (C7+)

C1–C6 from `IMPLEMENTATION_PLAN.md` remain in force unchanged.

### C7. Drift injection happens AFTER the σ-ratio multiplier

```python
scaled = (raw_block - mu_origin) * ratio + drift_target
#                                          ^^^^^^^^^^^^
#                                          C7: drift NOT scaled by ratio
```

Reasoning: `drift_target` represents the modeler's belief about the *next-period* expected log return given current regime — it is independent of the analog's historical vol. Multiplying it by `ratio` would re-introduce the exact pathology that motivated `mu_origin` in v1.1 (drift scaled by vol). The v1 `scale_block` already implements this; v2 must not move `drift_target` inside the ratio multiplier when implementing trailing-momentum.

### C8. Conditional re-matching is per-path AND per-block (not per-step)

When re-matching for block `k+1`:
- The effective sub-origin is `t_eff = origin_idx + k * block_length`.
- Features at `t_eff` are computed from `concat(real_returns[:origin_idx+1], simulated_path[:k*block_length])` — the per-path simulated tail is appended to the real history.
- Distances are computed against the candidate set re-restricted to `d + block_length < t_eff` (see open question 5).
- Probabilities are re-derived via `distances_to_probs(distances, n_eff)` with the SAME `n_eff` as for block 1.

Per-block (not per-step) keeps the design recognizable as an extension of v1's block-sampling architecture. Per-step would be a different algorithm entirely.

### C9. Causality of conditional features

For block `k+1`'s features computed at `t_eff`, the rolling windows must use only:
- Real returns at indices `≤ origin_idx` (known at forecast time)
- Simulated returns from this path at simulated-indices `1..k*block_length` (= future of origin, BUT future of *this* path, so causal *with respect to the simulation*)

The unit test: re-running `forecast(..., conditional_sampling=True)` with the same seed must produce bit-identical paths when called twice. Any non-determinism reveals a leakage path.

### C10. Trailing-momentum is computed at the ORIGIN, not at each block boundary

For v2.1 the trailing-momentum drift is computed once at `origin_idx` and applied unchanged to every block (just like `mu_origin`). For v2.2 with conditional sampling, the drift can either stay constant per forecast OR be re-evaluated at each sub-origin `t_eff` — **the former is the v2.2 default** because (a) it matches `mu_origin`'s semantics, and (b) re-evaluating drift from a partially-simulated path would compound regime-estimation error.

Reserve a `drift_per_block: bool` config knob for v2.3+ if diagnostics later argue for it.

---

## Module-level changes

### v2.1 changes (trailing-momentum drift)

| File | Change |
|---|---|
| `src/analog_mc/config.py` | `drift_mode` already accepts `"trailing_momentum"`; no schema change. |
| `src/analog_mc/features.py` | When `drift_mode != "zero"`, `compute_features` adds a SECOND trailing-mean column at the `momentum_lookback` horizon (separate from the `mu_origin` column at `max(zscore_horizons)`). Column name: `trailing_mean_<momentum_lookback>`. The two columns coexist; never reuse `mu_origin`'s long-horizon mean as the drift source — they serve different roles (regime baseline vs. recent-momentum estimate). |
| `src/analog_mc/simulate.py` | `forecast()` becomes the single point of drift policy. The existing `drift_target` kwarg changes from `float = 0.0` to `float \| None = None`. When `None` (the default), `forecast()` reads `config.drift_mode`: `"zero"` → `0.0`; `"trailing_momentum"` → `config.momentum_shrinkage * trailing_mean_<momentum_lookback>[origin_idx]`. An explicit float still works as a manual override (used by some tests). Search/walk_forward call sites unchanged. |
| `src/analog_mc/sampling.py` | No structural change — `generate_paths` already accepts `drift_target`. |
| `tests/analog_mc/test_simulate.py` | Add test: with `drift_mode="trailing_momentum"` on a synthetic upward-trending series, the forecast path's median end-cumulative-return is materially positive (vs ≈0 in `drift_mode="zero"`). |
| `tests/analog_mc/test_features.py` | Add test: when `drift_mode != "zero"` the bundle contains `trailing_mean_<momentum_lookback>` AND `trailing_mean_<max(zscore_horizons)>` as separate columns, and both are causal. |
| `configs/analog_mc/nasdaq100_v21.yaml` | New fast-preset config with `drift_mode: "trailing_momentum"` and otherwise identical knobs to `nasdaq100_fast.yaml`, for the acceptance gate. |
| `src/analog_mc/diagnostics.py` | `decision_rules` already expects v2 drift_mode in its recommendation text; no change. |

### v2.2 changes (conditional block sampling)

| File | Change |
|---|---|
| `src/analog_mc/config.py` | New field `conditional_block_sampling: bool = False`. Default off so v2.1 + v2.2 can be A/B-compared via config flip. |
| `src/analog_mc/sampling.py` | `generate_paths` gets a fast vectorized path for `conditional_block_sampling=True`: per-block re-match, per-path z-score buffer, batched `distances_to_probs`. |
| `src/analog_mc/distances.py` | New `distances_to_probs_batched(distances, n_eff)` that takes `(n_paths, n_candidates)` distances and returns `(n_paths, n_candidates)` probs. Internally vectorizes the τ root-find across paths. |
| `src/analog_mc/simulate.py` | `forecast` plumbs through `conditional_block_sampling` config flag. |
| `tests/analog_mc/test_sampling.py` | New tests: (a) determinism across re-runs; (b) when `conditional_block_sampling=True` and the candidate pool is degenerate (one candidate, p=1), result equals the v1 path; (c) per-block effective origin advances by `block_length`. |
| `tests/analog_mc/test_distances.py` | Tests for the batched `distances_to_probs_batched`: must agree with the scalar version row-wise. |

---

## Cost model and performance plan

### v2.1 cost

Negligible. Trailing-momentum is one rolling-mean lookup at the origin and one scalar add inside the inner loop. Expect ≤5% wall-time increase vs v1.

### v2.2 cost (the real concern)

Naive per-path per-block:
- 1000 paths × 6 blocks = 6000 extra `distances_to_probs` calls per forecast.
- Each `brentq` call ≈ 50–100 μs.
- Per forecast: ~300–600 ms extra.
- Per fold's search (≈20,000 forecasts in default config): ~6000–12000 s = 1.5–3 hours.
- × 76 folds = **5–10 days**. Intractable.

Vectorization path:
- Maintain `(n_paths, max_zscore_horizon)` rolling buffer of simulated returns per path. Update incrementally each block (O(block_length) per path per block, vectorized → O(block_length × n_paths)).
- Composite distance: `(K_candidates, 3) @ (3, n_paths) → (K, n_paths)` — single matmul.
- Batched τ solve: implement a vectorized Brent or Newton across paths in NumPy. Target ≤50 μs per path per block.

Vectorized target: ~10–20 ms extra per forecast (~3–5× v1, not 30×). Per-fold search: ~5–10 min in addition to v1's ~3 min. Total walk-forward: ~10 hours on default config. Tolerable.

If vectorization underperforms target, fall back to:
- Conditional sampling **only at test-time** with weights locked under v1 search. Search remains v1 (cheap); evaluation gets the v2.2 benefit. Reasonable approximation if v1 and v2 share the same optimal weights region — to be confirmed by the open-question-7 ablation.

---

## Implementation stages

### Stage A — v2.1: trailing-momentum drift

1. Wire `drift_mode` into `simulate.forecast`. Compute `drift_target` once at the origin from the trailing mean.
2. Add the v2.1 test (median forecast cum-return tracks the trailing momentum sign).
3. Add a small CLI integration test that runs a 2-fold walk-forward with `drift_mode="trailing_momentum"` and verifies the persisted forecasts have non-zero median.
4. Run `nasdaq100_fast.yaml` walk-forward with `drift_mode="trailing_momentum"`. Compare aggregate CRPS + decision rules to the v1 baseline. **Acceptance**: `sloped_global_pit` no longer fires; mean CRPS not materially worse.
5. Update `IMPLEMENTATION_PLAN.md` revision history with v2.1 details.

### Stage B — v2.2: conditional block sampling

6. Implement `distances_to_probs_batched` with a vectorized Brent solver. Unit-test against the scalar version.
7. Implement the per-path rolling z-score buffer in `generate_paths`. Cost-test it (microbenchmark) before integration.
8. Add the `conditional_block_sampling` config flag and wire it through.
9. Add the v2.2 tests (determinism, degenerate-pool equivalence to v1, effective-origin advance).
10. Run `nasdaq100_fast.yaml` walk-forward with both v2.1 + v2.2. Compare aggregate CRPS, ACF curve, and decision rules. **Acceptance**: `acf_seam_degradation` no longer fires; mean CRPS not materially worse than v2.1.
11. If v2.2 search becomes intractable, fall back to the locked-weights mode (search under v1, eval under v2.2) and document the deviation here.

### Stage C — Documentation + baseline refresh

12. Re-run `default.yaml` baseline under v2.1 + v2.2; archive as the new canonical run.
13. Update `README.md` "Status" section to reflect v2 completion.
14. Update `IMPLEMENTATION_PLAN.md` revision history with v2.2 details.
15. Decide whether to keep `drift_mode="zero"` as the *default* in `configs/analog_mc/default.yaml` or flip to `"trailing_momentum"` based on the v2.1 acceptance results.

---

## New / changed diagnostics

No new plots; the existing Stage-9 suite is sufficient to validate v2:

- `global_pit_histogram` validates v2.1 (slope should disappear).
- `acf_comparison` validates v2.2 (sim ACF should track realized at seam lags).
- `aggregate_crps_per_vol_regime` is the headline calibration measure — high-vol CRPS should drop relative to v1.

Two decision rules in `diagnostics.decision_rules` should change their **interpretation** but not their code:

- `sloped_global_pit` firing post-v2.1 is now a v3 trigger (e.g., richer drift model, regime-switching).
- `acf_seam_degradation` firing post-v2.2 is also a v3 trigger (e.g., per-step conditional sampling, or copula-based block joining).

No code change needed; just document the reinterpretation in v3 notes if/when that becomes relevant.

---

## What not to do

- **Do not** bundle v2.1 and v2.2 in a single PR. The whole point of the diagnostic-driven design is being able to attribute CRPS / PIT improvements to specific features. Bundling destroys that.
- **Do not** move `drift_target` inside the `ratio` multiplier (C7). The v1.1 fix to C3 was precisely about not scaling drift with vol.
- **Do not** implement per-step (rather than per-block) conditional re-matching in v2.2. That's a larger algorithm change reserved for v3 and would need its own design conversation.
- **Do not** widen `vol_clip_upper` or expand `n_eff_values` in v2 unless the `u_shaped_high_vol_pit` rule fires post-v2.1+v2.2 — the v1 baseline did not trigger this rule.
- **Do not** swap the trailing mean for an EWMA momentum or a regression-based estimator in v2.1. Keep the simplest possible v2 drift estimator; if it doesn't fix the PIT slope, that's diagnostic information for v3.
- **Do not** rebuild `distances_to_probs_batched` if the v1 vectorization budget can be met without it. Implement only if/when the cost-model says we need it (i.e., when conditional sampling lands in v2.2).

---

## Reference: v1 baseline numbers

Canonical v1 baseline: `runs/analog_mc/20260516T180000Z/` (76 folds, 273,120 origin × step pairs, 3h 47m wall time). The `nasdaq100_fast.yaml` proxy run is kept here for context; numbers agree within ~1% on aggregate CRPS but the decision-rule verdicts shifted (see notes below the table).

| | v1 fast (proxy) | v1 default (canonical) | v2.1 target | v2.2 target |
|---|---|---|---|---|
| Mean aggregate CRPS | 0.0521 | **0.05251** | ≤ v1 default | ≤ v2.1 |
| Median aggregate CRPS | — | **0.03121** | ≤ v1 default | ≤ v2.1 |
| High-vol-regime mean CRPS | 0.0888 | **0.09108** | ≤ v1 default high-vol | ≤ v2.1 high-vol |
| Low-vol-regime mean CRPS | — | 0.02770 | — | — |
| Mid-vol-regime mean CRPS | — | 0.03917 | — | — |
| h=1 / h=15 / h=30 / h=60 per-step CRPS | — | 0.0088 / 0.0340 / 0.0525 / 0.0905 | — | — |
| Fixed-weight baseline (⅓,⅓,⅓, n_eff=30) mean CRPS | — | 0.06182 (tuned beats by +17.84%) | — | — |
| `sloped_global_pit` fired | yes (+0.147) | **yes (+0.158)** | **no** | n/a |
| `u_shaped_high_vol_pit` fired | yes (+5.08) on partial | **no (+2.190)** | n/a | n/a |
| `acf_seam_degradation` fired | yes (−1.053) | **yes (−1.071)** | yes (expected; v2.1 doesn't address it) | **no** |
| `fixed_weight_close_to_tuned` fired | — | no (+0.178) | — | — |
| `clip_hit_excessive` fired | — | no (+0.100) | — | — |

**What changed from the partial (21-fold) read:**

- `u_shaped_high_vol_pit` *de-fired* on the full set (metric dropped from +5.08 → +2.19). This confirms the deferred status of the tail inflator in this plan — do NOT promote it to v2 scope.
- `fixed_weight_close_to_tuned` flipped sign cleanly: tuning now wins the baseline by ~18% (it was losing on the partial). The grid search is doing real work.
- v2 scope is unchanged from the original plan: **v2.1 trailing-momentum drift + v2.2 conditional block sampling** — and only those two. Tail inflator stays deferred.
