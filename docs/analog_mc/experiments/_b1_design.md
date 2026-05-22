# B1 (Platzer local-linear correction) — design scoping

Companion to [`V4_EXPERIMENTS_PLAN.md` §B1](../V4_EXPERIMENTS_PLAN.md#b1-platzer-local-linear-correction--p0-single-highest-priority-experiment-after-v35) and [`V3_5_RESULTS.md`](../V3_5_RESULTS.md). This doc nails down the implementation decisions before code lands. Each decision is made for a stated reason — surface deviations rather than silently change.

## What B1 must do

Apply the matcher's expected forecast a **conditional-mean bias correction** estimated by a locally-weighted linear regression on the analog candidates, addressing the V3.5.4 finding that the matcher's analog clusters systematically underestimate the realized 60-day magnitude.

Platzer–Yiou (`arXiv 2007.14216`) frames the correction as a Jacobian estimate: at high-Lyapunov regimes, the matcher's k-NN average is a biased estimator of the conditional mean; a locally-weighted linear regression over the same neighbours estimates the Jacobian and removes the leading bias term.

## Locked decisions

### D1. Regression target — scalar terminal cumulative log-return

For B1 v1, regress on **the analog's 60-day forward cumulative log-return** (a scalar):

```
y_i  =  sum(returns[d_i + 1 : d_i + 1 + H])      for analog i with origin index d_i
```

**Why scalar, not full path.** V4 plan §B1 says: *"Default to terminal-cumulative-return correction; add a B1-variant for full-path if needed."* A scalar target gives us a single, interpretable correction. The full-path variant — regressing each horizon step independently — multiplies implementation cost ~60× and risks over-fitting at near horizons where the matcher is already well-calibrated (V3.5.4 showed misses concentrated in the late horizon for V-recovery anchors). Variant deferred to B1-fullpath if v1 leaves residual coverage gap at far horizons.

### D2. Regression predictors — analog z-scores at analog origin

Predictors `X_i = [1, z_20(d_i), z_50(d_i), z_200(d_i)]` for analog i. Prediction point is `x_* = [1, z_20(origin), z_50(origin), z_200(origin)]`.

**Why z-scores, not raw features.** The matcher already uses these as the similarity dimensions; the regression should live in the same feature space so the bias correction is interpretable as "how does forward return co-vary with similarity-relevant state?". σ and the trailing means are *scale* variables that the C3 vol-scaling already handles; including them here would double-correct.

### D3. Regression weights — matcher probabilities (`probs`)

Weighted least squares with weights `w_i = probs_i` (the n_eff-determined softmax distances-to-probs that the matcher already computes). This makes the correction **locally-weighted** around the target's state — far analogs contribute proportionally less.

**Why not uniform.** Uniform weighting over the eligible pool would ignore the matcher's selection entirely and turn B1 into a global linear regression, which we know is biased for non-Markovian dynamics (V2_PLAN §Burov motivation for B2). Matching the matcher's weighting preserves the locality.

### D4. Correction application — uniform per-day drift adjustment

Compute the scalar correction:
```
correction  =  y_pred  -  E_probs[y_i]
           =  x_* · β   -  Σ probs_i · y_i
```
where `β` is the WLS coefficient vector. Then apply uniformly across the 60-step horizon:
```
paths[:, t]  +=  correction / H        for every path, every t
```

**Why uniform per-day.** The scalar correction must be allocated across H=60 days. Uniform allocation:
- Matches the existing `drift_target` semantics (a per-day log-return drift) — same hook, same units.
- Is unbiased about *when* in the horizon the correction should appear; the scalar target carries no time information.
- Preserves the analog blocks' intra-horizon shape — only the mean shifts.

**Alternatives rejected for v1.** Terminal-only spike (distorts path topology), block-end allocation (no theoretical preference over uniform, and requires more code).

### D5. WLS solver — closed-form normal equations with Tikhonov regularization

```
β = (Xᵀ W X + λ I)⁻¹ Xᵀ W y
```
with `λ = 1e-6 * trace(XᵀWX) / d` (scale-aware Tikhonov). Closed-form because:
- Problem is tiny: K candidates × 4 predictors. Even K=10,000 × 4 is a 40,000-flop matmul.
- No iterative convergence concerns.
- Tikhonov guards against near-singular `XᵀWX` (rare but possible when n_eff is tiny and the top-K all sit at the same z-score).

**Singular-case fallback.** If after regularization the predicted `y_pred` is more extreme than `[min(y_i), max(y_i)] × 1.5`, fall back to `correction = 0` (i.e., disable B1 for that origin) and log the clamp. Prevents adversarial extrapolation from corrupting paths at the rare regime-transition origins where the analog cluster is collinear.

### D6. Correction frozen at block 0

Compute `correction` once per forecast — at block 0, using the v1 (real-origin) probabilities. Apply uniformly across all 60 days regardless of conditional-sampling state.

**Why not re-estimate per block under conditional sampling.** Block-conditional re-estimation would require re-fitting WLS per block per path (n_paths × n_blocks = 6000 fits per forecast). The V4 plan §B1 lists this as a B1-variant ("full-path"); we defer it. For v1 the question is "does the bias correction help at all?" — a single global estimate answers it.

### D7. Config knob — `local_linear_correction: bool`

Add to `Config`:
```python
local_linear_correction: bool = False
```

Default off; the new experiment config `configs/analog_mc/ablation_B1_localreg.yaml` flips it on. Acceptance criterion that B1 is invisible when off: a unit test verifies path output is bit-identical to v2.4 when the knob is `False`.

### D8. Missing forward returns at analog candidates

Candidates near the end of the training pool may lack a complete 60d forward window (`d_i + H ≥ len(returns)`). Drop them from the regression set and renormalize `probs` over the survivors. Affects only fold 0 and the most-recent few candidates; quantify in the report.

### D9. C-constraint compatibility audit

| Constraint | Compatible? | Reasoning |
|---|---|---|
| **C1** causal features | Yes | Regression only uses features at d_i (causal at d_i) and at origin. No look-ahead. |
| **C2** n_eff parameterization | Yes | Uses the same `probs` the matcher produces — n_eff is baked in. |
| **C3** per-analog vol scaling | Yes | Correction adds a constant drift; vol-scaling is pre-correction. Order: scale → drift → +B1-correction-per-day. |
| **C4** running EWMA σ | Yes | Correction enters via `drift_target` (the natural hook), so it propagates into the EWMA-σ recursion via the `scaled = demeaned·ratio + drift_target` path. This matches v2.4's trailing-momentum drift exactly — that drift also enters σ recursion through `drift_target`. **Consequence**: block-1+ diffs between B1-on vs B1-off vary by ~1.5% around the per-day mean (σ recursion compounds). The per-day shift is *approximately* uniform; exact uniformity holds only at block 0. Acceptable: the V2.4 drift behaves the same way and is well-tested. Keeping correction *outside* σ recursion would require generating paths first then post-adjusting — more invasive code change for a numerical purity benefit that doesn't change forecast distributions materially. |
| **C5** strictly forward sampling | Yes | Regression targets `y_i` are strictly-forward windows from d_i. |
| **C6** walk-forward boundary discipline | Yes | All training pairs (X_i, y_i) come from `candidate_idx` ⊆ `train_idx`. No val/test leakage. |

### D10. Search-time vs test-time

**Decision: B1 active at both search and test time.** The matcher's weight grid was tuned against the baseline (no B1) loss. If we run B1 only at test, the search picks weights that are biased for the wrong objective. Run B1 in both phases so the optimal weights are re-discovered under the corrected sampler.

**Cost implication.** Search does ~150 forecasts per origin × 76 origins ≈ 11k weighted regressions. Each WLS fit is microseconds (K×4 problem); total search overhead ≪ 1% over baseline. No caching needed.

## Implementation surface

### New module: `src/analog_mc/local_linear.py`

```python
def fit_local_linear_correction(
    z_target: np.ndarray,            # (3,)
    z_candidates: np.ndarray,        # (K, 3)
    probs: np.ndarray,               # (K,) sums to 1
    forward_returns: np.ndarray,     # (K,) — y_i = forward 60d log-return sum
    tikhonov_rel: float = 1e-6,
    extrapolation_clamp: float = 1.5,
) -> tuple[float, dict]:
    """Returns (correction, diagnostics). correction is the scalar log-return
    shift to add to E_probs[y]; diagnostics has clamp_hit, beta, etc."""
```

### Modified: `src/analog_mc/simulate.py:forecast`

After computing `probs` (line ~256) and before `generate_paths*`:

```python
if config.local_linear_correction:
    forward_returns = _forward_logret_sums(returns, eligible, config.forecast_horizon)
    correction, b1_diag = fit_local_linear_correction(
        z_target, z_candidates, probs, forward_returns
    )
    drift_target = drift_target + correction / config.forecast_horizon
```

A single drift adjustment. Conditional sampling, GARCH, EWMA branches all use `drift_target` unchanged downstream.

### Modified: `src/analog_mc/data.py` or new helper

Precompute `forward_logret_sums(returns, horizon)` once per fold — O(N) cumsum — and pass to forecast. Avoids recomputing inside the search loop.

### New config: `configs/analog_mc/ablation_B1_localreg.yaml`

Inherits from `default.yaml` (v2.4 canonical) and flips `local_linear_correction: true`.

### Tests

`tests/analog_mc/test_local_linear.py`:
1. **Bit-identical when off**: same seed, knob off → paths identical to v2.4.
2. **Correction direction sanity**: synthetic toy where forward_returns linearly depends on z_50; verify correction has the right sign and magnitude.
3. **Tikhonov works**: degenerate input (all z_candidates equal) → β finite, correction = 0.
4. **Extrapolation clamp**: target z outside [min, max] of candidates by 5× → clamp fires, correction = 0.
5. **C1 causality smoke test**: assert `forward_returns` does not reference any returns at or after origin_idx.

### Report: `docs/analog_mc/experiments/_b1_local_linear.md`

Standard structure (status, setup, headline numbers, mechanistic reading, decision-rule verdict, implication). Includes the **mandatory fat-tail panel** rendered to `docs/analog_mc/experiments/figs/b1_local_linear_fat_tail/`.

## Out of scope for B1 v1

- Full-path regression (per-step y_i) — B1-fullpath variant if v1 has residual gap.
- Block-conditional re-estimation — B1-conditional variant if v1 over-shoots on conditional folds.
- Adding σ or other non-z predictors to X — risks double-correction with C3.
- Regularization tuning (Tikhonov λ) — fixed at scale-aware default; revisit only if diagnostics show numerical instability.
- Adapting B1 to GARCH (vol_model="garch") path — orthogonal; B1 lives at the drift layer, GARCH at the σ layer. Test that combination works but don't optimise it.

## Decision rules (recap from V4 plan §B1)

| Outcome | Action |
|---|---|
| `acf_seam_degradation` rule passes (≥ −0.30) | Structural win; close v3-residual failure inside analog primitive. |
| Rule moves toward zero but stays below threshold | Partial mechanism; quantify how much bias correction explains. Likely still CRPS win. |
| Rule unchanged | ACF gap is intra-block, not across-block mean bias. Refocus v5 on within-block primitive change. |
| Fat-tail panel: ≥3 of 5 V3.5 failure anchors recover 90%-band ≥45/60 | **Headline win** — promote to v2.5. |
| Fat-tail aggregate CRPS regresses on >2 of 15 anchors | Not promotable without explicit justification. |

Expected primary effect (independent of ACF): ≥1-2% mean CRPS gain in high-vol regime via the Jacobian correction; ≥2 of 5 V3.5 failure anchors materially improved.

## Build order

1. Scaffold `src/analog_mc/local_linear.py` with `fit_local_linear_correction` + helpers + docstring per design above.
2. Add `local_linear_correction: bool = False` to Config; validate; round-trip test.
3. Add `forward_logret_sums` helper (probably in `simulate.py` as a module-level cached fn keyed by `(id(returns), horizon)`).
4. Wire into `forecast()` gated on the knob.
5. Write `configs/analog_mc/ablation_B1_localreg.yaml`.
6. Tests (5 above).
7. Fast-preset run; sanity check the diff against v2.4.
8. Full canonical run.
9. Fat-tail panel + diff vs `fat_tail_baseline_v24.json`.
10. `_b1_local_linear.md` report with mechanistic reading and decision-rule verdict.

Steps 1–6 land in one PR-equivalent commit on `v4-b1-platzer`; runs land in subsequent commits with their artefacts.
