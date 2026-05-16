"""Block sampling and per-analog volatility scaling (Stage 4).

Implements:
  * **C3 (v1.1)** — per-analog vol scaling: subtract shared ``mu_origin``
    baseline → clip σ ratio → rescale → add drift. The baseline is a single
    constant per forecast (the trailing causal mean of returns at the
    forecast origin, over the longest z-score horizon). It is NOT recomputed
    per-block or per-analog; this preserves each analog's mean *relative to
    current regime*, so the path distribution at horizon end is non-degenerate.
    See ``docs/analog_mc/IMPLEMENTATION_PLAN.md`` C3 for the full rationale.
  * **C4** — running EWMA σ across the simulated path so block 2..N use the
    updated vol estimate from prior simulated returns, not the σ at the
    forecast origin.
  * **C5** — strictly forward sampling: an analog at index d contributes the
    block returns[d + 1 : d + 1 + block_length].

For v1, drift_target is always 0.0 (the plan defers any drift to v2, gated on
diagnostic findings). The running EWMA σ uses an uncentered recursion
``var_t = (1 - α) var_{t-1} + α r_t^2`` with ``α = 1 - 2^{-1/halflife}``. This
is an approximation of pandas' bias-corrected EWM variance, but the difference
is negligible for daily log returns (mean ≈ 0) and the recursion runs entirely
on simulated returns, so the absolute scale doesn't propagate from the
historical estimate beyond the warm-start.
"""

from __future__ import annotations

import numpy as np

from analog_mc.config import Config


# ---------------------------------------------------------------------------
# Per-block scaling (C3)
# ---------------------------------------------------------------------------


def scale_block(
    raw_block: np.ndarray,
    sigma_current: float,
    sigma_historical: float,
    mu_origin: float,
    drift_target: float,
    vol_clip_lower: float,
    vol_clip_upper: float,
) -> np.ndarray:
    """C3 (v1.1): subtract shared baseline → clip σ ratio → rescale → add drift.

    Args:
        raw_block:        (block_length,) historical analog block returns.
        sigma_current:    trailing causal σ at the forecast origin (or running
                          σ for blocks 2+).
        sigma_historical: trailing causal σ at the analog's origin date.
        mu_origin:        SHARED baseline — the trailing causal mean at the
                          forecast origin over the longest z-score horizon.
                          Must be the same value across every call to
                          scale_block within one forecast.
        drift_target:     per-day drift to inject after scaling (0.0 in v1).
        vol_clip_lower:   lower bound on the σ ratio.
        vol_clip_upper:   upper bound on the σ ratio.

    The baseline ``mu_origin`` removes only the *constant* component of the
    analog's returns (the current regime's drift) before the σ multiplier is
    applied. The analog's deviation FROM this constant is what gets scaled.
    """
    if sigma_historical <= 0:
        raise ValueError(f"sigma_historical must be > 0; got {sigma_historical}")
    demeaned = raw_block - mu_origin
    ratio = sigma_current / sigma_historical
    ratio = float(np.clip(ratio, vol_clip_lower, vol_clip_upper))
    return demeaned * ratio + drift_target


# ---------------------------------------------------------------------------
# Block sampling (C5)
# ---------------------------------------------------------------------------


def sample_analog_blocks(
    probs: np.ndarray,
    candidate_indices: np.ndarray,
    returns: np.ndarray,
    block_length: int,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n_paths analog candidates and return their forward blocks.

    Args:
        probs:             shape (K,) — probability per candidate; sums to 1.
        candidate_indices: shape (K,) — positional indices into ``returns``.
        returns:           shape (N,) — full log-returns array.
        block_length:      length of each sampled block.
        n_paths:           number of paths (= number of independent samples).
        rng:               np.random.Generator for reproducibility.

    Returns:
        chosen_local: shape (n_paths,) int — index into ``candidate_indices``
                      (so callers can look up matching σ_historical).
        raw_blocks:   shape (n_paths, block_length) — forward block returns
                      for each chosen analog: returns[d+1 : d+1+block_length].
    """
    if probs.shape != candidate_indices.shape:
        raise ValueError(
            f"probs {probs.shape} and candidate_indices {candidate_indices.shape} must match"
        )
    if candidate_indices.size == 0:
        raise ValueError("candidate_indices is empty")
    max_idx = int(candidate_indices.max())
    if max_idx + block_length >= returns.size:
        raise ValueError(
            f"Candidate index {max_idx} has no full forward block of length "
            f"{block_length} within returns of size {returns.size}; restrict "
            f"candidates so max(d) + block_length < N."
        )

    chosen_local = rng.choice(candidate_indices.size, size=n_paths, replace=True, p=probs)
    d = candidate_indices[chosen_local]  # (n_paths,)
    starts = d + 1
    offsets = np.arange(block_length, dtype=np.int64)
    raw_blocks = returns[starts[:, None] + offsets[None, :]]  # (n_paths, block_length)
    return chosen_local, raw_blocks


# ---------------------------------------------------------------------------
# Full path generation (C4)
# ---------------------------------------------------------------------------


def _alpha_from_halflife(halflife: float) -> float:
    return 1.0 - 2.0 ** (-1.0 / halflife)


def generate_paths(
    probs: np.ndarray,
    candidate_indices: np.ndarray,
    returns: np.ndarray,
    sigma_at_candidates: np.ndarray,
    sigma_init: float,
    mu_origin: float,
    config: Config,
    rng: np.random.Generator,
    drift_target: float = 0.0,
    record_ratios: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate n_paths Monte Carlo paths of length forecast_horizon.

    For v1, ``probs`` is constant across all blocks (per the plan: no
    conditional block sampling — that is reserved for v2 and gated on the
    squared-return ACF diagnostic). ``mu_origin`` is also constant across
    all blocks and all paths within one forecast (per C3 v1.1).

    The running EWMA σ is maintained per-path: it starts at ``sigma_init``
    (the σ at the forecast origin from real data) and is updated after each
    scaled-block element using the uncentered recursion.

    Args:
        probs:               shape (K,) candidate probabilities.
        candidate_indices:   shape (K,) positions into ``returns``.
        returns:             shape (N,) full log-returns array.
        sigma_at_candidates: shape (K,) σ at each candidate's origin date.
        sigma_init:          σ at the forecast origin (real data).
        mu_origin:           trailing causal mean at forecast origin over the
                             longest z-score horizon. The C3 shared baseline.
        config:              pipeline config.
        rng:                 np.random.Generator.
        drift_target:        per-day expected log return injected post-scaling.
                             v1 default 0.0; v2 may set this to a shrunk
                             trailing-momentum estimate.

    Returns:
        shape (n_paths, forecast_horizon) array of simulated log-return paths.
    """
    if sigma_init <= 0:
        raise ValueError(f"sigma_init must be > 0; got {sigma_init}")

    n_paths = config.n_paths
    horizon = config.forecast_horizon
    block_length = config.block_length
    n_blocks = config.n_blocks
    alpha = _alpha_from_halflife(config.ewma_halflife)
    one_minus_alpha = 1.0 - alpha

    paths = np.empty((n_paths, horizon), dtype=np.float64)
    running_var = np.full(n_paths, sigma_init * sigma_init, dtype=np.float64)
    ratios_out = np.empty((n_paths, n_blocks), dtype=np.float64) if record_ratios else None

    for b in range(n_blocks):
        chosen_local, raw_blocks = sample_analog_blocks(
            probs=probs,
            candidate_indices=candidate_indices,
            returns=returns,
            block_length=block_length,
            n_paths=n_paths,
            rng=rng,
        )
        sigma_current = np.sqrt(running_var)
        sigma_historical = sigma_at_candidates[chosen_local]
        # Vectorized C3 (v1.1): subtract SHARED mu_origin, not per-block mean.
        demeaned = raw_blocks - mu_origin
        raw_ratio = sigma_current / sigma_historical
        ratio = np.clip(raw_ratio, config.vol_clip_lower, config.vol_clip_upper)
        if ratios_out is not None:
            ratios_out[:, b] = raw_ratio  # record PRE-clip ratio for clip-hit diagnostic
        scaled = demeaned * ratio[:, None] + drift_target
        paths[:, b * block_length : (b + 1) * block_length] = scaled

        # Update running EWMA σ per path using the scaled returns from this block.
        # Uncentered recursion: var_t = (1-α) var_{t-1} + α r_t^2.
        for r_idx in range(block_length):
            r = scaled[:, r_idx]
            running_var = one_minus_alpha * running_var + alpha * r * r

    if ratios_out is not None:
        return paths, ratios_out
    return paths
