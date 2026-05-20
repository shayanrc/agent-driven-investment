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
from analog_mc.distances import (
    composite_distance,
    composite_distance_batched,
    distances_to_probs,
    distances_to_probs_batched,
)


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
    sigma_path: np.ndarray | None = None,
    sigma_at_returns: np.ndarray | None = None,
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

    # v3b (E9): per-step σ rescaling when a precomputed sigma_path is provided.
    # In that mode, σ_current[t] comes from the GARCH-simulated path (sigma_path
    # at step t) and σ_historical[t] comes from the analog's σ at the SAME
    # forward step (sigma_at_returns at index analog_origin + 1 + t). Block-
    # constant ratio (the EWMA branch below) is bypassed.
    per_step_sigma = sigma_path is not None
    if per_step_sigma:
        if sigma_at_returns is None:
            raise ValueError("sigma_path requires sigma_at_returns (σ per causal index)")
        if sigma_path.shape != (n_paths, horizon):
            raise ValueError(
                f"sigma_path shape {sigma_path.shape} must be ({n_paths}, {horizon})"
            )

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
        demeaned = raw_blocks - mu_origin

        if per_step_sigma:
            # σ_current per step from GARCH-simulated path.
            sigma_path_block = sigma_path[:, b * block_length : (b + 1) * block_length]
            # σ_historical per step: σ at analog_origin + 1 + r for each row.
            analog_origins = candidate_indices[chosen_local]  # (n_paths,)
            step_offsets = np.arange(1, block_length + 1)  # forward offsets
            hist_positions = analog_origins[:, None] + step_offsets[None, :]  # (n_paths, block_length)
            sigma_hist_block = sigma_at_returns[hist_positions]  # (n_paths, block_length)
            raw_ratio = sigma_path_block / sigma_hist_block
            ratio = np.clip(raw_ratio, config.vol_clip_lower, config.vol_clip_upper)
            scaled = demeaned * ratio + drift_target
            if ratios_out is not None:
                # Block-level summary: mean PRE-clip ratio across the block, per path.
                ratios_out[:, b] = raw_ratio.mean(axis=1)
        else:
            sigma_current = np.sqrt(running_var)
            sigma_historical = sigma_at_candidates[chosen_local]
            # Vectorized C3 (v1.1): subtract SHARED mu_origin, not per-block mean.
            raw_ratio = sigma_current / sigma_historical
            ratio = np.clip(raw_ratio, config.vol_clip_lower, config.vol_clip_upper)
            if ratios_out is not None:
                ratios_out[:, b] = raw_ratio  # record PRE-clip ratio for clip-hit diagnostic
            scaled = demeaned * ratio[:, None] + drift_target

        paths[:, b * block_length : (b + 1) * block_length] = scaled

        if not per_step_sigma:
            # Update running EWMA σ per path using the scaled returns from this block.
            # Skipped in v3b mode — GARCH σ_path is independent of simulated returns.
            for r_idx in range(block_length):
                r = scaled[:, r_idx]
                running_var = one_minus_alpha * running_var + alpha * r * r

    if ratios_out is not None:
        return paths, ratios_out
    return paths


# ---------------------------------------------------------------------------
# v2.2: Conditional block sampling
# ---------------------------------------------------------------------------


def _zscore_from_window(window: np.ndarray) -> np.ndarray:
    """Per-row z-score = mean / std(ddof=1) over the last axis.

    Mirrors features.causal_zscore semantics: same window convention, same
    NaN-on-constant-window behaviour. Returns NaN where std is zero so
    downstream comparisons fail loudly rather than silently producing 0.
    """
    mean = window.mean(axis=-1)
    std = window.std(axis=-1, ddof=1)
    z = np.where(std > 0, mean / np.where(std > 0, std, 1.0), np.nan)
    return z


def _sample_indices_from_probs_batched(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized per-row categorical sample.

    Args:
        probs: shape (n_paths, K), each row sums to 1.
        rng:   np.random.Generator.

    Returns:
        shape (n_paths,) int array — index into [0, K) for each row.
    """
    cum = probs.cumsum(axis=1)
    # Clamp small floating drift so the last column is exactly >= u.
    cum[:, -1] = 1.0
    u = rng.random(probs.shape[0])
    return (cum < u[:, None]).sum(axis=1)


def generate_paths_conditional(
    z_at_origin: np.ndarray,
    z_at_candidates: np.ndarray,
    candidate_indices: np.ndarray,
    returns: np.ndarray,
    sigma_at_candidates: np.ndarray,
    sigma_init: float,
    mu_origin: float,
    weights: np.ndarray,
    n_eff: float,
    origin_idx: int,
    config: Config,
    rng: np.random.Generator,
    drift_target: float = 0.0,
    record_ratios: bool = False,
    sigma_path: np.ndarray | None = None,
    sigma_at_returns: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """v2.2: Conditional block sampling — per-block re-match against per-path state.

    For block 0 the behaviour is identical to ``generate_paths`` (probs derived
    from the real-data origin's z-scores). For blocks 1..n_blocks-1 each path
    maintains its own effective sub-origin at ``t_eff = origin_idx + k * block_length``;
    z-scores are recomputed from a per-path return tail (real returns at
    indices ≤ origin_idx followed by the path's own simulated returns), distances
    are re-derived against the SAME v1 candidate set, and probabilities are
    re-solved via the batched n_eff parameterization.

    Architectural invariants preserved:

      * **C3** — ``mu_origin`` is per forecast, never per-block. The same constant
        is subtracted in every block (matching v1.1's semantics).
      * **C4** — running EWMA σ continues to use the path's scaled-block recursion.
      * **C5** — analog at index d still contributes block returns[d+1:d+1+block_length].
      * **C6** — the candidate set is the v1 eligibility set (``d + block_length <
        origin_idx``), UNCHANGED across blocks. The V2_PLAN.md open question 5
        recommendation to "re-restrict to d + block_length < t_eff" would expand
        the set and admit candidates whose forward block overlaps real post-origin
        returns. That's a walk-forward leak; the conservative resolution is to
        keep the v1 set.
      * **C7** — drift_target is added AFTER the σ ratio multiplier in every block.
      * **C10** — drift_target is constant per forecast (held at the origin value
        passed in by ``forecast``).
    """
    if sigma_init <= 0:
        raise ValueError(f"sigma_init must be > 0; got {sigma_init}")

    n_paths = config.n_paths
    horizon = config.forecast_horizon
    block_length = config.block_length
    n_blocks = config.n_blocks
    horizons = tuple(int(h) for h in config.zscore_horizons)
    max_h = max(horizons)
    alpha = _alpha_from_halflife(config.ewma_halflife)
    one_minus_alpha = 1.0 - alpha

    if origin_idx + 1 < max_h:
        raise ValueError(
            f"origin_idx={origin_idx} has insufficient prior history for "
            f"max(zscore_horizons)={max_h}."
        )

    # v3b (E9): per-step σ rescaling. See generate_paths for the same pattern.
    per_step_sigma = sigma_path is not None
    if per_step_sigma:
        if sigma_at_returns is None:
            raise ValueError("sigma_path requires sigma_at_returns (σ per causal index)")
        if sigma_path.shape != (n_paths, horizon):
            raise ValueError(
                f"sigma_path shape {sigma_path.shape} must be ({n_paths}, {horizon})"
            )

    paths = np.empty((n_paths, horizon), dtype=np.float64)
    running_var = np.full(n_paths, sigma_init * sigma_init, dtype=np.float64)
    ratios_out = np.empty((n_paths, n_blocks), dtype=np.float64) if record_ratios else None
    step_offsets = np.arange(1, block_length + 1)

    def _scale(raw_blocks: np.ndarray, chosen_local: np.ndarray, b: int) -> np.ndarray:
        """Local helper — block-constant or per-step σ scaling."""
        demeaned = raw_blocks - mu_origin
        if per_step_sigma:
            sigma_path_block = sigma_path[:, b * block_length : (b + 1) * block_length]
            analog_origins = candidate_indices[chosen_local]
            hist_positions = analog_origins[:, None] + step_offsets[None, :]
            sigma_hist_block = sigma_at_returns[hist_positions]
            raw_ratio = sigma_path_block / sigma_hist_block
            ratio = np.clip(raw_ratio, config.vol_clip_lower, config.vol_clip_upper)
            if ratios_out is not None:
                ratios_out[:, b] = raw_ratio.mean(axis=1)
            return demeaned * ratio + drift_target
        sigma_current = np.sqrt(running_var)
        sigma_historical = sigma_at_candidates[chosen_local]
        raw_ratio = sigma_current / sigma_historical
        ratio = np.clip(raw_ratio, config.vol_clip_lower, config.vol_clip_upper)
        if ratios_out is not None:
            ratios_out[:, b] = raw_ratio
        return demeaned * ratio[:, None] + drift_target

    # ---- Block 0 — same as v1: distances from the real-data origin ----
    distances0 = composite_distance(z_at_origin, z_at_candidates, weights)
    probs0 = distances_to_probs(distances0, target_n_eff=n_eff)
    chosen_local, raw_blocks = sample_analog_blocks(
        probs=probs0,
        candidate_indices=candidate_indices,
        returns=returns,
        block_length=block_length,
        n_paths=n_paths,
        rng=rng,
    )
    scaled = _scale(raw_blocks, chosen_local, 0)
    paths[:, :block_length] = scaled
    if not per_step_sigma:
        for r_idx in range(block_length):
            r = scaled[:, r_idx]
            running_var = one_minus_alpha * running_var + alpha * r * r

    # ---- Per-path tail buffer for blocks 1+ ----
    # Warm-start with the last (max_h - block_length) REAL returns ending at origin_idx,
    # then append the just-simulated block 0 returns. Length stays at max_h.
    real_tail = returns[origin_idx - (max_h - block_length) + 1 : origin_idx + 1]
    # Shape (n_paths, max_h)
    tail = np.empty((n_paths, max_h), dtype=np.float64)
    tail[:, : max_h - block_length] = real_tail[None, :]
    tail[:, max_h - block_length :] = scaled

    # Pre-cast candidate features for the batched paths.
    z_cand = np.asarray(z_at_candidates, dtype=np.float64)
    sigma_cand = np.asarray(sigma_at_candidates, dtype=np.float64)
    K = z_cand.shape[0]
    if K == 0:
        raise ValueError("z_at_candidates is empty")
    capped_n_eff = float(min(n_eff, K))

    # ---- Blocks 1..n_blocks-1 — per-path conditional re-match ----
    for b in range(1, n_blocks):
        # z-scores per path at the new effective sub-origin.
        z_per_path = np.empty((n_paths, len(horizons)), dtype=np.float64)
        for h_idx, h in enumerate(horizons):
            window = tail[:, -h:]
            z_per_path[:, h_idx] = _zscore_from_window(window)
        # If any path produced NaN (zero-std window), fall back to the
        # real-origin z for that path-axis. Rare; sampled returns degenerating
        # to a constant within max_h is essentially impossible at n_paths>=2.
        nan_mask = np.isnan(z_per_path)
        if nan_mask.any():
            broadcast_origin = np.broadcast_to(z_at_origin, z_per_path.shape)
            z_per_path = np.where(nan_mask, broadcast_origin, z_per_path)

        distances = composite_distance_batched(z_per_path, z_cand, weights)
        probs = distances_to_probs_batched(distances, target_n_eff=capped_n_eff)

        chosen_local = _sample_indices_from_probs_batched(probs, rng)
        d = candidate_indices[chosen_local]
        starts = d + 1
        offsets = np.arange(block_length, dtype=np.int64)
        raw_blocks = returns[starts[:, None] + offsets[None, :]]

        scaled = _scale(raw_blocks, chosen_local, b)
        paths[:, b * block_length : (b + 1) * block_length] = scaled

        if not per_step_sigma:
            for r_idx in range(block_length):
                r = scaled[:, r_idx]
                running_var = one_minus_alpha * running_var + alpha * r * r

        # Roll the tail buffer: drop oldest block_length, append the new block.
        tail = np.concatenate([tail[:, block_length:], scaled], axis=1)

    if ratios_out is not None:
        return paths, ratios_out
    return paths
