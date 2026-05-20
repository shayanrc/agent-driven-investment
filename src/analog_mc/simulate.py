"""Simulation orchestration: a single forecast for a single origin (Stage 5).

Wires together features (Stage 1), distances (Stage 3), and block sampling
(Stage 4) for one forecast origin date. Returns an (n_paths, forecast_horizon)
array of simulated log-return paths.

Two safety constraints are enforced here:

  * **Forward-block boundary** (extends C5): candidate analog dates must satisfy
    ``d + block_length < origin_idx`` so the sampled forward block lies strictly
    before the forecast origin. Without this, candidates near the end of the
    Train block could leak the forecast origin's realized return back into the
    sample.

  * **Feature completeness**: candidates whose z-scores or σ are NaN (e.g.
    too-early indices that lack 200-day history) are dropped from the eligible
    pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.distances import composite_distance, distances_to_probs
from analog_mc.distances_corrwindow import corrwindow_distance
from analog_mc.local_linear import (
    fit_local_linear_correction,
    forward_logret_sums,
)
from analog_mc.sampling import generate_paths, generate_paths_conditional


def _compute_block0_distances(
    z_target: np.ndarray,
    z_candidates: np.ndarray,
    weights: np.ndarray,
    returns: np.ndarray,
    origin_idx: int,
    eligible: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Block-0 distances under the configured matcher_distance.

    weighted_euclidean (default): composite over z-scores (uses `weights`).
    corrwindow (A2.1): Pearson-corr-window distance (ignores `weights`).
    """
    if config.matcher_distance == "weighted_euclidean":
        return composite_distance(z_target, z_candidates, weights)
    if config.matcher_distance == "corrwindow":
        return corrwindow_distance(
            returns=returns,
            origin_idx=origin_idx,
            candidate_idx=eligible,
            window_length=config.corrwindow_length,
        )
    raise ValueError(f"Unknown matcher_distance: {config.matcher_distance!r}")


# v4 B1: cache the H-day forward cumulative log-returns by (id(returns), horizon).
# `forward_logret_sums` is O(N) but search calls forecast() ~150×/origin × 76 folds;
# computing the array once per (returns, horizon) keeps overhead well under 1%.
_FORWARD_RET_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _forward_logret_sums_cached(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Per-worker cache for B1's forward-return precompute."""
    key = (id(returns), horizon)
    cached = _FORWARD_RET_CACHE.get(key)
    if cached is not None:
        return cached
    out = forward_logret_sums(returns, horizon)
    _FORWARD_RET_CACHE[key] = out
    return out


# Per-worker GARCH fit cache (E9). The fit depends only on (returns, origin_idx);
# search calls forecast() many times per origin with different weights/n_eff.
# Without this cache, GARCH fitting dominates wall time. See simulate.forecast.
_GARCH_FIT_CACHE: dict[tuple[int, int], object] = {}


def _garch_fit_cached(returns: np.ndarray, origin_idx: int) -> object:
    """Cache GARCHFit by (id(returns), origin_idx). Per-worker memory."""
    from analog_mc.vol import fit_garch

    key = (id(returns), origin_idx)
    cached = _GARCH_FIT_CACHE.get(key)
    if cached is not None:
        return cached
    fit = fit_garch(returns[: origin_idx + 1])
    _GARCH_FIT_CACHE[key] = fit
    return fit


def _z_columns(config: Config) -> list[str]:
    return [f"zscore_{h}" for h in config.zscore_horizons]


def _resolve_drift_target(config: Config, features: pd.DataFrame, origin_idx: int) -> float:
    """Compute the per-day drift target for a forecast given the config's drift_mode.

    C10: drift is read once at the origin and applied uniformly to all blocks
    (matching mu_origin's semantics — a per-forecast regime descriptor, not a
    per-block re-estimate). C7: this value is added AFTER the σ ratio in
    scale_block, so it is never multiplied by analog vol.
    """
    mode = config.drift_mode
    if mode == "zero":
        return 0.0
    if mode == "trailing_momentum":
        col = f"trailing_mean_{config.momentum_lookback}"
        if col not in features.columns:
            raise ValueError(
                f"drift_mode='trailing_momentum' requires features column "
                f"'{col}'. Pass momentum_lookback={config.momentum_lookback} "
                f"to compute_features()."
            )
        mu = float(features[col].to_numpy()[origin_idx])
        if np.isnan(mu):
            raise ValueError(
                f"trailing_mean_{config.momentum_lookback} is NaN at "
                f"origin_idx={origin_idx}; need at least {config.momentum_lookback} "
                f"prior returns."
            )
        return config.momentum_shrinkage * mu
    if mode == "scale_with_vol":
        # Reserved; never implemented in v1/v2.
        raise NotImplementedError("drift_mode='scale_with_vol' is reserved, not implemented.")
    raise ValueError(f"Unknown drift_mode: {mode!r}")


def eligible_candidates(
    candidate_idx: np.ndarray,
    features: pd.DataFrame,
    origin_idx: int,
    config: Config,
) -> np.ndarray:
    """Filter candidates to those usable for a forecast at ``origin_idx``.

    Drops (a) candidates whose forward block would overlap the origin or later,
    and (b) candidates with any NaN feature value.
    """
    z_cols = _z_columns(config)
    sigma_col = "ewma_vol"
    mu_col = f"trailing_mean_{max(config.zscore_horizons)}"
    feature_cols = z_cols + [sigma_col]
    if mu_col in features.columns:
        feature_cols.append(mu_col)

    # Forward-block boundary.
    forward_ok = candidate_idx + config.block_length < origin_idx
    eligible = candidate_idx[forward_ok]

    # Feature completeness.
    feat_block = features.iloc[eligible][feature_cols].to_numpy()
    valid = ~np.isnan(feat_block).any(axis=1)
    return eligible[valid]


def forecast(
    origin_idx: int,
    returns: np.ndarray,
    candidate_idx: np.ndarray,
    features: pd.DataFrame,
    weights: np.ndarray,
    n_eff: float,
    config: Config,
    rng: np.random.Generator,
    drift_target: float | None = None,
    record_ratios: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate Monte Carlo paths for a single forecast origin.

    Args:
        origin_idx:    positional index of the forecast origin in ``returns``.
        returns:       full log-returns array (1-D, length N).
        candidate_idx: positional indices of analog candidates (typically the
                       fold's train_idx; may include earlier val dates in v2).
        features:      causal feature DataFrame indexed identically to
                       ``returns`` (columns ``ewma_vol`` and ``zscore_<h>``).
        weights:       shape (3,) non-negative weights for the 3 z-score
                       horizons. Need not be normalized — final probabilities
                       are invariant to scalar rescaling of weights.
        n_eff:         target effective sample size for the analog probability
                       distribution (must be in (1, K] where K is the eligible
                       candidate count).
        config:        pipeline config.
        rng:           np.random.Generator.
        drift_target:  per-day expected log return injected after vol scaling.
                       If ``None`` (the default), it is computed from
                       ``config.drift_mode``: ``"zero"`` → 0.0;
                       ``"trailing_momentum"`` →
                       ``config.momentum_shrinkage *
                       trailing_mean_<momentum_lookback>[origin_idx]``.
                       A float overrides the config (used by tests that want
                       to assert sign/magnitude behaviour deterministically).

    Returns:
        shape (n_paths, forecast_horizon) array of simulated log returns.

    Raises:
        ValueError: if no eligible candidates remain, or if features at the
                    origin are NaN, or if drift_mode requires a feature column
                    that is missing.
    """
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1-D; got shape {returns.shape}")
    if origin_idx < 0 or origin_idx >= returns.size:
        raise ValueError(f"origin_idx {origin_idx} out of range for returns size {returns.size}")

    z_cols = _z_columns(config)
    sigma_col = "ewma_vol"
    mu_col = f"trailing_mean_{max(config.zscore_horizons)}"
    if mu_col not in features.columns:
        raise ValueError(
            f"features is missing required column '{mu_col}' (the C3 baseline "
            f"source). Use compute_features() to build the feature bundle."
        )

    z_all = features[z_cols].to_numpy()  # (N, 3)
    sigma_all = features[sigma_col].to_numpy()  # (N,)
    mu_all = features[mu_col].to_numpy()  # (N,)

    # Check the origin first — a NaN at the origin is a more diagnostic error
    # than "no eligible candidates" downstream.
    z_target = z_all[origin_idx]
    sigma_init = sigma_all[origin_idx]
    mu_origin = mu_all[origin_idx]
    if np.isnan(z_target).any() or np.isnan(sigma_init) or np.isnan(mu_origin):
        raise ValueError(
            f"Features at origin_idx={origin_idx} contain NaN; need at least "
            f"max(zscore_horizons)={max(config.zscore_horizons)} prior returns."
        )

    if drift_target is None:
        drift_target = _resolve_drift_target(config, features, origin_idx)

    eligible = eligible_candidates(candidate_idx, features, origin_idx, config)
    if eligible.size == 0:
        raise ValueError(
            f"No eligible candidates for origin_idx={origin_idx} "
            f"(after forward-block and NaN-feature filters)."
        )

    z_candidates = z_all[eligible]
    sigma_at_candidates = sigma_all[eligible]

    # n_eff must be <= K (eligible candidate count). Caller is responsible for
    # picking values that satisfy this for all origins in a fold; if not, we
    # cap at K rather than failing hard. (Aggressive search grids occasionally
    # ask for n_eff > K on the smallest folds.)
    target = min(float(n_eff), float(eligible.size))
    if target <= 1.0:
        raise ValueError(
            f"Effective candidate pool ({eligible.size}) is too small for n_eff>1."
        )

    # v4 B1 (decision D6): compute the local-linear conditional-mean bias
    # correction once per forecast, using the block-0 (real-origin)
    # probabilities. Distribute as a uniform per-day drift across the horizon.
    # No-op when local_linear_correction is False — paths must be bit-identical
    # to v2.4 in that case (tested in tests/analog_mc/test_local_linear.py).
    if config.local_linear_correction:
        distances0 = _compute_block0_distances(
            z_target, z_candidates, weights, returns, origin_idx, eligible, config,
        )
        probs0 = distances_to_probs(distances0, target_n_eff=target)
        forward_all = _forward_logret_sums_cached(returns, config.forecast_horizon)
        correction, _b1_diag = fit_local_linear_correction(
            z_target=z_target,
            z_candidates=z_candidates,
            probs=probs0,
            forward_returns=forward_all[eligible],
        )
        drift_target = drift_target + correction / config.forecast_horizon

    # v3b (E9): when vol_model='garch', fit GARCH(1,1) on causal returns up to
    # and through the forecast origin, then simulate per-path σ trajectories
    # over the forecast horizon. Per-step rescaling inside generate_paths swaps
    # the block-constant σ ratio for σ_path[t] / σ_historical_at_step[t].
    #
    # The GARCH fit depends ONLY on (returns, origin_idx) — not weights/n_eff/rng.
    # Search calls forecast() ~150 times per origin (weight×n_eff combos), so we
    # cache the fit per (id(returns), origin_idx) to avoid refitting. Per-worker
    # cache (multiprocessing has separate memory) — bounded growth: at most
    # ~76 folds × ~120 origins per fold / N_workers per-worker entries; the fit
    # object is tiny so total memory is negligible.
    sigma_path: np.ndarray | None = None
    if config.vol_model == "garch":
        from analog_mc.vol import fit_garch, simulate_garch_sigma_paths

        fit = _garch_fit_cached(returns, origin_idx)
        sigma_path = simulate_garch_sigma_paths(
            fit=fit,
            horizon=config.forecast_horizon,
            n_paths=config.n_paths,
            rng=rng,
        )

    # A2.1: corrwindow disables conditional block sampling — block-0 distance
    # is computed once and re-used for every block (per design _a2_design.md
    # §3). Conditional re-matching would require per-path simulated windows
    # which is out of A2.1 v1 scope.
    use_conditional = (
        config.conditional_block_sampling
        and config.matcher_distance == "weighted_euclidean"
    )
    if use_conditional:
        return generate_paths_conditional(
            z_at_origin=z_target,
            z_at_candidates=z_candidates,
            candidate_indices=eligible,
            returns=returns,
            sigma_at_candidates=sigma_at_candidates,
            sigma_init=float(sigma_init),
            mu_origin=float(mu_origin),
            weights=weights,
            n_eff=target,
            origin_idx=origin_idx,
            config=config,
            rng=rng,
            drift_target=drift_target,
            record_ratios=record_ratios,
            sigma_path=sigma_path,
            sigma_at_returns=sigma_all if sigma_path is not None else None,
        )

    distances = _compute_block0_distances(
        z_target, z_candidates, weights, returns, origin_idx, eligible, config,
    )
    probs = distances_to_probs(distances, target_n_eff=target)

    return generate_paths(
        probs=probs,
        candidate_indices=eligible,
        returns=returns,
        sigma_at_candidates=sigma_at_candidates,
        sigma_init=float(sigma_init),
        mu_origin=float(mu_origin),
        config=config,
        rng=rng,
        drift_target=drift_target,
        record_ratios=record_ratios,
        sigma_path=sigma_path,
        sigma_at_returns=sigma_all if sigma_path is not None else None,
    )
