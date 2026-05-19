"""Composite distance + n_eff-parameterized probability conversion (Stage 3).

Implements constraint **C2**: instead of a fixed softmax temperature, solve
for τ such that softmax(-distances / τ) has an effective sample size equal to
a user-specified target n_eff = exp(H(p)).

n_eff is the single most consequential parameter for forecast dispersion;
this lets the search loop tune it directly in interpretable units (number of
analogs effectively contributing) rather than in opaque temperature units.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def composite_distance(
    z_target: np.ndarray,
    z_candidates: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted Euclidean distance from target to each candidate.

    Args:
        z_target:     shape (H,) — z-scores for the forecast-origin date.
        z_candidates: shape (N, H) — z-scores for N candidate analog dates.
        weights:      shape (H,) — non-negative weights. Callers are
                      responsible for whatever normalization they need.

    Returns:
        shape (N,) array of distances sqrt(sum_h w_h * (z_target_h - z_cand_h)^2).
    """
    if z_target.ndim != 1:
        raise ValueError(f"z_target must be 1-D; got shape {z_target.shape}")
    if z_candidates.ndim != 2 or z_candidates.shape[1] != z_target.shape[0]:
        raise ValueError(
            f"z_candidates must be (N, {z_target.shape[0]}); got {z_candidates.shape}"
        )
    if weights.shape != z_target.shape:
        raise ValueError(f"weights must match z_target shape; got {weights.shape}")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    diff = z_candidates - z_target  # (N, H)
    sq = diff * diff
    weighted_sq_sum = sq @ weights  # (N,)
    # Numerical guard against tiny negatives from floating-point.
    return np.sqrt(np.maximum(weighted_sq_sum, 0.0))


def _softmax_neg(distances: np.ndarray, tau: float) -> np.ndarray:
    """Stable softmax of -distances/tau."""
    log_w = -distances / tau
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()


def _n_eff_of_tau(distances: np.ndarray, tau: float) -> float:
    """Effective sample size = exp(H(p)) where p = softmax(-d/tau)."""
    p = _softmax_neg(distances, tau)
    mask = p > 0
    entropy = -np.sum(p[mask] * np.log(p[mask]))
    return float(np.exp(entropy))


def distances_to_probs(
    distances: np.ndarray,
    target_n_eff: float,
    tol: float = 1e-4,
) -> np.ndarray:
    """Convert distances to probabilities via the n_eff parameterization.

    Solves for τ so that ``exp(entropy(softmax(-distances/τ))) ≈ target_n_eff``
    via Brent's method, then returns the resulting probability vector.

    Args:
        distances:    shape (N,) array of non-negative distances.
        target_n_eff: desired effective sample size in (1, N].
        tol:          absolute tolerance on tau for the brentq solve.

    Returns:
        shape (N,) probability vector summing to 1.

    Raises:
        ValueError: if target_n_eff is outside (1, N] or distances are invalid.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError(f"distances must be 1-D; got shape {distances.shape}")
    n = distances.size
    if n == 0:
        raise ValueError("distances is empty")
    if np.any(distances < 0):
        raise ValueError("distances must be non-negative")
    if not (1.0 < target_n_eff <= n):
        raise ValueError(
            f"target_n_eff ({target_n_eff}) must be in (1, {n}] for N={n} candidates"
        )

    # Degenerate: all distances equal => only the uniform distribution is
    # reachable, with n_eff == n. Honor that exactly when requested, else
    # raise (any other target is unattainable).
    d_range = float(distances.max() - distances.min())
    if d_range == 0.0:
        if abs(target_n_eff - n) > 1e-6:
            raise ValueError(
                f"All distances equal ({distances[0]}); only n_eff={n} is reachable, "
                f"not {target_n_eff}."
            )
        return np.full(n, 1.0 / n)

    # Bracket τ. n_eff is monotonically increasing in τ.
    #   small τ -> sharp p, n_eff -> count of minimum-distance candidates
    #   large τ -> uniform p, n_eff -> n
    d_scale = d_range
    tau_low = d_scale * 1e-6
    tau_high = d_scale * 1e6

    # Expand if necessary (rare). Bounded loop to avoid pathological inputs.
    for _ in range(40):
        n_low = _n_eff_of_tau(distances, tau_low)
        n_high = _n_eff_of_tau(distances, tau_high)
        if n_low >= target_n_eff:
            tau_low /= 10.0
        elif n_high <= target_n_eff:
            tau_high *= 10.0
        else:
            break
    else:
        raise RuntimeError("Failed to bracket tau for distances_to_probs")

    f = lambda tau: _n_eff_of_tau(distances, tau) - target_n_eff
    tau_star = brentq(f, tau_low, tau_high, xtol=tol)
    return _softmax_neg(distances, tau_star)


def composite_distance_batched(
    z_targets: np.ndarray,
    z_candidates: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Per-row composite distance for ``n_paths`` query points.

    Args:
        z_targets:    shape (n_paths, H) — z-scores for n_paths query origins
                      (e.g., the per-path effective sub-origins in conditional
                      block sampling).
        z_candidates: shape (K, H) — candidate analog z-scores.
        weights:      shape (H,) — non-negative weights.

    Returns:
        shape (n_paths, K) array of distances. Row i equals
        ``composite_distance(z_targets[i], z_candidates, weights)``.

    Used by v2.2 conditional block sampling, where each path has its own
    effective sub-origin and so its own distance vector to the candidate pool.

    Implementation: weighted-Euclidean identity ``‖x − y‖²_w = ‖x‖²_w + ‖y‖²_w
    − 2 (x · (w⊙y))``, which factors the (n_paths, K, H) broadcast into a
    single GEMM ``z_targets @ (z_candidates * weights).T`` plus two cheap
    norm vectors. Memory drops from O(n_paths·K·H) to O(n_paths·K) and the
    dominant work routes through BLAS instead of three NumPy elementwise
    passes — material at the n_paths≈500–1000, K≈5000 sizes the late
    walk-forward folds hit.
    """
    if z_targets.ndim != 2:
        raise ValueError(f"z_targets must be 2-D; got shape {z_targets.shape}")
    if z_candidates.ndim != 2 or z_candidates.shape[1] != z_targets.shape[1]:
        raise ValueError(
            f"z_candidates must be (K, {z_targets.shape[1]}); got {z_candidates.shape}"
        )
    if weights.shape != (z_targets.shape[1],):
        raise ValueError(f"weights must have shape ({z_targets.shape[1]},); got {weights.shape}")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    wz_cand = z_candidates * weights                       # (K, H)
    targ_sq = (z_targets * z_targets) @ weights            # (n_paths,)
    cand_sq = (z_candidates * z_candidates) @ weights      # (K,)
    # Cross is the only (n_paths, K) allocation; subsequent ops fold the
    # remaining identity terms into it in-place to avoid 4× the bandwidth
    # the naive expression incurs (each `+`/`*` on a (n_paths, K) array
    # allocates a fresh 44 MB at canonical resolution).
    out = z_targets @ wz_cand.T                            # (n_paths, K) — BLAS GEMM
    out *= -2.0
    out += targ_sq[:, None]
    out += cand_sq[None, :]
    # Cancellation on near-identical (target, candidate) pairs can produce
    # small negatives; clip before sqrt (semantics identical to the broadcast
    # form, which also clamped via maximum(0)).
    np.maximum(out, 0.0, out=out)
    np.sqrt(out, out=out)
    return out


def distances_to_probs_batched(
    distances: np.ndarray,
    target_n_eff: float,
    tol: float = 5e-3,
    max_iter: int = 22,
) -> np.ndarray:
    """Per-row n_eff-parameterized probability conversion (vectorized bisection).

    For each row of ``distances``, finds τ such that
    ``exp(entropy(softmax(-d/τ))) ≈ target_n_eff`` via log-space bisection,
    using fully vectorized NumPy ops at every iteration. ~50–100× faster than
    a naive per-row brentq loop, which is the difference between a tractable
    v2.2 walk-forward and an intractable one.

    Implementation: three (n_paths, K) buffers (``log_w``, ``w``, ``scratch``)
    are pre-allocated once and reused for every bisection iteration. The
    unfused expression form would allocate ~5 fresh (n_paths, K) arrays per
    iteration × max_iter ≈ 110 fresh allocations at canonical resolution;
    that allocation churn — not the floating-point work — dominates the
    function's cost at large K. The per-iteration entropy is computed via
    the identity ``H = log(Σ w) − Σ p · log_w_shifted`` so ``log_p`` never
    has to be materialized (and the ``p > 0`` guard the original carried is
    unnecessary: when ``exp(log_w_shifted)`` underflows, ``p · log_w_shifted``
    evaluates to ``0 · finite = 0`` in IEEE float — no NaN to mask).

    Args:
        distances:    shape (n_paths, K) array of non-negative distances.
        target_n_eff: scalar target effective sample size in (1, K].
        tol:          relative tolerance on log τ between bisection brackets.
        max_iter:     bisection step cap (~60 covers a 1e12-wide bracket).

    Returns:
        shape (n_paths, K) array of probability vectors (each row sums to 1).

    Raises:
        ValueError: bad shape, negative distances, out-of-range target, or
                    any row that is all-equal (only the uniform distribution
                    is reachable, which is target_n_eff == K only).
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError(f"distances must be 2-D; got shape {distances.shape}")
    n_paths, K = distances.shape
    if K == 0:
        raise ValueError("distances has zero candidates")
    if np.any(distances < 0):
        raise ValueError("distances must be non-negative")
    if not (1.0 < target_n_eff <= K):
        raise ValueError(
            f"target_n_eff ({target_n_eff}) must be in (1, {K}] for K={K} candidates"
        )

    d_min = distances.min(axis=1)  # (n_paths,)
    d_max = distances.max(axis=1)
    d_range = d_max - d_min  # (n_paths,)

    # Detect degenerate rows (all distances equal). Only the uniform p is
    # reachable on such rows -> only target_n_eff == K is satisfiable.
    degenerate = d_range == 0
    if degenerate.any() and abs(target_n_eff - K) > 1e-6:
        idxs = np.flatnonzero(degenerate)
        raise ValueError(
            f"Row(s) {idxs[:5].tolist()}{'...' if idxs.size > 5 else ''} have all-equal "
            f"distances; only target_n_eff={K} is reachable on those rows, not {target_n_eff}."
        )

    # Initial bracket per row in linear τ; log-space bisection inside.
    # Use a sane default for degenerate rows so the arithmetic doesn't NaN —
    # the row's final p is uniform regardless of τ.
    safe_range = np.where(degenerate, 1.0, d_range)
    log_tau_low = np.log(safe_range) + np.log(1e-6)
    log_tau_high = np.log(safe_range) + np.log(1e6)

    # Workhorse buffers — allocated once, reused for every bisection step.
    log_w = np.empty_like(distances)
    w = np.empty_like(distances)
    scratch = np.empty_like(distances)
    log_target_n_eff = np.log(target_n_eff)

    # Bisection on log τ. Each iter is O(n_paths * K) pure NumPy, allocating
    # only (n_paths,)-sized auxiliaries — no (n_paths, K) churn.
    for _ in range(max_iter):
        log_tau_mid = 0.5 * (log_tau_low + log_tau_high)
        # log_w := -distances * (1 / tau_mid) per row, in-place.
        # Computing inv_tau via exp(-log_tau_mid) saves one np.exp call vs
        # exp(log_tau_mid) then 1/.
        inv_tau = np.exp(-log_tau_mid)  # (n_paths,)
        np.multiply(distances, -inv_tau[:, None], out=log_w)
        log_w -= log_w.max(axis=1, keepdims=True)  # numerical-stability shift

        np.exp(log_w, out=w)                       # w := exp(log_w_shifted)
        w_sum = w.sum(axis=1)                       # (n_paths,)
        w /= w_sum[:, None]                         # w := p, in-place

        np.multiply(w, log_w, out=scratch)          # scratch := p · log_w_shifted
        # entropy = log(Σ w) − Σ p · log_w_shifted   (since log p = log_w − log Σ w)
        entropy = np.log(w_sum)
        entropy -= scratch.sum(axis=1)

        # Compare in log-space against log(target_n_eff) — saves one np.exp per
        # iter vs the original ``n_eff_mid > target_n_eff`` comparison.
        too_diffuse = entropy > log_target_n_eff
        log_tau_high = np.where(too_diffuse, log_tau_mid, log_tau_high)
        log_tau_low = np.where(too_diffuse, log_tau_low, log_tau_mid)
        if (log_tau_high - log_tau_low).max() < tol:
            break

    # Final p at the converged midpoint. Re-uses the same buffers — one more
    # in-place softmax pass, no fresh allocation.
    log_tau_final = 0.5 * (log_tau_low + log_tau_high)
    inv_tau = np.exp(-log_tau_final)
    np.multiply(distances, -inv_tau[:, None], out=log_w)
    log_w -= log_w.max(axis=1, keepdims=True)
    np.exp(log_w, out=w)
    w /= w.sum(axis=1, keepdims=True)

    # Degenerate rows get exact uniform.
    if degenerate.any():
        w[degenerate] = 1.0 / K
    return w
