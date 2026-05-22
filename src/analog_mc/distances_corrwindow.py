"""A2.1 — correlation-window matcher distance.

Replaces the weighted-Euclidean-on-z-scores distance with a Pearson-
correlation-based distance between L-day pre-origin causal windows.

Spec: docs/analog_mc/experiments/_a2_design.md (A2.1).

Distance:
    W_x = returns[x - L + 1 : x + 1]      # causal window of length L ending at x
    d(target, cand) = 1 - |Pearson_corr(W_target, W_cand)|

Larger correlation magnitude → smaller distance. A floor `ε` keeps the
softmax-via-n_eff well-behaved at perfect-correlation duplicates.

C1 (causality): W_x only references returns[≤x]. No look-ahead.

Vectorization: precompute the per-index window-z-score matrix
    Z[x] = (W_x - mean(W_x)) / std(W_x, ddof=1)
then distance = 1 - |Z[target] @ Z[cand].T / (L - 1)|. O(K·L) per call after
the precompute, O(N·L) for the precompute (done once per origin or cached).
"""

from __future__ import annotations

import numpy as np


def _window_zscores(returns: np.ndarray, indices: np.ndarray, L: int) -> np.ndarray:
    """Per-index L-window centered-and-scaled vectors, vectorized.

    Returns shape (K, L) where row k is `(W_{indices[k]} - mean) / std`. Uses
    population std (ddof=0) so each row has Euclidean norm sqrt(L). Rows where
    the window has zero std (constant returns over L days) are set to zero so
    correlation = 0 and distance = 1.
    """
    if L < 2:
        raise ValueError(f"window_length must be >= 2; got {L}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1-D; got shape {indices.shape}")
    starts = indices - L + 1
    if (starts < 0).any():
        raise ValueError(
            f"some indices in {indices.min()}..{indices.max()} have window "
            f"start < 0 (L={L}); filter candidate_idx upstream."
        )
    offsets = np.arange(L, dtype=np.int64)
    raw = returns[starts[:, None] + offsets[None, :]]  # (K, L)
    means = raw.mean(axis=1, keepdims=True)
    centered = raw - means
    stds = centered.std(axis=1, ddof=0, keepdims=True)
    # Avoid division-by-zero for degenerate (constant) windows.
    safe = np.where(stds > 0, stds, 1.0)
    z = np.where(stds > 0, centered / safe, 0.0)
    return z


def corrwindow_distance(
    returns: np.ndarray,
    origin_idx: int,
    candidate_idx: np.ndarray,
    window_length: int,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Pearson-correlation-window distance from origin to each candidate.

    Args:
        returns:        shape (N,) log returns.
        origin_idx:     position of the forecast origin in `returns`. Window
                        is `returns[origin_idx - L + 1 : origin_idx + 1]`.
        candidate_idx:  shape (K,) positions to score. Each must satisfy
                        candidate_idx - L + 1 >= 0.
        window_length:  L. The width of the matching window.
        epsilon:        floor on distances (keeps `distances_to_probs`'s
                        softmax-via-n_eff stable when |corr| ≈ 1).

    Returns:
        shape (K,) non-negative distances. Range is roughly [epsilon, 1].

    Causality (C1): only `returns[origin_idx - L + 1 : origin_idx + 1]` and
    each candidate's symmetric pre-window are read. Forward returns are
    untouched. The caller is responsible for the eligibility filter
    (`candidate_idx + block_length < origin_idx`).
    """
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1-D; got shape {returns.shape}")
    if origin_idx - window_length + 1 < 0:
        raise ValueError(
            f"origin_idx={origin_idx} insufficient for window_length={window_length}"
        )

    z_target_row = _window_zscores(returns, np.array([origin_idx]), window_length)[0]
    z_cands = _window_zscores(returns, candidate_idx, window_length)
    # Pearson correlation = (z_target · z_cand) / L for population-std rows.
    corr = (z_cands @ z_target_row) / window_length
    distances = 1.0 - np.abs(corr)
    # Numerical guard: clamp small negatives from floating-point.
    distances = np.maximum(distances, epsilon)
    return distances
