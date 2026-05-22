"""B1 — Platzer–Yiou local-linear conditional-mean correction.

Locally-weighted linear regression on the analog candidates' 60-day forward
cumulative log-returns, used to apply a Jacobian bias correction to the
matcher's expected forecast at high-Lyapunov regime transitions.

Spec: docs/analog_mc/experiments/_b1_design.md (decisions D1–D10).

Design summary:
    y_i  = sum(returns[d_i+1 : d_i+1+H])          # scalar terminal cum-log-return
    X_i  = [1, z_20(d_i), z_50(d_i), z_200(d_i)]  # 4-dim predictors
    w_i  = probs_i                                 # matcher's n_eff-softmax weights
    β    = (XᵀWX + λI)⁻¹ XᵀWy                     # scale-aware Tikhonov
    y_p  = x_*ᵀβ                                   # prediction at the origin's features
    corr = y_p − Σ w_i y_i                         # bias correction = pred − matcher mean

The scalar `corr` is then distributed across the H-day horizon as a uniform
per-day drift, added to the existing C3-scaled paths (decision D4).

C-constraint compatibility audited in the design doc §D9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalLinearDiagnostics:
    """Per-forecast B1 diagnostics. Cheap to construct; useful for run audits."""

    correction: float
    matcher_mean: float
    predicted_mean: float
    clamp_hit: bool
    n_candidates_used: int
    n_candidates_dropped_no_forward: int
    beta_norm: float


def forward_logret_sums(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Vectorized H-day forward cumulative log-return per index.

    Returns an array `y` of the same length as `returns` where:
        y[i] = sum(returns[i+1 : i+1+horizon])    if i + horizon < len(returns)
        y[i] = NaN                                  otherwise

    The NaN tail covers the last `horizon` indices, which cannot have a complete
    forward window. Callers must drop these from the regression set.
    """
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1-D; got shape {returns.shape}")
    n = returns.size
    if horizon < 1:
        raise ValueError(f"horizon must be ≥1; got {horizon}")

    out = np.full(n, np.nan, dtype=np.float64)
    if n <= horizon:
        return out

    # cumret[k] = sum(returns[0:k]); forward sum = cumret[i+1+H] − cumret[i+1].
    cumret = np.concatenate([[0.0], np.cumsum(returns)])
    last_valid = n - horizon - 1
    idx = np.arange(0, last_valid + 1)
    out[idx] = cumret[idx + horizon + 1] - cumret[idx + 1]
    return out


def fit_local_linear_correction(
    z_target: np.ndarray,
    z_candidates: np.ndarray,
    probs: np.ndarray,
    forward_returns: np.ndarray,
    tikhonov_rel: float = 1e-6,
    extrapolation_clamp: float = 1.5,
) -> tuple[float, LocalLinearDiagnostics]:
    """Compute the Platzer–Yiou bias correction scalar.

    Args:
        z_target:        shape (D,) — z-scores at the forecast origin.
        z_candidates:    shape (K, D) — z-scores at each eligible analog.
        probs:           shape (K,) — matcher probabilities (sum to 1).
        forward_returns: shape (K,) — H-day forward cumulative log-returns at
                         each analog. May contain NaN for candidates whose
                         forward window runs past the end of data; these are
                         dropped and `probs` is renormalized over survivors
                         (decision D8).
        tikhonov_rel:    relative Tikhonov regularization. `λ = tikhonov_rel ×
                         trace(XᵀWX) / D_aug`. Default scale-aware.
        extrapolation_clamp: if `|predicted_mean|` exceeds `extrapolation_clamp ×
                         max(|y_i|)`, fall back to `correction = 0`. Guards
                         against adversarial extrapolation when the analog
                         cluster is collinear.

    Returns:
        (correction, diagnostics). `correction` is the scalar log-return to
        distribute across the horizon (typically O(0.001)–O(0.1)).
    """
    if z_target.ndim != 1:
        raise ValueError(f"z_target must be 1-D; got shape {z_target.shape}")
    if z_candidates.ndim != 2 or z_candidates.shape[1] != z_target.shape[0]:
        raise ValueError(
            f"z_candidates must be (K, {z_target.shape[0]}); got {z_candidates.shape}"
        )
    if probs.shape != (z_candidates.shape[0],):
        raise ValueError(
            f"probs must be 1-D matching z_candidates rows; got {probs.shape}"
        )
    if forward_returns.shape != probs.shape:
        raise ValueError(
            f"forward_returns must match probs shape; got {forward_returns.shape}"
        )

    # D8: drop candidates lacking a forward window; renormalize probs.
    valid = ~np.isnan(forward_returns)
    n_dropped = int((~valid).sum())
    if n_dropped == probs.size:
        # No usable candidate (entire pool too close to end-of-data — should
        # never happen in walk-forward, but guard anyway).
        return 0.0, LocalLinearDiagnostics(
            correction=0.0,
            matcher_mean=0.0,
            predicted_mean=0.0,
            clamp_hit=True,
            n_candidates_used=0,
            n_candidates_dropped_no_forward=n_dropped,
            beta_norm=0.0,
        )

    X_raw = z_candidates[valid]                # (K', D)
    y = forward_returns[valid]                 # (K',)
    w = probs[valid]
    w_sum = w.sum()
    if w_sum <= 0:
        # All surviving candidates have zero probability — degenerate, skip.
        return 0.0, LocalLinearDiagnostics(
            correction=0.0,
            matcher_mean=0.0,
            predicted_mean=0.0,
            clamp_hit=True,
            n_candidates_used=int(valid.sum()),
            n_candidates_dropped_no_forward=n_dropped,
            beta_norm=0.0,
        )
    w = w / w_sum  # renormalize over survivors

    # Augment with intercept column.
    K_eff, D = X_raw.shape
    X = np.concatenate([np.ones((K_eff, 1)), X_raw], axis=1)  # (K', D+1)
    x_star = np.concatenate([[1.0], z_target])                # (D+1,)

    # Weighted normal equations: (XᵀWX + λI) β = XᵀWy.
    Xw = X * w[:, None]
    XtWX = X.T @ Xw                                            # (D+1, D+1)
    XtWy = Xw.T @ y                                            # (D+1,)
    trace = float(np.trace(XtWX))
    lam = tikhonov_rel * trace / (D + 1) if trace > 0 else tikhonov_rel
    XtWX_reg = XtWX + lam * np.eye(D + 1)

    try:
        beta = np.linalg.solve(XtWX_reg, XtWy)
    except np.linalg.LinAlgError:
        # Should be unreachable given Tikhonov, but fail safe.
        return 0.0, LocalLinearDiagnostics(
            correction=0.0,
            matcher_mean=float(w @ y),
            predicted_mean=0.0,
            clamp_hit=True,
            n_candidates_used=K_eff,
            n_candidates_dropped_no_forward=n_dropped,
            beta_norm=0.0,
        )

    matcher_mean = float(w @ y)
    predicted_mean = float(x_star @ beta)

    # D5 extrapolation clamp: if the local-linear prediction is wildly outside
    # the analogs' range, treat it as adversarial extrapolation and disable B1
    # for this origin.
    abs_y_max = float(np.max(np.abs(y))) if y.size else 0.0
    if abs_y_max == 0.0 or abs(predicted_mean) > extrapolation_clamp * abs_y_max:
        return 0.0, LocalLinearDiagnostics(
            correction=0.0,
            matcher_mean=matcher_mean,
            predicted_mean=predicted_mean,
            clamp_hit=True,
            n_candidates_used=K_eff,
            n_candidates_dropped_no_forward=n_dropped,
            beta_norm=float(np.linalg.norm(beta)),
        )

    correction = predicted_mean - matcher_mean
    return correction, LocalLinearDiagnostics(
        correction=correction,
        matcher_mean=matcher_mean,
        predicted_mean=predicted_mean,
        clamp_hit=False,
        n_candidates_used=K_eff,
        n_candidates_dropped_no_forward=n_dropped,
        beta_norm=float(np.linalg.norm(beta)),
    )
