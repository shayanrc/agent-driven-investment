"""Volatility models for v3b (E9) GARCH-conditional resampling.

Two-step API:
  1. ``fit_garch(returns)`` fits a GARCH(1,1) on causal returns (up to and
     including ``origin_idx``); returns a small dataclass.
  2. ``simulate_garch_sigma_paths(garch, horizon, n_paths, rng)`` simulates
     ``(n_paths, horizon)`` σ paths from that fit, starting from the fit's
     final conditional variance.

The σ paths are then consumed by ``sampling.scale_block`` (per-step rescaling)
in place of the single block-constant ratio used by the EWMA path. The analog
block selection is unchanged — only the σ-source flips parametric.

Causality: the fit must only see returns up to ``origin_idx``. Walk-forward
must call this once per training window (refit at fold boundary).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# arch rescales by default if it thinks magnitudes are off. We pass returns
# scaled by SCALE (basis points) and divide σ outputs back so callers see σ
# in the same units as raw log returns.
_SCALE = 100.0


@dataclass(frozen=True)
class GARCHFit:
    omega: float
    alpha: float
    beta: float
    last_resid: float  # raw-scale residual at the fit endpoint (returns[-1] - mean)
    last_sigma2: float  # raw-scale conditional variance at fit endpoint


def fit_garch(returns: np.ndarray) -> GARCHFit:
    """Fit a zero-mean GARCH(1,1) on a 1-D returns array.

    ``returns`` is treated as log returns. The model is::

        r_t = e_t,   e_t = σ_t z_t,   z_t ~ N(0,1)
        σ_t^2 = ω + α e_{t-1}^2 + β σ_{t-1}^2

    A nonzero unconditional mean is *not* modelled — drift is handled
    separately by ``simulate.forecast`` (C7 + C10). Mixing GARCH-mean with
    our drift estimator would double-count.

    Causality is the caller's responsibility: pass returns up to and including
    the forecast origin only.
    """
    import arch

    if returns.ndim != 1:
        raise ValueError(f"expected 1-D returns, got shape {returns.shape}")
    if returns.size < 100:
        raise ValueError(f"GARCH needs ≥100 obs, got {returns.size}")

    am = arch.arch_model(
        returns * _SCALE,
        vol="Garch",
        p=1,
        q=1,
        mean="Zero",
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)
    p = res.params
    omega = float(p["omega"]) / (_SCALE * _SCALE)
    alpha = float(p["alpha[1]"])
    beta = float(p["beta[1]"])
    # Last residual / σ on raw scale (returns are zero-mean by model assumption)
    last_resid = float(returns[-1])
    last_sigma2 = float(res.conditional_volatility[-1] ** 2) / (_SCALE * _SCALE)
    return GARCHFit(omega=omega, alpha=alpha, beta=beta, last_resid=last_resid, last_sigma2=last_sigma2)


def simulate_garch_sigma_paths(
    fit: GARCHFit,
    horizon: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate ``(n_paths, horizon)`` σ paths from a GARCHFit.

    Standard one-step recursion::

        σ_t^2 = ω + α e_{t-1}^2 + β σ_{t-1}^2
        e_t   = σ_t z_t,  z_t ~ N(0,1)

    The state at step 0 is initialised from ``fit.last_resid`` and
    ``fit.last_sigma2`` — so step 1's σ is fully determined by the fit
    (deterministic across paths). Steps ≥2 diverge because of the simulated
    innovations z_t.

    Returns the σ paths (NOT σ², NOT returns) so the caller can take ratios
    directly.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥1, got {horizon}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be ≥1, got {n_paths}")

    sigma2 = np.empty((n_paths, horizon), dtype=np.float64)
    # Step 1 σ²: fully determined by the fit's endpoint state
    sigma2_step1 = fit.omega + fit.alpha * fit.last_resid**2 + fit.beta * fit.last_sigma2
    sigma2[:, 0] = sigma2_step1

    if horizon > 1:
        # Pre-draw all innovations
        z = rng.standard_normal((n_paths, horizon - 1))
        for t in range(1, horizon):
            # e_{t-1} comes from path's previous step
            e_prev = np.sqrt(sigma2[:, t - 1]) * z[:, t - 1]
            sigma2[:, t] = fit.omega + fit.alpha * e_prev**2 + fit.beta * sigma2[:, t - 1]

    return np.sqrt(sigma2)


def constant_sigma_paths(sigma: float, horizon: int, n_paths: int) -> np.ndarray:
    """Trivial helper for vol_model='constant' — every path is a flat sigma."""
    return np.full((n_paths, horizon), sigma, dtype=np.float64)
