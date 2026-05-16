"""Probabilistic scoring (Stage 6).

CRPS = Continuous Ranked Probability Score. For an ensemble forecast
{x_1, ..., x_n} and observation y, the unbiased estimator is::

    CRPS = (1/n) Σ_i |x_i - y|  -  (1/(2 n^2)) Σ_i Σ_j |x_i - x_j|

The double sum admits an O(n log n) form after sorting::

    Σ_i Σ_j |x_i - x_j| = 2 Σ_{k=1}^{n} (2k - n - 1) x_(k)

so we use that. Lower CRPS is better; CRPS reduces to MAE for a point forecast.

For multi-step forecasts, we score the **marginal distribution at each
horizon step** of the cumulative log return, then return both the per-step
array and its mean (the plan: "mean is more interpretable").
"""

from __future__ import annotations

import numpy as np


def crps_ensemble(samples: np.ndarray, observation: float) -> float:
    """Unbiased ensemble CRPS for a single observation.

    Args:
        samples:     shape (n,) array of forecast samples.
        observation: scalar realized value.

    Returns:
        Non-negative scalar CRPS.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1-D; got shape {samples.shape}")
    n = samples.size
    if n < 1:
        raise ValueError("samples is empty")
    if n == 1:
        return float(abs(samples[0] - observation))

    mae = np.abs(samples - observation).mean()
    sorted_samples = np.sort(samples)
    k = np.arange(1, n + 1, dtype=np.float64)
    pairwise = (2.0 * k - n - 1.0) @ sorted_samples / (n * n)
    return float(mae - pairwise)


def crps_per_step(forecast_paths: np.ndarray, realized_path: np.ndarray) -> np.ndarray:
    """CRPS at each cumulative-log-return horizon step.

    Both inputs are log returns (not prices). The function cumulates them
    along the horizon axis before scoring.

    Args:
        forecast_paths: shape (n_paths, H) — simulated log returns per step.
        realized_path:  shape (H,) — realized log returns per step.

    Returns:
        shape (H,) array of CRPS values, one per horizon step.
    """
    if forecast_paths.ndim != 2:
        raise ValueError(f"forecast_paths must be 2-D; got shape {forecast_paths.shape}")
    if realized_path.ndim != 1:
        raise ValueError(f"realized_path must be 1-D; got shape {realized_path.shape}")
    if forecast_paths.shape[1] != realized_path.shape[0]:
        raise ValueError(
            f"horizon mismatch: forecast {forecast_paths.shape[1]} vs realized "
            f"{realized_path.shape[0]}"
        )

    fc_cum = np.cumsum(forecast_paths, axis=1)  # (n_paths, H)
    rl_cum = np.cumsum(realized_path)           # (H,)
    h = forecast_paths.shape[1]
    out = np.empty(h, dtype=np.float64)
    for step in range(h):
        out[step] = crps_ensemble(fc_cum[:, step], float(rl_cum[step]))
    return out


def crps_sample(forecast_paths: np.ndarray, realized_path: np.ndarray) -> float:
    """Mean per-step CRPS for a multi-step forecast.

    Convenience wrapper: ``crps_per_step(forecast_paths, realized_path).mean()``.
    """
    return float(crps_per_step(forecast_paths, realized_path).mean())
