"""Tests for analog_mc.scoring (CRPS).

The headline test: a large Gaussian ensemble must converge to the closed-form
Gaussian CRPS, validating both the formula and the O(n log n) implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from analog_mc.scoring import crps_ensemble, crps_per_step, crps_sample


def gaussian_crps_closed_form(mu: float, sigma: float, y: float) -> float:
    """Analytic CRPS for X ~ N(mu, sigma^2) and observation y."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


# ---------------------------------------------------------------------------
# crps_ensemble — basic + Gaussian convergence
# ---------------------------------------------------------------------------


def test_crps_point_forecast_reduces_to_abs_error() -> None:
    """With one sample, CRPS = |sample - obs|."""
    assert crps_ensemble(np.array([3.0]), 5.0) == pytest.approx(2.0)


def test_crps_perfect_forecast_is_zero() -> None:
    """An ensemble of identical samples matching the obs has CRPS == 0."""
    samples = np.full(50, 1.5)
    assert crps_ensemble(samples, 1.5) == pytest.approx(0.0, abs=1e-12)


def test_crps_against_naive_pairwise() -> None:
    """Sanity: O(n log n) formula must match the naive O(n^2) definition."""
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=50)
    y = 0.3
    fast = crps_ensemble(samples, y)
    n = samples.size
    mae = np.abs(samples - y).mean()
    pairwise = np.abs(samples[:, None] - samples[None, :]).sum() / (2 * n * n)
    naive = mae - pairwise
    assert fast == pytest.approx(naive, rel=1e-12)


@pytest.mark.parametrize("mu,sigma,y", [
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 1.5),
    (-0.5, 2.0, 1.0),
    (0.1, 0.05, 0.0),
])
def test_crps_gaussian_ensemble_converges_to_closed_form(mu, sigma, y) -> None:
    """Large Gaussian sample's CRPS must converge to the analytic value."""
    rng = np.random.default_rng(0)
    samples = rng.normal(mu, sigma, size=20_000)
    empirical = crps_ensemble(samples, y)
    analytic = gaussian_crps_closed_form(mu, sigma, y)
    # With 20k samples, ~1% tolerance is comfortable.
    assert empirical == pytest.approx(analytic, rel=0.01, abs=0.01)


def test_crps_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        crps_ensemble(np.array([[1.0]]), 0.0)
    with pytest.raises(ValueError):
        crps_ensemble(np.array([]), 0.0)


# ---------------------------------------------------------------------------
# crps_per_step / crps_sample
# ---------------------------------------------------------------------------


def test_crps_per_step_shape() -> None:
    rng = np.random.default_rng(1)
    fc = rng.normal(0, 0.01, size=(500, 20))
    rl = rng.normal(0, 0.01, size=20)
    per_step = crps_per_step(fc, rl)
    assert per_step.shape == (20,)
    assert (per_step >= 0).all()


def test_crps_per_step_uses_cumulative_returns() -> None:
    """At step h, the score must compare cumulative sums of log returns."""
    fc = np.array([
        [0.01, 0.02, 0.03],
        [0.02, 0.01, 0.0],
    ])
    rl = np.array([0.01, 0.02, 0.03])
    # Cumulative fc: [[0.01, 0.03, 0.06], [0.02, 0.03, 0.03]]
    # Cumulative rl: [0.01, 0.03, 0.06]
    per_step = crps_per_step(fc, rl)
    # Step 0: CRPS({0.01, 0.02}, 0.01)
    expected0 = crps_ensemble(np.array([0.01, 0.02]), 0.01)
    # Step 1: CRPS({0.03, 0.03}, 0.03)
    expected1 = crps_ensemble(np.array([0.03, 0.03]), 0.03)
    # Step 2: CRPS({0.06, 0.03}, 0.06)
    expected2 = crps_ensemble(np.array([0.06, 0.03]), 0.06)
    np.testing.assert_allclose(per_step, [expected0, expected1, expected2])


def test_crps_sample_is_mean_of_per_step() -> None:
    rng = np.random.default_rng(2)
    fc = rng.normal(0, 0.01, size=(200, 10))
    rl = rng.normal(0, 0.01, size=10)
    mean = crps_sample(fc, rl)
    arr = crps_per_step(fc, rl)
    assert mean == pytest.approx(arr.mean())


def test_crps_per_step_rejects_horizon_mismatch() -> None:
    with pytest.raises(ValueError, match="horizon mismatch"):
        crps_per_step(np.zeros((5, 10)), np.zeros(8))
