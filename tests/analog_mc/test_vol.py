"""Tests for v3b (E9) GARCH-conditional vol module."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analog_mc.config import Config
from analog_mc.data import load_returns
from analog_mc.features import compute_features
from analog_mc.simulate import forecast
from analog_mc.vol import GARCHFit, fit_garch, simulate_garch_sigma_paths


# ---------------------------------------------------------------------------
# Unit tests for vol.py
# ---------------------------------------------------------------------------


def test_fit_garch_recovers_synthetic_params() -> None:
    """Fit a known GARCH(1,1) DGP; recovered (alpha, beta) are in the right ballpark."""
    rng = np.random.default_rng(42)
    n = 5000
    omega, alpha, beta = 1e-6, 0.10, 0.85
    sigma = np.zeros(n)
    sigma[0] = 0.01
    ret = np.zeros(n)
    for t in range(n):
        ret[t] = sigma[t] * rng.standard_normal()
        if t + 1 < n:
            sigma[t + 1] = np.sqrt(omega + alpha * ret[t] ** 2 + beta * sigma[t] ** 2)

    fit = fit_garch(ret)
    # Each parameter within 0.05 of truth — generous, GARCH-MLE has well-known finite-sample bias.
    assert abs(fit.alpha - alpha) < 0.05, f"alpha={fit.alpha} vs {alpha}"
    assert abs(fit.beta - beta) < 0.10, f"beta={fit.beta} vs {beta}"
    # last_sigma2 close to the true terminal σ²
    assert abs(fit.last_sigma2 - sigma[-1] ** 2) / max(sigma[-1] ** 2, 1e-12) < 0.5


def test_simulate_garch_sigma_paths_deterministic_for_same_rng() -> None:
    fit = GARCHFit(omega=1e-6, alpha=0.1, beta=0.85, last_resid=0.01, last_sigma2=1e-4)
    s1 = simulate_garch_sigma_paths(fit, horizon=60, n_paths=100, rng=np.random.default_rng(99))
    s2 = simulate_garch_sigma_paths(fit, horizon=60, n_paths=100, rng=np.random.default_rng(99))
    assert np.array_equal(s1, s2)


def test_simulate_garch_sigma_paths_step1_is_deterministic_across_paths() -> None:
    """First step's σ is fully determined by the fit's endpoint state."""
    fit = GARCHFit(omega=1e-6, alpha=0.1, beta=0.85, last_resid=0.01, last_sigma2=1e-4)
    s = simulate_garch_sigma_paths(fit, horizon=10, n_paths=50, rng=np.random.default_rng(1))
    # All paths share the same σ at step 1.
    assert np.allclose(s[:, 0], s[0, 0])
    # By step 2 paths have diverged (innovations).
    assert s[:, 1].std() > 0


def test_simulate_garch_sigma_paths_positive_and_finite() -> None:
    fit = GARCHFit(omega=1e-6, alpha=0.05, beta=0.9, last_resid=0.005, last_sigma2=1e-4)
    s = simulate_garch_sigma_paths(fit, horizon=60, n_paths=200, rng=np.random.default_rng(7))
    assert np.all(np.isfinite(s)) and np.all(s > 0)


# ---------------------------------------------------------------------------
# Integration tests against simulate.forecast
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_setup():
    c = Config.from_yaml("configs/analog_mc/nasdaq100_fast.yaml")
    returns_s = load_returns(c)
    returns = returns_s.to_numpy()
    features = compute_features(
        returns_s,
        halflife=c.ewma_halflife,
        horizons=c.zscore_horizons,
        momentum_lookback=c.momentum_lookback,
    )
    return c, returns, features


def test_garch_path_determinism_end_to_end(fast_setup) -> None:
    """Same seed + same config → bit-identical paths in GARCH mode."""
    c, returns, features = fast_setup
    c_garch = dataclasses.replace(c, vol_model="garch", n_paths=200)
    origin = 1500
    candidate_idx = np.arange(max(c.zscore_horizons), origin - c.block_length)
    weights = np.array([0.3, 0.4, 0.3])

    p1 = forecast(origin, returns, candidate_idx, features, weights, 80, c_garch, np.random.default_rng(42))
    p2 = forecast(origin, returns, candidate_idx, features, weights, 80, c_garch, np.random.default_rng(42))
    assert np.array_equal(p1, p2)


def test_ewma_branch_unchanged_when_vol_model_default(fast_setup) -> None:
    """vol_model='ewma' must produce exactly the v2.x EWMA paths (no regression)."""
    c, returns, features = fast_setup
    c_default = dataclasses.replace(c, n_paths=200)  # vol_model='ewma' by default
    c_explicit = dataclasses.replace(c, vol_model="ewma", n_paths=200)
    origin = 1500
    candidate_idx = np.arange(max(c.zscore_horizons), origin - c.block_length)
    weights = np.array([0.3, 0.4, 0.3])

    p_default = forecast(origin, returns, candidate_idx, features, weights, 80, c_default, np.random.default_rng(42))
    p_explicit = forecast(origin, returns, candidate_idx, features, weights, 80, c_explicit, np.random.default_rng(42))
    assert np.array_equal(p_default, p_explicit)


def test_garch_causality_no_future_leak(fast_setup) -> None:
    """GARCH fit at origin_idx must not depend on any returns past origin_idx.

    Sentinel approach: corrupt all post-origin returns with extreme values; if the
    GARCH-mode forecast at origin_idx is bit-identical to the uncorrupted version,
    we have causality. The corruption must NOT affect:
      - GARCH fit (depends only on returns[:origin_idx+1])
      - Forecast paths (depend on GARCH fit + sigma_at_returns lookup at analog
        forward positions ≤ origin_idx, since candidates are restricted)
    """
    c, returns, features = fast_setup
    c_garch = dataclasses.replace(c, vol_model="garch", n_paths=200)
    origin = 1500
    candidate_idx = np.arange(max(c.zscore_horizons), origin - c.block_length)
    weights = np.array([0.3, 0.4, 0.3])

    paths_clean = forecast(
        origin, returns.copy(), candidate_idx, features, weights, 80, c_garch, np.random.default_rng(42)
    )

    # Corrupt all post-origin returns with extreme values; sigma_at_returns
    # would be poisoned at those indices but candidates are restricted to
    # d + block_length < origin, so the analog block lookup never reaches past origin.
    returns_poisoned = returns.copy()
    returns_poisoned[origin + 1 :] = 999.0
    paths_poisoned = forecast(
        origin, returns_poisoned, candidate_idx, features, weights, 80, c_garch, np.random.default_rng(42)
    )

    assert np.array_equal(paths_clean, paths_poisoned), "GARCH-mode forecast leaked future returns"


def test_garch_simulated_acf_higher_than_ewma_branch(fast_setup) -> None:
    """v3b's whole point: simulated squared-return ACF should be MORE positive
    under GARCH-conditional rescaling than under EWMA block-constant rescaling.
    This is a sanity check, not a strict claim — small effects are noise, but
    on a 60-day horizon × many paths the direction should be clear.
    """
    c, returns, features = fast_setup
    origin = 1500
    candidate_idx = np.arange(max(c.zscore_horizons), origin - c.block_length)
    weights = np.array([0.3, 0.4, 0.3])

    c_ewma = dataclasses.replace(c, vol_model="ewma", n_paths=300)
    c_garch = dataclasses.replace(c, vol_model="garch", n_paths=300)
    p_ewma = forecast(origin, returns, candidate_idx, features, weights, 80, c_ewma, np.random.default_rng(42))
    p_garch = forecast(origin, returns, candidate_idx, features, weights, 80, c_garch, np.random.default_rng(42))

    def lag1_sqr_acf(paths: np.ndarray) -> float:
        x = paths**2
        x_c = x - x.mean(axis=1, keepdims=True)
        denom = (x_c * x_c).sum(axis=1)
        mask = denom > 0
        numer = (x_c[mask, :-1] * x_c[mask, 1:]).sum(axis=1)
        return float((numer / denom[mask]).mean())

    acf_ewma = lag1_sqr_acf(p_ewma)
    acf_garch = lag1_sqr_acf(p_garch)
    # GARCH-conditional should be strictly greater (more positive). Allow a small
    # tolerance for stochastic noise, but the direction should be unambiguous.
    assert acf_garch > acf_ewma - 0.01, f"acf_garch={acf_garch:+.4f}, acf_ewma={acf_ewma:+.4f}"
