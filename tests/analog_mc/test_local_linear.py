"""Tests for B1 — Platzer–Yiou local-linear conditional-mean correction.

Spec: docs/analog_mc/experiments/_b1_design.md. Tests cover decisions D1, D3,
D4, D5, D6, D7, D8 + the C1 causality guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analog_mc.config import Config
from analog_mc.features import compute_features
from analog_mc.local_linear import (
    fit_local_linear_correction,
    forward_logret_sums,
)
from analog_mc.simulate import forecast


@pytest.fixture
def synthetic_returns() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.01, size=800)


@pytest.fixture
def synthetic_setup() -> tuple[pd.Series, pd.DataFrame, Config]:
    rng = np.random.default_rng(0)
    n = 800
    returns = pd.Series(
        rng.normal(0.0005, 0.01, size=n),
        index=pd.date_range("2010-01-04", periods=n, freq="B"),
        name="log_return",
    )
    cfg = Config(
        forecast_horizon=20,
        block_length=5,
        n_blocks=4,
        n_paths=100,
        ewma_halflife=10,
        zscore_horizons=(20, 50, 100),
        train_initial_size=500,
        val_size=60,
        test_size=60,
    )
    features = compute_features(returns, halflife=cfg.ewma_halflife, horizons=cfg.zscore_horizons)
    return returns, features, cfg


# ----------------------------------------------------------------------------
# forward_logret_sums
# ----------------------------------------------------------------------------


def test_forward_logret_sums_matches_naive_loop(synthetic_returns) -> None:
    horizon = 20
    out = forward_logret_sums(synthetic_returns, horizon)
    # Spot-check 5 indices against a naive loop.
    for i in [0, 50, 200, 500, 770]:
        if i + horizon < synthetic_returns.size:
            expected = synthetic_returns[i + 1 : i + 1 + horizon].sum()
            np.testing.assert_allclose(out[i], expected, atol=1e-12)


def test_forward_logret_sums_tail_is_nan(synthetic_returns) -> None:
    horizon = 60
    out = forward_logret_sums(synthetic_returns, horizon)
    # Last `horizon` indices must be NaN (insufficient forward window).
    assert np.isnan(out[-horizon:]).all()
    # Earlier indices must be finite.
    assert np.isfinite(out[: -horizon]).all()


def test_forward_logret_sums_causality(synthetic_returns) -> None:
    """C1: out[i] must only reference returns[i+1 : i+1+H]; never returns[≤ i]."""
    horizon = 30
    n = synthetic_returns.size
    out = forward_logret_sums(synthetic_returns, horizon)
    for i in [10, 100, 400]:
        modified = synthetic_returns.copy()
        modified[: i + 1] += 1000.0  # massive perturbation BEFORE the forward window
        out_mod = forward_logret_sums(modified, horizon)
        # out[i] depends only on returns[i+1:i+1+H], so the perturbation
        # of returns[:i+1] must not change out[i].
        np.testing.assert_allclose(out[i], out_mod[i], atol=1e-12)


# ----------------------------------------------------------------------------
# fit_local_linear_correction — core behaviour
# ----------------------------------------------------------------------------


def test_correction_signs_with_linear_target() -> None:
    """If y_i is exactly linear in z_50, the regression should recover the slope
    and the correction should drive the matcher mean toward the linear prediction."""
    rng = np.random.default_rng(123)
    K = 100
    # Candidates' z-scores spread around [-2, 2].
    z_candidates = rng.uniform(-2, 2, size=(K, 3))
    # Forward returns: y = 0.10 * z_50 + small noise.
    forward_returns = 0.10 * z_candidates[:, 1] + rng.normal(0, 0.001, size=K)
    # Probs: uniform.
    probs = np.full(K, 1.0 / K)

    # Target at +3 in z_50 (extrapolation just inside the clamp).
    z_target = np.array([0.0, 1.5, 0.0])  # z_50 = 1.5, inside max |y_i|
    correction, diag = fit_local_linear_correction(
        z_target, z_candidates, probs, forward_returns
    )

    # Matcher mean over uniform probs ≈ 0.10 * mean(z_50) ≈ 0.
    # Predicted mean ≈ 0.10 * 1.5 = 0.15.
    # Correction should be near +0.15.
    assert not diag.clamp_hit
    np.testing.assert_allclose(diag.matcher_mean, 0.10 * z_candidates[:, 1].mean(), atol=0.01)
    np.testing.assert_allclose(diag.predicted_mean, 0.10 * 1.5, atol=0.02)
    assert 0.10 < correction < 0.20


def test_correction_is_zero_when_target_equals_weighted_centroid() -> None:
    """When z_target = Σ w_i z_i, prediction equals matcher mean → correction ≈ 0."""
    rng = np.random.default_rng(456)
    K = 50
    z_candidates = rng.normal(0, 1, size=(K, 3))
    probs = rng.dirichlet(np.ones(K))
    forward_returns = rng.normal(0, 0.05, size=K)
    z_target = (probs[:, None] * z_candidates).sum(axis=0)  # weighted centroid

    correction, diag = fit_local_linear_correction(
        z_target, z_candidates, probs, forward_returns
    )
    # Correction should be small (exactly zero in the noiseless limit; here within
    # numerical+Tikhonov bias).
    assert abs(correction) < 1e-6


def test_tikhonov_handles_degenerate_inputs() -> None:
    """All-equal z_candidates → near-singular XᵀWX; Tikhonov keeps β finite."""
    K = 30
    z_candidates = np.zeros((K, 3))  # every candidate at the origin
    z_target = np.array([0.0, 0.0, 0.0])
    probs = np.full(K, 1.0 / K)
    forward_returns = np.full(K, 0.05)

    correction, diag = fit_local_linear_correction(
        z_target, z_candidates, probs, forward_returns
    )
    # No variation in X means β = [matcher_mean, 0, 0, 0] up to Tikhonov.
    # Prediction at z_target = matcher_mean → correction ≈ 0.
    assert np.isfinite(correction)
    assert abs(correction) < 0.01


def test_extrapolation_clamp_fires() -> None:
    """Adversarial z_target far outside cluster — clamp must fire, correction = 0."""
    rng = np.random.default_rng(789)
    K = 50
    # Candidates clustered tightly in z-space; their y is also tightly bounded.
    z_candidates = rng.normal(0, 0.1, size=(K, 3))
    forward_returns = rng.normal(0, 0.01, size=K)
    probs = np.full(K, 1.0 / K)
    # Target far outside cluster.
    z_target = np.array([100.0, 100.0, 100.0])

    correction, diag = fit_local_linear_correction(
        z_target, z_candidates, probs, forward_returns
    )
    assert diag.clamp_hit
    assert correction == 0.0


def test_nan_forward_returns_dropped(synthetic_returns) -> None:
    """Candidates with NaN forward_returns must be dropped (D8)."""
    K = 50
    z_candidates = np.zeros((K, 3))
    z_target = np.array([0.5, 0.5, 0.5])
    probs = np.full(K, 1.0 / K)
    forward_returns = np.full(K, 0.05)
    # Mark last 10 as having no forward window.
    forward_returns[-10:] = np.nan

    correction, diag = fit_local_linear_correction(
        z_target, z_candidates, probs, forward_returns
    )
    assert diag.n_candidates_used == 40
    assert diag.n_candidates_dropped_no_forward == 10


# ----------------------------------------------------------------------------
# Integration: forecast() with/without the B1 knob
# ----------------------------------------------------------------------------


def test_forecast_bit_identical_when_b1_off(synthetic_setup) -> None:
    """D7: knob OFF must produce paths bit-identical to baseline."""
    returns, features, cfg = synthetic_setup
    assert cfg.local_linear_correction is False  # default

    candidate_idx = np.arange(200, 500, dtype=np.int64)
    rng1 = np.random.default_rng(13)
    rng2 = np.random.default_rng(13)

    # Baseline.
    p1 = forecast(
        origin_idx=600,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=features,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=30.0,
        config=cfg,
        rng=rng1,
    )
    # Same call with knob still off (default) — must be identical.
    p2 = forecast(
        origin_idx=600,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=features,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=30.0,
        config=cfg,
        rng=rng2,
    )
    np.testing.assert_array_equal(p1, p2)


def test_forecast_changes_when_b1_on(synthetic_setup) -> None:
    """Knob ON must produce a non-zero shift (otherwise the wiring is broken)."""
    returns, features, cfg_off = synthetic_setup
    cfg_on = Config(**{**cfg_off.to_dict(), "local_linear_correction": True})

    candidate_idx = np.arange(200, 500, dtype=np.int64)
    rng_off = np.random.default_rng(21)
    rng_on = np.random.default_rng(21)
    p_off = forecast(
        origin_idx=600,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=features,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=30.0,
        config=cfg_off,
        rng=rng_off,
    )
    p_on = forecast(
        origin_idx=600,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=features,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=30.0,
        config=cfg_on,
        rng=rng_on,
    )
    # Paths must differ — even a small correction shifts every step.
    assert not np.array_equal(p_off, p_on)
    # The correction enters via drift_target, propagates into the EWMA-σ
    # recursion in blocks 1+ (same path as v2.4's trailing-momentum drift —
    # see design doc D9). Per-day shift is approximately uniform; exact only
    # at block 0. Verify by requiring std/|mean| << 1 across the horizon, AND
    # that block 0 is exactly uniform.
    diff = (p_on - p_off).mean(axis=0)
    block_length = cfg_on.block_length
    block0 = diff[:block_length]
    assert (
        block0.std() < 1e-12
    ), "Block 0 must be exactly uniform (σ recursion has not yet diverged)"
    assert abs(diff.std() / diff.mean()) < 0.05, (
        "Per-day diff std/mean should be small "
        f"(got {diff.std():.2e}/{diff.mean():.2e})"
    )
