"""Tests for analog_mc.sampling (constraints C3, C4, C5)."""

from __future__ import annotations

import numpy as np
import pytest

from analog_mc.config import Config
from analog_mc.sampling import (
    _alpha_from_halflife,
    generate_paths,
    sample_analog_blocks,
    scale_block,
)


# ---------------------------------------------------------------------------
# scale_block (C3)
# ---------------------------------------------------------------------------


def test_scale_block_subtracts_shared_baseline() -> None:
    """The SHARED baseline mu_origin must be subtracted before σ scaling.

    With ratio=2.0 and mu_origin=0.02, raw=[0.01, 0.02, 0.03] gives:
      demeaned = [-0.01, 0.0, 0.01]; scaled = [-0.02, 0.0, 0.02].
    Note this happens to match the per-block-demean answer here ONLY because
    the raw_mean coincides with mu_origin; that is the special case.
    """
    raw = np.array([0.01, 0.02, 0.03])
    out = scale_block(
        raw_block=raw,
        sigma_current=0.04,
        sigma_historical=0.02,
        mu_origin=0.02,
        drift_target=0.0,
        vol_clip_lower=0.5,
        vol_clip_upper=3.0,
    )
    np.testing.assert_allclose(out, [-0.02, 0.0, 0.02])


def test_scale_block_preserves_offset_when_baseline_differs_from_block_mean() -> None:
    """The block keeps a non-zero mean after scaling when raw_mean ≠ mu_origin.
    This is the key v1.1 property: per-block sums are NOT forced to zero.
    """
    raw = np.array([0.02, 0.04])  # raw_mean = 0.03
    out = scale_block(
        raw_block=raw,
        sigma_current=0.02,
        sigma_historical=0.02,  # ratio = 1.0
        mu_origin=0.01,         # not equal to raw_mean
        drift_target=0.0,
        vol_clip_lower=0.5,
        vol_clip_upper=3.0,
    )
    # demeaned = [0.01, 0.03]; scaled = same; mean = 0.02 (non-zero!)
    np.testing.assert_allclose(out, [0.01, 0.03])
    assert out.mean() == pytest.approx(0.02)


def test_scale_block_clips_ratio_upper() -> None:
    raw = np.array([0.01, -0.01])
    out = scale_block(
        raw_block=raw,
        sigma_current=10.0,
        sigma_historical=1.0,
        mu_origin=0.0,
        drift_target=0.0,
        vol_clip_lower=0.5,
        vol_clip_upper=3.0,
    )
    # ratio clipped to 3.0, demeaned = [0.01, -0.01], scaled = [0.03, -0.03]
    np.testing.assert_allclose(out, [0.03, -0.03])


def test_scale_block_clips_ratio_lower() -> None:
    raw = np.array([0.01, -0.01])
    out = scale_block(
        raw_block=raw,
        sigma_current=0.1,
        sigma_historical=1.0,
        mu_origin=0.0,
        drift_target=0.0,
        vol_clip_lower=0.5,
        vol_clip_upper=3.0,
    )
    # ratio clipped to 0.5
    np.testing.assert_allclose(out, [0.005, -0.005])


def test_scale_block_adds_drift_after_scaling() -> None:
    raw = np.array([0.02, 0.04])  # mean 0.03
    out = scale_block(
        raw_block=raw,
        sigma_current=0.02,
        sigma_historical=0.02,
        mu_origin=0.03,         # equal to raw_mean -> zero-mean demeaned
        drift_target=0.001,
        vol_clip_lower=0.5,
        vol_clip_upper=3.0,
    )
    # demeaned [-0.01, 0.01], ratio=1, + drift 0.001
    np.testing.assert_allclose(out, [-0.009, 0.011])


def test_scale_block_rejects_invalid_sigma_historical() -> None:
    with pytest.raises(ValueError):
        scale_block(np.array([0.01]), 0.02, 0.0, 0.0, 0.0, 0.5, 3.0)


# ---------------------------------------------------------------------------
# sample_analog_blocks (C5: strictly forward)
# ---------------------------------------------------------------------------


def test_sample_analog_blocks_forward_window() -> None:
    """Sampling candidate at index d must return returns[d+1 : d+1+L]."""
    returns = np.arange(100, dtype=np.float64)  # easy to identify
    candidate_indices = np.array([20], dtype=np.int64)
    probs = np.array([1.0])
    rng = np.random.default_rng(0)
    _, raw_blocks = sample_analog_blocks(
        probs, candidate_indices, returns, block_length=5, n_paths=3, rng=rng
    )
    # All paths must get the same block since there's only one candidate.
    expected = np.array([21, 22, 23, 24, 25], dtype=np.float64)
    for row in raw_blocks:
        np.testing.assert_allclose(row, expected)


def test_sample_analog_blocks_distribution_matches_probs() -> None:
    """Empirical frequencies must approximate the supplied probabilities."""
    returns = np.zeros(100)
    candidate_indices = np.array([10, 20, 30], dtype=np.int64)
    probs = np.array([0.6, 0.3, 0.1])
    rng = np.random.default_rng(42)
    chosen, _ = sample_analog_blocks(
        probs, candidate_indices, returns, block_length=5, n_paths=20_000, rng=rng
    )
    freq = np.bincount(chosen, minlength=3) / chosen.size
    np.testing.assert_allclose(freq, probs, atol=0.02)


def test_sample_analog_blocks_rejects_candidate_without_forward_window() -> None:
    returns = np.zeros(20)
    candidate_indices = np.array([18], dtype=np.int64)  # d+5=23 > 20
    probs = np.array([1.0])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="no full forward block"):
        sample_analog_blocks(
            probs, candidate_indices, returns, block_length=5, n_paths=1, rng=rng
        )


# ---------------------------------------------------------------------------
# generate_paths (C4: running σ across blocks)
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config() -> Config:
    return Config(
        forecast_horizon=20,
        block_length=5,
        n_blocks=4,
        n_paths=200,
        ewma_halflife=10,
        zscore_horizons=(20, 50, 100),
        train_initial_size=500,
    )


def test_generate_paths_shape(small_config: Config) -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=1000)
    candidate_indices = np.arange(100, 800, dtype=np.int64)
    probs = np.full(candidate_indices.size, 1.0 / candidate_indices.size)
    sigma_at_cand = np.full(candidate_indices.size, 0.01)
    paths = generate_paths(
        probs=probs,
        candidate_indices=candidate_indices,
        returns=returns,
        sigma_at_candidates=sigma_at_cand,
        sigma_init=0.012,
        mu_origin=0.0005,
        config=small_config,
        rng=rng,
    )
    assert paths.shape == (small_config.n_paths, small_config.forecast_horizon)
    assert np.isfinite(paths).all()


def test_generate_paths_no_block_boundary_collapse(small_config: Config) -> None:
    """v1.1 property: cumulative log return std must be > 0 at block boundaries.

    Under the old per-block demean, every path would have an identical
    cumulative log return at h = block_length, 2*block_length, ..., which
    collapsed dispersion to zero at those horizons. With the shared baseline,
    dispersion must remain non-degenerate.
    """
    rng = np.random.default_rng(0)
    # Generate returns with non-trivial vol so the analog pool is diverse.
    returns = rng.normal(0.0, 0.01, size=1000)
    candidate_indices = np.arange(100, 800, dtype=np.int64)
    probs = np.full(candidate_indices.size, 1.0 / candidate_indices.size)
    sigma_at_cand = np.full(candidate_indices.size, 0.01)
    paths = generate_paths(
        probs=probs,
        candidate_indices=candidate_indices,
        returns=returns,
        sigma_at_candidates=sigma_at_cand,
        sigma_init=0.01,
        mu_origin=0.0,
        config=small_config,
        rng=rng,
    )
    cum = paths.cumsum(axis=1)
    block_len = small_config.block_length
    for b in range(1, small_config.n_blocks + 1):
        h = b * block_len - 1  # 0-indexed end-of-block step
        assert cum[:, h].std() > 1e-6, (
            f"Cumulative-return std collapsed at block boundary h={h+1}; "
            f"per-block demean bug is back."
        )


def test_generate_paths_running_sigma_updates_between_blocks(small_config: Config) -> None:
    """If we artificially inject extreme returns via the analog block, the
    running σ in subsequent blocks must respond — visible as larger absolute
    returns in later blocks when σ_init is small relative to the block's
    realized vol.
    """
    rng = np.random.default_rng(1)
    returns = np.zeros(1000)
    # One candidate whose forward block has very high vol.
    big_block = np.array([0.05, -0.05, 0.05, -0.05, 0.05])
    cand = np.array([100], dtype=np.int64)
    returns[101:106] = big_block
    sigma_at_cand = np.array([0.005])  # tiny historical σ -> ratio clipped to upper
    probs = np.array([1.0])
    paths = generate_paths(
        probs=probs,
        candidate_indices=cand,
        returns=returns,
        sigma_at_candidates=sigma_at_cand,
        sigma_init=0.005,
        mu_origin=0.0,
        config=small_config,
        rng=rng,
    )
    block_abs_means = np.array(
        [
            np.abs(paths[:, b * small_config.block_length : (b + 1) * small_config.block_length]).mean()
            for b in range(small_config.n_blocks)
        ]
    )
    # Later blocks should have larger mean |return| than the first block.
    assert block_abs_means[-1] > block_abs_means[0]


def test_generate_paths_deterministic_with_seed(small_config: Config) -> None:
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    returns = np.random.default_rng(99).normal(0, 0.01, size=1000)
    candidate_indices = np.arange(100, 800, dtype=np.int64)
    probs = np.full(candidate_indices.size, 1.0 / candidate_indices.size)
    sigma_at_cand = np.full(candidate_indices.size, 0.01)
    p1 = generate_paths(
        probs, candidate_indices, returns, sigma_at_cand, 0.01, 0.0, small_config, rng1
    )
    p2 = generate_paths(
        probs, candidate_indices, returns, sigma_at_cand, 0.01, 0.0, small_config, rng2
    )
    np.testing.assert_array_equal(p1, p2)


def test_alpha_from_halflife() -> None:
    # halflife = N means var weight halves every N steps under the recursion.
    # (1-α)^N = 0.5 => α = 1 - 2^(-1/N).
    a = _alpha_from_halflife(20)
    assert a == pytest.approx(1.0 - 2.0 ** (-1.0 / 20.0))
    assert 0.0 < a < 1.0
