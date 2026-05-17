"""Tests for analog_mc.sampling (constraints C3, C4, C5)."""

from __future__ import annotations

import numpy as np
import pytest

from analog_mc.config import Config
from analog_mc.sampling import (
    _alpha_from_halflife,
    _sample_indices_from_probs_batched,
    generate_paths,
    generate_paths_conditional,
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


# ---------------------------------------------------------------------------
# v2.2 — conditional block sampling
# ---------------------------------------------------------------------------


def test_sample_indices_from_probs_batched_distribution() -> None:
    """With uniform probs, the sampled-index distribution should be ~uniform."""
    rng = np.random.default_rng(0)
    n_paths, K = 10_000, 5
    probs = np.full((n_paths, K), 1.0 / K)
    idx = _sample_indices_from_probs_batched(probs, rng)
    counts = np.bincount(idx, minlength=K)
    expected = n_paths / K
    assert np.abs(counts - expected).max() < expected * 0.10  # within 10%


def test_sample_indices_from_probs_batched_concentrated() -> None:
    """With probs ~ [1, 0, 0, ...], every row picks index 0."""
    n_paths, K = 1000, 5
    probs = np.zeros((n_paths, K))
    probs[:, 0] = 1.0
    idx = _sample_indices_from_probs_batched(probs, np.random.default_rng(0))
    assert np.all(idx == 0)


def _conditional_setup(seed: int = 0):
    """Build a fixture with enough history for max(zscore_horizons)=100."""
    rng = np.random.default_rng(seed)
    n = 1500
    returns = rng.normal(0.0005, 0.01, size=n)
    cfg = Config(
        forecast_horizon=20,
        block_length=5,
        n_blocks=4,
        n_paths=80,
        ewma_halflife=10,
        zscore_horizons=(20, 50, 100),
        train_initial_size=600,
        conditional_block_sampling=True,
    )
    origin_idx = 1200
    # Candidate set per v1 eligibility.
    candidate_idx = np.arange(150, origin_idx - cfg.block_length, dtype=np.int64)
    # Per-candidate z-scores: just reuse simple rolling stats on the returns.
    # Build features inline rather than relying on compute_features to avoid
    # coupling this unit test to that module.
    series = pd.Series(returns)
    zs = []
    for h in cfg.zscore_horizons:
        win = series.rolling(window=h, min_periods=h)
        zs.append((win.mean() / win.std(ddof=1)).to_numpy())
    z_all = np.stack(zs, axis=1)  # (N, 3)
    sigma_all = series.ewm(halflife=cfg.ewma_halflife, adjust=False).std().to_numpy()
    z_at_candidates = z_all[candidate_idx]
    sigma_at_candidates = sigma_all[candidate_idx]
    z_at_origin = z_all[origin_idx]
    sigma_init = float(sigma_all[origin_idx])
    mu_origin = float(series.iloc[origin_idx - max(cfg.zscore_horizons) + 1 : origin_idx + 1].mean())
    return dict(
        cfg=cfg,
        returns=returns,
        origin_idx=origin_idx,
        candidate_idx=candidate_idx,
        z_at_origin=z_at_origin,
        z_at_candidates=z_at_candidates,
        sigma_at_candidates=sigma_at_candidates,
        sigma_init=sigma_init,
        mu_origin=mu_origin,
    )


# pd is needed by the helper above.
import pandas as pd  # noqa: E402  -- after fixtures so test file structure stays readable


def test_generate_paths_conditional_deterministic() -> None:
    """Same seed must produce bit-identical paths (C9 leakage detector)."""
    s = _conditional_setup()
    weights = np.array([1.0, 1.0, 1.0])
    p1 = generate_paths_conditional(
        z_at_origin=s["z_at_origin"], z_at_candidates=s["z_at_candidates"],
        candidate_indices=s["candidate_idx"], returns=s["returns"],
        sigma_at_candidates=s["sigma_at_candidates"], sigma_init=s["sigma_init"],
        mu_origin=s["mu_origin"], weights=weights, n_eff=30.0,
        origin_idx=s["origin_idx"], config=s["cfg"], rng=np.random.default_rng(42),
    )
    p2 = generate_paths_conditional(
        z_at_origin=s["z_at_origin"], z_at_candidates=s["z_at_candidates"],
        candidate_indices=s["candidate_idx"], returns=s["returns"],
        sigma_at_candidates=s["sigma_at_candidates"], sigma_init=s["sigma_init"],
        mu_origin=s["mu_origin"], weights=weights, n_eff=30.0,
        origin_idx=s["origin_idx"], config=s["cfg"], rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(p1, p2)


def test_generate_paths_conditional_shape_and_finite() -> None:
    s = _conditional_setup()
    weights = np.array([1.0, 1.0, 1.0])
    paths = generate_paths_conditional(
        z_at_origin=s["z_at_origin"], z_at_candidates=s["z_at_candidates"],
        candidate_indices=s["candidate_idx"], returns=s["returns"],
        sigma_at_candidates=s["sigma_at_candidates"], sigma_init=s["sigma_init"],
        mu_origin=s["mu_origin"], weights=weights, n_eff=30.0,
        origin_idx=s["origin_idx"], config=s["cfg"], rng=np.random.default_rng(7),
    )
    assert paths.shape == (s["cfg"].n_paths, s["cfg"].forecast_horizon)
    assert np.isfinite(paths).all()


def test_generate_paths_conditional_first_block_matches_v1() -> None:
    """Block 0 of the conditional sampler must equal v1 generate_paths block 0
    when fed the same rng seed and same probs — the code paths diverge only
    from block 1 onward."""
    from analog_mc.distances import composite_distance, distances_to_probs

    s = _conditional_setup()
    weights = np.array([1.0, 1.0, 1.0])
    n_eff = 30.0

    # v1 path
    distances = composite_distance(s["z_at_origin"], s["z_at_candidates"], weights)
    probs = distances_to_probs(distances, target_n_eff=n_eff)
    rng_v1 = np.random.default_rng(99)
    v1_paths = generate_paths(
        probs=probs, candidate_indices=s["candidate_idx"], returns=s["returns"],
        sigma_at_candidates=s["sigma_at_candidates"], sigma_init=s["sigma_init"],
        mu_origin=s["mu_origin"], config=s["cfg"], rng=rng_v1,
    )

    rng_cond = np.random.default_rng(99)
    cond_paths = generate_paths_conditional(
        z_at_origin=s["z_at_origin"], z_at_candidates=s["z_at_candidates"],
        candidate_indices=s["candidate_idx"], returns=s["returns"],
        sigma_at_candidates=s["sigma_at_candidates"], sigma_init=s["sigma_init"],
        mu_origin=s["mu_origin"], weights=weights, n_eff=n_eff,
        origin_idx=s["origin_idx"], config=s["cfg"], rng=rng_cond,
    )

    bl = s["cfg"].block_length
    np.testing.assert_array_equal(v1_paths[:, :bl], cond_paths[:, :bl])


def test_generate_paths_conditional_diverges_from_v1_after_block_0() -> None:
    """After block 0, the conditional sampler must produce a different
    distribution than v1 — otherwise it isn't actually doing any conditioning."""
    from analog_mc.distances import composite_distance, distances_to_probs

    s = _conditional_setup()
    weights = np.array([1.0, 1.0, 1.0])
    n_eff = 30.0

    distances = composite_distance(s["z_at_origin"], s["z_at_candidates"], weights)
    probs = distances_to_probs(distances, target_n_eff=n_eff)
    v1_paths = generate_paths(
        probs=probs, candidate_indices=s["candidate_idx"], returns=s["returns"],
        sigma_at_candidates=s["sigma_at_candidates"], sigma_init=s["sigma_init"],
        mu_origin=s["mu_origin"], config=s["cfg"], rng=np.random.default_rng(99),
    )
    cond_paths = generate_paths_conditional(
        z_at_origin=s["z_at_origin"], z_at_candidates=s["z_at_candidates"],
        candidate_indices=s["candidate_idx"], returns=s["returns"],
        sigma_at_candidates=s["sigma_at_candidates"], sigma_init=s["sigma_init"],
        mu_origin=s["mu_origin"], weights=weights, n_eff=n_eff,
        origin_idx=s["origin_idx"], config=s["cfg"], rng=np.random.default_rng(99),
    )
    bl = s["cfg"].block_length
    # Block 1+ should differ. Compare cumulative log returns at horizon end —
    # any meaningful re-matching changes the distribution.
    v1_end = v1_paths[:, bl:].sum(axis=1)
    cond_end = cond_paths[:, bl:].sum(axis=1)
    # Different distributions (Kolmogorov via simple statistics).
    assert not np.allclose(np.sort(v1_end), np.sort(cond_end))


def test_generate_paths_conditional_records_ratios_per_block() -> None:
    s = _conditional_setup()
    weights = np.array([1.0, 1.0, 1.0])
    paths, ratios = generate_paths_conditional(
        z_at_origin=s["z_at_origin"], z_at_candidates=s["z_at_candidates"],
        candidate_indices=s["candidate_idx"], returns=s["returns"],
        sigma_at_candidates=s["sigma_at_candidates"], sigma_init=s["sigma_init"],
        mu_origin=s["mu_origin"], weights=weights, n_eff=30.0,
        origin_idx=s["origin_idx"], config=s["cfg"], rng=np.random.default_rng(0),
        record_ratios=True,
    )
    assert ratios.shape == (s["cfg"].n_paths, s["cfg"].n_blocks)
    assert (ratios > 0).all()


def test_forecast_dispatches_to_conditional_when_flag_set() -> None:
    """forecast() should produce different outputs with/without the conditional flag."""
    from analog_mc.features import compute_features
    from analog_mc.simulate import forecast

    rng_seed = np.random.default_rng(0)
    n = 1500
    returns_arr = rng_seed.normal(0.0005, 0.01, size=n)
    returns_s = pd.Series(returns_arr)
    cfg_v1 = Config(
        forecast_horizon=20, block_length=5, n_blocks=4, n_paths=100,
        ewma_halflife=10, zscore_horizons=(20, 50, 100), train_initial_size=600,
        conditional_block_sampling=False,
    )
    cfg_v22 = Config(
        forecast_horizon=20, block_length=5, n_blocks=4, n_paths=100,
        ewma_halflife=10, zscore_horizons=(20, 50, 100), train_initial_size=600,
        conditional_block_sampling=True,
    )
    feats = compute_features(returns_s, halflife=cfg_v1.ewma_halflife, horizons=cfg_v1.zscore_horizons)
    candidate_idx = np.arange(150, 1100, dtype=np.int64)
    v1 = forecast(
        origin_idx=1200, returns=returns_arr, candidate_idx=candidate_idx,
        features=feats, weights=np.array([1.0, 1.0, 1.0]), n_eff=30.0,
        config=cfg_v1, rng=np.random.default_rng(7),
    )
    v22 = forecast(
        origin_idx=1200, returns=returns_arr, candidate_idx=candidate_idx,
        features=feats, weights=np.array([1.0, 1.0, 1.0]), n_eff=30.0,
        config=cfg_v22, rng=np.random.default_rng(7),
    )
    bl = cfg_v1.block_length
    # Block 0 identical (same seed, same probs).
    np.testing.assert_array_equal(v1[:, :bl], v22[:, :bl])
    # Blocks 1+ differ.
    assert not np.allclose(v1[:, bl:], v22[:, bl:])
