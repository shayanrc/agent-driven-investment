"""Tests for analog_mc.simulate.forecast — single-origin orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analog_mc.config import Config
from analog_mc.features import compute_features
from analog_mc.simulate import eligible_candidates, forecast


@pytest.fixture
def synthetic_setup():
    """A modest synthetic series long enough for the default 200-day z-score."""
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


def test_eligible_candidates_excludes_overlapping_forward_block(synthetic_setup) -> None:
    returns, features, cfg = synthetic_setup
    candidate_idx = np.arange(200, 600, dtype=np.int64)
    origin_idx = 605
    elig = eligible_candidates(candidate_idx, features, origin_idx, cfg)
    # Each eligible d must satisfy d + block_length < origin_idx.
    assert (elig + cfg.block_length < origin_idx).all()
    # Confirm at least the cut-off where d + block_length == origin_idx - 1 is kept.
    assert origin_idx - cfg.block_length - 1 in set(int(i) for i in elig)


def test_eligible_candidates_drops_nan_feature_rows(synthetic_setup) -> None:
    returns, features, cfg = synthetic_setup
    # Indices 0..max(horizon)-1 have NaN z-scores; they should be dropped.
    candidate_idx = np.arange(0, 600, dtype=np.int64)
    elig = eligible_candidates(candidate_idx, features, origin_idx=650, config=cfg)
    # No eligible index can have NaN in any feature column.
    z_cols = [f"zscore_{h}" for h in cfg.zscore_horizons]
    rows = features.iloc[elig][z_cols + ["ewma_vol"]].to_numpy()
    assert not np.isnan(rows).any()


def test_forecast_shape(synthetic_setup) -> None:
    returns, features, cfg = synthetic_setup
    candidate_idx = np.arange(200, 500, dtype=np.int64)
    rng = np.random.default_rng(1)
    paths = forecast(
        origin_idx=600,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=features,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=30.0,
        config=cfg,
        rng=rng,
    )
    assert paths.shape == (cfg.n_paths, cfg.forecast_horizon)
    assert np.isfinite(paths).all()


def test_forecast_is_deterministic_with_seed(synthetic_setup) -> None:
    returns, features, cfg = synthetic_setup
    candidate_idx = np.arange(200, 500, dtype=np.int64)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    r1 = forecast(600, returns.to_numpy(), candidate_idx, features, np.array([1.0, 1.0, 1.0]), 30.0, cfg, rng1)
    r2 = forecast(600, returns.to_numpy(), candidate_idx, features, np.array([1.0, 1.0, 1.0]), 30.0, cfg, rng2)
    np.testing.assert_array_equal(r1, r2)


def test_forecast_probabilities_invariant_to_weight_scale(synthetic_setup) -> None:
    """Distance-to-probability is scale-invariant in weights, so a constant
    rescaling shouldn't change the (deterministic-seed) path distribution."""
    returns, features, cfg = synthetic_setup
    candidate_idx = np.arange(200, 500, dtype=np.int64)
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    w1 = np.array([1.0, 1.0, 1.0])
    w2 = np.array([7.0, 7.0, 7.0])  # rescaled, same direction
    p1 = forecast(600, returns.to_numpy(), candidate_idx, features, w1, 30.0, cfg, rng_a)
    p2 = forecast(600, returns.to_numpy(), candidate_idx, features, w2, 30.0, cfg, rng_b)
    np.testing.assert_allclose(p1, p2)


def test_forecast_rejects_origin_with_nan_features(synthetic_setup) -> None:
    returns, features, cfg = synthetic_setup
    candidate_idx = np.arange(200, 500, dtype=np.int64)
    with pytest.raises(ValueError, match="contain NaN"):
        forecast(
            origin_idx=50,  # too-early origin lacks 100-day z-score history
            returns=returns.to_numpy(),
            candidate_idx=candidate_idx,
            features=features,
            weights=np.array([1.0, 1.0, 1.0]),
            n_eff=10.0,
            config=cfg,
            rng=np.random.default_rng(0),
        )


def test_forecast_rejects_empty_eligible_pool(synthetic_setup) -> None:
    returns, features, cfg = synthetic_setup
    # All candidates fail the forward-block boundary.
    candidate_idx = np.arange(595, 600, dtype=np.int64)  # d+5 >= 600
    with pytest.raises(ValueError, match="No eligible candidates"):
        forecast(
            origin_idx=600,
            returns=returns.to_numpy(),
            candidate_idx=candidate_idx,
            features=features,
            weights=np.array([1.0, 1.0, 1.0]),
            n_eff=2.0,
            config=cfg,
            rng=np.random.default_rng(0),
        )


def test_forecast_caps_n_eff_at_pool_size(synthetic_setup) -> None:
    """If n_eff exceeds the eligible pool, it's silently capped (warning only)."""
    returns, features, cfg = synthetic_setup
    candidate_idx = np.arange(200, 300, dtype=np.int64)
    # Pool ~99; ask for n_eff=200.
    paths = forecast(
        origin_idx=600,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=features,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=200.0,
        config=cfg,
        rng=np.random.default_rng(0),
    )
    assert paths.shape == (cfg.n_paths, cfg.forecast_horizon)


# ---------------------------------------------------------------------------
# v2.1 — trailing-momentum drift
# ---------------------------------------------------------------------------


def _trending_setup(daily_drift: float, lookback: int = 20):
    """Build a long synthetic series with a known per-day drift."""
    rng = np.random.default_rng(0)
    n = 1500
    returns = pd.Series(
        rng.normal(daily_drift, 0.01, size=n),
        index=pd.date_range("2010-01-04", periods=n, freq="B"),
        name="log_return",
    )
    cfg_zero = Config(
        forecast_horizon=20,
        block_length=5,
        n_blocks=4,
        n_paths=300,
        ewma_halflife=10,
        zscore_horizons=(20, 50, 200),
        train_initial_size=600,
        val_size=60,
        test_size=60,
        drift_mode="zero",
        momentum_lookback=lookback,
    )
    cfg_mom = Config(
        forecast_horizon=20,
        block_length=5,
        n_blocks=4,
        n_paths=300,
        ewma_halflife=10,
        zscore_horizons=(20, 50, 200),
        train_initial_size=600,
        val_size=60,
        test_size=60,
        drift_mode="trailing_momentum",
        momentum_lookback=lookback,
        momentum_shrinkage=1.0,  # disable shrinkage so the drift sign is unambiguous
    )
    features_zero = compute_features(
        returns, halflife=cfg_zero.ewma_halflife, horizons=cfg_zero.zscore_horizons,
    )
    features_mom = compute_features(
        returns, halflife=cfg_mom.ewma_halflife, horizons=cfg_mom.zscore_horizons,
        momentum_lookback=cfg_mom.momentum_lookback,
    )
    return returns, features_zero, cfg_zero, features_mom, cfg_mom


def test_trailing_momentum_shifts_median_in_drift_direction() -> None:
    """C7+C10: on a synthetic series with positive drift, trailing-momentum forecasts
    have materially positive median end-cum-return; zero-drift forecasts do not."""
    daily_drift = 0.004  # large enough to dominate noise at 20-day horizon
    returns, feats_zero, cfg_zero, feats_mom, cfg_mom = _trending_setup(daily_drift)
    candidate_idx = np.arange(250, 1200, dtype=np.int64)
    origin = 1250

    paths_zero = forecast(
        origin_idx=origin,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=feats_zero,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=30.0,
        config=cfg_zero,
        rng=np.random.default_rng(1),
    )
    paths_mom = forecast(
        origin_idx=origin,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=feats_mom,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=30.0,
        config=cfg_mom,
        rng=np.random.default_rng(1),
    )

    cum_zero = paths_zero.sum(axis=1)  # end-cumulative log return per path
    cum_mom = paths_mom.sum(axis=1)
    # The drift contribution over 20 steps is ~ daily_drift * 20 = 0.08.
    # Mom forecast median should be materially above zero-drift median.
    assert np.median(cum_mom) > np.median(cum_zero) + 0.04, (
        f"momentum drift failed to lift median: zero={np.median(cum_zero):.4f}, "
        f"mom={np.median(cum_mom):.4f}"
    )
    assert np.median(cum_mom) > 0.03  # positive drift should produce positive median


def test_trailing_momentum_zero_when_drift_is_zero() -> None:
    """Sanity: if the underlying series has no drift, trailing_momentum reduces
    to noise around zero (no systematic bias from the implementation itself)."""
    returns, _, _, feats_mom, cfg_mom = _trending_setup(daily_drift=0.0)
    candidate_idx = np.arange(250, 1200, dtype=np.int64)
    paths_mom = forecast(
        origin_idx=1250,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=feats_mom,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=30.0,
        config=cfg_mom,
        rng=np.random.default_rng(0),
    )
    cum = paths_mom.sum(axis=1)
    # 20-day cumulative log return std ~ 0.01 * sqrt(20) ~ 0.045; median should be
    # within ~one std of zero. Loose tolerance since this is a single origin.
    assert abs(np.median(cum)) < 0.05


def test_trailing_momentum_requires_feature_column() -> None:
    """Clean error when drift_mode=trailing_momentum but the feature bundle
    was built without momentum_lookback (i.e., missing column)."""
    returns, _, _, _, cfg_mom = _trending_setup(daily_drift=0.001)
    # Re-compute features WITHOUT the momentum column.
    feats_no_mom = compute_features(returns, halflife=cfg_mom.ewma_halflife, horizons=cfg_mom.zscore_horizons)
    candidate_idx = np.arange(250, 1200, dtype=np.int64)
    with pytest.raises(ValueError, match="requires features column"):
        forecast(
            origin_idx=1250,
            returns=returns.to_numpy(),
            candidate_idx=candidate_idx,
            features=feats_no_mom,
            weights=np.array([1.0, 1.0, 1.0]),
            n_eff=30.0,
            config=cfg_mom,
            rng=np.random.default_rng(0),
        )


def test_explicit_drift_target_overrides_config() -> None:
    """An explicit drift_target float should bypass the config's drift_mode."""
    returns, feats_zero, cfg_zero, _, _ = _trending_setup(daily_drift=0.0)
    candidate_idx = np.arange(250, 1200, dtype=np.int64)
    # Even though config is drift_mode='zero', passing drift_target=0.01 should
    # shift the median up by ~0.01*20 = 0.2.
    paths = forecast(
        origin_idx=1250,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=feats_zero,
        weights=np.array([1.0, 1.0, 1.0]),
        n_eff=30.0,
        config=cfg_zero,
        rng=np.random.default_rng(0),
        drift_target=0.01,
    )
    assert np.median(paths.sum(axis=1)) > 0.15
