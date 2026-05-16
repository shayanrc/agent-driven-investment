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
