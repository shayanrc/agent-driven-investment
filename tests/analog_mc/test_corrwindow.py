"""Tests for A2.1 — correlation-window matcher distance.

Spec: docs/analog_mc/experiments/_a2_design.md (A2.1). Covers causality,
vectorization, identity-on-self, edge cases, and forecast() integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analog_mc.config import Config
from analog_mc.distances_corrwindow import (
    corrwindow_distance,
    _window_zscores,
)
from analog_mc.features import compute_features
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
    features = compute_features(
        returns, halflife=cfg.ewma_halflife, horizons=cfg.zscore_horizons
    )
    return returns, features, cfg


# ----------------------------------------------------------------------------
# corrwindow_distance core behaviour
# ----------------------------------------------------------------------------


def test_distance_to_self_is_floor(synthetic_returns) -> None:
    """Distance from origin to itself = 1 - |corr| = 0, clipped to epsilon."""
    origin = 100
    d = corrwindow_distance(
        synthetic_returns, origin, np.array([origin]), window_length=20, epsilon=0.05
    )
    assert d.shape == (1,)
    assert d[0] == pytest.approx(0.05, abs=1e-12)


def test_anti_correlated_self_is_floor(synthetic_returns) -> None:
    """A candidate whose window is the negation of the target → |corr|=1, d=ε."""
    origin = 100
    L = 20
    # Construct a returns array where window at index 600 = -window at index 100.
    returns = synthetic_returns.copy()
    returns[600 - L + 1 : 600 + 1] = -synthetic_returns[origin - L + 1 : origin + 1]
    d = corrwindow_distance(returns, origin, np.array([600]), window_length=L, epsilon=0.05)
    assert d[0] == pytest.approx(0.05, abs=1e-12)


def test_uncorrelated_random_far(synthetic_returns) -> None:
    """Random independent windows → correlations near 0 → distance near 1."""
    rng = np.random.default_rng(42)
    L = 100  # longer window: corr more concentrated near 0
    n = synthetic_returns.size
    cands = rng.choice(np.arange(L, n - L), size=20, replace=False)
    d = corrwindow_distance(synthetic_returns, 400, cands, window_length=L)
    # All distances should be > 0.5 on independent random data (|corr| < 0.5).
    assert (d > 0.4).mean() > 0.7, "Most independent windows should have |corr| < 0.6"


def test_distance_is_bounded_below_by_epsilon(synthetic_returns) -> None:
    """Epsilon floor must apply regardless of correlation."""
    origin = 100
    cands = np.array([100, 150, 200])
    d = corrwindow_distance(synthetic_returns, origin, cands, window_length=20, epsilon=0.1)
    assert (d >= 0.1).all()


def test_window_start_negative_raises(synthetic_returns) -> None:
    """Eligibility filter is caller's job; passing too-early candidates errors."""
    with pytest.raises(ValueError, match="window start < 0"):
        # origin is fine (300 - 20 + 1 >= 0); candidate at 10 is too early.
        corrwindow_distance(
            synthetic_returns, 300, np.array([10]), window_length=20
        )


def test_origin_too_early_raises(synthetic_returns) -> None:
    with pytest.raises(ValueError, match="insufficient for window_length"):
        corrwindow_distance(
            synthetic_returns, 5, np.array([100]), window_length=20
        )


def test_window_length_too_small_raises(synthetic_returns) -> None:
    with pytest.raises(ValueError, match="window_length"):
        corrwindow_distance(
            synthetic_returns, 100, np.array([200]), window_length=1
        )


def test_zero_std_window_yields_distance_one(synthetic_returns) -> None:
    """Constant window (zero std) → correlation undefined; we set z=0 → corr=0
    → distance=1."""
    returns = synthetic_returns.copy()
    L = 20
    # Make the window at index 600 a flat constant.
    returns[600 - L + 1 : 600 + 1] = 0.001
    d = corrwindow_distance(returns, 100, np.array([600]), window_length=L, epsilon=0.0)
    assert d[0] == pytest.approx(1.0, abs=1e-12)


def test_causality_origin_perturbation(synthetic_returns) -> None:
    """C1: only `returns[origin-L+1 : origin+1]` and `returns[cand-L+1 : cand+1]`
    matter. Perturbing returns OUTSIDE both windows must not change distance."""
    origin = 200
    L = 20
    cand = 100
    base = corrwindow_distance(synthetic_returns, origin, np.array([cand]), window_length=L)

    modified = synthetic_returns.copy()
    # Perturb everything outside [origin-L+1, origin+1] ∪ [cand-L+1, cand+1].
    mask = np.ones(len(modified), dtype=bool)
    mask[origin - L + 1 : origin + 1] = False
    mask[cand - L + 1 : cand + 1] = False
    modified[mask] += 1000.0  # massive perturbation

    perturbed = corrwindow_distance(modified, origin, np.array([cand]), window_length=L)
    np.testing.assert_allclose(base, perturbed, atol=1e-12)


def test_vectorized_matches_loop(synthetic_returns) -> None:
    """Compute distance one candidate at a time and compare to vectorized call."""
    L = 30
    origin = 400
    cands = np.array([100, 200, 300, 500, 700])
    vec = corrwindow_distance(synthetic_returns, origin, cands, window_length=L, epsilon=0.0)
    loop = np.array([
        corrwindow_distance(synthetic_returns, origin, np.array([c]), window_length=L, epsilon=0.0)[0]
        for c in cands
    ])
    np.testing.assert_allclose(vec, loop, atol=1e-12)


def test_window_zscores_norm_property(synthetic_returns) -> None:
    """Each row of _window_zscores should have variance=1 (population)."""
    L = 30
    cands = np.array([100, 200, 300])
    z = _window_zscores(synthetic_returns, cands, L)
    # Population std (ddof=0) along axis 1 should be ≈ 1.
    np.testing.assert_allclose(z.std(axis=1, ddof=0), 1.0, atol=1e-10)
    # And means ≈ 0.
    np.testing.assert_allclose(z.mean(axis=1), 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# Integration with forecast()
# ----------------------------------------------------------------------------


def test_forecast_corrwindow_smoke(synthetic_setup) -> None:
    """forecast() must run cleanly under matcher_distance='corrwindow' and
    produce finite paths."""
    returns, features, cfg_default = synthetic_setup
    cfg = Config(**{
        **cfg_default.to_dict(),
        "matcher_distance": "corrwindow",
        "corrwindow_length": 10,
    })
    candidate_idx = np.arange(200, 500, dtype=np.int64)
    rng = np.random.default_rng(7)
    paths = forecast(
        origin_idx=600,
        returns=returns.to_numpy(),
        candidate_idx=candidate_idx,
        features=features,
        weights=np.array([1.0, 1.0, 1.0]),  # should be ignored
        n_eff=30.0,
        config=cfg,
        rng=rng,
    )
    assert paths.shape == (cfg.n_paths, cfg.forecast_horizon)
    assert np.isfinite(paths).all()


def test_forecast_corrwindow_weights_invariant(synthetic_setup) -> None:
    """Under corrwindow, the weights argument must be ignored — different
    weight vectors should produce identical paths (with same rng seed)."""
    returns, features, cfg_default = synthetic_setup
    cfg = Config(**{
        **cfg_default.to_dict(),
        "matcher_distance": "corrwindow",
        "corrwindow_length": 10,
    })
    candidate_idx = np.arange(200, 500, dtype=np.int64)
    p1 = forecast(
        origin_idx=600, returns=returns.to_numpy(), candidate_idx=candidate_idx,
        features=features, weights=np.array([1.0, 0.0, 0.0]), n_eff=30.0,
        config=cfg, rng=np.random.default_rng(11),
    )
    p2 = forecast(
        origin_idx=600, returns=returns.to_numpy(), candidate_idx=candidate_idx,
        features=features, weights=np.array([0.0, 0.0, 1.0]), n_eff=30.0,
        config=cfg, rng=np.random.default_rng(11),
    )
    np.testing.assert_array_equal(p1, p2)


def test_forecast_corrwindow_disables_conditional(synthetic_setup) -> None:
    """When matcher_distance='corrwindow' AND conditional_block_sampling=True,
    the implementation must fall back to generate_paths (non-conditional). The
    sanity check: paths under (corrwindow, conditional=True) must equal paths
    under (corrwindow, conditional=False) with the same seed."""
    returns, features, cfg_default = synthetic_setup
    base = {**cfg_default.to_dict(), "matcher_distance": "corrwindow", "corrwindow_length": 10}
    cfg_cond = Config(**{**base, "conditional_block_sampling": True})
    cfg_noncond = Config(**{**base, "conditional_block_sampling": False})

    candidate_idx = np.arange(200, 500, dtype=np.int64)
    p_cond = forecast(
        origin_idx=600, returns=returns.to_numpy(), candidate_idx=candidate_idx,
        features=features, weights=np.array([1.0, 1.0, 1.0]), n_eff=30.0,
        config=cfg_cond, rng=np.random.default_rng(13),
    )
    p_noncond = forecast(
        origin_idx=600, returns=returns.to_numpy(), candidate_idx=candidate_idx,
        features=features, weights=np.array([1.0, 1.0, 1.0]), n_eff=30.0,
        config=cfg_noncond, rng=np.random.default_rng(13),
    )
    np.testing.assert_array_equal(p_cond, p_noncond)


def test_forecast_corrwindow_differs_from_euclidean(synthetic_setup) -> None:
    """Sanity: corrwindow paths should differ from weighted_euclidean paths
    (otherwise the wiring is broken)."""
    returns, features, cfg_default = synthetic_setup
    cfg_eu = Config(**{**cfg_default.to_dict(), "matcher_distance": "weighted_euclidean"})
    cfg_cw = Config(**{**cfg_default.to_dict(), "matcher_distance": "corrwindow", "corrwindow_length": 10})

    candidate_idx = np.arange(200, 500, dtype=np.int64)
    p_eu = forecast(
        origin_idx=600, returns=returns.to_numpy(), candidate_idx=candidate_idx,
        features=features, weights=np.array([1.0, 1.0, 1.0]), n_eff=30.0,
        config=cfg_eu, rng=np.random.default_rng(17),
    )
    p_cw = forecast(
        origin_idx=600, returns=returns.to_numpy(), candidate_idx=candidate_idx,
        features=features, weights=np.array([1.0, 1.0, 1.0]), n_eff=30.0,
        config=cfg_cw, rng=np.random.default_rng(17),
    )
    assert not np.array_equal(p_eu, p_cw)
