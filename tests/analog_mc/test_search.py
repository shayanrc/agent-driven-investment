"""Tests for analog_mc.search — grid + Nelder-Mead refine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analog_mc.config import Config
from analog_mc.data import generate_folds
from analog_mc.features import compute_features
from analog_mc.search import (
    evaluate,
    generate_weight_grid,
    grid_search,
    local_refine,
    run_search,
)


# ---------------------------------------------------------------------------
# generate_weight_grid
# ---------------------------------------------------------------------------


def test_weight_grid_default_resolution_has_66_points() -> None:
    grid = generate_weight_grid(0.1)
    assert grid.shape == (66, 3)


@pytest.mark.parametrize("res,expected", [(0.5, 6), (0.25, 15), (0.1, 66), (0.2, 21)])
def test_weight_grid_point_count(res: float, expected: int) -> None:
    grid = generate_weight_grid(res)
    assert grid.shape == (expected, 3)


def test_weight_grid_rows_sum_to_one() -> None:
    grid = generate_weight_grid(0.1)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)


def test_weight_grid_non_negative() -> None:
    grid = generate_weight_grid(0.1)
    assert (grid >= 0).all()


def test_weight_grid_includes_corners_and_center() -> None:
    grid = generate_weight_grid(0.1)
    rows = {tuple(row.round(6)) for row in grid}
    assert (1.0, 0.0, 0.0) in rows
    assert (0.0, 1.0, 0.0) in rows
    assert (0.0, 0.0, 1.0) in rows
    # The exact (1/3, 1/3, 1/3) isn't on a 0.1 grid; but (0.3, 0.3, 0.4) etc. is.
    assert (0.3, 0.3, 0.4) in rows


def test_weight_grid_rejects_non_dividing_resolution() -> None:
    with pytest.raises(ValueError, match="evenly divide"):
        generate_weight_grid(0.3)
    with pytest.raises(ValueError):
        generate_weight_grid(0.0)
    with pytest.raises(ValueError):
        generate_weight_grid(1.5)


# ---------------------------------------------------------------------------
# evaluate / grid_search / local_refine
# Use a small synthetic setup so tests stay <10s.
# ---------------------------------------------------------------------------


@pytest.fixture
def small_setup():
    rng = np.random.default_rng(0)
    n = 1500
    returns = pd.Series(
        rng.normal(0.0005, 0.01, size=n),
        index=pd.date_range("2010-01-04", periods=n, freq="B"),
    )
    cfg = Config(
        forecast_horizon=20,
        block_length=5,
        n_blocks=4,
        n_paths=50,                  # small for speed
        ewma_halflife=10,
        zscore_horizons=(20, 50, 100),
        train_initial_size=600,
        val_size=30,
        test_size=30,
        weight_grid_resolution=0.25, # 15 points -> faster
        n_eff_values=(10, 25),       # 2 values -> faster
        local_refine_top_k=2,
        nelder_mead_maxiter=8,       # cap iterations for test speed
    )
    features = compute_features(returns, halflife=cfg.ewma_halflife, horizons=cfg.zscore_horizons)
    folds = generate_folds(returns, cfg)
    return returns.to_numpy(), features, folds[0], cfg


def test_evaluate_returns_finite(small_setup) -> None:
    returns, features, fold, cfg = small_setup
    val_crps = evaluate(
        weights=np.array([0.33, 0.33, 0.34]),
        n_eff=25.0,
        fold=fold,
        returns=returns,
        features=features,
        config=cfg,
    )
    assert np.isfinite(val_crps)
    assert val_crps > 0


def test_evaluate_is_deterministic(small_setup) -> None:
    returns, features, fold, cfg = small_setup
    w = np.array([0.5, 0.25, 0.25])
    a = evaluate(w, 25.0, fold, returns, features, cfg)
    b = evaluate(w, 25.0, fold, returns, features, cfg)
    assert a == b  # exact equality — deterministic per-(weights, n_eff, origin) seeding


def test_grid_search_shape_and_sorting(small_setup) -> None:
    returns, features, fold, cfg = small_setup
    df = grid_search(fold, returns, features, cfg)
    # 15 weight points × 2 n_eff values
    assert len(df) == 15 * 2
    assert list(df.columns) == ["w0", "w1", "w2", "n_eff", "val_crps"]
    # Sorted ascending by val_crps.
    assert df["val_crps"].is_monotonic_increasing
    # All non-degenerate evaluations.
    assert np.isfinite(df["val_crps"]).any()


def test_grid_search_weights_on_simplex(small_setup) -> None:
    returns, features, fold, cfg = small_setup
    df = grid_search(fold, returns, features, cfg)
    sums = df[["w0", "w1", "w2"]].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0)
    assert (df[["w0", "w1", "w2"]] >= 0).all().all()


def test_local_refine_returns_finite_simplex_weights(small_setup) -> None:
    returns, features, fold, cfg = small_setup
    df = grid_search(fold, returns, features, cfg)
    top = df.head(cfg.local_refine_top_k)
    weights, n_eff, val_crps, n_evals = local_refine(top, fold, returns, features, cfg)
    assert weights.shape == (3,)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= -1e-9).all()
    assert np.isfinite(val_crps)
    assert val_crps <= top["val_crps"].min() + 1e-9  # at worst matches the grid best
    assert n_evals >= 1


def test_run_search_end_to_end(small_setup) -> None:
    returns, features, fold, cfg = small_setup
    result = run_search(fold, returns, features, cfg)
    assert result.weights.shape == (3,)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.isfinite(result.val_crps)
    assert result.n_eff in set(cfg.n_eff_values)
    # The grid_df has been populated.
    assert len(result.grid_df) == 15 * 2
    # Refined val_crps is no worse than the grid best.
    assert result.val_crps <= result.grid_df["val_crps"].min() + 1e-9


def test_evaluate_handles_origin_with_insufficient_future(small_setup) -> None:
    """Origins whose forecast horizon would run past the data are silently dropped."""
    returns, features, fold, cfg = small_setup
    # Truncate returns so the val origins can't realize the full horizon.
    short_returns = returns[: fold.val_idx[0] + 5]
    val_crps = evaluate(
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
        n_eff=25.0,
        fold=fold,
        returns=short_returns,
        features=features,
        config=cfg,
    )
    # No origin produces a forecast -> +inf sentinel.
    assert val_crps == float("inf")
