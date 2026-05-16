"""Tests for analog_mc.diagnostics — loaders + plots + decision rules.

Uses a tiny walk-forward run as a fixture so every diagnostic exercises real
persisted artifacts.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from analog_mc.config import Config
from analog_mc.diagnostics import (
    acf_comparison,
    aggregate_crps_overall,
    aggregate_crps_per_fold,
    aggregate_crps_per_step,
    aggregate_crps_per_vol_regime,
    clip_hit_summary,
    concatenate_oos,
    conditional_pit_by_vol_regime,
    crps_surface_plot,
    decision_rules,
    fixed_weight_baseline_crps,
    generate_report,
    global_pit_histogram,
    load_run,
    per_step_crps,
    pit_ranks,
    reliability_diagram,
    weight_trajectory_plot,
)
from analog_mc.walk_forward import run_walk_forward


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One small walk-forward run shared across all diagnostic tests."""
    rng = np.random.default_rng(0)
    n = 1500
    returns = pd.Series(
        rng.normal(0.0005, 0.01, size=n),
        index=pd.date_range("2010-01-04", periods=n, freq="B"),
        name="log_return",
    )
    cfg = Config(
        forecast_horizon=20,
        block_length=5,
        n_blocks=4,
        n_paths=50,
        ewma_halflife=10,
        zscore_horizons=(20, 50, 100),
        train_initial_size=600,
        val_size=30,
        test_size=30,
        weight_grid_resolution=0.5,
        n_eff_values=(10, 30),
        local_refine_top_k=1,
        nelder_mead_maxiter=3,
        runs_dir=str(tmp_path_factory.mktemp("runs")),
    )
    run_dir = run_walk_forward(returns, cfg)
    return returns, cfg, run_dir


# ---------------------------------------------------------------------------
# Loader + primitives
# ---------------------------------------------------------------------------


def test_load_run_returns_populated_artifacts(small_run) -> None:
    _, _, run_dir = small_run
    run = load_run(run_dir)
    assert run.n_folds >= 2
    assert not run.summary.empty
    for fold in run.folds:
        assert fold.paths.ndim == 3
        assert fold.ratios.ndim == 3
        assert fold.realized.ndim == 2
        assert fold.origin_idx.ndim == 1
        assert "val_crps" in fold.summary
        assert not fold.search_grid.empty


def test_load_run_raises_on_incomplete(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path)


def test_load_run_tolerates_missing_summary_parquet(small_run) -> None:
    """Simulate an in-progress run: delete summary.parquet but leave per-fold artifacts.

    load_run must rebuild the summary table from per-fold summary.json files so
    the dashboard's diagnostics view still works on a run that's still going.
    """
    _, _, run_dir = small_run
    parquet = run_dir / "summary.parquet"
    backup = parquet.read_bytes()
    try:
        parquet.unlink()
        run = load_run(run_dir)
        assert run.n_folds >= 2
        assert set(run.summary.columns) >= {
            "fold_index", "w0", "w1", "w2", "n_eff", "val_crps", "test_crps", "n_test_origins"
        }
        # All listed folds have actual artifacts.
        assert len(run.folds) == len(run.summary)
    finally:
        parquet.write_bytes(backup)


def test_load_run_raises_when_no_folds_complete(tmp_path) -> None:
    """If config.yaml is present but no folds have completed, load_run raises clearly."""
    cfg_dir = tmp_path / "20260101T000000Z"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text("ticker: X\n")
    (cfg_dir / "folds").mkdir()
    # No completed folds.
    from analog_mc.config import Config
    Config().to_yaml(cfg_dir / "config.yaml")
    with pytest.raises(FileNotFoundError, match="No completed folds"):
        load_run(cfg_dir)


def test_pit_ranks_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    paths = rng.normal(0, 1, size=(5, 200, 8))
    realized = rng.normal(0, 1, size=(5, 8))
    ranks = pit_ranks(paths, realized)
    assert ranks.shape == (5, 8)
    assert ((ranks >= 0) & (ranks <= 1)).all()


def test_pit_ranks_uniform_when_realized_drawn_from_forecast() -> None:
    """If realized is drawn from the same Gaussian as the forecast, PIT must be ~uniform."""
    rng = np.random.default_rng(0)
    n_origins = 500
    paths = rng.normal(0, 1, size=(n_origins, 500, 1))
    realized = rng.normal(0, 1, size=(n_origins, 1))
    ranks = pit_ranks(paths, realized).flatten()
    # Roughly uniform: mean ~0.5, var ~1/12, and counts in bins of width 0.1 should be
    # near n/10 (loose tolerance because of finite samples).
    assert 0.45 < ranks.mean() < 0.55
    counts, _ = np.histogram(ranks, bins=10, range=(0, 1))
    # No bin should be much further than ±35% from the expected count.
    expected = n_origins / 10
    assert np.abs(counts - expected).max() < expected * 0.45


def test_per_step_crps_shape(small_run) -> None:
    _, _, run_dir = small_run
    run = load_run(run_dir)
    fold = run.folds[0]
    crps = per_step_crps(fold.paths, fold.realized)
    assert crps.shape == (fold.paths.shape[0], fold.paths.shape[2])
    assert (crps >= 0).all()


def test_concatenate_oos(small_run) -> None:
    _, _, run_dir = small_run
    run = load_run(run_dir)
    paths, realized, origin_idx = concatenate_oos(run.folds)
    total = sum(f.paths.shape[0] for f in run.folds)
    assert paths.shape[0] == total
    assert realized.shape[0] == total
    assert origin_idx.size == total


# ---------------------------------------------------------------------------
# Plots — smoke test that figures render without error
# ---------------------------------------------------------------------------


def test_weight_trajectory_plot_runs(small_run) -> None:
    _, _, run_dir = small_run
    fig = weight_trajectory_plot(load_run(run_dir))
    assert fig is not None
    assert len(fig.axes) == 1


def test_crps_surface_plot_runs(small_run) -> None:
    _, _, run_dir = small_run
    fig = crps_surface_plot(load_run(run_dir), fold_index=0)
    assert fig is not None


def test_global_pit_histogram_runs(small_run) -> None:
    _, _, run_dir = small_run
    fig = global_pit_histogram(load_run(run_dir))
    assert fig is not None


def test_conditional_pit_by_vol_regime_runs(small_run) -> None:
    returns, _, run_dir = small_run
    fig = conditional_pit_by_vol_regime(load_run(run_dir), returns)
    assert fig is not None
    assert len(fig.axes) == 3


def test_reliability_diagram_runs(small_run) -> None:
    _, _, run_dir = small_run
    fig = reliability_diagram(load_run(run_dir))
    assert fig is not None


def test_acf_comparison_runs(small_run) -> None:
    returns, _, run_dir = small_run
    fig = acf_comparison(load_run(run_dir), returns)
    assert fig is not None


def test_clip_hit_summary_runs(small_run) -> None:
    _, _, run_dir = small_run
    fig = clip_hit_summary(load_run(run_dir))
    assert fig is not None


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------


def test_decision_rules_structure(small_run) -> None:
    returns, _, run_dir = small_run
    rules = decision_rules(load_run(run_dir), returns, fixed_baseline=None)
    expected_keys = {
        "sloped_global_pit",
        "u_shaped_high_vol_pit",
        "acf_seam_degradation",
        "clip_hit_excessive",
    }
    assert set(rules) >= expected_keys
    for name, body in rules.items():
        assert {"fired", "metric", "recommendation"} <= set(body)
        assert isinstance(body["fired"], (bool, np.bool_))


def test_decision_rules_with_baseline(small_run) -> None:
    returns, cfg, run_dir = small_run
    baseline = fixed_weight_baseline_crps(returns, cfg)
    rules = decision_rules(load_run(run_dir), returns, fixed_baseline=baseline)
    assert "fixed_weight_close_to_tuned" in rules
    assert np.isfinite(rules["fixed_weight_close_to_tuned"]["metric"])


# ---------------------------------------------------------------------------
# Aggregate report (Stage 10)
# ---------------------------------------------------------------------------


def test_aggregate_crps_overall(small_run) -> None:
    _, _, run_dir = small_run
    out = aggregate_crps_overall(load_run(run_dir))
    assert {"mean_crps", "median_crps", "n_origin_step_pairs"} <= set(out)
    assert out["mean_crps"] > 0
    assert out["n_origin_step_pairs"] > 0


def test_aggregate_crps_per_fold(small_run) -> None:
    _, _, run_dir = small_run
    df = aggregate_crps_per_fold(load_run(run_dir))
    assert {"fold_index", "val_crps", "test_crps"} <= set(df.columns)
    assert (df["test_crps"] > 0).all()


def test_aggregate_crps_per_step(small_run) -> None:
    _, cfg, run_dir = small_run
    df = aggregate_crps_per_step(load_run(run_dir))
    assert len(df) == cfg.forecast_horizon
    assert df["step"].tolist() == list(range(1, cfg.forecast_horizon + 1))
    assert (df["mean_crps"] > 0).all()


def test_aggregate_crps_per_vol_regime(small_run) -> None:
    returns, _, run_dir = small_run
    df = aggregate_crps_per_vol_regime(load_run(run_dir), returns)
    assert list(df["regime"]) == ["low_vol", "mid_vol", "high_vol"]
    # At least one regime must have origins.
    assert (df["n_origins"] > 0).any()


def test_generate_report_one_stop(small_run) -> None:
    returns, cfg, run_dir = small_run
    baseline = fixed_weight_baseline_crps(returns, cfg)
    rep = generate_report(load_run(run_dir), returns, fixed_baseline=baseline)
    assert set(rep) == {"overall", "per_fold", "per_step", "per_regime", "fixed_baseline", "decision_rules"}
    assert isinstance(rep["per_fold"], pd.DataFrame)
    assert isinstance(rep["per_step"], pd.DataFrame)
    assert isinstance(rep["per_regime"], pd.DataFrame)
    assert rep["fixed_baseline"] is baseline
    assert "fixed_weight_close_to_tuned" in rep["decision_rules"]
