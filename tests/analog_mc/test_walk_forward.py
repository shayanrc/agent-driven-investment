"""Tests for analog_mc.walk_forward — orchestration + persistence + resume."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analog_mc.config import Config
from analog_mc.walk_forward import (
    _config_hash,
    _git_commit_hash,
    create_run_dir,
    run_walk_forward,
)


@pytest.fixture
def small_returns_and_config(tmp_path):
    """Series long enough for 2-3 folds with a tiny config."""
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
        weight_grid_resolution=0.5,   # 6 points -> fast
        n_eff_values=(10, 30),
        local_refine_top_k=1,
        nelder_mead_maxiter=3,        # cap for test speed
        runs_dir=str(tmp_path / "runs"),
    )
    return returns, cfg


# ---------------------------------------------------------------------------
# Run directory + meta
# ---------------------------------------------------------------------------


def test_create_run_dir_writes_config_meta_lock(small_returns_and_config) -> None:
    _, cfg = small_returns_and_config
    run_dir = create_run_dir(cfg)
    try:
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "meta.json").exists()
        assert (run_dir / "lock").exists()
        assert (run_dir / "folds").is_dir()
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["config_hash"] == _config_hash(cfg)
        assert meta["finished_at"] is None
    finally:
        # Best-effort cleanup; tmp_path also handles it.
        pass


def test_config_hash_is_stable() -> None:
    a = Config(n_paths=100)
    b = Config(n_paths=100)
    assert _config_hash(a) == _config_hash(b)
    c = Config(n_paths=101)
    assert _config_hash(a) != _config_hash(c)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_run_walk_forward_end_to_end(small_returns_and_config) -> None:
    returns, cfg = small_returns_and_config
    run_dir = run_walk_forward(returns, cfg)

    # Lock should be gone on success.
    assert not (run_dir / "lock").exists()

    # Summary table should exist with one row per fold.
    summary = pd.read_parquet(run_dir / "summary.parquet")
    assert len(summary) >= 2
    assert set(summary.columns) >= {
        "fold_index", "w0", "w1", "w2", "n_eff", "val_crps", "test_crps", "n_test_origins"
    }
    assert (summary["val_crps"] > 0).all()
    assert (summary["test_crps"] > 0).all()
    np.testing.assert_allclose(summary[["w0", "w1", "w2"]].sum(axis=1).to_numpy(), 1.0, atol=1e-6)

    # meta.json should have a finished timestamp + wall seconds.
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["finished_at"] is not None
    assert meta["wall_seconds"] > 0


def test_run_walk_forward_persists_per_fold_artifacts(small_returns_and_config) -> None:
    returns, cfg = small_returns_and_config
    run_dir = run_walk_forward(returns, cfg)

    fold_dirs = sorted((run_dir / "folds").iterdir())
    assert len(fold_dirs) >= 2
    for fd in fold_dirs:
        assert (fd / "summary.json").exists()
        assert (fd / "search.parquet").exists()
        assert (fd / "forecasts.npz").exists()
        summary = json.loads((fd / "summary.json").read_text())
        assert "weights" in summary and len(summary["weights"]) == 3
        assert summary["n_test_origins"] >= 1

        # forecasts.npz shape check
        with np.load(fd / "forecasts.npz") as data:
            paths = data["paths"]
            ratios = data["ratios"]
            realized = data["realized"]
            origin_idx = data["origin_idx"]
        assert paths.dtype == np.float32
        assert paths.shape == (origin_idx.size, cfg.n_paths, cfg.forecast_horizon)
        assert ratios.shape == (origin_idx.size, cfg.n_paths, cfg.n_blocks)
        assert realized.shape == (origin_idx.size, cfg.forecast_horizon)


def test_run_walk_forward_resumes_completed_folds(small_returns_and_config) -> None:
    """A second invocation on the same run dir must reuse persisted folds."""
    returns, cfg = small_returns_and_config
    run_dir = run_walk_forward(returns, cfg)
    summary_first = pd.read_parquet(run_dir / "summary.parquet")
    fold0_mtime = (run_dir / "folds" / "0" / "summary.json").stat().st_mtime

    # Re-run on same dir with resume=True.
    run_walk_forward(returns, cfg, run_dir=run_dir, resume=True)
    summary_second = pd.read_parquet(run_dir / "summary.parquet")
    fold0_mtime_after = (run_dir / "folds" / "0" / "summary.json").stat().st_mtime

    # Per-fold files were not regenerated.
    assert fold0_mtime == fold0_mtime_after
    # Summaries match.
    pd.testing.assert_frame_equal(summary_first, summary_second)


def test_run_walk_forward_invokes_progress_callback(small_returns_and_config) -> None:
    returns, cfg = small_returns_and_config
    seen: list[tuple[int, int]] = []

    def cb(fold_idx, n_folds, outcome):
        seen.append((fold_idx, n_folds))

    run_walk_forward(returns, cfg, progress_callback=cb)
    assert len(seen) >= 2
    # n_folds must be consistent.
    assert len({n for _, n in seen}) == 1


def test_git_commit_hash_returns_string_or_none() -> None:
    h = _git_commit_hash()
    # Either a 40-char SHA or None if not in a git repo. The repo IS git
    # so this should be a hex string.
    assert h is None or (isinstance(h, str) and len(h) == 40)


def test_cli_subcommand_dispatcher() -> None:
    """The `python -m analog_mc` entry routes to walk-forward without warning."""
    from analog_mc.__main__ import SUBCOMMANDS, main

    assert "walk-forward" in SUBCOMMANDS
    # --help short-circuits before doing any real work.
    assert main(["--help"]) == 0
    # Unknown subcommand exits 2.
    assert main(["does-not-exist"]) == 2
