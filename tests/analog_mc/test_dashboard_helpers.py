"""Smoke tests for dashboard helper functions.

Full Streamlit views require the streamlit runtime; this module verifies the
pure-Python helpers and that view modules import cleanly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analog_mc.config import Config
from analog_mc.walk_forward import run_walk_forward


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 1200
    returns = pd.Series(
        rng.normal(0.0005, 0.01, size=n),
        index=pd.date_range("2010-01-04", periods=n, freq="B"),
    )
    cfg = Config(
        forecast_horizon=20,
        block_length=5,
        n_blocks=4,
        n_paths=30,
        ewma_halflife=10,
        zscore_horizons=(20, 50, 100),
        train_initial_size=600,
        val_size=30,
        test_size=30,
        weight_grid_resolution=0.5,
        n_eff_values=(10,),
        local_refine_top_k=1,
        nelder_mead_maxiter=2,
        runs_dir=str(tmp_path_factory.mktemp("runs")),
    )
    run_dir = run_walk_forward(returns, cfg)
    return Path(cfg.runs_dir), run_dir


def test_diagnostics_view_helpers(small_run) -> None:
    runs_root, run_dir = small_run
    from dashboards.analog_mc.views.diagnostics import _list_runs, _run_status

    runs = _list_runs(runs_root)
    assert run_dir in runs
    assert _run_status(run_dir) == "complete"


def test_run_experiment_view_helpers(small_run) -> None:
    runs_root, run_dir = small_run
    from dashboards.analog_mc.views.run_experiment import (
        _list_active_runs, _list_all_runs, _progress_summary,
    )

    # Completed run -> not active.
    assert run_dir not in _list_active_runs(runs_root)
    # But listed in all-runs.
    assert run_dir in _list_all_runs(runs_root)

    n_done, summaries = _progress_summary(run_dir)
    assert n_done > 0
    assert len(summaries) == n_done
    assert "test_crps" in summaries[0]


def test_view_modules_import_cleanly() -> None:
    """The three view modules and the entry points must import without side effects."""
    import dashboards.analog_mc.views.config_editor  # noqa: F401
    import dashboards.analog_mc.views.diagnostics  # noqa: F401
    import dashboards.analog_mc.views.run_experiment  # noqa: F401
    import dashboards.analog_mc.app  # noqa: F401
    import dashboards.app as launcher

    modules = launcher._discover_modules()
    assert "analog_mc" in modules
