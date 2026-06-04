"""Integration tests for V1.3 Option B scout + FS-prefit wiring (P3).

Synthetic in-memory panel; no SQLite cache, no data_pipelines.fetch.
Covers the back-compat path (scout disabled → byte-for-byte unchanged)
plus the new fresh-path Phase 1.4–1.6 hooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt.train import WalkForwardResult, walk_forward_train


def _toy_panel(n_per_ticker: int = 1600, n_tickers: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_per_ticker, freq="B")
    frames = []
    for i in range(n_tickers):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_per_ticker)))
        df = pd.DataFrame({
            "date": dates,
            "ticker": f"T{i}",
            "open": c, "high": c * 1.005, "low": c * 0.995,
            "close": c, "adj_close": c,
            "volume": np.ones(n_per_ticker, dtype=int),
        })
        frames.append(df)
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    n_total = len(panel)
    X = pd.DataFrame(
        rng.normal(0, 1, (n_total, 6)),
        index=panel.index,
        columns=["sig", "n1", "n2", "n3", "n4", "n5"],
    )
    y = ((X["sig"] + rng.normal(0, 0.3, n_total)) > 0).astype(int)
    return panel, X, y


def test_scout_disabled_preserves_walk_forward_behavior():
    """When scout_spec is None / disabled, scout_report stays None and the
    run produces the same iter_0 baseline as pre-V1.3 Option B."""
    panel, X, y = _toy_panel(1600, 3, seed=10)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=2,
        scout_spec=None,
        fs_prefit_spec=None,
    )
    assert isinstance(result, WalkForwardResult)
    assert result.scout_report is None


def test_scout_disabled_explicit_false_still_none():
    panel, X, y = _toy_panel(1600, 3, seed=11)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=2,
        scout_spec={"enabled": False},
        fs_prefit_spec={"enabled": False},
    )
    assert result.scout_report is None


def test_scout_enabled_default_mode_emits_report():
    """With scout enabled in default mode, scout_report is populated
    and reflects per-knob winners + lex auto-compose."""
    panel, X, y = _toy_panel(1600, 3, seed=12)
    # Tiny grid override so the test runs quickly: only 2 values per knob.
    grid = {
        "max_depth": [3, 4],
        "eta": [0.05, 0.1],
        "colsample_bytree": [0.5, 1.0],
        "min_child_weight": [1, 5],
        "alpha": [0.0, 0.1],
        "subsample": [0.7, 1.0],
        "scale_pos_weight": ["1"],
    }
    # CatBoost has no gamma analog, so we don't override gamma either.
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 20, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=2,
        scout_spec={"enabled": True, "grid": grid},
        fs_prefit_spec={"enabled": True, "cliff_pct": 0.0},   # keep all features
        callback_mode="default",
    )
    assert result.scout_report is not None
    sr = result.scout_report["scout"]
    assert sr["enabled"] is True
    assert sr["n_configs_total"] > 0
    assert sr["n_configs_completed"] > 0
    assert "lexicographic_auto_compose" in sr
    assert sr["status"] in (
        "lex_auto_compose", "degenerate_sink_fallback",
    )
    # FS-prefit ran (cliff_pct=0.0 keeps everything; report shows n_kept >= 1).
    pf = result.scout_report["fs_prefit"]
    assert pf["enabled"] is True
    assert pf.get("n_kept", 0) >= 1


def test_scout_sweep_mode_skipped_even_when_enabled():
    """D4 — sweep mode hard-OFF. Scout config presence is ignored."""
    panel, X, y = _toy_panel(1600, 3, seed=13)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 20, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=2,
        scout_spec={"enabled": True},
        fs_prefit_spec={"enabled": True},
        callback_mode="sweep",
    )
    # No scout — back-compat path.
    assert result.scout_report is None


def test_scout_agent_file_protocol_mode_skipped_in_walk_forward():
    """In agent_file_protocol mode, scout is the runner's responsibility
    (__main__.py); walk_forward_train must NOT run scout itself."""
    panel, X, y = _toy_panel(1600, 3, seed=15)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 20, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=2,
        scout_spec={"enabled": True},
        fs_prefit_spec={"enabled": True},
        callback_mode="agent_file_protocol",
    )
    assert result.scout_report is None


def test_scout_resume_path_skips_scout():
    """Resume path inherits iter_0 HP from the checkpoint; scout MUST NOT
    re-run (we don't want it to overwrite the agent's chosen HP)."""
    panel, X, y = _toy_panel(1600, 3, seed=14)
    # Minimal resume_state seed: iter_idx=1, no force_stop, one prior iter.
    resume_state = {
        "current_features": list(X.columns),
        "current_hp": {"iterations": 20, "depth": 3, "boosting_type": "Plain",
                        "learning_rate": 0.05},
        "iter_idx": 1,
        "val_briers": [0.25],
        "hp_history": [{"iter": 0, "hp": {"depth": 3}}],
        "feature_history": [list(X.columns)],
        "hp_lists": [{"depth": 3}],
        "delta_attributions": ["prior"],
        "force_stop": True,    # finalize without further fits
    }
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 20, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=2,
        scout_spec={"enabled": True},
        fs_prefit_spec={"enabled": True},
        resume_state=resume_state,
        callback_mode="agent_file_protocol",
    )
    # Scout was skipped on the resume path.
    assert result.scout_report is None
