"""V1.3 Option B P7 — end-to-end smoke for default-mode scout artifacts.

Aggregates the P3 + P5 wiring: default mode + scout enabled writes the
scout report to ``WalkForwardResult.scout_report`` AND the scout/ subdir
files when surfaced through ``run_experiment``-style emission (which P5
does post-loop).

Synthetic in-memory data only; no SQLite cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbdt import scout as gbdt_scout
from gbdt import scout_io as gbdt_scout_io
from gbdt.train import walk_forward_train


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


def test_default_mode_scout_report_carries_required_keys():
    """Confirms ``scout_report['scout']`` has the D7.1 field set the runner
    needs to build ``metrics.json::scout``."""
    panel, X, y = _toy_panel(1600, 3, seed=42)
    grid = {
        "max_depth": [3, 4],
        "eta": [0.05, 0.1],
        "colsample_bytree": [0.5, 1.0],
        "min_child_weight": [1, 5],
        "alpha": [0.0, 0.1],
        "subsample": [0.7, 1.0],
        "scale_pos_weight": ["1"],
    }
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 20, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=2,
        scout_spec={"enabled": True, "grid": grid},
        fs_prefit_spec={"enabled": True, "cliff_pct": 0.0},
        callback_mode="default",
    )
    sr = result.scout_report["scout"]
    for key in (
        "enabled", "backend", "n_configs_total", "n_configs_completed",
        "runtime_seconds", "lexicographic_auto_compose",
        "status", "degenerate_sink_fallback",
    ):
        assert key in sr, f"missing key {key!r} in scout_report['scout']"
    # FS-prefit block always present.
    assert "fs_prefit" in result.scout_report
    # Combine block always present.
    assert "combine" in result.scout_report
    cb = result.scout_report["combine"]
    assert cb["status"] in (
        "lex_auto_compose", "degenerate_sink_fallback",
    )


def test_default_mode_scout_results_raw_rows_are_serializable():
    """Each raw scout row in scout_report must be JSON-serializable —
    the runner writes them to scout/scout_results.jsonl in P5."""
    panel, X, y = _toy_panel(1600, 3, seed=43)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 20, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=2,
        scout_spec={"enabled": True, "grid": {"max_depth": [3]}},
        fs_prefit_spec={"enabled": True, "cliff_pct": 0.0},
        callback_mode="default",
    )
    raw = result.scout_report["_scout_results_raw"]
    assert isinstance(raw, list)
    assert len(raw) >= 1
    # All rows serialize to JSON without error.
    for row in raw:
        json.dumps(row)


def test_default_mode_scout_lex_winner_format_matches_combine_decision_schema():
    """The lex auto-compose overlay should be a flat HP dict — the same
    shape combine_decision.json::configs[*]['hp'] uses (D3b.A).
    """
    panel, X, y = _toy_panel(1600, 3, seed=44)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 20, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=2,
        scout_spec={"enabled": True, "grid": {"max_depth": [3, 4]}},
        fs_prefit_spec={"enabled": True, "cliff_pct": 0.0},
        callback_mode="default",
    )
    lex = result.scout_report["scout"]["lexicographic_auto_compose"]
    overlay = lex["hp_overlay"]
    assert isinstance(overlay, dict)
    # Every key/value must be a scalar OR a dict (for CatBoost class_weights);
    # i.e. JSON-serializable.
    json.dumps(overlay)
