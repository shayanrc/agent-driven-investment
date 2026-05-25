"""Stage 8 — report renderer + CLI atom smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gbdt.diagnostics import DiagnosticBundle
from gbdt.report import emit_figures, render_report


def _toy_bundle(iter_idx=0):
    return DiagnosticBundle(
        iter=iter_idx,
        hp={"iterations": 100, "depth": 6, "boosting_type": "Plain",
            "l2_leaf_reg": 3.0, "learning_rate": 0.05},
        features=["a", "b", "c"], n_features=3,
        train_brier=0.20 - 0.01 * iter_idx,
        val_brier=0.22 - 0.005 * iter_idx,
        train_val_gap=0.02,
        eval_brier_provisional=0.23,
        spiegelhalter_z=1.0, spiegelhalter_p=0.32,
        reliability={"n_bins": 10, "points": []},
        positive_prevalence_val=0.4, positive_recall_val=0.5,
        early_stop_iteration=80, iteration_cap_hit=False,
        importance_native={"a": 1.0, "b": 0.5, "c": 0.1},
        importance_permutation=None, top_feature_correlation={},
        learning_curve={"learn_BrierScore": [0.3, 0.25, 0.22],
                         "validation_BrierScore": [0.32, 0.28, 0.25]},
        rationale=f"iteration {iter_idx}",
    )


def _toy_predictions():
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "ticker": ["A"] * n,
        "p_raw": p, "p_calibrated": p, "y_true": y,
    })
    return {"train": df.iloc[:100], "val": df.iloc[100:140],
             "eval": df.iloc[140:180], "test": df.iloc[180:]}


def test_emit_figures_writes_pngs(tmp_path):
    iterations = [_toy_bundle(0), _toy_bundle(1)]
    preds = _toy_predictions()
    paths = emit_figures(tmp_path, iterations, preds)
    assert any(p.name == "reliability_diagram.png" for p in paths)
    assert any(p.name == "feature_importance_final.png" for p in paths)
    assert any(p.name == "train_val_gap_history.png" for p in paths)
    assert any(p.name.startswith("learning_curve_iter_") for p in paths)
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_render_report_writes_markdown(tmp_path):
    spec = {
        "target": {"universe": "nifty50", "direction": "up",
                    "threshold_pct": 10, "horizon_days": 20},
    }
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    metrics = {
        "experiment_name": "test_exp",
        "data": {"n_tickers_in_universe": 50, "n_tickers_used": 48,
                  "tickers_excluded": ["JIOFIN", "MAXHEALTH"],
                  "n_rows_train": 38400, "n_rows_val": 19200,
                  "n_rows_eval": 9600, "n_rows_test": 4800,
                  "positive_prevalence_train": 0.42,
                  "positive_prevalence_eval": 0.39},
        "loop": {"n_iterations_run": 3, "best_iteration": 1,
                  "inner_stop_signal": "plateau"},
        "calibration": {"method": "conditional_isotonic", "decision": "isotonic",
                          "spiegelhalter_z": 2.5, "spiegelhalter_p": 0.012},
        "headline_eval": {"brier": 0.23, "brier_baseline_baserate": 0.24,
                           "brier_improvement_vs_baseline": 0.01,
                           "log_loss": 0.65, "roc_auc": 0.61},
        "headline_test": {"brier": 0.24, "brier_baseline_baserate": 0.24,
                           "brier_improvement_vs_baseline": 0.00,
                           "log_loss": 0.66, "roc_auc": 0.60},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    with open(tmp_path / "iterations.jsonl", "w") as f:
        for b in [_toy_bundle(0), _toy_bundle(1)]:
            f.write(json.dumps(b.to_dict()) + "\n")
    out = render_report(tmp_path)
    assert out.exists()
    md = out.read_text()
    assert "# gbdt experiment" in md
    assert "## Spec" in md
    assert "## Calibration" in md
    assert "## Headline metrics" in md
    assert "## Per-experiment verdict" in md


def test_load_spec_validates_and_merges(tmp_path):
    """Quick spec validation smoke test (the validation lives in __main__)."""
    from gbdt.__main__ import load_spec

    # Defaults
    defaults = {
        "split": {"train_rows": 800, "val_rows": 400, "eval_rows": 200,
                   "test_rows": 100, "min_rows_per_ticker": 1600},
        "backend": {"library": "catboost", "calibration_method": "conditional_isotonic",
                     "fs_hp_loop": {"max_iterations": 8}},
        "random_seed": 42,
    }
    (tmp_path / "default.yaml").write_text(yaml.safe_dump(defaults))

    spec = {"target": {"universe": "nifty50", "direction": "up",
                         "threshold_pct": 10, "horizon_days": 20}}
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))

    merged = load_spec(spec_path, default_path=tmp_path / "default.yaml")
    assert merged["target"]["universe"] == "nifty50"
    assert merged["backend"]["fs_hp_loop"]["max_iterations"] == 8


def test_load_spec_rejects_bad_direction(tmp_path):
    from gbdt.__main__ import load_spec

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(
        {"target": {"universe": "nifty50", "direction": "sideways",
                     "threshold_pct": 10, "horizon_days": 20}}
    ))
    with pytest.raises(ValueError, match="direction"):
        load_spec(spec_path, default_path=tmp_path / "nope.yaml")


def test_load_spec_rejects_bad_calibration(tmp_path):
    from gbdt.__main__ import load_spec

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(
        {"target": {"universe": "nifty50", "direction": "up",
                     "threshold_pct": 10, "horizon_days": 20},
          "backend": {"calibration_method": "nope"}}
    ))
    with pytest.raises(ValueError, match="calibration_method"):
        load_spec(spec_path, default_path=tmp_path / "nope.yaml")
