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


# ---------------------------------------------------------------------------
# Issue #30 — spec.yaml snapshot must be the per-experiment spec only
# ---------------------------------------------------------------------------


def test_load_spec_stashes_per_experiment_authored_content(tmp_path):
    """``load_spec`` must preserve the on-disk spec under
    ``__per_experiment_spec__`` so the runner can write a snapshot that
    isn't polluted by defaults (issue #30).
    """
    from gbdt.__main__ import load_spec

    defaults = {
        "universes": {
            "nifty50": {"source": "x", "index_ticker": "NIFTY:50",
                         "annualization_factor": 250},
            "nasdaq100": {"source": "y", "index_ticker": "INDEX:^NDX",
                           "annualization_factor": 252},
        },
        "split": {"train_rows": 800, "val_rows": 400, "eval_rows": 200,
                   "test_rows": 100, "min_rows_per_ticker": 1600},
        "backend": {"library": "catboost",
                     "calibration_method": "conditional_isotonic",
                     "fs_hp_loop": {"max_iterations": 8}},
        "random_seed": 42,
    }
    (tmp_path / "default.yaml").write_text(yaml.safe_dump(defaults))
    spec = {
        "target": {"universe": "nasdaq100", "direction": "up",
                    "threshold_pct": 10, "horizon_days": 100,
                    "max_drawdown": 0.05},
        "backend": {"fs_hp_loop": {"max_iterations": 3}},
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))

    merged = load_spec(spec_path, default_path=tmp_path / "default.yaml")
    # The merged spec carries the snapshot under an internal key.
    assert "__per_experiment_spec__" in merged
    snap = merged["__per_experiment_spec__"]
    # The snapshot is exactly what the author wrote — no universes block.
    assert "universes" not in snap
    assert snap["target"]["universe"] == "nasdaq100"
    assert snap["backend"]["fs_hp_loop"]["max_iterations"] == 3


def test_spec_hash_ignores_internal_keys(tmp_path):
    """``_spec_hash`` must skip the snapshot key — adding the snapshot
    helper must not perturb the recorded hash of historical experiments.
    """
    from gbdt.__main__ import _spec_hash

    base = {"target": {"universe": "x", "direction": "up",
                         "threshold_pct": 10, "horizon_days": 20}}
    h1 = _spec_hash(base)
    base_with_internal = dict(base)
    base_with_internal["__per_experiment_spec__"] = {"target": {"x": "y"}}
    h2 = _spec_hash(base_with_internal)
    assert h1 == h2


# ---------------------------------------------------------------------------
# Issue #31 — test segment projection + warning
# ---------------------------------------------------------------------------


def _toy_panel(n_tickers: int = 3, n_rows: int = 200) -> pd.DataFrame:
    """Build a minimal MultiIndex(date, ticker) panel for projection tests."""
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    rows = []
    for ti in range(n_tickers):
        for d in dates:
            rows.append({"date": d, "ticker": f"T{ti}", "close": 100.0})
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


def test_project_test_rows_zero_when_horizon_eats_window():
    """At ``horizon_days == split.test_rows`` every ticker's test rows
    are NaN-target → expected_test_rows = 0 (the H=100 bug from issue #31).
    """
    from gbdt.__main__ import _project_test_rows

    panel = _toy_panel(n_tickers=3)
    proj = _project_test_rows(
        panel, test_rows_per_ticker=100, horizon_days=100,
    )
    assert proj["expected_test_rows"] == 0
    assert proj["per_ticker_usable"] == 0
    assert proj["n_tickers"] == 3


def test_project_test_rows_normal_case():
    """At ``horizon_days << split.test_rows`` we recover the expected
    ``(test_rows - horizon) × n_tickers`` row count.
    """
    from gbdt.__main__ import _project_test_rows

    panel = _toy_panel(n_tickers=4)
    proj = _project_test_rows(
        panel, test_rows_per_ticker=100, horizon_days=25,
    )
    # 100 - 25 = 75 usable per ticker × 4 tickers = 300
    assert proj["per_ticker_usable"] == 75
    assert proj["expected_test_rows"] == 300


def test_format_test_split_warning_empty_segment_explicit():
    """The empty-segment warning string names both knobs (horizon, test_rows)
    so a first-time reader can diagnose without source-code lookup.
    """
    from gbdt.__main__ import _format_test_split_warning

    proj = {"per_ticker_usable": 0, "horizon_days": 100,
             "test_rows_per_ticker": 100, "n_tickers": 5,
             "expected_test_rows": 0}
    msg = _format_test_split_warning(proj, threshold=100)
    assert "EMPTY" in msg
    assert "horizon_days=100" in msg
    assert "test_rows=100" in msg


def test_format_test_split_warning_slim_but_nonzero():
    from gbdt.__main__ import _format_test_split_warning

    proj = {"per_ticker_usable": 10, "horizon_days": 90,
             "test_rows_per_ticker": 100, "n_tickers": 5,
             "expected_test_rows": 50}
    msg = _format_test_split_warning(proj, threshold=100)
    assert "SMALL" in msg
    assert "50" in msg


def test_render_report_surfaces_test_split_warning(tmp_path):
    """When ``data.test_split_warning`` is present in ``metrics.json``,
    the rendered report must include a top-level ``## Warnings`` section.
    """
    spec = {"target": {"universe": "x", "direction": "up",
                         "threshold_pct": 10, "horizon_days": 100}}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    metrics = {
        "experiment_name": "warn_exp",
        "data": {"n_tickers_in_universe": 5, "n_tickers_used": 5,
                  "n_rows_train": 4000, "n_rows_val": 2000,
                  "n_rows_eval": 1000, "n_rows_test": 0,
                  "test_split_warning":
                      "Test segment expected to be EMPTY: horizon_days=100 >= "
                      "split.test_rows=100, …"},
        "loop": {"n_iterations_run": 3, "best_iteration": 0,
                  "inner_stop_signal": "cap",
                  "hp_search_active": True, "max_iterations": 8,
                  "hp_search_iter_threshold": 5},
        "calibration": {"method": "conditional_isotonic", "decision": "native",
                          "spiegelhalter_z": 0.5, "spiegelhalter_p": 0.6},
        "headline_eval": {"brier": 0.2, "brier_baseline_baserate": 0.2,
                            "brier_improvement_vs_baseline": 0.0,
                            "log_loss": 0.6, "roc_auc": 0.5},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "iterations.jsonl").write_text("")
    out = render_report(tmp_path)
    md = out.read_text()
    assert "## Warnings" in md
    assert "test_split" in md
    assert "EMPTY" in md


# ---------------------------------------------------------------------------
# Issue #32 — sweep-mode hp_search_active flag + report warning
# ---------------------------------------------------------------------------


def test_render_report_surfaces_hp_search_inactive_warning(tmp_path):
    """``loop.hp_search_active=False`` must surface in the report's
    ``## Warnings`` section so artifact readers know the FS+HP loop
    actually ran FS only.
    """
    spec = {"target": {"universe": "x", "direction": "up",
                         "threshold_pct": 10, "horizon_days": 20}}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    metrics = {
        "experiment_name": "sweep_exp",
        "data": {"n_tickers_in_universe": 5, "n_tickers_used": 5,
                  "n_rows_train": 4000, "n_rows_val": 2000,
                  "n_rows_eval": 1000, "n_rows_test": 500},
        "loop": {"n_iterations_run": 3, "best_iteration": 0,
                  "inner_stop_signal": "cap",
                  "hp_search_active": False, "max_iterations": 3,
                  "hp_search_iter_threshold": 5},
        "calibration": {"method": "conditional_isotonic", "decision": "native",
                          "spiegelhalter_z": 0.5, "spiegelhalter_p": 0.6},
        "headline_eval": {"brier": 0.2, "brier_baseline_baserate": 0.2,
                            "brier_improvement_vs_baseline": 0.0,
                            "log_loss": 0.6, "roc_auc": 0.5},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "iterations.jsonl").write_text("")
    out = render_report(tmp_path)
    md = out.read_text()
    assert "## Warnings" in md
    assert "hp_search" in md
    assert "max_iter=3" in md
    assert "threshold=5" in md


def test_render_report_no_warnings_section_when_clean(tmp_path):
    """When nothing is wrong, the warnings section must NOT appear at all
    (no empty header, no noise)."""
    spec = {"target": {"universe": "x", "direction": "up",
                         "threshold_pct": 10, "horizon_days": 20}}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    metrics = {
        "experiment_name": "clean_exp",
        "data": {"n_tickers_in_universe": 5, "n_tickers_used": 5,
                  "n_rows_train": 4000, "n_rows_val": 2000,
                  "n_rows_eval": 1000, "n_rows_test": 500,
                  "test_split_warning": None},
        "loop": {"n_iterations_run": 8, "best_iteration": 0,
                  "inner_stop_signal": "cap",
                  "hp_search_active": True, "max_iterations": 8,
                  "hp_search_iter_threshold": 5},
        "calibration": {"method": "conditional_isotonic", "decision": "native",
                          "spiegelhalter_z": 0.5, "spiegelhalter_p": 0.6},
        "headline_eval": {"brier": 0.2, "brier_baseline_baserate": 0.2,
                            "brier_improvement_vs_baseline": 0.0,
                            "log_loss": 0.6, "roc_auc": 0.5},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "iterations.jsonl").write_text("")
    out = render_report(tmp_path)
    md = out.read_text()
    assert "## Warnings" not in md
