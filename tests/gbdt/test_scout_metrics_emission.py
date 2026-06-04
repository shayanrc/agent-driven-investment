"""Tests for V1.3 Option B P5 — metrics.json::scout + metrics.json::combine
emission.

The blocks land in metrics.json from one of two sources:
- Default mode: the in-train scout (walk_forward_train's scout_report).
- Agent mode: ``scout/scout_bundle.json`` + ``scout/combine_results.json`` +
  ``scout/iter_0_decision.json``.

We test the helper ``_build_scout_metrics_blocks`` directly with synthetic
inputs so no live model fit is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _import_helper():
    """Import the helper lazily so test collection doesn't trigger a
    heavy import chain via ``gbdt.__main__``."""
    from gbdt.__main__ import _build_scout_metrics_blocks
    return _build_scout_metrics_blocks


def test_metrics_blocks_none_when_scout_disabled():
    build = _import_helper()
    scout_block, combine_block = build(
        result=None, out_dir=Path("/tmp/x"), cycle_outcome=None,
        callback_mode="default", scout_enabled=False,
    )
    assert scout_block is None
    assert combine_block is None


def test_metrics_blocks_default_mode_from_in_train_report():
    build = _import_helper()
    # Synthetic in-train scout report.
    fake_result = SimpleNamespace(scout_report={
        "scout": {
            "enabled": True, "backend": "xgboost",
            "n_configs_total": 35, "n_configs_completed": 35,
            "runtime_seconds": 12.5,
            "defaults_metrics": {"val_brier": 0.25},
            "per_knob_winner": {"max_depth": {"knob_value": 2}},
            "lexicographic_auto_compose": {"hp_overlay": {"max_depth": 2}},
            "status": "lex_auto_compose",
            "degenerate_sink_fallback": False,
            "grid_spec": {},
        },
        "fs_prefit": {
            "enabled": True, "n_kept": 130, "n_dropped": 149,
            "top_importance": 50.5, "cliff_threshold": 0.505,
            "fit_seconds": 2.1, "cliff_pct": 0.01, "backend": "xgboost",
        },
        "combine": {
            "status": "lex_auto_compose",
            "composed_overlay": {"max_depth": 2},
            "n_mix_configs_completed": 0,
        },
    })
    scout_block, combine_block = build(
        result=fake_result, out_dir=Path("/tmp/x"), cycle_outcome=None,
        callback_mode="default", scout_enabled=True,
    )
    assert scout_block is not None
    assert scout_block["enabled"] is True
    assert scout_block["backend"] == "xgboost"
    assert scout_block["n_configs_total"] == 35
    assert scout_block["lexicographic_auto_compose"] == {
        "hp_overlay": {"max_depth": 2},
    }
    assert scout_block["fs_prefit"]["n_kept"] == 130
    assert combine_block is not None
    assert combine_block["mode"] == "default"
    assert combine_block["exit_resume_rounds"] == 0


def test_metrics_blocks_agent_mode_from_cycle_outcome(tmp_path):
    build = _import_helper()
    # Synthetic cycle outcome (cycle 3 stitches the bundle into the dict).
    cycle_outcome = {
        "hp_starting": {"max_depth": 2},
        "features": ["a", "b"],
        "scout_report": {
            "backend": "xgboost",
            "n_configs_total": 35,
            "n_configs_completed": 35,
            "runtime_seconds": 12.5,
            "defaults_metrics": {"val_brier": 0.25},
            "per_knob_winner": {"max_depth": {"knob_value": 2}},
            "lexicographic_auto_compose": {"hp_overlay": {"max_depth": 2}},
            "fs_prefit": {"enabled": True, "n_kept": 130, "n_dropped": 149},
            "grid_spec": {},
        },
    }
    # Synthetic combine_results.json (cycle 2 output).
    scout_dir = tmp_path / "scout"
    scout_dir.mkdir(parents=True, exist_ok=True)
    (scout_dir / "combine_results.json").write_text(json.dumps({
        "configs": [
            {"index": 0, "status": "ok", "hp_overlay": {"max_depth": 2}},
            {"index": 1, "status": "ok", "hp_overlay": {"max_depth": 3}},
        ],
    }))
    # Synthetic iter_0_decision.json — the agent picked a DIFFERENT HP
    # from the lex auto-compose, so vs_lex should be False.
    (scout_dir / "iter_0_decision.json").write_text(json.dumps({
        "hp": {"max_depth": 3},
    }))
    scout_block, combine_block = build(
        result=None, out_dir=tmp_path, cycle_outcome=cycle_outcome,
        callback_mode="agent_file_protocol", scout_enabled=True,
    )
    assert scout_block is not None
    assert scout_block["status"] == "agent_combine"
    assert scout_block["n_configs_total"] == 35
    assert combine_block is not None
    assert combine_block["mode"] == "agent_file_protocol"
    assert combine_block["exit_resume_rounds"] == 2
    assert combine_block["n_mix_configs_proposed"] == 2
    assert combine_block["n_mix_configs_completed"] == 2
    assert combine_block["agent_winner"] == {"max_depth": 3}
    assert combine_block["vs_lexicographic_auto_compose"] is not None
    assert combine_block["vs_lexicographic_auto_compose"]["is_lex"] is False


def test_metrics_blocks_agent_mode_lex_equals_winner(tmp_path):
    """When the agent picks the lex auto-compose, vs_lex is True."""
    build = _import_helper()
    cycle_outcome = {
        "hp_starting": {"max_depth": 2},
        "features": ["a"],
        "scout_report": {
            "backend": "xgboost",
            "lexicographic_auto_compose": {"hp_overlay": {"max_depth": 2}},
            "fs_prefit": {"enabled": True},
            "grid_spec": {},
        },
    }
    scout_dir = tmp_path / "scout"
    scout_dir.mkdir(parents=True, exist_ok=True)
    (scout_dir / "iter_0_decision.json").write_text(json.dumps({
        "hp": {"max_depth": 2},
    }))
    _scout_block, combine_block = build(
        result=None, out_dir=tmp_path, cycle_outcome=cycle_outcome,
        callback_mode="agent_file_protocol", scout_enabled=True,
    )
    assert combine_block["vs_lexicographic_auto_compose"]["is_lex"] is True
