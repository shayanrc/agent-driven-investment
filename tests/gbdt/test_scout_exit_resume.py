"""Tests for V1.3 Option B P4 — agent_file_protocol scout cycles.

Covers the file I/O contract in :mod:`gbdt.scout_io` + the cycle detection
machinery. The end-to-end runner integration is covered indirectly via the
scout_io helpers (the runner just orchestrates them).

No SQLite cache, no data_pipelines.fetch — these are pure file-I/O tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gbdt.scout_io import (
    COMBINE_DECISION_MAX_CONFIGS,
    CombineDecisionError,
    CycleState,
    Iter0DecisionError,
    combine_decision_path,
    combine_request_path,
    combine_results_path,
    detect_cycle_state,
    iter_0_decision_path,
    read_combine_decision,
    read_iter_0_decision,
    scout_bundle_path,
    scout_results_path,
    write_combine_request,
    write_combine_results,
    write_scout_bundle,
    write_scout_results,
)


# ---------------------------------------------------------------------------
# Atomic-write contract
# ---------------------------------------------------------------------------


def test_write_scout_results_creates_jsonl(tmp_path):
    rows = [
        {"config": {"knob_name": "defaults"}, "val_brier": 0.25,
          "status": "ok"},
        {"config": {"knob_name": "max_depth"}, "val_brier": 0.24,
          "status": "ok"},
    ]
    path = write_scout_results(tmp_path, rows)
    assert path == scout_results_path(tmp_path)
    assert path.exists()
    content = path.read_text().strip().split("\n")
    assert len(content) == 2
    # Each line parses as JSON.
    for line, row in zip(content, rows):
        assert json.loads(line) == row


def test_write_scout_bundle_pretty_prints(tmp_path):
    summary = {"run_id": "test", "n_configs_total": 35}
    path = write_scout_bundle(tmp_path, summary)
    assert path == scout_bundle_path(tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded["run_id"] == "test"
    assert loaded["n_configs_total"] == 35


def test_write_combine_request_carries_speed_biased_prompt(tmp_path):
    payload = {
        "run_id": "test", "prompt": "fast configs please",
        "lex_auto_compose_overlay": {"max_depth": 2},
    }
    path = write_combine_request(tmp_path, payload)
    assert path == combine_request_path(tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded["prompt"] == "fast configs please"


def test_write_combine_results_groups_under_configs(tmp_path):
    rows = [{"index": 0, "status": "ok"}, {"index": 1, "status": "ok"}]
    path = write_combine_results(tmp_path, rows)
    loaded = json.loads(path.read_text())
    assert loaded == {"configs": rows}


# ---------------------------------------------------------------------------
# read_combine_decision — validation
# ---------------------------------------------------------------------------


def test_read_combine_decision_missing_file_raises(tmp_path):
    with pytest.raises(CombineDecisionError, match="not found"):
        read_combine_decision(tmp_path)


def test_read_combine_decision_malformed_json_raises(tmp_path):
    p = combine_decision_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    with pytest.raises(CombineDecisionError, match="not valid JSON"):
        read_combine_decision(tmp_path)


def test_read_combine_decision_non_list_configs_raises(tmp_path):
    p = combine_decision_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"configs": "not a list"}))
    with pytest.raises(CombineDecisionError, match="must be a list"):
        read_combine_decision(tmp_path)


def test_read_combine_decision_over_cap_raises(tmp_path):
    """D3b.A — > 50 configs is rejected at parse time."""
    p = combine_decision_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfgs = [{"hp": {"max_depth": i}}
              for i in range(COMBINE_DECISION_MAX_CONFIGS + 1)]
    p.write_text(json.dumps({"configs": cfgs}))
    with pytest.raises(CombineDecisionError, match="exceeds the D3b.A cap"):
        read_combine_decision(tmp_path)


def test_read_combine_decision_missing_hp_key_raises(tmp_path):
    p = combine_decision_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"configs": [{"label": "no_hp"}]}))
    with pytest.raises(CombineDecisionError, match="missing 'hp' key"):
        read_combine_decision(tmp_path)


def test_read_combine_decision_non_dict_hp_raises(tmp_path):
    p = combine_decision_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"configs": [{"hp": "not a dict"}]}))
    with pytest.raises(CombineDecisionError, match="must be a"):
        read_combine_decision(tmp_path)


def test_read_combine_decision_at_cap_accepted(tmp_path):
    """Exactly 50 configs is OK (cap is inclusive)."""
    p = combine_decision_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfgs = [{"hp": {"max_depth": i}}
              for i in range(COMBINE_DECISION_MAX_CONFIGS)]
    p.write_text(json.dumps({"configs": cfgs}))
    decision = read_combine_decision(tmp_path)
    assert len(decision["configs"]) == COMBINE_DECISION_MAX_CONFIGS


def test_read_combine_decision_happy_path(tmp_path):
    p = combine_decision_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"configs": [
        {"label": "lex_auto_compose", "hp": {"max_depth": 2, "eta": 0.1}},
        {"label": "alt", "hp": {"max_depth": 3, "eta": 0.05}},
    ]}
    p.write_text(json.dumps(payload))
    decision = read_combine_decision(tmp_path)
    assert decision == payload


# ---------------------------------------------------------------------------
# read_iter_0_decision — validation
# ---------------------------------------------------------------------------


def test_read_iter_0_decision_missing_file_raises(tmp_path):
    with pytest.raises(Iter0DecisionError, match="not found"):
        read_iter_0_decision(tmp_path)


def test_read_iter_0_decision_non_dict_hp_raises(tmp_path):
    p = iter_0_decision_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"hp": "not a dict"}))
    with pytest.raises(Iter0DecisionError, match="must be a dict"):
        read_iter_0_decision(tmp_path)


def test_read_iter_0_decision_happy_path(tmp_path):
    p = iter_0_decision_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"hp": {"max_depth": 2, "eta": 0.1, "n_estimators": 6}}
    p.write_text(json.dumps(payload))
    decision = read_iter_0_decision(tmp_path)
    assert decision == payload


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def test_detect_cycle_state_empty_dir(tmp_path):
    state = detect_cycle_state(tmp_path)
    assert state == CycleState(
        has_combine_request=False,
        has_combine_decision=False,
        has_combine_results=False,
        has_iter_0_decision=False,
    )


def test_detect_cycle_state_after_cycle1(tmp_path):
    write_combine_request(tmp_path, {"prompt": "test"})
    state = detect_cycle_state(tmp_path)
    assert state.has_combine_request is True
    assert state.has_combine_decision is False
    assert state.has_combine_results is False
    assert state.has_iter_0_decision is False


def test_detect_cycle_state_after_agent_writes_combine_decision(tmp_path):
    write_combine_request(tmp_path, {"prompt": "test"})
    p = combine_decision_path(tmp_path)
    p.write_text(json.dumps({"configs": [{"hp": {"max_depth": 2}}]}))
    state = detect_cycle_state(tmp_path)
    assert state.has_combine_request is True
    assert state.has_combine_decision is True
    assert state.has_combine_results is False


def test_detect_cycle_state_after_cycle2(tmp_path):
    write_combine_request(tmp_path, {"prompt": "test"})
    p = combine_decision_path(tmp_path)
    p.write_text(json.dumps({"configs": [{"hp": {"max_depth": 2}}]}))
    write_combine_results(tmp_path, [{"index": 0, "status": "ok"}])
    state = detect_cycle_state(tmp_path)
    assert state.has_combine_results is True


def test_detect_cycle_state_at_cycle3(tmp_path):
    write_combine_request(tmp_path, {"prompt": "test"})
    p = combine_decision_path(tmp_path)
    p.write_text(json.dumps({"configs": [{"hp": {"max_depth": 2}}]}))
    write_combine_results(tmp_path, [{"index": 0, "status": "ok"}])
    p2 = iter_0_decision_path(tmp_path)
    p2.write_text(json.dumps({"hp": {"max_depth": 2}}))
    state = detect_cycle_state(tmp_path)
    assert state.has_iter_0_decision is True
