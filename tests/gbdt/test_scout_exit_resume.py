"""Tests for V1.3 Option B P4 — agent_file_protocol scout cycles.

Covers the file I/O contract in :mod:`gbdt.scout_io` + the cycle detection
machinery. The end-to-end runner integration is covered indirectly via the
scout_io helpers (the runner just orchestrates them).

No SQLite cache, no data_pipelines.fetch — these are pure file-I/O tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gbdt.__main__ import _handle_scout_cycles_agent_mode
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


# ---------------------------------------------------------------------------
# Bug #248 regression — cycle-2-complete pause path
# ---------------------------------------------------------------------------


def test_post_combine_results_pauses_for_iter_0_decision_no_scout_rerun(
    tmp_path, monkeypatch,
):
    """Bug #248 — when combine_results.json is present but iter_0_decision.json
    is not, ``_handle_scout_cycles_agent_mode`` must PAUSE (return None) and
    leave the cycle-1 artifacts (combine_request.json, scout_results.jsonl)
    untouched. Pre-fix, the conditional ladder fell through to cycle 1 and
    re-ran FS-prefit + scout response curves every --resume, clobbering
    combine_request.json + bumping its mtime each time.
    """
    import os
    import time

    # Stage the artifact dir: cycle 1 complete (combine_request +
    # scout_bundle + scout_results), cycle 2 complete (combine_decision +
    # combine_results), but NO iter_0_decision.json yet.
    write_scout_results(tmp_path, [
        {"config": {"knob_name": "defaults"}, "val_brier": 0.25, "status": "ok"},
    ])
    write_scout_bundle(tmp_path, {"run_id": "test", "n_configs_total": 1})
    write_combine_request(tmp_path, {"prompt": "test", "run_id": "test"})
    cdp = combine_decision_path(tmp_path)
    cdp.write_text(json.dumps({"configs": [{"hp": {"max_depth": 2}}]}))
    write_combine_results(tmp_path, [{"index": 0, "status": "ok"}])
    assert not iter_0_decision_path(tmp_path).exists()

    # Capture pre-call mtimes + content of the cycle-1 artifacts. Sleep a
    # touch so any rewrite would advance mtime past the resolution floor.
    cr_path = combine_request_path(tmp_path)
    sr_path = scout_results_path(tmp_path)
    cr_mtime_before = cr_path.stat().st_mtime_ns
    sr_mtime_before = sr_path.stat().st_mtime_ns
    cr_text_before = cr_path.read_text()
    sr_text_before = sr_path.read_text()
    time.sleep(0.01)

    # Hard-fail the carve so if the cycle-1 fall-through path runs we
    # see it loudly. The new pause branch sits BEFORE the carve, so this
    # patch must not be exercised.
    from gbdt import train as gbdt_train

    def _carve_should_not_run(*_args, **_kwargs):
        raise AssertionError(
            "_carve_X_y was called — the cycle-2-complete pause branch did "
            "NOT trigger; cycle 1 fall-through happened (Bug #248 regression)."
        )

    monkeypatch.setattr(gbdt_train, "_carve_X_y", _carve_should_not_run)

    milestones: list[str] = []

    def milestone(msg: str) -> None:
        milestones.append(msg)

    status = MagicMock()
    heartbeat = MagicMock()
    progress_log = MagicMock()

    # Most other params are unused in the pause path. Sentinels make any
    # accidental dereference obvious.
    sentinel = MagicMock()
    result = _handle_scout_cycles_agent_mode(
        out_dir=tmp_path,
        spec={},
        run_id="test",
        spec_path=Path("unused.yaml"),
        panel=sentinel, X=sentinel, y=sentinel, sample_weights=None,
        split=sentinel, universe_calendar=None,
        backend_library="catboost",
        hp_starting={"iterations": 30, "depth": 3},
        calibration_method="native",
        calibration_z_threshold=2.0,
        random_seed=0,
        scout_cfg={"enabled": True},
        fs_prefit_cfg={"enabled": True},
        milestone=milestone,
        heartbeat=heartbeat,
        status=status,
        progress_log=progress_log,
    )

    # 1. The pause returned None (= cleanly exit + wait for agent).
    assert result is None
    # 2. status.update was called with awaiting_decision=True (visually
    # parallel to cycle 1's status.update).
    status.update.assert_any_call(
        phase="scout_cycle_2", awaiting_decision=True,
    )
    # 3. A milestone log line surfaces the pause reason.
    assert any(
        "cycle 2 already complete" in m
        and "awaiting iter_0_decision.json" in m
        for m in milestones
    ), f"expected pause milestone in {milestones!r}"
    # 4. Cycle-1 artifacts UNTOUCHED — no rewrite, no mtime bump.
    assert cr_path.stat().st_mtime_ns == cr_mtime_before, (
        "combine_request.json mtime advanced — scout cycle 1 re-ran"
    )
    assert sr_path.stat().st_mtime_ns == sr_mtime_before, (
        "scout_results.jsonl mtime advanced — scout cycle 1 re-ran"
    )
    assert cr_path.read_text() == cr_text_before
    assert sr_path.read_text() == sr_text_before
