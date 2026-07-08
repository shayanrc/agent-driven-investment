"""Regression: ``--overwrite`` must clear stale agent-authored decision files.

The bug: in ``agent_file_protocol`` mode the runner only ever WRITES
request/checkpoint files — the ``loop/iter_<N>_decision.json`` (and the scout
``combine_decision.json`` / ``iter_0_decision.json``) are agent-authored, so an
``--overwrite`` that re-trains iter 0 leaves a prior run's decision files in
place. The next ``--resume`` (without a fresh decision written) then reads that
stale decision and applies its OLD ``hp_changes`` / ``prune_features`` instead
of a clean start — silently contaminating a result that looks legitimate.

The fix: on ``--overwrite`` (a fresh run), the runner purges those decision
files so the loop dir is pristine and a bare ``--resume`` fails fast until the
agent writes a new decision. These tests exercise the clearing logic directly
against a hand-built loop dir (no data cache / panel build needed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gbdt import loop_protocol
from gbdt.agent_cycles import _load_and_apply_resume
from gbdt.checkpoint import write_checkpoint
from gbdt.experiment_runner import _clear_stale_loop_decisions
from gbdt.loop_protocol import DecisionError
from gbdt.scout_io import combine_decision_path, iter_0_decision_path


def _write_decision(out_dir: Path, iter_n: int, decision: dict) -> Path:
    path = loop_protocol.decision_path(out_dir, iter_n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision))
    return path


def _seed_prior_run(out_dir: Path) -> dict[str, Path]:
    """Mimic the on-disk state after a completed agent_file_protocol run.

    Returns a map of the files it wrote so a test can assert per-file survival.
    """
    loop_dir = loop_protocol.decision_path(out_dir, 0).parent
    loop_dir.mkdir(parents=True, exist_ok=True)

    # Runner-authored files — MUST survive a clear.
    ckpt = write_checkpoint(
        out_dir,
        {
            "iter_idx": 0,
            "current_features": ["f1", "f2", "f3"],
            "current_hp": {"depth": 6},
        },
    )
    req0 = loop_protocol.request_path(out_dir, 0)
    req0.write_text(json.dumps({"iter": 0}))

    # Agent-authored decisions — MUST be cleared on overwrite.
    dec0 = _write_decision(
        out_dir, 0, {"hp_changes": {"depth": 4}, "prune_features": ["f3"]},
    )
    dec1 = _write_decision(out_dir, 1, {"hp_changes": {"depth": 3}})

    return {"ckpt": ckpt, "req0": req0, "dec0": dec0, "dec1": dec1}


def test_clear_removes_all_iter_decisions_keeps_runner_files(tmp_path: Path) -> None:
    files = _seed_prior_run(tmp_path)

    removed = _clear_stale_loop_decisions(tmp_path)

    # Both decision files gone; both runner files survive.
    assert not files["dec0"].exists()
    assert not files["dec1"].exists()
    assert files["ckpt"].exists()
    assert files["req0"].exists()
    assert set(removed) == {files["dec0"], files["dec1"]}


def test_clear_removes_scout_decisions(tmp_path: Path) -> None:
    scout_combine = combine_decision_path(tmp_path)
    scout_iter0 = iter_0_decision_path(tmp_path)
    scout_combine.parent.mkdir(parents=True, exist_ok=True)
    scout_combine.write_text(json.dumps({"configs": []}))
    scout_iter0.write_text(json.dumps({"hp": {}}))
    # A non-decision scout artifact that must be left alone.
    keep = scout_combine.parent / "combine_request.json"
    keep.write_text("{}")

    removed = _clear_stale_loop_decisions(tmp_path)

    assert not scout_combine.exists()
    assert not scout_iter0.exists()
    assert keep.exists()
    assert set(removed) == {scout_combine, scout_iter0}


def test_clear_is_noop_when_no_decisions(tmp_path: Path) -> None:
    # Default-mode run: no decision files anywhere -> nothing removed, no error.
    (tmp_path / "loop").mkdir()
    assert _clear_stale_loop_decisions(tmp_path) == []


def test_stale_decision_applied_without_clear_then_rejected_after(tmp_path: Path) -> None:
    """The bug and its fix, end-to-end at the loop-protocol layer.

    Before clearing, ``_load_and_apply_resume`` reads the leftover
    ``iter_0_decision.json`` and applies its stale ``hp_changes`` /
    ``prune_features`` (the contamination). After ``_clear_stale_loop_decisions``
    (what ``--overwrite`` now does), the same resume fails fast because the stale
    decision is gone.
    """
    _seed_prior_run(tmp_path)
    spec: dict = {}

    # BUG: a resume with no fresh decision silently applies the prior run's
    # iter_0 decision (depth 6 -> 4, feature f3 pruned).
    contaminated = _load_and_apply_resume(tmp_path, spec, run_id="cell")
    assert contaminated["current_hp"]["depth"] == 4
    assert "f3" not in contaminated["current_features"]

    # FIX: overwrite clears the decisions, so the next resume can't apply them.
    _clear_stale_loop_decisions(tmp_path)
    with pytest.raises(DecisionError):
        _load_and_apply_resume(tmp_path, spec, run_id="cell")
