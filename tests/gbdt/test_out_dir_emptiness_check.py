"""Runner-side regression for #193 bug 2: the non-empty-out-dir guard
must ignore dotfile entries so the agent-loop wrapper's ``.wrapper/``
sidecar dir doesn't force callers to pass ``--overwrite`` on first launch.

The helper under test is ``gbdt.__main__._has_runner_artifacts``.
"""

from __future__ import annotations

from pathlib import Path

from gbdt.__main__ import _has_runner_artifacts


def test_empty_dir_returns_false(tmp_path: Path) -> None:
    assert _has_runner_artifacts(tmp_path) is False


def test_only_wrapper_sidecars_returns_false(tmp_path: Path) -> None:
    # Simulate scripts/gbdt/run_agent_loop_resumable.sh after #193 bug 2:
    # state lives at out_dir/.wrapper/{pid,status.json,log}.
    sidecar = tmp_path / ".wrapper"
    sidecar.mkdir()
    (sidecar / "pid").write_text("12345\n")
    (sidecar / "status.json").write_text('{"state":"starting"}\n')
    (sidecar / "log").write_text("[wrapper] start\n")
    assert _has_runner_artifacts(tmp_path) is False


def test_multiple_dotfile_entries_returns_false(tmp_path: Path) -> None:
    # Future-proofing: any dotfile/dotdir is ignored, not just .wrapper/.
    (tmp_path / ".wrapper").mkdir()
    (tmp_path / ".dvc").mkdir()
    (tmp_path / ".lock").write_text("")
    assert _has_runner_artifacts(tmp_path) is False


def test_visible_file_returns_true(tmp_path: Path) -> None:
    # Any non-dot entry is a runner artifact and triggers the guard.
    (tmp_path / ".wrapper").mkdir()
    (tmp_path / "metrics.json").write_text("{}")
    assert _has_runner_artifacts(tmp_path) is True


def test_visible_dir_returns_true(tmp_path: Path) -> None:
    (tmp_path / "predictions").mkdir()
    assert _has_runner_artifacts(tmp_path) is True
