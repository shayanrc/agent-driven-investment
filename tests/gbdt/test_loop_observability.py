"""Task #177 — persistent agent-loop observability.

Covers the three artifacts added by ``gbdt.loop_observability`` + the
``scripts.gbdt.loop_status`` reader:

1. ``loop/progress.log`` APPENDS across an initial run + a ``--resume`` run
   (same file, both invocations' lines present) — the survives-the-resume-
   boundary contract.
2. ``loop/status.json`` reflects ``awaiting_decision=true`` after a pause and
   updates iter/phase across the cycle; carries the v1 schema.
3. ``scripts.gbdt.loop_status`` parses + renders a single cell + scans all
   cells, and the shared ``liveness_verdict`` computes ALIVE/STALE/DONE.
4. The heartbeat-disabled path (``GBDT_HEARTBEAT_INTERVAL=0``) does not crash
   and still produces progress.log milestone lines + a status.json.

Reuses the tiny synthetic Phase-4 smoke fixture (monkeypatch the data seam;
everything downstream is real code; sub-minute end-to-end).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from gbdt import loop_observability as obs
from gbdt.__main__ import run_experiment
from gbdt.loop_protocol import decision_path, request_path
from scripts.gbdt import loop_status as reader

# Reuse the Phase-4 smoke harness verbatim (synthetic panel + spec writer +
# the data-seam monkeypatch fixture).
from tests.gbdt.test_phase4_smoke import _synthetic_panel, _write_spec, smoke_env  # noqa: F401


# ---------------------------------------------------------------------------
# 1) progress.log appends across the resume boundary
# ---------------------------------------------------------------------------


def test_progress_log_appends_across_resume_boundary(smoke_env):
    spec_path, cell_dir, run_id = smoke_env

    # --- fresh launch -> pauses at iter 0 ---
    run_experiment(spec_path, resume=None)
    log_path = obs.progress_log_path(cell_dir)
    assert log_path.exists(), "progress.log not written on fresh run"
    text_after_fresh = log_path.read_text()
    lines_fresh = [ln for ln in text_after_fresh.splitlines() if ln.strip()]

    # The fresh run's milestones are present (phase banners + the pause).
    assert any("[experiment] start" in ln for ln in lines_fresh)
    assert any("[data] complete" in ln for ln in lines_fresh)
    assert any("[loop] start" in ln for ln in lines_fresh)
    assert any("PAUSED iter 0" in ln for ln in lines_fresh)
    # Every line is UTC-ISO timestamped (parseable leading token).
    for ln in lines_fresh:
        stamp = ln.split(" ", 1)[0]
        datetime.fromisoformat(stamp)  # raises if not ISO

    # --- scripted iter-0 decision + resume ---
    decision_path(cell_dir, 0).write_text(json.dumps({
        "iter": 0,
        "prune_features": ["realized_vol_20"],
        "hp_changes": {"depth": 3},
        "should_stop": False,
        "rationale": "prune one, deepen.",
    }))
    run_experiment(spec_path, resume=run_id)

    text_after_resume = log_path.read_text()
    lines_resume = [ln for ln in text_after_resume.splitlines() if ln.strip()]

    # APPEND, not overwrite: the resume invocation's text strictly extends the
    # fresh run's text (same physical file across two separate run_experiment
    # calls = the separate-process resume model).
    assert text_after_resume.startswith(text_after_fresh), (
        "progress.log was truncated/overwritten on resume — must append"
    )
    assert len(lines_resume) > len(lines_fresh)
    # Both invocations' signature lines coexist in the one file.
    assert any("PAUSED iter 0" in ln for ln in lines_resume)        # invocation 1
    assert any("RESUMED iter 1" in ln for ln in lines_resume)       # invocation 2
    assert any("PAUSED iter 1" in ln for ln in lines_resume)        # invocation 2


# ---------------------------------------------------------------------------
# 2) status.json reflects awaiting_decision + updates iter/phase
# ---------------------------------------------------------------------------


def test_status_json_awaiting_decision_after_pause(smoke_env):
    spec_path, cell_dir, run_id = smoke_env

    run_experiment(spec_path, resume=None)
    status = json.loads(obs.status_path(cell_dir).read_text())

    assert status["schema_version"] == obs.STATUS_SCHEMA_VERSION
    assert status["run_id"] == run_id
    assert status["iter_idx"] == 0
    assert status["phase"] == "loop"
    assert status["awaiting_decision"] is True
    assert status["stop_reason"] is None
    # best_val_brier is recorded at the pause (iter 0 trained one model).
    assert status["best_val_brier"] is not None
    # last_update_utc parseable; with the heartbeat disabled in the smoke
    # fixture last_heartbeat_utc stays None.
    datetime.fromisoformat(status["last_update_utc"])
    assert status["last_heartbeat_utc"] is None


def test_status_json_updates_iter_and_phase_across_cycle(smoke_env):
    spec_path, cell_dir, run_id = smoke_env

    run_experiment(spec_path, resume=None)
    s0 = json.loads(obs.status_path(cell_dir).read_text())
    assert s0["iter_idx"] == 0 and s0["awaiting_decision"] is True

    decision_path(cell_dir, 0).write_text(json.dumps({
        "iter": 0, "prune_features": ["realized_vol_20"],
        "hp_changes": {"depth": 3}, "should_stop": False,
        "rationale": "prune one, deepen.",
    }))
    run_experiment(spec_path, resume=run_id)
    s1 = json.loads(obs.status_path(cell_dir).read_text())
    # Iter advanced to 1 and the run is again parked on the agent.
    assert s1["iter_idx"] == 1
    assert s1["phase"] == "loop"
    assert s1["awaiting_decision"] is True
    assert s1["stop_reason"] is None

    # should_stop -> finalize: terminal status, stop_reason set.
    decision_path(cell_dir, 1).write_text(json.dumps({
        "iter": 1, "should_stop": True, "rationale": "stop.",
    }))
    run_experiment(spec_path, resume=run_id)
    s2 = json.loads(obs.status_path(cell_dir).read_text())
    assert s2["phase"] == "complete"
    assert s2["awaiting_decision"] is False
    assert s2["stop_reason"] == "agent_should_stop"
    assert s2["best_val_brier"] is not None


# ---------------------------------------------------------------------------
# 3) loop_status reader: parse + render + scan + liveness verdict
# ---------------------------------------------------------------------------


def test_reader_renders_single_cell(smoke_env, capsys):
    spec_path, cell_dir, run_id = smoke_env
    run_experiment(spec_path, resume=None)

    rc = reader.main([str(cell_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(cell_dir) in out
    assert "awaiting_decision: True" in out
    assert "phase            : loop" in out
    # The verdict line is present (heartbeat disabled -> NO-HEARTBEAT branch).
    assert "verdict          :" in out
    assert "NO-HEARTBEAT" in out
    # The progress.log tail is rendered.
    assert "progress.log (last" in out
    assert "PAUSED iter 0" in out


def test_reader_scan_summarizes_all_cells(smoke_env, capsys):
    spec_path, cell_dir, run_id = smoke_env
    run_experiment(spec_path, resume=None)

    # The scan root is the cell's parent (artifacts.experiment_dir == art_dir).
    experiments_root = cell_dir.parent
    rc = reader.main(["--experiments-root", str(experiments_root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 cell(s) with status.json" in out
    assert run_id in out
    assert "iter=0" in out
    assert "AWAIT-DECISION" in out


def test_reader_missing_status_is_graceful(tmp_path, capsys):
    empty = tmp_path / "no_loop_here"
    empty.mkdir()
    rc = reader.main([str(empty)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no status.json" in out


def test_reader_scan_empty_root_is_graceful(tmp_path, capsys):
    rc = reader.main(["--experiments-root", str(tmp_path / "does_not_exist")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no agent-loop cells" in out


# ---------------------------------------------------------------------------
# liveness_verdict unit cases (the derived ALIVE/STALE/DONE logic)
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)


def test_liveness_alive_when_recent_heartbeat():
    hb = (_now() - timedelta(seconds=20)).isoformat()
    v = obs.liveness_verdict({"last_heartbeat_utc": hb}, now=_now())
    assert v.startswith("ALIVE")
    assert "20s ago" in v


def test_liveness_stale_when_old_heartbeat():
    hb = (_now() - timedelta(seconds=300)).isoformat()
    v = obs.liveness_verdict({"last_heartbeat_utc": hb}, now=_now())
    assert v.startswith("STALE")
    assert "300s ago" in v


def test_liveness_done_when_stop_reason_set():
    v = obs.liveness_verdict(
        {"stop_reason": "agent_should_stop",
         "last_heartbeat_utc": (_now() - timedelta(seconds=999)).isoformat()},
        now=_now(),
    )
    assert v.startswith("DONE")
    assert "agent_should_stop" in v


def test_liveness_no_heartbeat_falls_back_to_last_update():
    upd = (_now() - timedelta(seconds=42)).isoformat()
    v = obs.liveness_verdict(
        {"last_heartbeat_utc": None, "last_update_utc": upd}, now=_now(),
    )
    assert v.startswith("NO-HEARTBEAT")
    assert "42s ago" in v


# ---------------------------------------------------------------------------
# 4) heartbeat-disabled path does not crash + still writes artifacts;
#    + the TeeStream / StatusFile.heartbeat integration when ENABLED.
# ---------------------------------------------------------------------------


def test_heartbeat_disabled_path_still_writes_artifacts(smoke_env):
    """The smoke fixture sets GBDT_HEARTBEAT_INTERVAL=0; the run must still
    complete the pause, write progress.log + status.json, and leave
    last_heartbeat_utc=None (the file heartbeat no-ops when disabled)."""
    spec_path, cell_dir, run_id = smoke_env
    # smoke_env already exports GBDT_HEARTBEAT_INTERVAL=0.
    returned = run_experiment(spec_path, resume=None)
    assert returned == cell_dir
    assert obs.progress_log_path(cell_dir).exists()
    status = json.loads(obs.status_path(cell_dir).read_text())
    assert status["last_heartbeat_utc"] is None
    # No [heartbeat] lines leaked into progress.log (heartbeat thread off).
    assert "[heartbeat]" not in obs.progress_log_path(cell_dir).read_text()


def test_heartbeat_tee_and_status_refresh_when_enabled(tmp_path):
    """When the heartbeat IS enabled, the TeeStream mirrors [heartbeat] lines
    into progress.log AND on_tick refreshes status.json's last_heartbeat_utc —
    without disturbing the underlying stream's output. Driven directly (no full
    experiment) so it's fast + deterministic."""
    import io
    import time

    from gbdt.heartbeat import Heartbeat

    art = tmp_path / "cell"
    progress = obs.ProgressLog(art)
    status = obs.StatusFile(art, run_id="hb_test")
    underlying = io.StringIO()
    tee = obs.TeeStream(underlying, progress)

    hb = Heartbeat(interval=0.05, stream=tee, on_tick=status.heartbeat).start()
    try:
        hb.set_phase("loop")
        time.sleep(0.18)  # ~3 ticks
    finally:
        hb.stop()
    progress.close()

    # Underlying stream (== stdout in production) still got the heartbeat lines.
    assert "[heartbeat]" in underlying.getvalue()
    # And they were mirrored into progress.log (timestamped).
    log_text = obs.progress_log_path(art).read_text()
    hb_lines = [ln for ln in log_text.splitlines() if "[heartbeat]" in ln]
    assert len(hb_lines) >= 2
    datetime.fromisoformat(hb_lines[-1].split(" ", 1)[0])
    # status.json's last_heartbeat_utc got refreshed by on_tick.
    s = json.loads(obs.status_path(art).read_text())
    assert s["last_heartbeat_utc"] is not None
    datetime.fromisoformat(s["last_heartbeat_utc"])


def test_progress_log_and_status_path_helpers(tmp_path):
    """Path helpers co-locate under loop/ (alongside checkpoint.json)."""
    assert obs.progress_log_path(tmp_path) == tmp_path / "loop" / "progress.log"
    assert obs.status_path(tmp_path) == tmp_path / "loop" / "status.json"


def test_tail_lines_returns_last_n(tmp_path):
    p = tmp_path / "loop" / "progress.log"
    p.parent.mkdir(parents=True)
    p.write_text("\n".join(f"line{i}" for i in range(50)) + "\n")
    tail = obs.tail_lines(p, n=15)
    assert len(tail) == 15
    assert tail[-1] == "line49"
    assert tail[0] == "line35"
    # Missing file -> [].
    assert obs.tail_lines(tmp_path / "nope.log") == []
