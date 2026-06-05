"""V1.3 Option B PR #125 Medium 5 — unit tests for the
``--snapshot-end REQUIRED when scout enabled in agent mode`` rule
(:mod:`gbdt.__main__` runner-side validation).

The fresh-launch path doesn't require ``--snapshot-end`` (it pins the
date on first run); but every ``--resume`` of a scout-enabled,
agent_file_protocol spec MUST re-pass the SAME ``--snapshot-end`` value
the fresh run used. Without it, the universe-feature-cache key drifts
between scout cycles (the panel signature changes whenever the cache
auto-fetches) and the cliff-cut feature pool from cycle 1 won't match
the feature matrix of cycle 2/3 — silent miscompare, hard to debug.

Validation lives at src/gbdt/__main__.py around the
``_scout_enabled_in_spec AND _is_agent_mode AND snapshot_end is None``
gate (the new ValueError). The pre-V1.3 Option B agent loop (no scout)
does NOT require ``--snapshot-end`` — back-compat for the V1.1 phase4
smoke flow, which has been resuming without it since V1.1.

These tests construct minimal specs that ONLY exercise the validation
gate; they never load data (the validation fires before load_panel).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gbdt.__main__ import run_experiment


def _write_spec(
    out_dir: Path, artifact_dir: Path, *, scout_enabled: bool,
    callback_mode: str = "agent_file_protocol",
) -> Path:
    """Build a tiny synthetic spec that's syntactically valid through
    ``load_spec`` + ``_validate_spec``. The validation gate under test
    fires BEFORE any data load, so the spec doesn't need a real universe.
    """
    backend: dict = {
        "library": "catboost",
        "calibration_method": "conditional_isotonic",
        "fs_hp_loop": {
            "max_iterations": 4,
            "callback_mode": callback_mode,
        },
        "hp_starting": {
            "iterations": 10, "depth": 2, "learning_rate": 0.1,
            "boosting_type": "Plain",
        },
    }
    if scout_enabled:
        backend["scout"] = {"enabled": True}
        backend["fs_prefit"] = {"enabled": True, "cliff_pct": 0.01}

    spec = {
        "target": {
            "universe": "smoke_synth",
            "direction": "up",
            "threshold_pct": 5,
            "horizon_days": 10,
            "max_drawdown": None,
        },
        "split": {
            "train_rows": 180, "val_rows": 90, "eval_rows": 50,
            "test_rows": 30, "min_rows_per_ticker": 350,
        },
        "features": {
            "candidates": ["F2"], "lookback_windows": [5, 10, 20],
            "exclude": [],
        },
        "backend": backend,
        "artifacts": {"experiment_dir": str(artifact_dir)},
        "random_seed": 42,
    }
    spec_path = out_dir / "snap_end_validate.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


# ---------------------------------------------------------------------------
# POSITIVE — scout-enabled agent_file_protocol resume WITHOUT --snapshot-end
# raises ValueError.
# ---------------------------------------------------------------------------


def test_resume_without_snapshot_end_raises_when_scout_enabled(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "0")
    art_dir = tmp_path / "artifacts"
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    spec_path = _write_spec(spec_dir, art_dir, scout_enabled=True)

    # Pre-create the artifact dir + a fake "loop/progress.log" so the
    # ``--resume`` path enters the gate.
    cell_dir = art_dir / spec_path.stem
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "loop").mkdir(exist_ok=True)

    with pytest.raises(ValueError, match=r"--snapshot-end is REQUIRED"):
        run_experiment(spec_path, resume=spec_path.stem, snapshot_end=None)


# ---------------------------------------------------------------------------
# NEGATIVE — pre-V1.3 Option B specs (scout disabled) MUST NOT trip the gate.
# This protects the V1.1 phase4 smoke flow, which resumes without
# ``--snapshot-end`` and has done so since V1.1.
# ---------------------------------------------------------------------------


def test_resume_without_snapshot_end_ok_when_scout_disabled(
    tmp_path, monkeypatch,
):
    """When scout is NOT enabled in the spec, the new gate stays silent —
    the V1.1 ``--resume`` flow continues to work without ``--snapshot-end``.

    We can't easily assert "no ValueError raised AND the run continues"
    without standing up the synthetic-panel fixture from
    ``test_phase4_smoke.py``; the load_panel call WILL fail since we
    haven't patched it. So instead we assert that the SPECIFIC ValueError
    under test is NOT raised — any OTHER exception type (e.g., a data
    loading error) is fine and confirms we got PAST the gate.
    """
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "0")
    art_dir = tmp_path / "artifacts"
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    spec_path = _write_spec(spec_dir, art_dir, scout_enabled=False)

    cell_dir = art_dir / spec_path.stem
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "loop").mkdir(exist_ok=True)

    # We expect SOME exception (loading the panel, missing checkpoint, etc.)
    # but NOT the new --snapshot-end ValueError.
    with pytest.raises(Exception) as excinfo:
        run_experiment(spec_path, resume=spec_path.stem, snapshot_end=None)
    msg = str(excinfo.value)
    assert "--snapshot-end is REQUIRED" not in msg, (
        f"Pre-V1.3 Option B specs (scout disabled) MUST NOT trip the "
        f"new gate. Got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# NEGATIVE — scout-enabled non-agent (default mode) specs MUST NOT trip
# the gate either (the validation is restricted to agent_file_protocol).
# ---------------------------------------------------------------------------


def test_resume_without_snapshot_end_ok_when_default_mode(
    tmp_path, monkeypatch,
):
    """Default-mode specs (lex auto-compose inside walk_forward_train) don't
    have the multi-cycle exit-and-resume that motivates the rule. The gate
    must not fire."""
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "0")
    art_dir = tmp_path / "artifacts"
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    spec_path = _write_spec(
        spec_dir, art_dir, scout_enabled=True, callback_mode="default",
    )

    cell_dir = art_dir / spec_path.stem
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "loop").mkdir(exist_ok=True)

    with pytest.raises(Exception) as excinfo:
        run_experiment(spec_path, resume=spec_path.stem, snapshot_end=None)
    msg = str(excinfo.value)
    assert "--snapshot-end is REQUIRED" not in msg, (
        f"Default-mode specs MUST NOT trip the new gate (it's "
        f"agent_file_protocol-only). Got: {msg!r}"
    )
