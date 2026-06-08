"""Issue #251 — ``metrics.json::loop`` loop-history metadata regression test.

In ``agent_file_protocol`` callback mode the FS+HP loop runs across multiple
``--resume`` invocations: each iter (or batch of iters) trains, writes a
checkpoint, pauses, and exits. Finalization happens in the LAST resume process,
which is the only one that writes ``metrics.json``. Prior-iter ``DiagnosticBundle``
objects are NOT carried in the checkpoint (no model blobs; plan § 0.2), so
``WalkForwardResult.iterations`` reflects ONLY the in-process bundles at
finalize — which on a should_stop resume is 0, and on a one-iter-per-resume
cadence is 1 regardless of how many iters the loop actually ran.

Six production cells were observed with the metadata misreporting the loop's
shape (issue #251 ticket body): ``n_iterations_run = 0`` despite multi-iter
runs evidenced by the checkpoint's ``val_briers`` list + ``progress.log``.

The fix surfaces ``WalkForwardResult.n_iterations_total`` (== ``len(val_briers)``,
the prior-seeded + in-process count) into ``metrics.json::loop.n_iterations_run``
and lifts ``best_val_brier`` (already on the result) into the same block so the
loop block self-describes without forcing a cross-reference to ``status.json``
or ``iterations.jsonl``.

The integration-level coverage lives in ``test_phase4_smoke.py`` (the full
fresh -> resume -> resume(should_stop) cycle exercises the real
``run_experiment`` and finalize paths); this file is the targeted regression
that pins the per-field behaviour at the metrics-emission seam — fast +
hermetic via a constructed ``WalkForwardResult`` so a regression in field
naming/type lands as a one-test failure instead of a noisy smoke break.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import gbdt.data as gbdt_data
from gbdt.__main__ import run_experiment
from gbdt.checkpoint import read_checkpoint
from gbdt.loop_protocol import decision_path
from tests.gbdt.test_phase4_smoke import _synthetic_panel, _write_spec  # type: ignore


# ---------------------------------------------------------------------------
# Targeted unit: WalkForwardResult.n_iterations_total threads cleanly
# ---------------------------------------------------------------------------


def test_walk_forward_result_carries_n_iterations_total():
    """``WalkForwardResult`` exposes ``n_iterations_total`` as a typed int
    field defaulting to 0 (back-compat for tests constructing the result
    directly). ``train.py``'s ``walk_forward_train`` populates it with
    ``len(val_briers)`` — the prior-seeded + in-process count — which on the
    default path equals ``len(history)`` and on the agent_file_protocol
    finalize path can exceed it."""
    from gbdt.train import WalkForwardResult

    # The field exists on the dataclass + defaults to 0.
    fields = {f.name for f in WalkForwardResult.__dataclass_fields__.values()}
    assert "n_iterations_total" in fields

    # A constructed-without-the-field instance defaults the count to 0 (the
    # back-compat guarantee for downstream code that builds results inline).
    from gbdt.calibration import CalibrationDecision

    class _StubModel:
        def predict_proba(self, X):
            return np.zeros(len(X))

        def save(self, p):
            Path(p).write_bytes(b"")

    r = WalkForwardResult(
        best_iteration=0,
        best_model=_StubModel(),
        best_features=["f"],
        best_hp={},
        best_val_brier=0.25,
        iterations=[],
        calibration=CalibrationDecision(
            method="native", spiegelhalter_z=0.0, spiegelhalter_p=1.0,
            z_threshold=2.0, calibrator=None, rationale="",
        ),
        inner_stop_signal="cap",
    )
    assert r.n_iterations_total == 0

    # Explicit construction respects the override.
    r2 = WalkForwardResult(
        best_iteration=0,
        best_model=_StubModel(),
        best_features=["f"],
        best_hp={},
        best_val_brier=0.25,
        iterations=[],
        calibration=CalibrationDecision(
            method="native", spiegelhalter_z=0.0, spiegelhalter_p=1.0,
            z_threshold=2.0, calibrator=None, rationale="",
        ),
        inner_stop_signal="cap",
        n_iterations_total=5,
    )
    assert r2.n_iterations_total == 5


# ---------------------------------------------------------------------------
# Integration: agent_file_protocol finalize writes the full loop block
# ---------------------------------------------------------------------------


@pytest.fixture()
def _smoke_env(tmp_path, monkeypatch):
    """Mirror of ``test_phase4_smoke.smoke_env`` — synthetic panel + spec so
    the multi-call ``run_experiment`` cycle is fast + hermetic."""
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "0")

    def _fake_load_panel(universe, *args, **kwargs):
        assert universe == "smoke_synth"
        return _synthetic_panel()

    monkeypatch.setattr(gbdt_data, "load_panel", _fake_load_panel)

    art_dir = tmp_path / "artifacts"
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    spec_path = _write_spec(spec_dir, art_dir)
    run_id = spec_path.stem
    cell_dir = art_dir / run_id
    return spec_path, cell_dir, run_id


def test_agent_file_protocol_finalize_writes_loop_metadata(_smoke_env):
    """Drive a 3-iter agent_file_protocol cycle:

      iter 0: fresh launch -> pauses (checkpoint has 1 val_brier)
      iter 1: scripted continue decision -> trains iter 1 -> pauses
              (checkpoint has 2 val_briers)
      iter 2: scripted should_stop=true -> finalize (force_stop path; no new
              iter trained — best_checkpoint retrains a prior config)

    Assert ``metrics.json::loop`` reports:

      n_iterations_run == 2    (the loop's TOTAL count — NOT the 0 the bug
                                surfaced; NOT the 1 a 1-resume-per-iter cadence
                                would surface if only ``len(history)`` was
                                consulted)
      best_iteration in (0, 1) (the best_checkpoint pick across the full prior
                                history)
      best_val_brier is a float (the val Brier at best_iteration, surfaced
                                 into metrics.json::loop by the #251 fix —
                                 previously this lived in status.json only)
      inner_stop_signal == "agent_should_stop"  (the force_stop terminal)
      tiebreak_path is a known label  (V1.4 P2's plumbing, sanity-check the
                                       loop block is fully populated)
    """
    spec_path, cell_dir, run_id = _smoke_env

    # --- iter 0: fresh launch, pauses ---
    run_experiment(spec_path, resume=None)
    ckpt0 = read_checkpoint(cell_dir)
    assert ckpt0 is not None and len(ckpt0["val_briers"]) == 1
    feats0 = list(ckpt0["current_features"])

    # iter 0 decision: prune one feature, in-bounds HP nudge, continue.
    decision_path(cell_dir, 0).write_text(json.dumps({
        "iter": 0,
        "prune_features": ["realized_vol_20"],
        "hp_changes": {"depth": 3},
        "should_stop": False,
        "rationale": "prune one, deepen.",
    }))

    # --- iter 1: resume, trains iter 1, pauses ---
    run_experiment(spec_path, resume=run_id)
    ckpt1 = read_checkpoint(cell_dir)
    assert ckpt1 is not None and len(ckpt1["val_briers"]) == 2

    # iter 1 decision: should_stop=true triggers force_stop finalize path
    # (loop body skipped — no in-process bundle built; ``history`` stays empty
    # at finalize, which is the exact scenario that surfaced #251).
    decision_path(cell_dir, 1).write_text(json.dumps({
        "iter": 1,
        "prune_features": [],
        "hp_changes": {},
        "should_stop": True,
        "rationale": "stop now.",
    }))

    # --- iter 2: resume(should_stop) -> finalize ---
    run_experiment(spec_path, resume=run_id)

    metrics_path = cell_dir / "metrics.json"
    assert metrics_path.exists(), "finalize did not emit metrics.json"
    metrics = json.loads(metrics_path.read_text())
    loop = metrics.get("loop", {})

    # Issue #251 — the core regression: total iter count across the loop's
    # full history (NOT just the in-process slice at finalize).
    assert loop.get("n_iterations_run") == 2, (
        f"n_iterations_run should be the TOTAL loop iters (2), got "
        f"{loop.get('n_iterations_run')}; checkpoint had "
        f"{len(ckpt1['val_briers'])} val_briers"
    )

    # Issue #251 — best_val_brier surfaced into metrics.json::loop. The
    # field was previously only in status.json; the bug ticket showed it as
    # null in the metrics block. It's a float (NOT None) here because the
    # loop completed at least one iter.
    assert "best_val_brier" in loop, (
        "best_val_brier missing from metrics.json::loop (#251 surfacing)"
    )
    assert loop["best_val_brier"] is not None
    assert isinstance(loop["best_val_brier"], float)

    # Sanity: best_iteration + inner_stop_signal + tiebreak_path remain
    # populated correctly (these were not the regressed fields, but the bug
    # report quoted all four as null — pin them here so a future refactor
    # that breaks any single field surfaces as a focused failure).
    assert loop.get("best_iteration") in (0, 1)
    assert loop.get("inner_stop_signal") == "agent_should_stop"
    # V1.4 P2's tiebreak_path label is one of the documented 5; just assert
    # it's a non-empty string here (the smoke test asserts the specific
    # branch).
    assert isinstance(loop.get("tiebreak_path"), str)
    assert loop["tiebreak_path"]
