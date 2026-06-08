"""V1.1 Phase 4 — end-to-end agent-loop smoke test.

Exercises the FULL exit -> decide -> resume -> finalize cycle through the real
``run_experiment(...)`` entry point (plan ``docs/gbdt/V1.1_agent_driven_fs_hp_
loop_plan.md`` § 10 Phase 4 row + § 11 Phase 4 test strategy). This is the first
integration test that drives the whole loop: the genuine
``PauseForAgentDecision`` catch (exit-and-resume's *exit* half), the
``--resume`` reload + decision validate/apply, and the ``force_stop``
finalization with a best-checkpoint retrain on a prior (None-placeholder) iter.

Harness level (per § 11 + the Phase-4 brief)
--------------------------------------------
**In-process, multi-call ``run_experiment(...)``** — three sequential calls per
cycle: a fresh launch (``resume=None``) and two ``--resume`` relaunches,
simulating the exit-and-relaunch the harness does in production. This is the
truest integration short of a subprocess: it runs the *actual* pause-catch (the
``except PauseForAgentDecision`` in ``run_experiment`` that returns cleanly,
NOT an error) and the *actual* ``_load_and_apply_resume`` reload across process
boundaries-modeled-as-calls.

A subprocess-level test (``python -m gbdt experiment ... --resume <id>``) would
be the literal end-to-end, but it needs a *real* seeded universe in the SQLite
cache for the subprocess to load — that is neither hermetic nor fast (NSE/US
feature builds + cache I/O are minutes). So we keep it fast + hermetic by
injecting a tiny synthetic panel at the smallest viable seam: monkeypatch
``gbdt.data.load_panel`` to return a synthetic ``UniversePanel``. EVERYTHING
downstream is real code — the real ``build_feature_matrix`` (6 cols, lookbacks
5/10/20), real ``build_target``, real uniqueness weights, real CatBoost fits
(iterations=15, depth=2, Plain boosting => sub-second each), real calibration,
real artifact emit + ``render_report``. The whole smoke runs in well under a
minute.

The 5 areas asserted (Phase-4 brief)
------------------------------------
1. Fresh launch pauses cleanly at iter 0 (returns the artifact dir, no error),
   writes the Phase-3 diagnose-shaped ``loop/iter_0_request.json`` (+ envelope
   keys) and ``loop/checkpoint.json``.
2. Scripted iter-0 decision (prune a real feature + an in-bounds ``hp_changes``)
   -> resume -> loads the checkpoint, validates+applies (feature pruned, HP
   changed), trains ONLY iter 1 (iters 0..N not re-trained), pauses at iter 1.
3. ``should_stop`` iter-1 decision -> resume -> finalizes WITHOUT training a new
   iter (``force_stop`` / ``agent_should_stop``), the best-checkpoint retrain
   works when the winner is a prior None-placeholder iter (``_fit_one``), and
   the final artifacts land (report.md / metrics.json / predictions / model /
   spec.yaml snapshot).
4. Determinism of the finalization retrain: re-running the whole cycle
   reproduces the prior config's model predictions bit-for-bit.
5. Negative path: an out-of-bounds / pinned-HP / unknown-feature decision on
   resume raises a clear ``DecisionError`` and does NOT corrupt state (the user
   fixes the file + relaunches).

All synthetic. No network, no real universe, no cache writes (artifacts land in
a tmp dir via ``artifacts.experiment_dir`` override).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import gbdt.data as gbdt_data
from gbdt.__main__ import run_experiment
from gbdt.checkpoint import read_checkpoint
from gbdt.data import TickerStatus, UniversePanel
from gbdt.loop_protocol import (
    DecisionError,
    REQUEST_SCHEMA_VERSION,
    decision_path,
    request_path,
)


# ---------------------------------------------------------------------------
# Synthetic universe panel (the only seam we patch) + tiny synthetic spec
# ---------------------------------------------------------------------------


def _synthetic_panel(n_per_ticker: int = 360, n_tickers: int = 4,
                     seed: int = 3) -> UniversePanel:
    """A tiny synthetic NSE-shaped panel: a handful of tickers, short history.

    Real OHLCV columns (so the real feature/target builders run unmodified) +
    a flat index series. ``stock_return_*`` / ``realized_vol_*`` over lookbacks
    5/10/20 yield 6 non-NaN feature columns; the +5%/10d target lands a usable
    positive prevalence. Each CatBoost fit on this panel is sub-second.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_per_ticker, freq="B")
    frames = []
    statuses = []
    for i in range(n_tickers):
        rets = rng.normal(0.0003, 0.012, n_per_ticker)
        c = 100.0 * np.exp(np.cumsum(rets))
        ticker = f"NSE:T{i}"
        frames.append(pd.DataFrame({
            "date": dates, "ticker": ticker,
            "open": c,
            "high": c * (1 + np.abs(rng.normal(0, 0.004, n_per_ticker))),
            "low": c * (1 - np.abs(rng.normal(0, 0.004, n_per_ticker))),
            "close": c, "adj_close": c,
            "volume": rng.integers(100_000, 500_000, n_per_ticker),
        }))
        statuses.append(TickerStatus(
            ticker=ticker, rows=n_per_ticker, kept=True, reason="",
            cache_last_date="2016-05-01", cache_age_days=1, is_stale=False,
        ))
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()

    ic = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n_per_ticker)))
    index_series = pd.DataFrame({
        "date": dates, "open": ic, "high": ic * 1.003, "low": ic * 0.997,
        "close": ic, "adj_close": ic,
        "volume": rng.integers(1_000_000, 5_000_000, n_per_ticker),
    }).set_index("date")

    return UniversePanel(
        universe="smoke_synth",
        panel=panel,
        index_series=index_series,
        annualization_factor=250,
        statuses=statuses,
        stale_tickers=[],
        staleness_days_threshold=gbdt_data.DEFAULT_STALENESS_DAYS,
    )


def _write_spec(out_dir, artifact_dir) -> "object":
    """Write a tiny synthetic spec; return its path.

    Overrides keep every fit sub-second (iterations=15, depth=2, Plain boosting)
    and confine artifacts to ``artifact_dir`` (an absolute tmp path — pathlib's
    ``repo_root / "/abs"`` collapses to the absolute path, so the real worktree
    repo_root still resolves ``default.yaml`` + preflight). ``max_iterations=8``
    keeps the loop in the agent's hands (no early inner-stop) so the scripted
    decisions drive the pauses, not the plateau gate.
    """
    spec = {
        "target": {
            "universe": "smoke_synth",
            "direction": "up",
            "threshold_pct": 5,
            "horizon_days": 10,
            "max_drawdown": None,
        },
        "split": {
            "train_rows": 180, "val_rows": 90, "eval_rows": 50, "test_rows": 30,
            "min_rows_per_ticker": 350,
        },
        "features": {
            "candidates": ["F2", "F4"],          # stock_return_* + realized_vol_*
            "lookback_windows": [5, 10, 20],     # 6 cols total
            "exclude": [],
        },
        "backend": {
            "library": "catboost",
            "calibration_method": "conditional_isotonic",
            "fs_hp_loop": {
                "max_iterations": 8,
                "callback_mode": "agent_file_protocol",
                # Loosen the inner-stop gates so plateau/degradation never fires
                # before the agent's scripted decisions do (the loop must hand
                # control back, not self-terminate, for the smoke to exercise
                # the pause/resume cycle).
                "plateau_threshold": 0.0,
                "degradation_gate": 1.0,
            },
            "hp_starting": {
                "iterations": 15, "depth": 2, "learning_rate": 0.1,
                "l2_leaf_reg": 3.0, "boosting_type": "Plain",
                "early_stopping_rounds": 10,
            },
        },
        "artifacts": {"experiment_dir": str(artifact_dir)},
        "random_seed": 42,
    }
    spec_path = out_dir / "smoke_synth.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


@pytest.fixture()
def smoke_env(tmp_path, monkeypatch):
    """Patch the data seam + quiet the heartbeat; yield (spec_path, art_dir).

    Only ``gbdt.data.load_panel`` is patched (the network/cache seam). Feature
    build, target, uniqueness, training, calibration, and artifact emit are all
    the real code paths. ``repo_root`` is the real worktree so ``default.yaml``
    + the preflight git fingerprint resolve; ``artifacts.experiment_dir`` is an
    absolute tmp path so nothing lands in the checked-in results tree.
    """
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "0")

    def _fake_load_panel(universe, *args, **kwargs):
        assert universe == "smoke_synth"
        return _synthetic_panel()

    monkeypatch.setattr(gbdt_data, "load_panel", _fake_load_panel)

    art_dir = tmp_path / "artifacts"
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    spec_path = _write_spec(spec_dir, art_dir)
    # The run_id printed in the pause hint == spec stem (run_experiment uses
    # ``name = spec_path.stem``); the artifact dir is <art_dir>/<stem>.
    run_id = spec_path.stem
    cell_dir = art_dir / run_id
    return spec_path, cell_dir, run_id


# ---------------------------------------------------------------------------
# Area 1 — fresh launch pauses cleanly at iter 0 with request + checkpoint
# ---------------------------------------------------------------------------


def test_fresh_launch_pauses_cleanly_at_iter0(smoke_env):
    spec_path, cell_dir, run_id = smoke_env

    # No exception escapes: run_experiment catches PauseForAgentDecision and
    # RETURNS the artifact dir (exit 0, a clean pause — not an error).
    returned = run_experiment(spec_path, resume=None)
    assert returned == cell_dir

    # The Phase-3 diagnose-shaped request landed at loop/iter_0_request.json.
    req_path = request_path(cell_dir, 0)
    assert req_path.exists(), f"request not written at {req_path}"
    req = json.loads(req_path.read_text())
    # Envelope keys (Phase-2 set, unchanged through Phase 3).
    assert set(req) == {
        "schema_version", "run_id", "iter", "max_iterations",
        "available_features", "diagnostics",
    }
    assert req["schema_version"] == REQUEST_SCHEMA_VERSION
    assert req["iter"] == 0
    assert req["run_id"] == run_id
    # 6 synthetic features available to prune (F2+F4 over 3 lookbacks).
    assert len(req["available_features"]) == 6
    assert "stock_return_5" in req["available_features"]
    # Diagnostics = the diagnose.json-shaped payload (Phase 3), not a raw dump.
    diag = req["diagnostics"]
    assert diag["source"] == "in_memory_iteration"
    assert diag["full_diagnose_available"] is False
    for key in ("overfit", "prevalence_drift", "calibration", "top_features",
                "per_day_p_at_k", "r_precision", "tuning_guidance",
                "feature_importance"):
        assert key in diag, f"missing diagnose key {key!r}"
    assert diag["metrics"]["val_brier"] is not None

    # The resume checkpoint landed (full loop state, NO model blob — plan § 0.2).
    ckpt = read_checkpoint(cell_dir)
    assert ckpt is not None
    assert ckpt["iter_idx"] == 0
    assert "models" not in ckpt
    assert "model.cbm" not in json.dumps(ckpt)
    assert ckpt["current_features"] == req["available_features"]
    assert len(ckpt["val_briers"]) == 1

    # No final artifacts yet — the run paused, it did not finalize.
    assert not (cell_dir / "metrics.json").exists()
    assert not (cell_dir / "model.cbm").exists()


# ---------------------------------------------------------------------------
# Area 2 + 3 — full cycle: resume(iter0 decision) -> pause(iter1) ->
#              resume(should_stop) -> finalize. The integration backbone.
# ---------------------------------------------------------------------------


def _drive_full_cycle(spec_path, cell_dir, run_id):
    """Run the complete fresh -> resume -> resume(should_stop) cycle.

    Returns the finalized artifact dir. The scripted "agent" writes an iter-0
    decision (prune one real feature + an in-bounds HP change) and an iter-1
    decision (should_stop=true).
    """
    # --- iter 0: fresh launch, pauses ---
    run_experiment(spec_path, resume=None)
    req0 = json.loads(request_path(cell_dir, 0).read_text())
    feats0 = req0["available_features"]
    pruned_feature = "realized_vol_20"      # a real, present feature
    assert pruned_feature in feats0

    # Scripted iter-0 decision: prune a real feature + a valid in-bounds HP
    # change (l2_leaf_reg 3.0 -> 5.0, within [0, 100]; depth 2 -> 3, within
    # [1, 16]). The agent plays data scientist.
    decision_path(cell_dir, 0).write_text(json.dumps({
        "iter": 0,
        "prune_features": [pruned_feature],
        "hp_changes": {"l2_leaf_reg": 5.0, "depth": 3},
        "should_stop": False,
        "rationale": "drop the slowest realized-vol window; deepen + raise L2.",
    }))

    # --- resume: applies iter-0 decision, trains iter 1, pauses at iter 1 ---
    run_experiment(spec_path, resume=run_id)
    req1_path = request_path(cell_dir, 1)
    assert req1_path.exists(), "iter 1 request not written on resume"
    req1 = json.loads(req1_path.read_text())
    assert req1["iter"] == 1
    # The pruned feature is gone from iter 1's active set; HP change applied.
    assert pruned_feature not in req1["available_features"]
    assert len(req1["available_features"]) == len(feats0) - 1
    assert req1["diagnostics"]["hp"]["depth"] == 3
    assert req1["diagnostics"]["hp"]["l2_leaf_reg"] == pytest.approx(5.0)

    # Checkpoint now at iter 1 with 2 prior val_briers (iter 0 + iter 1).
    ckpt1 = read_checkpoint(cell_dir)
    assert ckpt1["iter_idx"] == 1
    assert len(ckpt1["val_briers"]) == 2

    # --- iter-1 decision: agent stops ---
    decision_path(cell_dir, 1).write_text(json.dumps({
        "iter": 1,
        "should_stop": True,
        "rationale": "two iters explored; stopping per the agent's judgment.",
    }))

    # --- resume(should_stop): finalize without a new iteration ---
    final_dir = run_experiment(spec_path, resume=run_id)
    return final_dir


def test_full_cycle_resume_applies_decision_and_trains_only_next_iter(smoke_env):
    """Area 2: resume loads checkpoint, validates+applies the decision (feature
    pruned, HP changed), runs iter 1 ONLY (iters 0..N not re-trained), pauses
    again at iter 1."""
    spec_path, cell_dir, run_id = smoke_env

    run_experiment(spec_path, resume=None)
    req0 = json.loads(request_path(cell_dir, 0).read_text())
    feats0 = req0["available_features"]

    decision_path(cell_dir, 0).write_text(json.dumps({
        "iter": 0,
        "prune_features": ["realized_vol_20"],
        "hp_changes": {"depth": 3},
        "should_stop": False,
        "rationale": "prune one, deepen.",
    }))

    run_experiment(spec_path, resume=run_id)

    # Only iter 1 newly trained: the iter-1 request exists; the checkpoint shows
    # exactly 2 val_briers (the prior iter 0 threaded back + the new iter 1) —
    # iter 0 was NOT re-trained (its val_brier carried over from the checkpoint).
    req1 = json.loads(request_path(cell_dir, 1).read_text())
    assert req1["iter"] == 1
    assert "realized_vol_20" not in req1["available_features"]
    assert req1["diagnostics"]["hp"]["depth"] == 3
    ckpt = read_checkpoint(cell_dir)
    assert ckpt["iter_idx"] == 1
    assert len(ckpt["val_briers"]) == 2
    assert len(ckpt["feature_history"]) == 2
    # iter 0's recorded feature list keeps the full 6; iter 1's is the pruned 5.
    assert len(ckpt["feature_history"][0]) == len(feats0)
    assert len(ckpt["feature_history"][1]) == len(feats0) - 1


def test_full_cycle_should_stop_finalizes_and_emits_artifacts(smoke_env):
    """Area 3: the should_stop resume finalizes WITHOUT training a new iteration
    (force_stop / agent_should_stop), the best-checkpoint retrain works on a
    prior None-placeholder iter, and the full artifact set lands."""
    spec_path, cell_dir, run_id = smoke_env
    final_dir = _drive_full_cycle(spec_path, cell_dir, run_id)
    assert final_dir == cell_dir

    # Final artifacts present (the runner emits the full set on finalize).
    for name in ("metrics.json", "model.cbm", "calibration.pkl", "spec.yaml",
                 "features.yaml", "hp.yaml", "iterations.jsonl", "report.md"):
        assert (cell_dir / name).exists(), f"missing final artifact {name}"
    for seg in ("train", "val", "eval", "test"):
        assert (cell_dir / "predictions" / f"{seg}.csv").exists()

    metrics = json.loads((cell_dir / "metrics.json").read_text())
    # force_stop path: the loop finalized on the agent's should_stop signal.
    assert metrics["loop"]["inner_stop_signal"] == "agent_should_stop"
    # Issue #251: ``n_iterations_run`` is the TOTAL iters seen across the
    # loop's full history (prior resume-seeded iters + in-process iters).
    # On this should_stop resume the finalizing process trains none, but
    # iters 0 + 1 ran in earlier exit-resume cycles and survive in the
    # checkpoint — the metrics block reports the full count, not just the
    # in-process slice.
    assert metrics["loop"]["n_iterations_run"] == 2
    # The best checkpoint indexes into the FULL prior history (iters 0 + 1).
    assert metrics["loop"]["best_iteration"] in (0, 1)
    # Issue #251: best_val_brier now surfaced into metrics.json::loop.
    assert metrics["loop"]["best_val_brier"] is not None
    assert isinstance(metrics["loop"]["best_val_brier"], float)

    # The spec.yaml snapshot reflects the synthetic cell + agent_file_protocol.
    snap = yaml.safe_load((cell_dir / "spec.yaml").read_text())
    assert snap["target"]["universe"] == "smoke_synth"
    assert snap["backend"]["fs_hp_loop"]["callback_mode"] == "agent_file_protocol"

    # Eval predictions are populated + carry the calibrated-probability schema.
    eval_df = pd.read_csv(cell_dir / "predictions" / "eval.csv")
    assert len(eval_df) > 0
    assert {"date", "ticker", "p_raw", "p_calibrated", "y_true"}.issubset(
        eval_df.columns
    )

    # report.md mentions the run (sanity: render_report ran on the tiny panel).
    report = (cell_dir / "report.md").read_text()
    assert run_id in report


# ---------------------------------------------------------------------------
# Area 4 — determinism of the finalization retrain
# ---------------------------------------------------------------------------


def test_finalization_retrain_is_deterministic(tmp_path, monkeypatch):
    """Re-running the whole fresh->resume->resume cycle with identical scripted
    decisions reproduces the prior config's model predictions bit-for-bit. This
    is the load-bearing assumption behind the best-checkpoint ``_fit_one``
    retrain (the prior winning iter's model isn't carried in the checkpoint —
    it's re-fit at finalization, so the re-fit MUST be deterministic)."""
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "0")

    def _fake_load_panel(universe, *args, **kwargs):
        return _synthetic_panel()

    monkeypatch.setattr(gbdt_data, "load_panel", _fake_load_panel)

    def _run_cycle(tag: str):
        art_dir = tmp_path / f"art_{tag}"
        spec_dir = tmp_path / f"spec_{tag}"
        spec_dir.mkdir()
        spec_path = _write_spec(spec_dir, art_dir)
        run_id = spec_path.stem
        cell_dir = art_dir / run_id
        _drive_full_cycle(spec_path, cell_dir, run_id)
        return pd.read_csv(cell_dir / "predictions" / "eval.csv")

    a = _run_cycle("a")
    b = _run_cycle("b")

    # Same picks, same calibrated + raw probabilities, bit-for-bit.
    pd.testing.assert_frame_equal(
        a.sort_values(["date", "ticker"]).reset_index(drop=True),
        b.sort_values(["date", "ticker"]).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Area 5 — negative paths: a bad decision on resume raises a clear DecisionError
#          and does NOT corrupt state (the user fixes the file + relaunches).
# ---------------------------------------------------------------------------


def _pause_at_iter0(smoke_env):
    spec_path, cell_dir, run_id = smoke_env
    run_experiment(spec_path, resume=None)
    return spec_path, cell_dir, run_id


def _assert_state_uncorrupted_and_recoverable(spec_path, cell_dir, run_id):
    """After a rejected decision: the iter-0 request + checkpoint are intact and
    a CORRECTED decision lets the resume succeed (state was never corrupted)."""
    # Checkpoint + request still present + unchanged in shape.
    ckpt = read_checkpoint(cell_dir)
    assert ckpt is not None and ckpt["iter_idx"] == 0
    assert request_path(cell_dir, 0).exists()
    # No partial finalization happened.
    assert not (cell_dir / "metrics.json").exists()

    # The user fixes the decision file -> resume now succeeds (advances to iter 1).
    decision_path(cell_dir, 0).write_text(json.dumps({
        "iter": 0, "prune_features": ["realized_vol_20"],
        "hp_changes": {"depth": 3}, "should_stop": False,
        "rationale": "corrected decision.",
    }))
    run_experiment(spec_path, resume=run_id)
    assert request_path(cell_dir, 1).exists()
    assert read_checkpoint(cell_dir)["iter_idx"] == 1


def test_negative_out_of_bounds_hp_raises_and_state_recoverable(smoke_env):
    spec_path, cell_dir, run_id = _pause_at_iter0(smoke_env)
    # depth max is 16; 99 is out of bounds.
    decision_path(cell_dir, 0).write_text(json.dumps({
        "iter": 0, "hp_changes": {"depth": 99},
    }))
    with pytest.raises(DecisionError, match="outside the allowed range"):
        run_experiment(spec_path, resume=run_id)
    _assert_state_uncorrupted_and_recoverable(spec_path, cell_dir, run_id)


def test_negative_pinned_hp_raises_and_state_recoverable(smoke_env):
    spec_path, cell_dir, run_id = _pause_at_iter0(smoke_env)
    decision_path(cell_dir, 0).write_text(json.dumps({
        "iter": 0, "hp_changes": {"has_time": False},
    }))
    with pytest.raises(DecisionError, match="pinned"):
        run_experiment(spec_path, resume=run_id)
    _assert_state_uncorrupted_and_recoverable(spec_path, cell_dir, run_id)


def test_negative_unknown_feature_raises_and_state_recoverable(smoke_env):
    spec_path, cell_dir, run_id = _pause_at_iter0(smoke_env)
    decision_path(cell_dir, 0).write_text(json.dumps({
        "iter": 0, "prune_features": ["not_a_real_feature"],
    }))
    with pytest.raises(DecisionError, match="unknown feature"):
        run_experiment(spec_path, resume=run_id)
    _assert_state_uncorrupted_and_recoverable(spec_path, cell_dir, run_id)


def test_negative_malformed_json_raises_and_state_recoverable(smoke_env):
    spec_path, cell_dir, run_id = _pause_at_iter0(smoke_env)
    dp = decision_path(cell_dir, 0)
    dp.write_text("{ not valid json")
    with pytest.raises(DecisionError, match="not valid JSON"):
        run_experiment(spec_path, resume=run_id)
    _assert_state_uncorrupted_and_recoverable(spec_path, cell_dir, run_id)


def test_negative_missing_decision_raises_and_state_recoverable(smoke_env):
    spec_path, cell_dir, run_id = _pause_at_iter0(smoke_env)
    # No decision file written at all.
    with pytest.raises(DecisionError, match="not found"):
        run_experiment(spec_path, resume=run_id)
    _assert_state_uncorrupted_and_recoverable(spec_path, cell_dir, run_id)
