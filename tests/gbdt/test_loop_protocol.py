"""V1.1 Phase 2 — exit-and-resume control flow + decision validation.

Covers (plan § 11 Phase 2, recast for exit-and-resume per § 0):
  - The agent_file_protocol callback writes ``iter_<N>_request.json`` + a
    checkpoint, then raises ``PauseForAgentDecision``.
  - ``run_experiment`` catches the pause and returns cleanly (exit 0) with the
    copy-pasteable ``--resume`` hint logged.
  - ``--resume``: given a checkpoint + a valid decision file, the loaded
    decision is validated, applied (features pruned + HP changed), and the loop
    advances to iter N+1 (seeded, not re-training 0..N).
  - ``validate_decision``: rejects out-of-bounds HP, a pinned-HP change, an
    unknown ``prune_features`` name, a bad ``should_stop`` type — each with a
    distinct error; accepts a valid decision; ``should_stop=true`` stops.
  - Round-trip: a checkpoint written at iter N reloads to the same loop state.

All synthetic / mocked — no real feature build. The genuine ``walk_forward_train``
resume-seeding test uses a 3-feature toy matrix with ``max_iterations`` tiny so
it runs in seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gbdt import loop_protocol as LP
from gbdt.checkpoint import read_checkpoint, write_checkpoint
from gbdt.diagnostics import DiagnosticBundle
from gbdt.loop_protocol import (
    DecisionError,
    PauseForAgentDecision,
    apply_decision,
    build_request_bundle,
    decision_path,
    read_decision,
    request_path,
    validate_decision,
    write_request,
)
from gbdt.train import SplitSpec, walk_forward_train


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _toy_bundle(iter_n: int = 0, features=None, val_brier: float = 0.16):
    """A minimal real DiagnosticBundle (no model required) for I/O tests."""
    features = features or ["sig", "n1", "n2"]
    return DiagnosticBundle(
        iter=iter_n,
        hp={"depth": 6, "learning_rate": 0.05, "iterations": 100,
            "boosting_type": "Plain"},
        features=list(features), n_features=len(features),
        train_brier=0.14, val_brier=val_brier, train_val_gap=0.02,
        eval_brier_provisional=None,
        spiegelhalter_z=1.0, spiegelhalter_p=0.3,
        reliability={}, positive_prevalence_val=0.4, positive_recall_val=0.4,
        early_stop_iteration=80, iteration_cap_hit=False,
        importance_native={"sig": 1.0, "n1": 0.5, "n2": 0.001},
        importance_permutation=None, top_feature_correlation={},
        learning_curve={},
    )


def _toy_panel(n_per_ticker: int = 320, n_tickers: int = 2, seed: int = 0):
    """A tiny panel + 3-feature matrix so a couple of real fits run in seconds."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_per_ticker, freq="B")
    frames = []
    for i in range(n_tickers):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_per_ticker)))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i}",
            "open": c, "high": c * 1.005, "low": c * 0.995,
            "close": c, "adj_close": c,
            "volume": np.ones(n_per_ticker, dtype=int),
        }))
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    n = len(panel)
    X = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=panel.index,
                     columns=["sig", "n1", "n2"])
    y = ((X["sig"] + rng.normal(0, 0.3, n)) > 0).astype(int)
    return panel, X, y


_TINY_SPLIT = SplitSpec(160, 80, 40, 20)  # sums to 300 ≤ 320 rows/ticker
_TINY_HP = {"iterations": 20, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.1}


# ---------------------------------------------------------------------------
# build_request_bundle / write_request / read_decision I/O
# ---------------------------------------------------------------------------


def test_build_request_bundle_wraps_diagnostics(tmp_path):
    b = _toy_bundle(iter_n=2, features=["sig", "n1"])
    payload = build_request_bundle(
        b, iter_n=2, run_id="cell_x", max_iterations=8,
        available_features=["sig", "n1"],
    )
    # Envelope (Phase 2 keys, unchanged in Phase 3).
    assert payload["schema_version"] == LP.REQUEST_SCHEMA_VERSION
    assert payload["iter"] == 2
    assert payload["run_id"] == "cell_x"
    assert payload["max_iterations"] == 8
    assert payload["available_features"] == ["sig", "n1"]
    # Phase 3: ``diagnostics`` is now the diagnose.json-shaped payload, not a
    # raw DiagnosticBundle.to_dict() dump.
    diag = payload["diagnostics"]
    assert diag["source"] == "in_memory_iteration"
    assert diag["iter"] == 2
    assert diag["metrics"]["val_brier"] == pytest.approx(0.16)
    # Richer diagnose-shaped keys are present (the Phase-3 swap).
    for key in ("overfit", "prevalence_drift", "calibration", "top_features",
                "per_day_p_at_k", "r_precision", "tuning_guidance",
                "full_diagnose_available"):
        assert key in diag, f"missing diagnose key {key!r}"


def test_write_request_lands_in_loop_subdir(tmp_path):
    payload = build_request_bundle(
        _toy_bundle(1), iter_n=1, run_id="r", max_iterations=8,
        available_features=["sig", "n1", "n2"],
    )
    p = write_request(tmp_path, 1, payload)
    assert p == request_path(tmp_path, 1)
    assert p == tmp_path / "loop" / "iter_1_request.json"
    rt = json.loads(p.read_text())
    assert rt["iter"] == 1


def test_read_decision_missing_raises(tmp_path):
    with pytest.raises(DecisionError, match="not found"):
        read_decision(tmp_path, 0)


def test_read_decision_malformed_json_raises(tmp_path):
    p = decision_path(tmp_path, 0)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ this is not json")
    with pytest.raises(DecisionError, match="not valid JSON"):
        read_decision(tmp_path, 0)


def test_read_decision_round_trip(tmp_path):
    p = decision_path(tmp_path, 3)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"iter": 3, "should_stop": False}))
    d = read_decision(tmp_path, 3)
    assert d["iter"] == 3 and d["should_stop"] is False


# ---------------------------------------------------------------------------
# validate_decision — distinct errors per violation class (plan § 5 + § 9)
# ---------------------------------------------------------------------------


_KNOWN = ["sig", "n1", "n2", "realized_vol_200"]


def test_validate_accepts_valid_decision():
    decision = {
        "prune_features": ["n2"],
        "hp_changes": {"learning_rate": 0.03, "depth": 8},
        "should_stop": False,
        "rationale": "drop noise, slow + deepen",
    }
    validate_decision(decision, spec=None, known_features=_KNOWN)  # no raise


def test_validate_rejects_out_of_bounds_hp():
    # depth max is 16 (model.py TUNABLE_HP_RANGES); 99 is out.
    decision = {"hp_changes": {"depth": 99}}
    with pytest.raises(DecisionError, match="outside the allowed range"):
        validate_decision(decision, spec=None, known_features=_KNOWN)


def test_validate_rejects_pinned_hp_change():
    decision = {"hp_changes": {"has_time": False}}
    with pytest.raises(DecisionError, match="pinned"):
        validate_decision(decision, spec=None, known_features=_KNOWN)


def test_validate_rejects_calibration_method_change():
    # calibration_method is pinned-by-policy for the loop (plan § 5).
    decision = {"hp_changes": {"calibration_method": "platt"}}
    with pytest.raises(DecisionError, match="pinned"):
        validate_decision(decision, spec=None, known_features=_KNOWN)


def test_validate_rejects_unknown_prune_feature():
    decision = {"prune_features": ["sig", "not_a_real_feature"]}
    with pytest.raises(DecisionError, match="unknown feature"):
        validate_decision(decision, spec=None, known_features=_KNOWN)


def test_validate_rejects_unknown_hp():
    decision = {"hp_changes": {"made_up_knob": 1.0}}
    with pytest.raises(DecisionError, match="unknown HP"):
        validate_decision(decision, spec=None, known_features=_KNOWN)


def test_validate_rejects_bad_should_stop_type():
    decision = {"should_stop": "yes"}
    with pytest.raises(DecisionError, match="should_stop"):
        validate_decision(decision, spec=None, known_features=_KNOWN)


def test_validate_rejects_non_dict_decision():
    with pytest.raises(DecisionError, match="JSON object"):
        validate_decision(["not", "a", "dict"], spec=None, known_features=_KNOWN)


def test_validate_rejects_non_list_prune():
    with pytest.raises(DecisionError, match="must be a list"):
        validate_decision({"prune_features": "sig"}, spec=None, known_features=_KNOWN)


def test_validate_rejects_bool_as_numeric_hp():
    # bool is a subclass of int — must not slip through the numeric check.
    with pytest.raises(DecisionError, match="must be numeric"):
        validate_decision({"hp_changes": {"learning_rate": True}}, spec=None,
                          known_features=_KNOWN)


def test_validate_enum_hp_accepts_valid_value():
    validate_decision({"hp_changes": {"bootstrap_type": "MVS"}}, spec=None,
                      known_features=_KNOWN)  # no raise


def test_validate_enum_hp_rejects_invalid_value():
    with pytest.raises(DecisionError, match="not one of"):
        validate_decision({"hp_changes": {"bootstrap_type": "Nonsense"}}, spec=None,
                          known_features=_KNOWN)


def test_validate_respects_spec_search_space_narrowing():
    # Spec narrows learning_rate to [0.01, 0.05]; 0.10 (within the canonical
    # 1e-4..1.0) is now out of bounds.
    spec = {"backend": {"fs_hp_loop": {"search_space": {
        "learning_rate": {"min": 0.01, "max": 0.05},
    }}}}
    with pytest.raises(DecisionError, match="outside the allowed range"):
        validate_decision({"hp_changes": {"learning_rate": 0.10}}, spec,
                          known_features=_KNOWN)
    # Within the narrowed band is fine.
    validate_decision({"hp_changes": {"learning_rate": 0.03}}, spec,
                      known_features=_KNOWN)


# ---------------------------------------------------------------------------
# V1.2 Phase 2 — backend-conditional HP validation (plan D3 + § 5.2)
# ---------------------------------------------------------------------------


def test_validate_default_backend_is_catboost():
    """No backend arg ⇒ CatBoost tables (back-compat). A CatBoost name validates;
    an XGBoost-only name is rejected as unknown."""
    validate_decision({"hp_changes": {"depth": 8}}, spec=None,
                      known_features=_KNOWN)  # CatBoost name OK by default
    with pytest.raises(DecisionError, match="unknown HP"):
        validate_decision({"hp_changes": {"max_depth": 8}}, spec=None,
                          known_features=_KNOWN)


def test_validate_xgboost_accepts_xgb_named_hp():
    validate_decision(
        {"hp_changes": {"max_depth": 8, "eta": 0.03, "lambda": 5.0}},
        spec=None, known_features=_KNOWN, backend="xgboost",
    )  # no raise


def test_validate_xgboost_rejects_catboost_only_hp():
    # `depth` is a CatBoost-only name; under the XGBoost backend it is unknown.
    with pytest.raises(DecisionError, match="unknown HP"):
        validate_decision({"hp_changes": {"depth": 8}}, spec=None,
                          known_features=_KNOWN, backend="xgboost")


def test_validate_xgboost_rejects_has_time_as_unknown():
    # XGBoost has no `has_time` concept — it is neither tunable nor a pin here,
    # so it is rejected as an unknown HP under the XGBoost backend.
    with pytest.raises(DecisionError, match="unknown HP"):
        validate_decision({"hp_changes": {"has_time": False}}, spec=None,
                          known_features=_KNOWN, backend="xgboost")


def test_validate_xgboost_rejects_pinned_determinism_knob():
    # tree_method is an XGBoost determinism pin — never overridable. The pin
    # is ``hist`` (post the speedup swap), so ``exact`` is the value an agent
    # might propose and that must be rejected.
    with pytest.raises(DecisionError, match="pinned"):
        validate_decision({"hp_changes": {"tree_method": "exact"}}, spec=None,
                          known_features=_KNOWN, backend="xgboost")


def test_validate_xgboost_out_of_range():
    with pytest.raises(DecisionError, match="outside the allowed range"):
        validate_decision({"hp_changes": {"max_depth": 99}}, spec=None,
                          known_features=_KNOWN, backend="xgboost")


def test_validate_xgboost_enum_accepts_and_rejects():
    validate_decision({"hp_changes": {"grow_policy": "lossguide"}}, spec=None,
                      known_features=_KNOWN, backend="xgboost")  # no raise
    with pytest.raises(DecisionError, match="not one of"):
        validate_decision({"hp_changes": {"grow_policy": "Nonsense"}}, spec=None,
                          known_features=_KNOWN, backend="xgboost")


def test_validate_catboost_rejects_xgboost_pinned_objective():
    # `objective` is an XGBoost pin, not in the CatBoost vocab ⇒ unknown HP.
    with pytest.raises(DecisionError, match="unknown HP"):
        validate_decision({"hp_changes": {"objective": "binary:logistic"}},
                          spec=None, known_features=_KNOWN, backend="catboost")


# ---------------------------------------------------------------------------
# apply_decision
# ---------------------------------------------------------------------------


def test_apply_decision_prunes_and_merges_hp():
    decision = {"prune_features": ["n2"], "hp_changes": {"depth": 8}}
    feats, hp, stop = apply_decision(
        decision, ["sig", "n1", "n2"], {"depth": 6, "learning_rate": 0.05},
    )
    assert feats == ["sig", "n1"]            # n2 pruned, order preserved
    assert hp == {"depth": 8, "learning_rate": 0.05}  # depth overwritten
    assert stop is False


def test_apply_decision_should_stop():
    feats, hp, stop = apply_decision(
        {"should_stop": True}, ["sig", "n1"], {"depth": 6},
    )
    assert stop is True
    assert feats == ["sig", "n1"] and hp == {"depth": 6}


def test_apply_decision_empty_is_noop():
    feats, hp, stop = apply_decision({}, ["sig", "n1"], {"depth": 6})
    assert feats == ["sig", "n1"] and hp == {"depth": 6} and stop is False


# ---------------------------------------------------------------------------
# PauseForAgentDecision exception shape
# ---------------------------------------------------------------------------


def test_pause_exception_carries_paths_and_run_id(tmp_path):
    exc = PauseForAgentDecision(
        iter_n=3, request_path=tmp_path / "r.json",
        checkpoint_path=tmp_path / "c.json", run_id="cell_x",
    )
    assert exc.iter_n == 3
    assert exc.run_id == "cell_x"
    assert "iter 3" in str(exc)
    assert exc.request_path.name == "r.json"


# ---------------------------------------------------------------------------
# Callback writes request + checkpoint, then raises (the exit half)
# ---------------------------------------------------------------------------


def _make_wired_callback(tmp_path, run_id="cell_x", max_iterations=8):
    """Build the agent_file_protocol callback with a populated loop_state_sink,
    mirroring how run_experiment wires it. Returns (callback, sink)."""
    from gbdt.__main__ import _resolve_callback

    sink = {}
    cb = _resolve_callback(
        {"callback_mode": "agent_file_protocol"}, run_id=run_id,
        artifact_dir=tmp_path, loop_state_sink=sink, max_iterations=max_iterations,
    )
    return cb, sink


def test_callback_writes_request_and_checkpoint_then_raises(tmp_path):
    cb, sink = _make_wired_callback(tmp_path, run_id="cell_x", max_iterations=8)
    # Populate the sink as walk_forward_train would before invoking the cb.
    sink.update({
        "iter_idx": 0,
        "current_features": ["sig", "n1", "n2"],
        "current_hp": {"depth": 6, "learning_rate": 0.05},
        "val_briers": [0.165],
        "hp_history": [{"iter": 0, "hp": {"depth": 6}}],
        "feature_history": [["sig", "n1", "n2"]],
        "hp_lists": [{"depth": 6, "learning_rate": 0.05}],
        "delta_attributions": [],
        "max_iterations": 8,
    })
    bundle = _toy_bundle(iter_n=0, val_brier=0.165)

    with pytest.raises(PauseForAgentDecision) as ei:
        cb(bundle, ["sig", "n1", "n2"])

    # Request file written with the bundle + envelope.
    req = json.loads(request_path(tmp_path, 0).read_text())
    assert req["iter"] == 0
    assert req["available_features"] == ["sig", "n1", "n2"]
    # Phase 3: diagnose.json-shaped payload under "diagnostics".
    assert req["diagnostics"]["metrics"]["val_brier"] == pytest.approx(0.165)
    assert req["diagnostics"]["source"] == "in_memory_iteration"

    # Checkpoint written with the full loop state (no model blobs).
    ckpt = read_checkpoint(tmp_path)
    assert ckpt is not None
    assert ckpt["iter_idx"] == 0
    assert ckpt["current_features"] == ["sig", "n1", "n2"]
    assert ckpt["val_briers"] == [0.165]
    assert "model" not in json.dumps(ckpt).lower() or "model.cbm" not in ckpt
    # No CatBoost blob serialized.
    assert all(k != "models" for k in ckpt)

    # Exception carries the iter + paths.
    assert ei.value.iter_n == 0
    assert ei.value.run_id == "cell_x"


# ---------------------------------------------------------------------------
# walk_forward_train pauses through the callback (control-flow integration,
# mocked training so it runs instantly)
# ---------------------------------------------------------------------------


def test_walk_forward_pauses_via_callback(monkeypatch, tmp_path):
    """With the agent_file_protocol callback wired + loop_state_sink, the loop
    trains iter 0, then the callback raises PauseForAgentDecision (after writing
    request + checkpoint). Training is mocked so no CatBoost fit runs."""
    cb, sink = _make_wired_callback(tmp_path, run_id="cell_x", max_iterations=8)

    # Mock the per-iteration fit + bundle so no real CatBoost runs.
    import gbdt.train as T

    class _FakeModel:
        pass

    # train.py builds the per-iteration model via make_model(backend, ...)
    # (V1.2 backend seam) — patch the factory in train's namespace so no real
    # CatBoost fit runs.
    monkeypatch.setattr(T, "make_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(_FakeModel, "fit", lambda self, *a, **k: None, raising=False)

    def _fake_bundle(**kwargs):
        return _toy_bundle(iter_n=kwargs["iter_idx"], val_brier=0.16)

    monkeypatch.setattr(T, "build_diagnostic_bundle", _fake_bundle)

    panel, X, y = _toy_panel()

    with pytest.raises(PauseForAgentDecision) as ei:
        walk_forward_train(
            panel=panel, X=X, y=y, features=["sig", "n1", "n2"],
            hp={"depth": 3}, split=_TINY_SPLIT, max_iterations=8,
            fs_hp_callback=cb, loop_state_sink=sink,
        )
    assert ei.value.iter_n == 0
    # Checkpoint + request landed.
    assert read_checkpoint(tmp_path) is not None
    assert request_path(tmp_path, 0).exists()


# ---------------------------------------------------------------------------
# walk_forward_train resume seeding — real (tiny) fits, runs in seconds
# ---------------------------------------------------------------------------


def test_walk_forward_resume_seeds_iter_n_plus_1():
    """Given a resume_state seeded at iter 1 with prior history, the loop trains
    iter 1 (not 0), uses the seeded features/HP, and finalizes. Best checkpoint
    can land on the prior (non-retrained) iter and gets retrained."""
    panel, X, y = _toy_panel(seed=7)
    resume_state = {
        "iter_idx": 1,
        "current_features": ["sig", "n1"],          # n2 pruned by the decision
        "current_hp": dict(_TINY_HP),
        "val_briers": [0.30],                         # prior iter 0 val_brier
        "hp_history": [{"iter": 0, "hp": dict(_TINY_HP)}],
        "feature_history": [["sig", "n1", "n2"]],     # iter 0 had all 3
        "hp_lists": [dict(_TINY_HP)],
        "delta_attributions": ["agent pruned n2"],
        "force_stop": False,
    }

    def stop_cb(bundle, available):
        # Force a stop after iter 1 by emitting should_stop semantics via the
        # default cap path: return same so plateau/cap fires, OR just shrink.
        return list(available), dict(bundle.hp), "noop"

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=["sig", "n1", "n2"],
        hp=dict(_TINY_HP), split=_TINY_SPLIT, max_iterations=2,
        fs_hp_callback=stop_cb, resume_state=resume_state,
    )
    # max_iterations=2 + seeding at iter 1 => caps after iter 1 (n=2 val_briers).
    assert len(result.iterations) == 1                # only iter 1 trained here
    assert result.iterations[0].iter == 1
    # Iter 1 used the seeded (pruned) feature set.
    assert set(result.iterations[0].features) == {"sig", "n1"}
    # val_briers history is [prior, this-iter]; best checkpoint over both.
    assert 0 <= result.best_iteration <= 1
    # A model is always emitted (retrained if best lands on prior iter 0).
    assert result.best_model is not None
    assert "test" in result.predictions


def test_walk_forward_resume_force_stop_finalizes_without_new_iter():
    """resume_state.force_stop (agent should_stop=true) finalizes at the prior
    history without training a new iteration; the best prior config is
    retrained for calibration/prediction."""
    panel, X, y = _toy_panel(seed=11)
    resume_state = {
        "iter_idx": 2,
        "current_features": ["sig", "n1"],
        "current_hp": dict(_TINY_HP),
        "val_briers": [0.30, 0.25],                   # two prior iters
        "hp_history": [{"iter": 0, "hp": dict(_TINY_HP)},
                       {"iter": 1, "hp": dict(_TINY_HP)}],
        "feature_history": [["sig", "n1", "n2"], ["sig", "n1"]],
        "hp_lists": [dict(_TINY_HP), dict(_TINY_HP)],
        "delta_attributions": ["d0", "d1"],
        "force_stop": True,
    }

    def boom_cb(bundle, available):  # must NOT be called when force_stop
        raise AssertionError("callback invoked despite force_stop")

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=["sig", "n1", "n2"],
        hp=dict(_TINY_HP), split=_TINY_SPLIT, max_iterations=8,
        fs_hp_callback=boom_cb, resume_state=resume_state,
    )
    assert result.inner_stop_signal == "agent_should_stop"
    assert len(result.iterations) == 0                # no new iteration trained
    # Best checkpoint = lowest prior val_brier (iter 1 @ 0.25) -> retrained.
    assert result.best_iteration == 1
    assert result.best_model is not None
    assert result.best_val_brier == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Checkpoint round-trip of loop state (plan § 11)
# ---------------------------------------------------------------------------


def test_checkpoint_loop_state_round_trip(tmp_path):
    state = {
        "run_id": "cell_x",
        "iter_idx": 2,
        "max_iterations": 8,
        "current_features": ["sig", "n1"],
        "current_hp": {"depth": 8, "learning_rate": 0.03},
        "val_briers": [0.30, 0.25, 0.24],
        "hp_history": [{"iter": 0, "hp": {"depth": 6}}],
        "feature_history": [["sig", "n1", "n2"], ["sig", "n1", "n2"], ["sig", "n1"]],
        "hp_lists": [{"depth": 6}, {"depth": 6}, {"depth": 8}],
        "delta_attributions": ["d0", "d1"],
    }
    write_checkpoint(tmp_path, state)
    loaded = read_checkpoint(tmp_path)
    for k, v in state.items():
        assert loaded[k] == v


# ---------------------------------------------------------------------------
# run_experiment-level resume loading (mocked, no data/feature build)
# ---------------------------------------------------------------------------


def test_load_and_apply_resume_applies_valid_decision(tmp_path):
    """_load_and_apply_resume reads the checkpoint + a valid decision, validates,
    applies (prune + hp), and returns the resume_state for iter N+1."""
    from gbdt.__main__ import _load_and_apply_resume

    out_dir = tmp_path / "cell_x"
    # Checkpoint paused at iter 0 with all 3 features.
    write_checkpoint(out_dir, {
        "run_id": "cell_x", "iter_idx": 0, "max_iterations": 8,
        "current_features": ["sig", "n1", "n2"],
        "current_hp": {"depth": 6, "learning_rate": 0.05},
        "val_briers": [0.30], "hp_history": [{"iter": 0, "hp": {"depth": 6}}],
        "feature_history": [["sig", "n1", "n2"]],
        "hp_lists": [{"depth": 6, "learning_rate": 0.05}],
        "delta_attributions": [],
    })
    # Agent decision for iter 0.
    dp = decision_path(out_dir, 0)
    dp.parent.mkdir(parents=True, exist_ok=True)
    dp.write_text(json.dumps({
        "iter": 0, "prune_features": ["n2"],
        "hp_changes": {"depth": 8, "learning_rate": 0.03},
        "should_stop": False, "rationale": "drop noise, slow + deepen",
    }))

    rs = _load_and_apply_resume(out_dir, spec={}, run_id="cell_x")
    assert rs["iter_idx"] == 1
    assert rs["current_features"] == ["sig", "n1"]       # n2 pruned
    assert rs["current_hp"]["depth"] == 8                # applied
    assert rs["current_hp"]["learning_rate"] == 0.03
    assert rs["force_stop"] is False
    # Prior history threaded back for the inner-stop check + best-checkpoint.
    assert rs["val_briers"] == [0.30]
    assert rs["feature_history"] == [["sig", "n1", "n2"]]


def test_load_and_apply_resume_rejects_bad_decision(tmp_path):
    from gbdt.__main__ import _load_and_apply_resume

    out_dir = tmp_path / "cell_x"
    write_checkpoint(out_dir, {
        "run_id": "cell_x", "iter_idx": 0, "max_iterations": 8,
        "current_features": ["sig", "n1", "n2"],
        "current_hp": {"depth": 6}, "val_briers": [0.30],
        "hp_history": [], "feature_history": [["sig", "n1", "n2"]],
        "hp_lists": [{"depth": 6}], "delta_attributions": [],
    })
    dp = decision_path(out_dir, 0)
    dp.parent.mkdir(parents=True, exist_ok=True)
    # depth=99 is out of bounds.
    dp.write_text(json.dumps({"iter": 0, "hp_changes": {"depth": 99}}))
    with pytest.raises(DecisionError, match="outside the allowed range"):
        _load_and_apply_resume(out_dir, spec={}, run_id="cell_x")


def test_load_and_apply_resume_missing_checkpoint_raises(tmp_path):
    from gbdt.__main__ import _load_and_apply_resume

    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        _load_and_apply_resume(tmp_path / "never_ran", spec={}, run_id="x")


def test_load_and_apply_resume_should_stop_sets_force_stop(tmp_path):
    from gbdt.__main__ import _load_and_apply_resume

    out_dir = tmp_path / "cell_x"
    write_checkpoint(out_dir, {
        "run_id": "cell_x", "iter_idx": 1, "max_iterations": 8,
        "current_features": ["sig", "n1"], "current_hp": {"depth": 6},
        "val_briers": [0.30, 0.25], "hp_history": [],
        "feature_history": [["sig", "n1", "n2"], ["sig", "n1"]],
        "hp_lists": [{"depth": 6}, {"depth": 6}], "delta_attributions": ["d0"],
    })
    dp = decision_path(out_dir, 1)
    dp.parent.mkdir(parents=True, exist_ok=True)
    dp.write_text(json.dumps({"iter": 1, "should_stop": True,
                              "rationale": "plateau, stopping"}))
    rs = _load_and_apply_resume(out_dir, spec={}, run_id="cell_x")
    assert rs["force_stop"] is True


# ---------------------------------------------------------------------------
# Task #204 — disable_plateau gate in walk_forward_train
#
# In default (sweep) mode the runner auto-stops on a single-knob val_brier
# plateau; in agent_file_protocol mode the agent should be allowed to pivot
# to a different knob first. The gate lives on walk_forward_train's
# ``disable_plateau`` kwarg, set by run_experiment from callback_mode.
# These tests exercise the kwarg directly with mocked training so we can
# control the val_brier history precisely.
# ---------------------------------------------------------------------------


def _patch_fit_with_briers(monkeypatch, briers):
    """Patch model + bundle + finalization so iter_idx i emits val_brier=briers[i].

    Also stubs ``_fit_one`` (the finalization retrain) + calibration helpers so
    the post-loop emit doesn't require a real fitted model — these tests only
    care about which inner_stop_signal is reached.
    """
    import numpy as np

    import gbdt.train as T

    class _FakeModel:
        def fit(self, *a, **k):
            return None

        def predict_proba(self, X):
            # 1-D P(positive) per BaseGBDTModel contract — calibration / metrics
            # don't affect inner_stop_signal, which is what these tests assert on.
            # Use a non-degenerate distribution so spiegelhalter_z variance > 0
            # and conditional_isotonic doesn't divide by zero.
            n = len(X)
            return np.linspace(0.2, 0.8, n)

    monkeypatch.setattr(T, "make_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(T, "_fit_one",
                        lambda *a, **k: _FakeModel(), raising=False)

    def _fake_bundle(**kwargs):
        i = kwargs["iter_idx"]
        vb = briers[i] if i < len(briers) else briers[-1]
        return _toy_bundle(iter_n=i, val_brier=vb)

    monkeypatch.setattr(T, "build_diagnostic_bundle", _fake_bundle)


def _noop_callback(bundle, available):
    """FS+HP callback that proposes no changes — keeps loop alive so plateau
    or cap is what stops it."""
    return list(available), dict(bundle.hp), "noop"


def test_walk_forward_default_mode_plateau_fires(monkeypatch):
    """Default (sweep) mode: tiny val_brier improvements MUST trigger plateau-stop.

    History [0.30, 0.28, 0.279, 0.278]: d_last=0.001, d_prev=0.001, both <
    plateau_threshold=0.005 → plateau fires at iter 3.
    """
    briers = [0.30, 0.28, 0.279, 0.278, 0.277, 0.276]
    _patch_fit_with_briers(monkeypatch, briers)
    panel, X, y = _toy_panel()

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=["sig", "n1", "n2"],
        hp=dict(_TINY_HP), split=_TINY_SPLIT, max_iterations=8,
        plateau_threshold=0.005,
        fs_hp_callback=_noop_callback,
        # disable_plateau defaults to False — explicit for the test contract.
        disable_plateau=False,
    )
    assert result.inner_stop_signal == "plateau"
    assert len(result.iterations) == 4  # iters 0..3; plateau at 3


def test_walk_forward_agent_mode_plateau_suppressed_runs_to_cap(monkeypatch):
    """Agent mode (disable_plateau=True): same tiny-improvement history MUST
    NOT trigger plateau-stop — the loop runs to ``max_iterations`` (cap)
    because the noop callback never proposes a structural pivot. In the real
    runner the agent would write a should_stop decision and the loop would
    finalize via force_stop instead; here we just prove plateau is gated off
    and ``cap`` (or another non-plateau signal) takes over.
    """
    briers = [0.30, 0.28, 0.279, 0.278, 0.277, 0.276]
    _patch_fit_with_briers(monkeypatch, briers)
    panel, X, y = _toy_panel()

    # max_iterations=5 so the loop terminates promptly under cap, not plateau.
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=["sig", "n1", "n2"],
        hp=dict(_TINY_HP), split=_TINY_SPLIT, max_iterations=5,
        plateau_threshold=0.005,
        fs_hp_callback=_noop_callback,
        disable_plateau=True,
    )
    assert result.inner_stop_signal == "cap"
    assert len(result.iterations) == 5  # ran past the plateau window


def test_walk_forward_agent_mode_degradation_still_fires(monkeypatch):
    """Agent mode: degradation_gate MUST still fire — regression off best val
    is a real stop signal the agent can't recover from without backtracking,
    so the runner keeps that gate active in both modes.

    History [0.30, 0.25, 0.30]: latest 0.30 > 1.01*0.25 → degradation at iter 2.
    """
    briers = [0.30, 0.25, 0.30, 0.30, 0.30]
    _patch_fit_with_briers(monkeypatch, briers)
    panel, X, y = _toy_panel()

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=["sig", "n1", "n2"],
        hp=dict(_TINY_HP), split=_TINY_SPLIT, max_iterations=8,
        plateau_threshold=0.005,
        degradation_gate=0.01,
        fs_hp_callback=_noop_callback,
        disable_plateau=True,
    )
    assert result.inner_stop_signal == "degradation"
    assert len(result.iterations) == 3  # iters 0..2; degradation at 2


def test_runner_disable_plateau_gate_present_in_source():
    """Source-level contract: ``run_experiment`` MUST gate ``disable_plateau``
    on ``callback_mode == "agent_file_protocol"`` and forward it as a kwarg to
    ``walk_forward_train``. End-to-end mocking of ``run_experiment`` would
    require stubbing panel + feature build + calibration; instead we assert
    the wiring is present in source so a future refactor that accidentally
    drops it (e.g. by hardcoding ``disable_plateau=False`` or removing the
    kwarg) fails this test loudly. The behaviour itself is covered by the
    unit + ``walk_forward_train`` integration tests above.
    """
    from pathlib import Path

    # run_experiment lives in gbdt.experiment_runner since the runner split.
    src = (Path(__file__).resolve().parents[2]
           / "src" / "gbdt" / "experiment_runner.py")
    text = src.read_text()
    # Gate computed from callback_mode.
    assert 'disable_plateau = (callback_mode == "agent_file_protocol")' in text
    # Forwarded to walk_forward_train.
    assert "disable_plateau=disable_plateau," in text
