"""V1.1 Phase 3 — per-iteration request bundle = diagnose.json payload.

Covers (plan § 10 Phase 3 row + § 11 Phase 3 tests, recast for the in-memory
reuse design):

  - :func:`gbdt.diagnose_payload.build_diagnose_payload` assembles the
    diagnose.json-shaped dict from an in-memory ``DiagnosticBundle`` ALONE
    (no model, no rebuilt matrix), reusing the /gbdt-diagnose pure helpers
    (overfit read, prevalence-drift flag, tuning-guidance lines).
  - With a per-segment prediction frame threaded, the per-day P@k (min(R(d), k)
    denominator) + per-day variable-K R-precision + per-ticker + prediction-
    range sections populate from the canonical functions; without one they
    carry ``available: false`` (the in-loop default — the runner only carves
    calibrated predictions at finalization).
  - ``build_request_bundle`` now emits the diagnose-shaped ``diagnostics``
    payload while the loop-control envelope keys are byte-for-byte the Phase-2
    set; ``REQUEST_SCHEMA_VERSION`` is unchanged (the envelope didn't change).
  - The whole request is JSON-serializable + NaN/Inf-safe (round-trips through
    ``json.dumps`` / ``json.loads`` with no ``ValueError`` and no NaN tokens).
  - The Phase-2 pause -> checkpoint -> resume round-trip still works with the
    richer bundle (mocked training; no real CatBoost / feature build).

All synthetic / mocked — NO real feature build, NO model fit.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from gbdt import loop_protocol as LP
from gbdt.checkpoint import read_checkpoint
from gbdt.diagnose_payload import (
    DIAGNOSE_PAYLOAD_VERSION,
    build_diagnose_payload,
)
from gbdt.diagnostics import DiagnosticBundle
from gbdt.loop_protocol import (
    PauseForAgentDecision,
    build_request_bundle,
    request_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bundle(
    iter_n: int = 0,
    val_brier: float = 0.164,
    train_brier: float = 0.150,
    importance=None,
    spiegelhalter_z: float = 5.93,
    prevalence_val: float = 0.17,
):
    """A realistic in-memory DiagnosticBundle (no model needed)."""
    importance = importance or {
        "realized_vol_200": 0.0234,
        "index_return_50": 0.0189,
        "stock_return_5": 0.0001,   # below the 0.01 importance floor
        "dollar_move_z": 0.0080,    # below the floor
    }
    feats = list(importance)
    return DiagnosticBundle(
        iter=iter_n,
        hp={"depth": 6, "learning_rate": 0.05, "iterations": 2000,
            "boosting_type": "Plain"},
        features=feats, n_features=len(feats),
        train_brier=train_brier, val_brier=val_brier,
        train_val_gap=val_brier - train_brier,
        eval_brier_provisional=0.171,
        spiegelhalter_z=spiegelhalter_z, spiegelhalter_p=1e-4,
        reliability={"n_bins": 10, "points": []},
        positive_prevalence_val=prevalence_val, positive_recall_val=0.42,
        early_stop_iteration=180, iteration_cap_hit=False,
        importance_native=importance,
        importance_permutation=None,
        top_feature_correlation={"realized_vol_200": {"index_return_50": 0.3}},
        learning_curve={"validation_Logloss": [0.5, 0.45, 0.40]},
        effective_sample_size_train=1000.0, effective_sample_size_val=300.0,
        n_rows_train=1200, n_rows_val=350,
    )


def _pred_frame(n_days: int = 6, n_tickers: int = 8, seed: int = 0):
    """A synthetic (date, ticker, p_calibrated, y_true) val-prediction frame.

    Signal in p_calibrated correlates with y_true so per-day P@k / R-precision
    come out non-trivial (and finite).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for d in dates:
        for t in range(n_tickers):
            score = rng.uniform(0, 1)
            y = int(rng.uniform(0, 1) < (0.15 + 0.4 * score))  # score-correlated
            rows.append({"date": d, "ticker": f"NSE:T{t}",
                         "p_calibrated": score, "y_true": y})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_diagnose_payload — in-memory only (no predictions)
# ---------------------------------------------------------------------------


def test_payload_shape_from_bundle_only():
    p = build_diagnose_payload(
        _bundle(iter_n=3, val_brier=0.164),
        artifact_dir="/abs/results/gbdt/experiments/cell_x",
        cell={"universe": "nifty50", "direction": "up", "threshold_pct": 10,
              "horizon_days": 25, "max_drawdown": 0.05},
    )
    assert p["payload_version"] == DIAGNOSE_PAYLOAD_VERSION
    assert p["source"] == "in_memory_iteration"
    assert p["iter"] == 3
    assert p["cell"]["universe"] == "nifty50"
    assert p["artifact_dir"].endswith("cell_x")
    assert p["metrics"]["val_brier"] == pytest.approx(0.164)
    assert p["metrics"]["train_val_gap"] == pytest.approx(0.014)
    # Native importance carried through (free, in-memory).
    assert p["feature_importance"]["realized_vol_200"] == pytest.approx(0.0234)
    assert any(f["feature"] == "realized_vol_200" for f in p["top_features"])
    # Matrix-dependent analyses are deferred to the full on-disk diagnose.
    assert p["full_diagnose_available"] is False
    assert "marginal_monotonicity" in p["deferred_to_full_diagnose"]


def test_payload_reuses_overfit_helper():
    # gap = 0.164 - 0.150 = 0.014 <= 0.02 threshold => no overfit.
    p = build_diagnose_payload(_bundle(val_brier=0.164, train_brier=0.150))
    assert p["overfit"]["no_overfit"] is True
    assert any("NO OVERFIT" in g for g in p["tuning_guidance"])
    # A wide gap flips it.
    p2 = build_diagnose_payload(_bundle(val_brier=0.20, train_brier=0.12))
    assert p2["overfit"]["no_overfit"] is False
    assert any("OVERFIT signal" in g for g in p2["tuning_guidance"])


def test_payload_pruned_split_by_importance_floor():
    # Two of the four toy features are below the 0.01 floor.
    p = build_diagnose_payload(_bundle(), importance_threshold=0.01)
    assert p["pruned_summary"]["kept_count"] == 2
    assert p["pruned_summary"]["pruned_count"] == 2
    assert set(p["pruned_summary"]["pruned_features"]) == {
        "stock_return_5", "dollar_move_z"
    }


def test_payload_per_day_sections_unavailable_without_predictions():
    p = build_diagnose_payload(_bundle())
    for sect in ("per_day_p_at_k", "r_precision", "per_ticker_hit_rate",
                 "prediction_range"):
        assert p[sect]["available"] is False
        assert "reason" in p[sect]


# ---------------------------------------------------------------------------
# build_diagnose_payload — with a threaded prediction frame
# ---------------------------------------------------------------------------


def test_payload_per_day_sections_populate_with_predictions():
    preds = _pred_frame()
    p = build_diagnose_payload(_bundle(), val_predictions=preds)

    pk = p["per_day_p_at_k"]
    assert pk["available"] is True
    assert pk["base_rate"] is not None
    # min(R(d), k) denominator: by_k carries n_denom + n_days_R_lt_k per k.
    assert "1" in pk["by_k"] and "5" in pk["by_k"]
    assert pk["by_k"]["5"]["n_denom"] >= 0
    assert "n_days_R_lt_k" in pk["by_k"]["5"]
    # CLAUDE.md reporting convention: no lift field in the structured payload.
    assert "lift" not in pk["by_k"]["5"]

    rp = p["r_precision"]
    assert rp["available"] is True
    assert rp["weighted"] is None or 0.0 <= rp["weighted"] <= 1.0
    assert "per_day_quantiles" in rp

    pt = p["per_ticker_hit_rate"]
    assert pt["available"] is True
    assert isinstance(pt["rows"], list)

    pr = p["prediction_range"]
    assert pr["available"] is True
    assert pr["min"] is not None and pr["max"] is not None


def test_payload_empty_prediction_frame_is_unavailable():
    empty = pd.DataFrame(columns=["date", "ticker", "p_calibrated", "y_true"])
    p = build_diagnose_payload(_bundle(), val_predictions=empty)
    assert p["per_day_p_at_k"]["available"] is False


# ---------------------------------------------------------------------------
# JSON-serializable + NaN/Inf-safe
# ---------------------------------------------------------------------------


def _assert_no_nan_inf(obj):
    """Recursively assert no float NaN/Inf survived the _json_safe coercion."""
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_nan_inf(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_nan_inf(v)
    elif isinstance(obj, float):
        assert not (math.isnan(obj) or math.isinf(obj)), f"NaN/Inf leaked: {obj}"


def test_payload_is_json_serializable_and_nan_safe():
    # Inject NaN/Inf where a real bundle might carry them.
    b = _bundle()
    b.spiegelhalter_z = float("nan")
    b.importance_native = {"a": float("inf"), "b": 0.5, "c": float("nan")}
    b.val_brier = float("nan")
    p = build_diagnose_payload(b, val_predictions=_pred_frame())
    # strict json.dumps would raise on NaN/Inf if allow_nan default weren't
    # relied on; force allow_nan=False so any leak is a hard failure.
    s = json.dumps(p, allow_nan=False)
    rt = json.loads(s)
    _assert_no_nan_inf(rt)
    # The NaN/Inf fields coerced to None.
    assert rt["calibration"]["spiegelhalter_z"] is None
    assert rt["feature_importance"]["a"] is None   # was +inf
    assert rt["feature_importance"]["c"] is None    # was NaN
    assert rt["feature_importance"]["b"] == pytest.approx(0.5)


def test_full_request_bundle_is_json_serializable():
    payload = build_request_bundle(
        _bundle(iter_n=1), iter_n=1, run_id="cell_x", max_iterations=8,
        available_features=["realized_vol_200", "index_return_50"],
        artifact_dir="/abs/cell_x",
        cell={"universe": "nifty50"},
        val_predictions=_pred_frame(),
    )
    s = json.dumps(payload, allow_nan=False)
    rt = json.loads(s)
    _assert_no_nan_inf(rt)
    assert rt["diagnostics"]["source"] == "in_memory_iteration"


# ---------------------------------------------------------------------------
# build_request_bundle — envelope unchanged, payload richer
# ---------------------------------------------------------------------------


def test_request_envelope_keys_unchanged_payload_richer():
    payload = build_request_bundle(
        _bundle(iter_n=4), iter_n=4, run_id="cell_x", max_iterations=8,
        available_features=["realized_vol_200", "index_return_50"],
    )
    # Envelope = exactly the Phase-2 key set.
    assert set(payload) == {
        "schema_version", "run_id", "iter", "max_iterations",
        "available_features", "diagnostics",
    }
    assert payload["schema_version"] == LP.REQUEST_SCHEMA_VERSION
    assert payload["iter"] == 4
    assert payload["available_features"] == ["realized_vol_200", "index_return_50"]
    # The diagnostics payload is the diagnose-shaped dict.
    diag = payload["diagnostics"]
    assert diag["source"] == "in_memory_iteration"
    assert "tuning_guidance" in diag and "overfit" in diag


def test_request_schema_version_unchanged_at_v1():
    # The envelope didn't change in Phase 3, so REQUEST_SCHEMA_VERSION stays.
    assert LP.REQUEST_SCHEMA_VERSION == "v1"


# ---------------------------------------------------------------------------
# Pause -> checkpoint -> resume round-trip still works with the richer bundle
# ---------------------------------------------------------------------------


def _wired_callback(tmp_path, run_id="cell_x", max_iterations=8, cell=None):
    from gbdt.__main__ import _resolve_callback

    sink: dict = {}
    cb = _resolve_callback(
        {"callback_mode": "agent_file_protocol"}, run_id=run_id,
        artifact_dir=tmp_path, loop_state_sink=sink,
        max_iterations=max_iterations, cell=cell,
    )
    return cb, sink


def test_callback_writes_richer_request_and_checkpoint_then_pauses(tmp_path):
    cell = {"universe": "nifty50", "direction": "up", "threshold_pct": 10,
            "horizon_days": 25, "max_drawdown": 0.05}
    cb, sink = _wired_callback(tmp_path, cell=cell)
    sink.update({
        "iter_idx": 0,
        "current_features": ["realized_vol_200", "index_return_50"],
        "current_hp": {"depth": 6, "learning_rate": 0.05},
        "val_briers": [0.164],
        "hp_history": [{"iter": 0, "hp": {"depth": 6}}],
        "feature_history": [["realized_vol_200", "index_return_50"]],
        "hp_lists": [{"depth": 6, "learning_rate": 0.05}],
        "delta_attributions": [],
        "max_iterations": 8,
    })

    with pytest.raises(PauseForAgentDecision) as ei:
        cb(_bundle(iter_n=0, val_brier=0.164),
           ["realized_vol_200", "index_return_50"])

    # The richer diagnose-shaped payload landed in the request file.
    req = json.loads(request_path(tmp_path, 0).read_text())
    diag = req["diagnostics"]
    assert diag["source"] == "in_memory_iteration"
    assert diag["cell"]["universe"] == "nifty50"        # cell threaded through
    assert str(tmp_path) in diag["artifact_dir"]        # artifact_dir threaded
    assert diag["metrics"]["val_brier"] == pytest.approx(0.164)
    assert "tuning_guidance" in diag
    # In-loop default: no per-segment frame, so per-day sections unavailable.
    assert diag["per_day_p_at_k"]["available"] is False

    # Checkpoint still written with the full loop state (no model blobs).
    ckpt = read_checkpoint(tmp_path)
    assert ckpt is not None and ckpt["iter_idx"] == 0
    assert "models" not in ckpt
    assert ei.value.iter_n == 0


def test_resume_round_trip_unaffected_by_richer_bundle(monkeypatch, tmp_path):
    """The Phase-2 pause->checkpoint->resume control flow is untouched: a mocked
    walk_forward pauses at iter 0 (writing the richer request + checkpoint), and
    _load_and_apply_resume reads the checkpoint + a valid decision and seeds
    iter 1. No real CatBoost / feature build."""
    from gbdt.__main__ import _load_and_apply_resume
    from gbdt.loop_protocol import decision_path
    import gbdt.train as T

    cb, sink = _wired_callback(tmp_path)

    class _FakeModel:
        pass

    # train.py builds the per-iteration model via make_model(backend, ...)
    # (V1.2 backend seam) — patch the factory in train's namespace.
    monkeypatch.setattr(T, "make_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(_FakeModel, "fit", lambda self, *a, **k: None,
                        raising=False)
    monkeypatch.setattr(
        T, "build_diagnostic_bundle",
        lambda **kw: _bundle(iter_n=kw["iter_idx"], val_brier=0.164),
    )

    from gbdt.train import SplitSpec, walk_forward_train

    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=320, freq="B")
    frames = []
    for i in range(2):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 320)))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i}", "open": c, "high": c * 1.005,
            "low": c * 0.995, "close": c, "adj_close": c,
            "volume": np.ones(320, dtype=int)}))
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    n = len(panel)
    X = pd.DataFrame(rng.normal(0, 1, (n, 2)), index=panel.index,
                     columns=["realized_vol_200", "index_return_50"])
    y = ((X["realized_vol_200"] + rng.normal(0, 0.3, n)) > 0).astype(int)

    with pytest.raises(PauseForAgentDecision):
        walk_forward_train(
            panel=panel, X=X, y=y,
            features=["realized_vol_200", "index_return_50"],
            hp={"depth": 3}, split=SplitSpec(160, 80, 40, 20),
            max_iterations=8, fs_hp_callback=cb, loop_state_sink=sink,
        )

    # Agent writes a decision for iter 0; resume validates + applies it.
    dp = decision_path(tmp_path, 0)
    dp.write_text(json.dumps({
        "iter": 0, "prune_features": ["index_return_50"],
        "hp_changes": {"depth": 8}, "should_stop": False,
        "rationale": "drop the weaker feature, deepen",
    }))
    rs = _load_and_apply_resume(tmp_path, spec={}, run_id="cell_x")
    assert rs["iter_idx"] == 1
    assert rs["current_features"] == ["realized_vol_200"]   # pruned
    assert rs["current_hp"]["depth"] == 8                   # applied
    assert rs["force_stop"] is False
