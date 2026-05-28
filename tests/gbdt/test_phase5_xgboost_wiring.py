"""V1.2 Phase 5 — XGBoost end-to-end wiring smoke.

Locks in the Phase-5 acceptance row (``docs/gbdt/V1.2_xgboost_feature_
interactions_plan.md`` § 8 Phase-5 + § 6.1/§ 6.2/§ 6.4 + D7):

> *end-to-end smoke: an ``xgboost`` spec runs through ``walk_forward_train``;
> ``diagnose.json`` carries the ``interactions`` block; the ``/gbdt-diagnose``
> loader picks the right backend from ``spec.yaml``. Resume-checkpoint round-trip
> with an XGBoost HP dict.*

This is a TINY synthetic-panel smoke (3 tickers, one short fold, low iterations),
NOT a data-backed run — the panel ingestion is tested elsewhere; here we exercise
the *wiring*: backend threads into ``walk_forward_train``; the persisted model is
``model.ubj`` (backend-determined filename, single source of truth
``gbdt.model.model_filename``); the diagnose loader dispatches on
``spec.yaml::backend.library`` and loads the ``.ubj`` via ``XGBoostModel.load``;
the D7 ``interactions`` block lands in ``diagnose.json`` with ``method="shap"``
(SHAP for the xgboost artifact); and a resume checkpoint (which stores NO model
blob) round-trips, with the finalization retrain reproducing the fit
bit-identically (the Phase-3 determinism guarantee).

Contract item-by-item:
  1. runner accepts xgboost specs              → test_xgboost_spec_accepted_by_validator
  2. backend-determined model filename         → test_walk_forward_xgboost_persists_ubj
  3. /gbdt-diagnose loader backend dispatch    → test_diagnose_loads_xgboost_via_dispatch
  4. D7 interactions block (method="shap")     → test_diagnose_carries_d7_interactions_block
  resume round-trip + determinism              → test_resume_roundtrip_with_xgboost_hp
  CatBoost path unchanged                      → test_catboost_path_unchanged_end_to_end
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from gbdt import checkpoint as gbdt_checkpoint
from gbdt import loop_protocol
from gbdt.__main__ import _load_and_apply_resume, _validate_spec
from gbdt.model import CatBoostModel, XGBoostModel, model_filename
from gbdt.train import SplitSpec, walk_forward_train

# Tiny deterministic XGBoost HP — small + fast (smallest viable iterations/depth).
_HP_XGB = {"n_estimators": 40, "max_depth": 3, "eta": 0.3, "lambda": 1.0}
_HP_CB = {"iterations": 30, "depth": 3, "boosting_type": "Plain", "learning_rate": 0.1}

# A short fold keeps the smoke fast while preserving walk-forward boundary order.
_SPLIT = SplitSpec(train_rows=180, val_rows=80, eval_rows=40, test_rows=20)
_MIN_ROWS = 320  # = 180 + 80 + 40 + 20


def _toy_panel(n_per_ticker: int = _MIN_ROWS, n_tickers: int = 3, seed: int = 0):
    """Synthetic OHLCV panel + a small (1 signal + 1 interaction-partner + noise)
    feature matrix with an XOR-flavoured target so there is *some* interaction
    structure for the SHAP pass to surface (the smoke only asserts the block
    exists + is shap-typed, not a particular ranking)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_per_ticker, freq="B")
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
    n_total = len(panel)
    a = rng.normal(0, 1, n_total)
    b = rng.normal(0, 1, n_total)
    X = pd.DataFrame(
        {"a": a, "b": b,
         "n1": rng.normal(0, 1, n_total), "n2": rng.normal(0, 1, n_total)},
        index=panel.index,
    )
    y = pd.Series(((a > 0) ^ (b > 0)).astype(int), index=panel.index)
    return panel, X, y


def _run_xgb(seed: int = 0):
    panel, X, y = _toy_panel(seed=seed)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp=dict(_HP_XGB), split=_SPLIT, max_iterations=2,
        backend="xgboost", random_seed=42,
    )
    return result, panel, X, y


# ---------------------------------------------------------------------------
# Contract 1 — the runner's spec validator accepts xgboost specs
# ---------------------------------------------------------------------------


def _xgb_spec() -> dict:
    return {
        "target": {"universe": "nifty50", "direction": "up",
                   "threshold_pct": 10, "horizon_days": 25},
        "backend": {"library": "xgboost", "hp_starting": dict(_HP_XGB)},
    }


def test_xgboost_spec_accepted_by_validator():
    # The old hard reject ("v1 supports backend.library='catboost' only") is
    # lifted — an xgboost spec with XGBoost-named hp_starting validates clean.
    _validate_spec(_xgb_spec())  # no raise


def test_xgboost_spec_rejects_out_of_range_hp():
    # The runner validates an xgboost spec's hp_starting against the *_XGB tables
    # at parse time (fail-fast before any data is loaded). An out-of-range
    # XGBoost-named HP (eta above the documented [1e-4, 1.0] range) is rejected.
    spec = _xgb_spec()
    spec["backend"]["hp_starting"] = {"eta": 5.0, "max_depth": 4}
    with pytest.raises(ValueError, match="hp_starting is invalid"):
        _validate_spec(spec)


def test_xgboost_spec_rejects_determinism_pin_override():
    # Overriding an XGBoost determinism pin (tree_method) in hp_starting is a
    # hard-fail at parse time (the Phase-3 determinism guarantee, plan § 5.1).
    spec = _xgb_spec()
    spec["backend"]["hp_starting"] = {"eta": 0.3, "tree_method": "hist"}
    with pytest.raises(ValueError, match="hp_starting is invalid"):
        _validate_spec(spec)


def test_unknown_backend_rejected():
    spec = _xgb_spec()
    spec["backend"]["library"] = "lightgbm"
    with pytest.raises(ValueError, match="backend.library must be in"):
        _validate_spec(spec)


# ---------------------------------------------------------------------------
# Contract 2 — end-to-end run + backend-determined model filename (model.ubj)
# ---------------------------------------------------------------------------


def test_walk_forward_xgboost_runs_to_completion():
    result, _panel, _X, _y = _run_xgb()
    assert isinstance(result.best_model, XGBoostModel)
    assert result.best_model.fitted
    assert result.inner_stop_signal in ("plateau", "cap", "degradation")
    for seg in ("train", "val", "eval", "test"):
        df = result.predictions[seg]
        assert set(df.columns) >= {"date", "ticker", "p_raw", "p_calibrated", "y_true"}


def test_walk_forward_xgboost_persists_ubj(tmp_path):
    result, _panel, _X, _y = _run_xgb()
    # The backend-determined filename is the single source of truth in gbdt.model.
    assert model_filename("xgboost") == "model.ubj"
    model_path = tmp_path / model_filename("xgboost")
    result.best_model.save(model_path)
    assert model_path.exists() and model_path.name == "model.ubj"


# ---------------------------------------------------------------------------
# Helper — build a minimal on-disk xgboost artifact dir
# ---------------------------------------------------------------------------


def _write_xgb_artifact(artifact_dir, result, X, y) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "predictions").mkdir(exist_ok=True)
    # backend-determined model file
    result.best_model.save(artifact_dir / model_filename("xgboost"))
    # spec.yaml carries backend.library=xgboost — the loader-dispatch key.
    spec = {
        "target": {"universe": "toy_smoke", "direction": "up",
                   "threshold_pct": 10, "horizon_days": 25},
        "split": {"test_rows": _SPLIT.test_rows},
        "backend": {"library": "xgboost"},
    }
    (artifact_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (artifact_dir / "features.yaml").write_text(
        yaml.safe_dump({"features": list(X.columns)})
    )
    for seg, df in result.predictions.items():
        df.to_csv(artifact_dir / "predictions" / f"{seg}.csv", index=False)


# ---------------------------------------------------------------------------
# Contract 3 — /gbdt-diagnose loader picks the backend from spec.yaml
# ---------------------------------------------------------------------------


def test_diagnose_loads_xgboost_via_dispatch(tmp_path):
    from scripts.gbdt import diagnose as diag

    result, _panel, X, y = _run_xgb()
    art = tmp_path / "xgb_cell"
    _write_xgb_artifact(art, result, X, y)

    cell = diag.load_cell(art)
    # The loader read backend.library from spec.yaml, built the XGBoost wrapper
    # via the shared dispatch, and loaded model.ubj back through XGBoostModel.load.
    assert cell["backend"] == "xgboost"
    assert isinstance(cell["model"], XGBoostModel)
    assert cell["model"].fitted
    # backend-neutral surface: 1-D predict_proba, feature_names (NOT feature_names_)
    proba = cell["model"].predict_proba(X[list(X.columns)].head(10))
    assert proba.ndim == 1 and ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------
# Contract 4 — D7 interactions block (method="shap" for an xgboost artifact)
# ---------------------------------------------------------------------------


def test_diagnose_carries_d7_interactions_block(tmp_path, monkeypatch):
    from scripts.gbdt import diagnose as diag

    result, _panel, X, y = _run_xgb()
    art = tmp_path / "xgb_cell"
    _write_xgb_artifact(art, result, X, y)

    # Avoid the data-cache panel rebuild — feed the smoke's own in-sample matrix.
    feats = list(X.columns)
    X_in = X.reset_index(drop=True)[feats]
    y_in = y.reset_index(drop=True).astype(int).values
    monkeypatch.setattr(diag, "build_insample", lambda *a, **k: (X_in, y_in))

    bundle = diag.diagnose(art, top_n=4, do_pdp=False, do_figs=False)

    # bundle records which backend produced it
    assert bundle["backend"] == "xgboost"

    # D7 block present + shap-typed (SHAP for the xgboost artifact).
    inter = bundle["interactions"]
    assert inter["method"] == "shap"
    assert inter["n_rows_used"] > 0
    assert set(inter["per_feature_involvement"]) == set(feats)
    # main-effect map carried (the drop-only-if-low-main-AND-low-interaction input).
    assert set(inter["per_feature_main_effect"]) == set(feats)
    # top_pairs are [a, b, strength, sign] quads; sign is a float or None.
    assert inter["top_pairs"]
    for a, b, s, sign in inter["top_pairs"]:
        assert a in feats and b in feats
        assert isinstance(s, float)
        assert sign is None or isinstance(sign, float)

    # The on-disk diagnose.json is JSON-serialisable + carries the block.
    disk = json.loads((art / "diagnose" / "diagnose.json").read_text())
    assert disk["interactions"]["method"] == "shap"


def test_diagnose_interactions_full_flag_lifts_row_cap(tmp_path, monkeypatch):
    """`--interactions full` lifts the SHAP row cap so the whole in-sample slice
    is scored (default `summary` caps the streamed aggregate at 5000 rows)."""
    from scripts.gbdt import diagnose as diag

    result, _panel, X, y = _run_xgb()
    art = tmp_path / "xgb_cell"
    _write_xgb_artifact(art, result, X, y)

    feats = list(X.columns)
    X_in = X.reset_index(drop=True)[feats]
    y_in = y.reset_index(drop=True).astype(int).values
    monkeypatch.setattr(diag, "build_insample", lambda *a, **k: (X_in, y_in))

    bundle = diag.diagnose(art, top_n=4, do_pdp=False, do_figs=False,
                           interactions="full")
    # full → every in-sample row scored (panel is < 5000 rows, so both paths
    # score all rows here; the assertion is that "full" honours the whole slice).
    assert bundle["interactions"]["n_rows_used"] == len(X_in)


# ---------------------------------------------------------------------------
# Resume-checkpoint round-trip with an XGBoost HP dict + determinism
# ---------------------------------------------------------------------------


def test_resume_roundtrip_with_xgboost_hp(tmp_path):
    """A resume checkpoint stores NO model blob — the finalization retrain
    reproduces the fit, leaning on the Phase-3 determinism guarantee. Drive the
    real runner resume seam: write a checkpoint (XGBoost HP dict) + a valid
    XGBoost decision, apply it, and finalize. The retrained best model is a
    fitted XGBoostModel and its predictions match a fresh from-scratch run
    bit-identically (same features, hp, seed, row order)."""
    panel, X, y = _toy_panel(seed=1)
    feats = list(X.columns)

    # (1) An xgboost spec for backend-aware decision validation on resume.
    spec = {
        "target": {"universe": "toy_smoke", "direction": "up",
                   "threshold_pct": 10, "horizon_days": 25},
        "backend": {"library": "xgboost", "hp_starting": dict(_HP_XGB)},
    }

    # (2) Write a checkpoint with NO model blob (the contract) at iter 0 — the
    # XGBoost HP dict round-trips through the JSON checkpoint identically.
    art = tmp_path / "resume_cell"
    art.mkdir(parents=True, exist_ok=True)
    ckpt_state = {
        "run_id": "resume_cell",
        "iter_idx": 0,
        "max_iterations": 2,
        "current_features": list(feats),
        "current_hp": dict(_HP_XGB),
        "val_briers": [0.25],
        "hp_history": [{"iter": 0, "hp": dict(_HP_XGB)}],
        "feature_history": [list(feats)],
        "hp_lists": [dict(_HP_XGB)],
        "delta_attributions": ["iteration 0"],
    }
    ckpt_path = gbdt_checkpoint.write_checkpoint(art, ckpt_state)
    assert ckpt_path.exists()
    # no model blob beside the checkpoint
    assert not (art / model_filename("xgboost")).exists()
    # the HP dict survives the JSON round-trip unchanged
    reread = gbdt_checkpoint.read_checkpoint(art)
    assert reread["current_hp"] == dict(_HP_XGB)

    # (3) Agent decision: stop now (force finalize), with a valid XGBoost HP
    # change (eta — an XGBoost-named tunable, accepted only under backend=xgboost).
    decision = {"prune_features": [], "hp_changes": {"eta": 0.25},
                "should_stop": True, "rationale": "smoke: finalize at iter 0"}
    (art / "loop").mkdir(exist_ok=True)
    loop_protocol.decision_path(art, 0).write_text(json.dumps(decision))

    # (4) The runner's resume seam: validate (backend-aware) + apply the decision.
    resume_state = _load_and_apply_resume(art, spec, run_id="resume_cell")
    assert resume_state["force_stop"] is True
    assert resume_state["current_hp"]["eta"] == 0.25  # decision applied

    # (5) Finalize: walk_forward_train with the resume_state retrains the best
    # prior config (no blob) and emits predictions — a fitted XGBoostModel.
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(feats),
        hp=dict(_HP_XGB), split=_SPLIT, max_iterations=2,
        backend="xgboost", random_seed=42, resume_state=resume_state,
    )
    assert isinstance(result.best_model, XGBoostModel)
    assert result.best_model.fitted

    # (6) Determinism: the retrained best config reproduces a fresh fit of the
    # SAME (features, hp, seed, row order) bit-identically (Phase-3 guarantee).
    from gbdt.model import make_model
    from gbdt.train import _carve_X_y

    best_hp = result.best_model.hp
    best_feats = result.best_model.feature_names
    parts = _carve_X_y(X, y, panel, _SPLIT, best_feats, None)
    X_tr, y_tr, _, _ = parts["train"]
    X_val, y_val, _, _ = parts["val"]
    fresh = make_model("xgboost", dict(best_hp), feature_names=best_feats,
                       random_seed=42)
    fresh.fit(X_tr, y_tr, X_val, y_val)
    np.testing.assert_array_equal(
        result.best_model.predict_proba(X_val), fresh.predict_proba(X_val)
    )


# ---------------------------------------------------------------------------
# CatBoost path unchanged — same wiring, .cbm filename, co-occurrence interactions
# ---------------------------------------------------------------------------


def test_catboost_path_unchanged_end_to_end(tmp_path, monkeypatch):
    """The catboost spec still writes model.cbm, the diagnose loader still loads
    it via dispatch, and the interaction path is the native co-occurrence
    (method="cooccurrence") — byte-for-byte the v1 behaviour, now flowing through
    the shared InteractionResult shape."""
    from scripts.gbdt import diagnose as diag

    panel, X, y = _toy_panel(seed=2)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp=dict(_HP_CB), split=_SPLIT, max_iterations=2,
        backend="catboost", random_seed=42,
    )
    assert isinstance(result.best_model, CatBoostModel)
    assert model_filename("catboost") == "model.cbm"

    art = tmp_path / "cb_cell"
    art.mkdir(parents=True, exist_ok=True)
    (art / "predictions").mkdir(exist_ok=True)
    result.best_model.save(art / model_filename("catboost"))
    assert (art / "model.cbm").exists()
    spec = {
        "target": {"universe": "toy_smoke", "direction": "up",
                   "threshold_pct": 10, "horizon_days": 25},
        "split": {"test_rows": _SPLIT.test_rows},
        "backend": {"library": "catboost"},
    }
    (art / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    for seg, df in result.predictions.items():
        df.to_csv(art / "predictions" / f"{seg}.csv", index=False)

    cell = diag.load_cell(art)
    assert cell["backend"] == "catboost"
    assert isinstance(cell["model"], CatBoostModel)

    feats = list(X.columns)
    X_in = X.reset_index(drop=True)[feats]
    y_in = y.reset_index(drop=True).astype(int).values
    monkeypatch.setattr(diag, "build_insample", lambda *a, **k: (X_in, y_in))
    bundle = diag.diagnose(art, top_n=4, do_pdp=False, do_figs=False)
    assert bundle["backend"] == "catboost"
    # catboost interactions are the native co-occurrence cross-check.
    assert bundle["interactions"]["method"] == "cooccurrence"
    assert bundle["interactions"]["per_feature_main_effect"] == {}
