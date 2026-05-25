"""Tests for forecasters.dispatch — backend routing + contract validation."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from forecasters import dispatch as dispatch_mod
from forecasters.dispatch import (
    _validate_result_contract,
    dispatch_forecast,
    dispatch_tune,
    known_backends,
)
from forecasters.errors import ResultContractError, UnknownBackendError


# ----------------------------------------------------------------------------
# Helpers — fake backend installation for isolation
# ----------------------------------------------------------------------------


def _good_result(horizon: int = 4, n_paths: int = 5) -> dict[str, Any]:
    return {
        "paths": np.zeros((n_paths, horizon), dtype=np.float64),
        "anchors": {
            "origin_date": "2020-01-02",
            "horizon_dates": [f"2020-01-{3 + i:02d}" for i in range(horizon)],
        },
        "summary": {
            "median": [0.0] * horizon,
            "p05": [-1.0] * horizon,
            "p25": [-0.5] * horizon,
            "p75": [0.5] * horizon,
            "p95": [1.0] * horizon,
            "crps": None,
        },
        "metadata": {
            "backend_name": "fake_backend",
            "preset_name": "fake",
            "preset_hash": "sha256:abc",
            "config_hash": "sha256:xyz",
            "n_paths": n_paths,
            "seed_used": 42,
        },
        "warnings": [],
    }


def _install_fake_backend(
    name: str,
    forecast_fn=None,
    tune_fn=None,
) -> None:
    mod = types.ModuleType(name)
    if forecast_fn is not None:
        mod.forecast = forecast_fn
    if tune_fn is not None:
        mod.tune = tune_fn
    sys.modules[name] = mod
    dispatch_mod._BACKENDS[name.split(".")[-1]] = lambda mod=mod: mod


@pytest.fixture
def fake_backend_clean():
    """Snapshot/restore the dispatch table around each test."""
    saved = dict(dispatch_mod._BACKENDS)
    yield
    dispatch_mod._BACKENDS.clear()
    dispatch_mod._BACKENDS.update(saved)


# ----------------------------------------------------------------------------
# _validate_result_contract
# ----------------------------------------------------------------------------


def test_good_result_validates() -> None:
    _validate_result_contract(_good_result(horizon=4), backend_name="x", expected_horizon=4)


def test_horizon_mismatch_raises() -> None:
    bad = _good_result(horizon=4)
    with pytest.raises(ResultContractError, match="paths.shape"):
        _validate_result_contract(bad, backend_name="x", expected_horizon=5)


def test_missing_top_level_key_raises() -> None:
    bad = _good_result()
    del bad["summary"]
    with pytest.raises(ResultContractError, match="summary"):
        _validate_result_contract(bad, backend_name="x", expected_horizon=4)


def test_paths_must_be_ndarray() -> None:
    bad = _good_result()
    bad["paths"] = [[0.0] * 4] * 5
    with pytest.raises(ResultContractError, match="np.ndarray"):
        _validate_result_contract(bad, backend_name="x", expected_horizon=4)


def test_horizon_dates_length_mismatch_raises() -> None:
    bad = _good_result(horizon=4)
    bad["anchors"]["horizon_dates"] = ["2020-01-03"]  # too short
    with pytest.raises(ResultContractError, match="horizon_dates"):
        _validate_result_contract(bad, backend_name="x", expected_horizon=4)


def test_warnings_none_raises() -> None:
    bad = _good_result()
    bad["warnings"] = None
    with pytest.raises(ResultContractError, match="warnings"):
        _validate_result_contract(bad, backend_name="x", expected_horizon=4)


def test_summary_percentile_length_mismatch_raises() -> None:
    bad = _good_result(horizon=4)
    bad["summary"]["p05"] = [0.0, 0.0, 0.0]  # wrong length
    with pytest.raises(ResultContractError, match=r"summary\.p05"):
        _validate_result_contract(bad, backend_name="x", expected_horizon=4)


def test_missing_metadata_key_raises() -> None:
    bad = _good_result()
    del bad["metadata"]["seed_used"]
    with pytest.raises(ResultContractError, match="metadata"):
        _validate_result_contract(bad, backend_name="x", expected_horizon=4)


# ----------------------------------------------------------------------------
# dispatch_forecast — backend routing + drift detection
# ----------------------------------------------------------------------------


def test_dispatch_forecast_happy_path(fake_backend_clean) -> None:
    def fake_forecast(input_dict: dict) -> dict:
        return _good_result(horizon=input_dict["horizon"])
    _install_fake_backend("fake_backend", forecast_fn=fake_forecast)
    preset = {
        "name": "fake-p",
        "backend": "fake_backend",
        "fitted_on": {"data_hash": "sha256:abc", "identifier": "X", "start": "2020-01-01", "end": "2020-12-31"},
        "__content_hash__": "sha256:phash",
    }
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5), "adj_close": [100.0] * 5})
    input_dict = {"data": df, "origin": "2020-01-02", "horizon": 4, "hyperparameters": {}}
    result = dispatch_forecast(preset, input_dict)
    assert result["paths"].shape == (5, 4)
    # No drift warning when the test backend's input data matches the preset hash —
    # except our fake preset's hash won't match the actual df. So one drift warning expected.
    assert any("Hyperparameters may be uncalibrated" in w for w in result["warnings"])


def test_dispatch_forecast_no_drift_when_hashes_match(fake_backend_clean) -> None:
    from forecasters.data import data_hash
    def fake_forecast(input_dict: dict) -> dict:
        return _good_result(horizon=input_dict["horizon"])
    _install_fake_backend("fake_backend", forecast_fn=fake_forecast)
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5), "adj_close": [100.0 + i for i in range(5)]})
    preset = {
        "name": "fake-p",
        "backend": "fake_backend",
        "fitted_on": {
            "data_hash": data_hash(df),
            "identifier": "X", "start": "2020-01-01", "end": "2020-12-31",
        },
        "__content_hash__": "sha256:phash",
    }
    input_dict = {"data": df, "origin": "2020-01-02", "horizon": 4, "hyperparameters": {}}
    result = dispatch_forecast(preset, input_dict)
    assert result["warnings"] == []


def test_dispatch_forecast_unknown_backend_raises() -> None:
    preset = {"name": "x", "backend": "not_a_real_backend",
              "fitted_on": {"data_hash": "sha256:a", "identifier": "Z",
                            "start": "x", "end": "y"}}
    with pytest.raises(UnknownBackendError) as ei:
        dispatch_forecast(preset, {"data": pd.DataFrame(), "origin": "2020-01-01", "horizon": 1, "hyperparameters": {}})
    assert "not_a_real_backend" in str(ei.value)


def test_dispatch_forecast_bad_result_shape_raises(fake_backend_clean) -> None:
    def bad_forecast(input_dict: dict) -> dict:
        r = _good_result(horizon=input_dict["horizon"])
        r["paths"] = np.zeros((3, 99))  # wrong horizon
        return r
    _install_fake_backend("fake_backend", forecast_fn=bad_forecast)
    preset = {"name": "x", "backend": "fake_backend",
              "fitted_on": {"data_hash": "sha256:a", "identifier": "Z", "start": "x", "end": "y"}}
    with pytest.raises(ResultContractError):
        dispatch_forecast(preset, {"data": pd.DataFrame({"date":[pd.Timestamp('2020-01-01')], "adj_close":[1.0]}), "origin": "2020-01-02", "horizon": 4, "hyperparameters": {}})


def test_dispatch_forecast_injects_preset_name_and_hash(fake_backend_clean) -> None:
    captured: dict[str, dict] = {}
    def fake_forecast(input_dict: dict) -> dict:
        captured["seen"] = dict(input_dict)
        return _good_result(horizon=input_dict["horizon"])
    _install_fake_backend("fake_backend", forecast_fn=fake_forecast)
    preset = {
        "name": "v24-default",
        "backend": "fake_backend",
        "fitted_on": {"data_hash": "sha256:a", "identifier": "Z", "start": "x", "end": "y"},
        "__content_hash__": "sha256:HASH",
    }
    df = pd.DataFrame({"date":[pd.Timestamp('2020-01-01')], "adj_close":[1.0]})
    dispatch_forecast(preset, {"data": df, "origin": "2020-01-02", "horizon": 4, "hyperparameters": {}})
    assert captured["seen"]["preset_name"] == "v24-default"
    assert captured["seen"]["preset_hash"] == "sha256:HASH"


# ----------------------------------------------------------------------------
# dispatch_tune
# ----------------------------------------------------------------------------


def _good_preset_dict() -> dict:
    return {
        "name": "ok",
        "backend": "fake_backend",
        "schema_version": 1,
        "hyperparameters": {"n_eff": 50},
        "fitted_on": {
            "identifier": "X",
            "start": "2020-01-01",
            "end": "2020-12-31",
            "data_hash": "sha256:abc",
            "n_observations": 250,
        },
        "fitted_at": "2026-05-24T18:00:00Z",
        "validation_metrics": {"crps_mean": 0.05},
    }


def test_dispatch_tune_happy_path(fake_backend_clean) -> None:
    def fake_tune(input_dict: dict) -> dict:
        return _good_preset_dict()
    _install_fake_backend("fake_backend", tune_fn=fake_tune)
    p = dispatch_tune("fake_backend", {"data": pd.DataFrame(), "identifier": "X", "range": ("2020-01-01", "2020-12-31")})
    assert p["name"] == "ok"


def test_dispatch_tune_rejects_wrong_backend(fake_backend_clean) -> None:
    def fake_tune(input_dict: dict) -> dict:
        bad = _good_preset_dict()
        bad["backend"] = "other_backend"
        return bad
    _install_fake_backend("fake_backend", tune_fn=fake_tune)
    with pytest.raises(ResultContractError, match="backend"):
        dispatch_tune("fake_backend", {"data": pd.DataFrame(), "identifier": "X", "range": ("a", "b")})


def test_known_backends_includes_analog_mc() -> None:
    assert "analog_mc" in known_backends()
