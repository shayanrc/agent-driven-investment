"""Backend dispatch + wire-format validation.

This is the framework's thinnest layer: read the preset's ``backend`` field,
import that backend's public ``forecast``/``tune`` function, call it, validate
the returned shape, and return.

v1 ships **one** backend; the dispatch table is a hand-written dict. Per
``docs/forecasters/V1_PLAN.md`` §"Anti-goals" item 1, no ABC and no registry
crystallize until backend #2 lands.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np
import pandas as pd

from forecasters.data import data_hash
from forecasters.errors import ResultContractError, UnknownBackendError


# ----------------------------------------------------------------------------
# Backend table — one entry per shipped backend
# ----------------------------------------------------------------------------
#
# Each entry is a lambda returning the imported module — deferred so importing
# ``forecasters.dispatch`` doesn't pull in every backend's heavyweight deps at
# framework load time.

_BACKENDS: dict[str, Callable[[], Any]] = {
    "analog_mc": lambda: importlib.import_module("analog_mc.forecaster"),
}


def known_backends() -> list[str]:
    return sorted(_BACKENDS.keys())


def _load_backend(backend_name: str):
    if backend_name not in _BACKENDS:
        raise UnknownBackendError(backend_name, known_backends())
    return _BACKENDS[backend_name]()


# ----------------------------------------------------------------------------
# Result-shape contract (V1_PLAN §"Wire-format contract")
# ----------------------------------------------------------------------------


_REQUIRED_TOP = ("paths", "anchors", "summary", "metadata", "warnings")
_REQUIRED_ANCHORS = ("origin_date", "horizon_dates")
_REQUIRED_SUMMARY = ("median", "p05", "p25", "p75", "p95")
_REQUIRED_METADATA = ("backend_name", "preset_name", "preset_hash",
                      "config_hash", "n_paths", "seed_used")


def _validate_result_contract(
    result: Any,
    *,
    backend_name: str,
    expected_horizon: int,
) -> None:
    """Assert that ``result`` conforms to the V1_PLAN wire-format contract.

    Raises ``ResultContractError`` with backend + violated assertion on
    failure. This is a backend bug, not a user bug; the dispatcher refuses
    to write a malformed result to the cache.
    """
    if not isinstance(result, dict):
        raise ResultContractError(backend_name, f"result is not a dict (got {type(result).__name__})")
    missing = [k for k in _REQUIRED_TOP if k not in result]
    if missing:
        raise ResultContractError(backend_name, f"missing top-level keys: {missing}")

    paths = result["paths"]
    if not isinstance(paths, np.ndarray):
        raise ResultContractError(backend_name, f"paths must be np.ndarray (got {type(paths).__name__})")
    if paths.ndim != 2:
        raise ResultContractError(backend_name, f"paths.ndim must be 2 (got {paths.ndim})")
    if paths.shape[1] != expected_horizon:
        raise ResultContractError(
            backend_name,
            f"paths.shape[1]={paths.shape[1]} does not match horizon={expected_horizon}",
        )

    anchors = result["anchors"]
    missing_a = [k for k in _REQUIRED_ANCHORS if k not in anchors]
    if missing_a:
        raise ResultContractError(backend_name, f"anchors missing keys: {missing_a}")
    horizon_dates = anchors["horizon_dates"]
    if not isinstance(horizon_dates, list) or len(horizon_dates) != expected_horizon:
        raise ResultContractError(
            backend_name,
            f"anchors.horizon_dates must be a list of length {expected_horizon}; "
            f"got {len(horizon_dates) if isinstance(horizon_dates, list) else type(horizon_dates).__name__}",
        )

    summary = result["summary"]
    missing_s = [k for k in _REQUIRED_SUMMARY if k not in summary]
    if missing_s:
        raise ResultContractError(backend_name, f"summary missing keys: {missing_s}")
    for k in _REQUIRED_SUMMARY:
        v = summary[k]
        if not isinstance(v, list) or len(v) != expected_horizon:
            raise ResultContractError(
                backend_name,
                f"summary.{k} must be a list of length {expected_horizon}; "
                f"got len={len(v) if isinstance(v, list) else type(v).__name__}",
            )

    metadata = result["metadata"]
    missing_m = [k for k in _REQUIRED_METADATA if k not in metadata]
    if missing_m:
        raise ResultContractError(backend_name, f"metadata missing keys: {missing_m}")

    if not isinstance(result["warnings"], list):
        raise ResultContractError(
            backend_name, f"warnings must be a list (got {type(result['warnings']).__name__})"
        )


# ----------------------------------------------------------------------------
# Drift detection
# ----------------------------------------------------------------------------


def _detect_drift(preset: dict, df: pd.DataFrame) -> str | None:
    """Compare preset.fitted_on.data_hash to a fresh hash of the input data.

    Returns a human-readable warning string if drift is detected, else None.
    """
    try:
        current_hash = data_hash(df)
    except Exception:  # pragma: no cover - defensive; data_hash failures shouldn't block forecasting
        return None

    fitted = preset.get("fitted_on", {})
    fitted_hash = fitted.get("data_hash")
    if not fitted_hash or fitted_hash == current_hash:
        return None
    return (
        f"preset fitted on identifier={fitted.get('identifier')!r} "
        f"range=[{fitted.get('start')} .. {fitted.get('end')}] "
        f"(data_hash {fitted_hash}); forecasting on data with "
        f"data_hash {current_hash}. Hyperparameters may be uncalibrated."
    )


# ----------------------------------------------------------------------------
# Public dispatch entry points
# ----------------------------------------------------------------------------


def dispatch_forecast(
    preset: dict[str, Any],
    input_dict: dict[str, Any],
) -> dict[str, Any]:
    """Route a forecast call to the right backend and validate the result.

    ``input_dict`` must already carry ``data``, ``origin``, ``horizon``,
    ``hyperparameters``, and optionally ``seed`` — i.e., the framework has
    already done the preset → input_dict translation.
    """
    backend_name = preset["backend"]
    backend = _load_backend(backend_name)
    if not hasattr(backend, "forecast"):
        raise UnknownBackendError(backend_name, known_backends())

    # Inject preset identity into the input so the backend can echo it back
    # in metadata (the framework owns these strings; the backend doesn't
    # need to know about preset_hash semantics).
    input_dict = dict(input_dict)
    input_dict.setdefault("preset_name", preset.get("name", "<inline>"))
    input_dict.setdefault("preset_hash", preset.get("__content_hash__", "<inline>"))

    result = backend.forecast(input_dict)
    _validate_result_contract(
        result,
        backend_name=backend_name,
        expected_horizon=int(input_dict["horizon"]),
    )

    # Inject framework-side warnings AFTER validation so the backend can't
    # silently drop drift warnings by overwriting `warnings`.
    drift_msg = _detect_drift(preset, input_dict["data"])
    if drift_msg:
        result["warnings"] = list(result.get("warnings") or []) + [drift_msg]

    return result


def dispatch_tune(
    backend_name: str,
    input_dict: dict[str, Any],
) -> dict[str, Any]:
    """Route a tune call to the right backend; validate the produced preset.

    The returned preset_dict is validated against the preset schema
    (``forecasters.presets.validate_preset``) before being returned.
    """
    backend = _load_backend(backend_name)
    if not hasattr(backend, "tune"):
        raise UnknownBackendError(backend_name, known_backends())
    preset = backend.tune(input_dict)
    if not isinstance(preset, dict):
        raise ResultContractError(backend_name, f"tune returned {type(preset).__name__}, not dict")
    if preset.get("backend") != backend_name:
        raise ResultContractError(
            backend_name,
            f"tune produced preset with backend={preset.get('backend')!r}, "
            f"expected {backend_name!r}",
        )
    from forecasters.presets import validate_preset
    validate_preset(preset)
    return preset
