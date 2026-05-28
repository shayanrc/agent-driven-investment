"""V1.1 Phase 1 — ``callback_mode`` spec field + ``_resolve_callback`` dispatch.

Covers (plan § 11 Phase 1):
  - ``_resolve_callback`` with ``callback_mode="default"`` (and absent) → None.
  - ``_resolve_callback`` with ``callback_mode="agent_file_protocol"`` →
    a non-None callable that raises ``NotImplementedError`` when invoked.
  - Spec validation rejects an unknown ``callback_mode``; accepts the two
    valid values + absence.
  - CLI ``--callback-mode`` override beats the spec value (resolution logic).

No experiment / training is run here — these are pure unit tests on the
resolution + validation surface, fast (sub-second).
"""

from __future__ import annotations

import pytest

from gbdt.__main__ import (
    _DEFAULT_CALLBACK_MODE,
    _VALID_CALLBACK_MODES,
    _resolve_callback,
    _validate_spec,
)


# ---------------------------------------------------------------------------
# _resolve_callback dispatch
# ---------------------------------------------------------------------------


def test_resolve_callback_default_returns_none():
    # Explicit "default" → None so walk_forward_train keeps its built-in
    # default_fs_hp_callback (v1 behaviour preserved byte-for-byte).
    assert _resolve_callback({"callback_mode": "default"}, run_id="r") is None


def test_resolve_callback_absent_returns_none():
    # Absent callback_mode defaults to "default" → None.
    assert _resolve_callback({}, run_id="r") is None
    assert _resolve_callback({"max_iterations": 8}, run_id="r") is None


def test_resolve_callback_none_loop_cfg_returns_none():
    # Defensive: a None loop_cfg is treated as absent.
    assert _resolve_callback(None, run_id="r") is None


def test_default_callback_mode_constant_is_default():
    assert _DEFAULT_CALLBACK_MODE == "default"
    assert _DEFAULT_CALLBACK_MODE in _VALID_CALLBACK_MODES


def test_resolve_callback_agent_file_protocol_returns_callable():
    cb = _resolve_callback({"callback_mode": "agent_file_protocol"}, run_id="r")
    assert cb is not None
    assert callable(cb)


def test_resolve_callback_agent_file_protocol_body_not_implemented():
    cb = _resolve_callback({"callback_mode": "agent_file_protocol"}, run_id="r")
    # Phase 1: the body is a placeholder; the exit-and-resume protocol lands
    # in Phase 2. Calling it (with the (bundle, current_features) signature)
    # must raise NotImplementedError.
    with pytest.raises(NotImplementedError, match="Phase 2"):
        cb(object(), ["feat_a", "feat_b"])


def test_resolve_callback_unknown_mode_raises():
    with pytest.raises(ValueError, match="callback_mode"):
        _resolve_callback({"callback_mode": "totally_bogus"}, run_id="r")


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def _minimal_spec(**loop_overrides) -> dict:
    """A minimal spec that passes _validate_spec, with an fs_hp_loop block."""
    loop = {"max_iterations": 8}
    loop.update(loop_overrides)
    return {
        "target": {
            "universe": "nifty50",
            "direction": "up",
            "threshold_pct": 10,
            "horizon_days": 25,
        },
        "backend": {"library": "catboost", "fs_hp_loop": loop},
    }


def test_validate_spec_accepts_absent_callback_mode():
    # No callback_mode key at all — must pass.
    spec = _minimal_spec()
    spec["backend"]["fs_hp_loop"].pop("callback_mode", None)
    _validate_spec(spec)  # no raise


@pytest.mark.parametrize("mode", sorted(_VALID_CALLBACK_MODES))
def test_validate_spec_accepts_valid_callback_modes(mode):
    _validate_spec(_minimal_spec(callback_mode=mode))  # no raise


def test_validate_spec_rejects_unknown_callback_mode():
    with pytest.raises(ValueError, match="callback_mode"):
        _validate_spec(_minimal_spec(callback_mode="not_a_mode"))


# ---------------------------------------------------------------------------
# CLI override beats the spec value (resolution logic)
# ---------------------------------------------------------------------------


def _apply_override(loop_cfg: dict, override: str | None) -> dict:
    """Mirror run_experiment's override semantics on an fs_hp_loop dict.

    Tests the override-resolution logic without running a full experiment:
    when an override is supplied it replaces the spec's callback_mode, then
    _resolve_callback is dispatched on the resulting effective config.
    """
    cfg = dict(loop_cfg)
    if override is not None:
        assert override in _VALID_CALLBACK_MODES
        cfg["callback_mode"] = override
    return cfg


def test_cli_override_beats_spec_default_to_agent():
    # Spec says default; CLI override forces agent_file_protocol.
    spec_loop = {"max_iterations": 8, "callback_mode": "default"}
    effective = _apply_override(spec_loop, "agent_file_protocol")
    cb = _resolve_callback(effective, run_id="r")
    assert cb is not None and callable(cb)


def test_cli_override_beats_spec_agent_to_default():
    # Spec says agent_file_protocol; CLI override forces default.
    spec_loop = {"max_iterations": 8, "callback_mode": "agent_file_protocol"}
    effective = _apply_override(spec_loop, "default")
    assert _resolve_callback(effective, run_id="r") is None


def test_no_cli_override_uses_spec_value():
    spec_loop = {"max_iterations": 8, "callback_mode": "agent_file_protocol"}
    effective = _apply_override(spec_loop, None)
    cb = _resolve_callback(effective, run_id="r")
    assert cb is not None and callable(cb)
