"""Tests for V1.3 Option B P6 — spec validation of the new
``backend.scout`` + ``backend.fs_prefit`` blocks.
"""

from __future__ import annotations

import pytest

from gbdt.__main__ import _validate_spec


def _base_spec(**overrides):
    spec = {
        "target": {
            "universe": "nifty50",
            "direction": "up",
            "threshold_pct": 10,
            "horizon_days": 25,
        },
        "backend": {"library": "catboost"},
    }
    backend_extras = overrides.pop("backend", None)
    if backend_extras:
        spec["backend"].update(backend_extras)
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------------
# Absence is fine
# ---------------------------------------------------------------------------


def test_validate_spec_accepts_absent_scout_block():
    _validate_spec(_base_spec())   # no raise


def test_validate_spec_accepts_absent_fs_prefit_block():
    _validate_spec(_base_spec())   # no raise


# ---------------------------------------------------------------------------
# Happy-path opt in
# ---------------------------------------------------------------------------


def test_validate_spec_accepts_scout_enabled_only():
    _validate_spec(_base_spec(backend={"scout": {"enabled": True}}))


def test_validate_spec_accepts_full_scout_block():
    _validate_spec(_base_spec(backend={"scout": {
        "enabled": True,
        "grid": {"max_depth": [2, 3], "eta": [0.1, 0.2]},
        "n_configs_cap": 50,
        "per_config_timeout_seconds": 30,
        "wall_clock_cap_seconds": 300,
    }}))


def test_validate_spec_accepts_fs_prefit_block():
    _validate_spec(_base_spec(backend={"fs_prefit": {
        "enabled": True,
        "cliff_pct": 0.01,
    }}))


def test_validate_spec_accepts_scout_disabled_explicit():
    _validate_spec(_base_spec(backend={"scout": {"enabled": False}}))


# ---------------------------------------------------------------------------
# Rejections — type errors
# ---------------------------------------------------------------------------


def test_validate_spec_rejects_scout_enabled_non_bool():
    with pytest.raises(ValueError, match="scout.enabled must be bool"):
        _validate_spec(_base_spec(backend={"scout": {"enabled": "yes"}}))


def test_validate_spec_rejects_scout_negative_n_configs_cap():
    with pytest.raises(ValueError, match="scout.n_configs_cap"):
        _validate_spec(_base_spec(backend={"scout": {
            "enabled": True, "n_configs_cap": -1,
        }}))


def test_validate_spec_rejects_scout_zero_timeout():
    with pytest.raises(ValueError, match="per_config_timeout_seconds"):
        _validate_spec(_base_spec(backend={"scout": {
            "enabled": True, "per_config_timeout_seconds": 0,
        }}))


def test_validate_spec_rejects_scout_negative_wall_clock_cap():
    with pytest.raises(ValueError, match="wall_clock_cap_seconds"):
        _validate_spec(_base_spec(backend={"scout": {
            "enabled": True, "wall_clock_cap_seconds": -10,
        }}))


def test_validate_spec_rejects_fs_prefit_enabled_non_bool():
    with pytest.raises(ValueError, match="fs_prefit.enabled must be bool"):
        _validate_spec(_base_spec(backend={"fs_prefit": {"enabled": 1}}))


def test_validate_spec_rejects_fs_prefit_cliff_pct_out_of_range():
    with pytest.raises(ValueError, match="cliff_pct must be in"):
        _validate_spec(_base_spec(backend={"fs_prefit": {
            "enabled": True, "cliff_pct": 1.5,
        }}))


def test_validate_spec_rejects_fs_prefit_cliff_pct_negative():
    with pytest.raises(ValueError, match="cliff_pct must be in"):
        _validate_spec(_base_spec(backend={"fs_prefit": {
            "enabled": True, "cliff_pct": -0.1,
        }}))


def test_validate_spec_accepts_cliff_pct_boundary_values():
    _validate_spec(_base_spec(backend={"fs_prefit": {
        "enabled": True, "cliff_pct": 0,
    }}))
    _validate_spec(_base_spec(backend={"fs_prefit": {
        "enabled": True, "cliff_pct": 1,
    }}))


# ---------------------------------------------------------------------------
# Co-existence with other backend blocks
# ---------------------------------------------------------------------------


def test_validate_spec_accepts_scout_with_fs_hp_loop():
    """The scout block doesn't interfere with the V1.1 fs_hp_loop block."""
    _validate_spec(_base_spec(backend={
        "scout": {"enabled": True},
        "fs_prefit": {"enabled": True, "cliff_pct": 0.01},
        "fs_hp_loop": {
            "callback_mode": "agent_file_protocol",
            "max_iterations": 8,
        },
    }))
