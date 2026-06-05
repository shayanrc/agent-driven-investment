"""Tests for V1.3 Option B PR #125 Medium 2 fix —
backend-vocabulary translation on agent-supplied HP overlays
(combine_decision + iter_0_decision).

The speed-biased combine prompt is written in XGBoost-canonical knob names
(``max_depth``, ``eta``, ``colsample_bytree``, ``gamma``, ...). The agent
echoes those names. For CatBoost specs, those names would be silently
dropped by ``make_model``. The fix routes agent overlays through
``scout._translate_for_backend`` before merging into ``hp_starting``.

These tests cover the translation helper as used at the runner's agent
mode call sites. The runner-side wiring is tested via a thin direct
invocation of the inline ``_translate_agent_overlay`` closure (constructed
the same way the production code constructs it) — full end-to-end
agent-mode test requires the scout cycles 1-3 file dance which is
covered separately in test_scout_exit_resume.py.
"""

from __future__ import annotations

import pytest

from gbdt import scout as gbdt_scout


# ---------------------------------------------------------------------------
# Helper — mirrors the inline closure in
# __main__.py::_handle_scout_cycles_agent_mode::_translate_agent_overlay.
# Pure function (no logging) so we can assert the translation alone; the
# logging side-effect is verified by test_translation_drop_logged below.
# ---------------------------------------------------------------------------


def _translate_overlay(overlay: dict, backend: str) -> dict:
    """The translation half of ``_translate_agent_overlay`` (pure)."""
    if not overlay:
        return {}
    return gbdt_scout._translate_for_backend(dict(overlay), backend)


# ---------------------------------------------------------------------------
# CatBoost translation (the load-bearing path)
# ---------------------------------------------------------------------------


def test_agent_overlay_catboost_xgboost_names_translated():
    """Agent submits XGBoost-named keys against a CatBoost spec; runner
    translates to CatBoost equivalents BEFORE fit."""
    overlay = {"max_depth": 3, "eta": 0.1}
    out = _translate_overlay(overlay, "catboost")
    assert out == {"depth": 3, "learning_rate": 0.1}


def test_agent_overlay_catboost_drops_gamma():
    """Agent submits ``gamma`` against a CatBoost spec; gamma is dropped
    (no CatBoost analog per scout._translate_for_backend)."""
    overlay = {"max_depth": 4, "gamma": 0.5}
    out = _translate_overlay(overlay, "catboost")
    assert "gamma" not in out
    assert out == {"depth": 4}


def test_agent_overlay_catboost_scale_pos_weight_to_class_weights():
    """spw → class_weights dict for CatBoost."""
    overlay = {"scale_pos_weight": 2.5}
    out = _translate_overlay(overlay, "catboost")
    assert out == {"class_weights": {0: 1.0, 1: 2.5}}


def test_agent_overlay_catboost_unknown_key_dropped():
    """Unknown-vocab keys are silently dropped (caller logs)."""
    overlay = {"max_depth": 3, "not_a_real_knob": 99}
    out = _translate_overlay(overlay, "catboost")
    assert "not_a_real_knob" not in out
    assert out == {"depth": 3}


# ---------------------------------------------------------------------------
# XGBoost translation — pass-through (no translation needed).
# ---------------------------------------------------------------------------


def test_agent_overlay_xgboost_passthrough():
    """Agent submits XGBoost-named keys against an XGBoost spec; no
    translation needed; output equals input."""
    overlay = {"max_depth": 3, "eta": 0.1, "gamma": 0.2}
    out = _translate_overlay(overlay, "xgboost")
    assert out == overlay


# ---------------------------------------------------------------------------
# Empty overlay edge case.
# ---------------------------------------------------------------------------


def test_agent_overlay_empty_passthrough():
    assert _translate_overlay({}, "catboost") == {}
    assert _translate_overlay({}, "xgboost") == {}


# ---------------------------------------------------------------------------
# Inline closure logging — verify the runner logs dropped keys.
# ---------------------------------------------------------------------------


def test_translation_drop_logged():
    """Reconstruct the inline ``_translate_agent_overlay`` closure with a
    capturing milestone fn; submit an overlay with ``gamma`` + an unknown
    key; assert both drops are logged for agent feedback."""
    # Mirror the closure shape from __main__.py.
    backend_library = "catboost"
    logged: list[str] = []

    def milestone(msg: str) -> None:
        logged.append(msg)

    def _translate_agent_overlay(overlay: dict, *, source_label: str) -> dict:
        if not overlay:
            return {}
        translated = gbdt_scout._translate_for_backend(
            dict(overlay), backend_library,
        )
        if backend_library == "xgboost":
            return translated
        translated_xgb_names = set()
        for k_in in overlay.keys():
            if k_in == "gamma":
                continue
            if k_in in gbdt_scout._XGB_TO_CATBOOST:
                translated_xgb_names.add(k_in)
            elif k_in == "scale_pos_weight":
                translated_xgb_names.add(k_in)
        dropped = [
            k for k in overlay.keys()
            if k != "gamma" and k not in translated_xgb_names
        ]
        if "gamma" in overlay:
            milestone(
                f"[scout] {source_label}: dropped 'gamma' "
                f"(no CatBoost analog) — translation per V1.3 Option B D1"
            )
        if dropped:
            milestone(
                f"[scout] {source_label}: dropped unknown-vocab key(s) "
                f"{dropped} for backend={backend_library!r}"
            )
        return translated

    overlay = {"max_depth": 3, "gamma": 0.5, "weird_key": 1}
    out = _translate_agent_overlay(overlay, source_label="cycle 2 cfg[0]")
    # gamma drop logged.
    assert any("dropped 'gamma'" in m for m in logged)
    # weird_key drop logged.
    assert any("unknown-vocab key(s)" in m and "weird_key" in m for m in logged)
    # Translation correct (max_depth → depth).
    assert out == {"depth": 3}
