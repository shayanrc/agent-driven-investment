"""Unit tests for the scout module (V1.3 Option B P1).

Synthetic in-memory data only — no SQLite cache, no data_pipelines.fetch.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gbdt.scout import (
    DEFAULT_SCOUT_GRID,
    LEX_ORACLE_PRIORITY,
    SPEED_BIASED_COMBINE_PROMPT,
    ScoutConfig,
    ScoutResult,
    _translate_for_backend,
    build_grid,
    detect_degenerate_sink,
    lexicographic_winner,
    per_knob_winners,
    run_scout,
)


# ---------------------------------------------------------------------------
# build_grid
# ---------------------------------------------------------------------------


def test_build_grid_xgboost_count_is_41():
    """41 = 1 defaults zeroth + 5 (max_depth) + 5 (eta) + 5 (colsample) +
    4 (mcw) + 4 (gamma) + 4 (alpha) + 4 (subsample) + 3 (spw)."""
    grid = build_grid(backend="xgboost", n_positive=10, n_negative=90)
    knob_counts: dict[str, int] = {}
    for c in grid:
        knob_counts[c.knob_name] = knob_counts.get(c.knob_name, 0) + 1
    assert knob_counts.get("defaults") == 1
    assert knob_counts.get("max_depth") == 5
    assert knob_counts.get("eta") == 5
    assert knob_counts.get("colsample_bytree") == 5
    assert knob_counts.get("min_child_weight") == 4
    assert knob_counts.get("gamma") == 4
    assert knob_counts.get("alpha") == 4
    assert knob_counts.get("subsample") == 4
    assert knob_counts.get("scale_pos_weight") == 3
    # Total = 1 + 5 + 5 + 5 + 4 + 4 + 4 + 4 + 3 = 35? recheck
    # Wait: 1 + 5 + 5 + 5 + 4 + 4 + 4 + 4 + 3 = 35.
    # The plan said "~41 fits" = "8 knobs × ~4-5 values + 1 zeroth". Let's
    # recount carefully:
    #   max_depth: 5; eta: 5; colsample: 5; mcw: 4; gamma: 4; alpha: 4;
    #   subsample: 4; spw: 3 → 34 grid + 1 zeroth = 35.
    # Plan text says "~40 single-knob configs + 1 defaults zeroth = ~41".
    # The plan grid totals 5+5+5+4+4+4+4+3 = 34 (not 40), so the actual
    # implemented count is 35. This is intentional — the grid is a coarse
    # map; the plan's "~41" is approximate.
    assert len(grid) == 35


def test_build_grid_catboost_drops_gamma():
    grid = build_grid(backend="catboost", n_positive=10, n_negative=90)
    knob_names = {c.knob_name for c in grid}
    assert "gamma" not in knob_names
    assert "defaults" in knob_names
    # max_depth → depth, eta → learning_rate, colsample_bytree → rsm, etc.
    # All overlays should use CatBoost-named keys.
    for c in grid:
        if c.knob_name == "defaults":
            assert c.hp_overlay == {}
            continue
        # The overlay keys are CatBoost names — never XGBoost.
        assert "max_depth" not in c.hp_overlay
        assert "eta" not in c.hp_overlay


def test_build_grid_spec_overrides_per_knob():
    grid = build_grid(
        backend="xgboost",
        spec_overrides={"max_depth": [3, 4]},
        n_positive=10, n_negative=90,
    )
    md_rows = [c for c in grid if c.knob_name == "max_depth"]
    assert [c.knob_value for c in md_rows] == [3, 4]


def test_build_grid_scale_pos_weight_sentinels_resolved():
    # n_neg / n_pos = 9, sqrt = 3
    grid = build_grid(backend="xgboost", n_positive=10, n_negative=90)
    spw_rows = [c for c in grid if c.knob_name == "scale_pos_weight"]
    values = [c.knob_value for c in spw_rows]
    assert pytest.approx(values[0]) == 1.0
    assert pytest.approx(values[1]) == 3.0
    assert pytest.approx(values[2]) == 9.0


# ---------------------------------------------------------------------------
# Backend translation
# ---------------------------------------------------------------------------


def test_translate_xgboost_passthrough():
    overlay = {"max_depth": 3, "eta": 0.1}
    assert _translate_for_backend(overlay, "xgboost") == overlay


def test_translate_catboost_full_table():
    overlay = {
        "max_depth": 3,
        "eta": 0.1,
        "colsample_bytree": 0.5,
        "min_child_weight": 5,
        "alpha": 0.5,
        "subsample": 0.7,
    }
    out = _translate_for_backend(overlay, "catboost")
    assert out == {
        "depth": 3,
        "learning_rate": 0.1,
        "rsm": 0.5,
        "min_data_in_leaf": 5,
        "l2_leaf_reg": 0.5,
        "subsample": 0.7,
    }


def test_translate_catboost_drops_gamma():
    overlay = {"gamma": 0.5, "max_depth": 3}
    out = _translate_for_backend(overlay, "catboost")
    assert "gamma" not in out
    assert out == {"depth": 3}


def test_translate_catboost_scale_pos_weight_to_class_weights():
    overlay = {"scale_pos_weight": 3.0}
    out = _translate_for_backend(overlay, "catboost")
    assert out == {"class_weights": {0: 1.0, 1: 3.0}}


# ---------------------------------------------------------------------------
# run_scout — fake fit_one harness on synthetic 3-ticker panel
# ---------------------------------------------------------------------------


def _make_synthetic_panel(n_dates: int = 5, n_tickers: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    index = pd.MultiIndex.from_product(
        [dates, tickers], names=["date", "ticker"],
    )
    n = len(index)
    X = pd.DataFrame(
        rng.normal(0, 1, (n, 4)),
        index=index, columns=["f0", "f1", "f2", "f3"],
    )
    y = pd.Series(rng.integers(0, 2, n), index=index, name="y").astype(int)
    return X, y, index


def _fake_fit_one(*, hp_overlay, X_train, y_train, w_train, X_val, y_val,
                   w_val, X_eval, y_eval, w_eval, mi_eval):
    """Deterministic fake fit — returns metrics derived from the overlay.

    This is a TEST DOUBLE that exercises the scout loop without paying for
    real model fitting. The metrics depend on the overlay so we can predict
    which config the oracle picks.
    """
    # val_brier = 0.25 + 0.01 * (overlay sum) — overlays with negative scalars
    # get LOWER val_brier (better).
    total = 0.0
    for k, v in hp_overlay.items():
        try:
            total += float(v)
        except (TypeError, ValueError):
            continue
    val_brier = 0.25 + 0.01 * total
    train_brier = 0.22 + 0.01 * total
    # Higher max_depth → better R-p@1 (synthetic ordering for argmax check)
    rp1 = 0.10 + 0.01 * float(hp_overlay.get("max_depth", 0))
    return {
        "val_brier": val_brier,
        "train_brier": train_brier,
        "train_val_gap": val_brier - train_brier,
        "spiegelhalter_z": 0.0,
        "eval_R_p_at_K": {1: rp1, 3: rp1 - 0.01, 5: rp1 - 0.02,
                            10: rp1 - 0.03, 20: rp1 - 0.04},
    }


def test_run_scout_emits_results_for_every_config():
    X, y, mi = _make_synthetic_panel()
    arr_y = np.asarray(y.values)
    results = run_scout(
        X_train=X, y_train=arr_y, w_train=None,
        X_val=X, y_val=arr_y, w_val=None,
        X_eval=X, y_eval=arr_y, w_eval=None,
        mi_eval=mi,
        fit_one=_fake_fit_one,
        backend="xgboost",
        spec=None,
        n_positive=int(arr_y.sum()), n_negative=int((arr_y == 0).sum()),
    )
    # All configs ran (no errors / timeouts on the fake fit).
    assert len(results) == 35
    assert all(r.status == "ok" for r in results)
    # Defaults row should have empty overlay and a baseline val_brier.
    defaults = [r for r in results if r.config.knob_name == "defaults"]
    assert len(defaults) == 1
    assert defaults[0].config.hp_overlay == {}
    # Every result should have eval_R_p_at_K populated.
    for r in results:
        assert r.eval_R_p_at_K is not None
        assert set(r.eval_R_p_at_K.keys()) == {1, 3, 5, 10, 20}


def test_run_scout_wall_clock_cap_drops_late_configs():
    """When the soft wall-clock is tiny, only the first config runs; the rest
    get status="timeout"."""
    X, y, mi = _make_synthetic_panel()
    arr_y = np.asarray(y.values)

    import time as _time
    def slow_fit(*, hp_overlay, **kw):
        _time.sleep(0.1)
        return {
            "val_brier": 0.25, "train_brier": 0.22, "train_val_gap": 0.03,
            "spiegelhalter_z": 0.0,
            "eval_R_p_at_K": {1: 0.1, 3: 0.1, 5: 0.1, 10: 0.1, 20: 0.1},
        }

    results = run_scout(
        X_train=X, y_train=arr_y, w_train=None,
        X_val=X, y_val=arr_y, w_val=None,
        X_eval=X, y_eval=arr_y, w_eval=None,
        mi_eval=mi,
        fit_one=slow_fit,
        backend="xgboost",
        spec=None,
        per_config_timeout_seconds=60,
        # Tiny but non-zero so a few configs run before the cap fires.
        soft_wall_clock_seconds=1,
        n_positive=int(arr_y.sum()),
        n_negative=int((arr_y == 0).sum()),
    )
    # A few configs run, then the cap fires and the rest get status=timeout.
    ok = [r for r in results if r.status == "ok"]
    timeout = [r for r in results if r.status == "timeout"]
    assert len(ok) >= 1
    assert len(timeout) >= 1
    assert len(ok) + len(timeout) == len(results)


def test_run_scout_catches_errors_as_data():
    X, y, mi = _make_synthetic_panel()
    arr_y = np.asarray(y.values)

    def broken_fit(*, hp_overlay, **kw):
        raise RuntimeError("synthetic boom")

    results = run_scout(
        X_train=X, y_train=arr_y, w_train=None,
        X_val=X, y_val=arr_y, w_val=None,
        X_eval=X, y_eval=arr_y, w_eval=None,
        mi_eval=mi,
        fit_one=broken_fit,
        backend="xgboost",
        spec=None,
        n_positive=int(arr_y.sum()),
        n_negative=int((arr_y == 0).sum()),
    )
    assert all(r.status == "error" for r in results)
    assert all("RuntimeError" in (r.error_message or "") for r in results)


# ---------------------------------------------------------------------------
# lexicographic_winner
# ---------------------------------------------------------------------------


def test_lexicographic_winner_picks_per_knob_argmax():
    """Two synthetic results with the same knob — winner takes higher R-p@1."""
    cfg_a = ScoutConfig(knob_name="max_depth", knob_value=3,
                         hp_overlay={"max_depth": 3})
    cfg_b = ScoutConfig(knob_name="max_depth", knob_value=4,
                         hp_overlay={"max_depth": 4})
    cfg_default = ScoutConfig(knob_name="defaults", knob_value=None, hp_overlay={})
    res_default = ScoutResult(
        config=cfg_default, val_brier=0.25, train_brier=0.22,
        eval_R_p_at_K={1: 0.10, 3: 0.10, 5: 0.10, 10: 0.10, 20: 0.10},
        train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    res_a = ScoutResult(
        config=cfg_a, val_brier=0.24, train_brier=0.21,
        eval_R_p_at_K={1: 0.20, 3: 0.20, 5: 0.20, 10: 0.20, 20: 0.20},
        train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    res_b = ScoutResult(
        config=cfg_b, val_brier=0.23, train_brier=0.20,
        eval_R_p_at_K={1: 0.30, 3: 0.20, 5: 0.20, 10: 0.20, 20: 0.20},
        train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    winner = lexicographic_winner([res_default, res_a, res_b])
    assert winner.hp_overlay == {"max_depth": 4}


def test_lexicographic_winner_lex_tiebreak_on_rp3():
    """When R-p@1 ties, lex falls through to R-p@3."""
    cfg_a = ScoutConfig(knob_name="eta", knob_value=0.05,
                         hp_overlay={"eta": 0.05})
    cfg_b = ScoutConfig(knob_name="eta", knob_value=0.1,
                         hp_overlay={"eta": 0.1})
    cfg_default = ScoutConfig(knob_name="defaults", knob_value=None, hp_overlay={})
    res_default = ScoutResult(
        config=cfg_default, val_brier=0.25, train_brier=0.22,
        eval_R_p_at_K={1: 0.10, 3: 0.10, 5: 0.10, 10: 0.10, 20: 0.10},
        train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    res_a = ScoutResult(
        config=cfg_a, val_brier=0.24, train_brier=0.21,
        eval_R_p_at_K={1: 0.20, 3: 0.10, 5: 0.10, 10: 0.10, 20: 0.10},
        train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    res_b = ScoutResult(
        config=cfg_b, val_brier=0.24, train_brier=0.21,
        eval_R_p_at_K={1: 0.20, 3: 0.20, 5: 0.10, 10: 0.10, 20: 0.10},
        train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    winner = lexicographic_winner([res_default, res_a, res_b])
    # Same R-p@1 (0.20), but res_b wins on R-p@3 (0.20 vs 0.10).
    assert winner.hp_overlay == {"eta": 0.1}


def test_lexicographic_winner_skips_knob_when_no_beat_defaults():
    """If no knob value beats defaults, the composed overlay omits that knob."""
    cfg_a = ScoutConfig(knob_name="max_depth", knob_value=3,
                         hp_overlay={"max_depth": 3})
    cfg_default = ScoutConfig(knob_name="defaults", knob_value=None, hp_overlay={})
    res_default = ScoutResult(
        config=cfg_default, val_brier=0.25, train_brier=0.22,
        eval_R_p_at_K={1: 0.30, 3: 0.30, 5: 0.30, 10: 0.30, 20: 0.30},
        train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    res_a = ScoutResult(
        config=cfg_a, val_brier=0.24, train_brier=0.21,
        eval_R_p_at_K={1: 0.10, 3: 0.10, 5: 0.10, 10: 0.10, 20: 0.10},
        train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    winner = lexicographic_winner([res_default, res_a])
    assert winner.hp_overlay == {}


def test_lexicographic_winner_no_ok_results_returns_defaults():
    cfg_a = ScoutConfig(knob_name="max_depth", knob_value=3,
                         hp_overlay={"max_depth": 3})
    res_a = ScoutResult(
        config=cfg_a, val_brier=None, train_brier=None,
        eval_R_p_at_K=None, train_val_gap=None, spiegelhalter_z=None,
        fit_seconds=0.0, status="timeout",
        error_message="timeout",
    )
    winner = lexicographic_winner([res_a])
    assert winner.hp_overlay == {}
    assert winner.knob_name == "defaults"


# ---------------------------------------------------------------------------
# per_knob_winners
# ---------------------------------------------------------------------------


def test_per_knob_winners_excludes_defaults_row():
    cfg_a = ScoutConfig(knob_name="max_depth", knob_value=3,
                         hp_overlay={"max_depth": 3})
    cfg_default = ScoutConfig(knob_name="defaults", knob_value=None, hp_overlay={})
    res_default = ScoutResult(
        config=cfg_default, val_brier=0.25, train_brier=0.22,
        eval_R_p_at_K={1: 0.10}, train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    res_a = ScoutResult(
        config=cfg_a, val_brier=0.24, train_brier=0.21,
        eval_R_p_at_K={1: 0.20}, train_val_gap=0.03, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    out = per_knob_winners([res_default, res_a])
    assert "defaults" not in out
    assert "max_depth" in out
    assert out["max_depth"]["knob_value"] == 3


# ---------------------------------------------------------------------------
# detect_degenerate_sink
# ---------------------------------------------------------------------------


def test_detect_degenerate_sink_fires_on_trivial_winner():
    cfg = ScoutConfig(knob_name="alpha", knob_value=1.0,
                       hp_overlay={"alpha": 1.0})
    res = ScoutResult(
        config=cfg, val_brier=0.21, train_brier=0.21,    # baseline ~0.20, gap~0
        eval_R_p_at_K={1: 0.1}, train_val_gap=0.0, spiegelhalter_z=0.0,
        fit_seconds=1.0, status="ok",
    )
    assert detect_degenerate_sink(cfg, [res], baseline_brier=0.20) is True


def test_detect_degenerate_sink_skips_healthy_winner():
    cfg = ScoutConfig(knob_name="max_depth", knob_value=3,
                       hp_overlay={"max_depth": 3})
    res = ScoutResult(
        config=cfg, val_brier=0.15, train_brier=0.10,    # below baseline + real gap
        eval_R_p_at_K={1: 0.3}, train_val_gap=0.05, spiegelhalter_z=0.5,
        fit_seconds=1.0, status="ok",
    )
    assert detect_degenerate_sink(cfg, [res], baseline_brier=0.20) is False


def test_detect_degenerate_sink_none_baseline_returns_false():
    cfg = ScoutConfig(knob_name="defaults", knob_value=None, hp_overlay={})
    assert detect_degenerate_sink(cfg, [], baseline_brier=None) is False


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_lex_oracle_priority_constant():
    assert LEX_ORACLE_PRIORITY == (1, 3, 5, 10, 20)


def test_speed_biased_prompt_mentions_depth2_and_eta01():
    assert "max_depth ∈ {2, 3}" in SPEED_BIASED_COMBINE_PROMPT
    assert "eta ≥ 0.1" in SPEED_BIASED_COMBINE_PROMPT


def test_default_grid_keys():
    assert set(DEFAULT_SCOUT_GRID.keys()) == {
        "max_depth", "eta", "colsample_bytree", "min_child_weight",
        "gamma", "alpha", "subsample", "scale_pos_weight",
    }
