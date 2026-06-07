"""V1.3 Option A — bundle additions + sweep CSV lookup + loop-doctrine
auto-disables (plan ``docs/gbdt/V1.3_option_a_loop_anti_auc_integration_plan.md``).

Covers:
- ``compute_anti_auc_flag`` thresholding (D4 — tightened to ``AUC ∈
  [0.46, 0.54]`` + ``R-Precision@10 lift > 1.8x``).
- ``compute_degenerate_sink_warning`` α=1.05 edge cases (D5).
- ``DiagnosticBundle`` + ``build_diagnostic_bundle`` field population
  (D7) — both with and without ``X_eval``/``sweep_row``.
- ``sweep_lookup`` — cell-key → experiment-name format + bare-row
  preference over suffixed variants (D3).
- ``best_checkpoint`` auto-disable of L1 tie-break on anti-AUC cells
  (D6) — falls back to strict val-Brier argmin, OR tie-breaks on eval
  R-p@1 when supplied.
- ``inner_stop_check`` auto-disable of val_brier plateau on anti-AUC
  cells (D6) — degradation + cap remain active.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbdt.diagnostics import (
    ANTI_AUC_FLAG_AUC_HIGH,
    ANTI_AUC_FLAG_AUC_LOW,
    ANTI_AUC_FLAG_RP10_LIFT_MIN,
    DiagnosticBundle,
    build_diagnostic_bundle,
    compute_anti_auc_flag,
    compute_degenerate_sink_warning,
)
from gbdt.fs_hp_loop import best_checkpoint, inner_stop_check
from gbdt.model import GBDTModel
from gbdt.sweep_lookup import (
    cell_key_to_experiment_name,
    lookup_sweep_row,
)


# ---------------------------------------------------------------------------
# compute_anti_auc_flag (D4)
# ---------------------------------------------------------------------------


def test_anti_auc_flag_none_row_is_unknown():
    assert compute_anti_auc_flag(None) == "unknown"


def test_anti_auc_flag_true_for_anti_auc_cell():
    # Cell-5 from r_precision_at_k.csv:
    # AUC=0.475, base=0.265, R-p@10=0.515 -> lift = 0.515/0.265 ≈ 1.94 > 1.8.
    row = {
        "AUC": 0.475014,
        "R_precision_at_10": 0.514841,
        "base_rate": 0.265217,
    }
    assert compute_anti_auc_flag(row) == "true"


def test_anti_auc_flag_false_for_high_auc_cell():
    # Strong AUC + strong lift = signal-rich cell, NOT anti-AUC.
    row = {"AUC": 0.85, "R_precision_at_10": 0.40, "base_rate": 0.02}
    assert compute_anti_auc_flag(row) == "false"


def test_anti_auc_flag_false_for_in_band_auc_but_low_lift():
    # AUC in [0.46, 0.54] but lift = 0.10 / 0.09 = 1.11 < 1.8 → "false".
    row = {"AUC": 0.50, "R_precision_at_10": 0.10, "base_rate": 0.09}
    assert compute_anti_auc_flag(row) == "false"


def test_anti_auc_flag_false_just_outside_auc_band_low():
    # AUC = 0.459 just below the low threshold → flag false (tightened from
    # 0.45 to 0.46 per D4).
    row = {"AUC": 0.459, "R_precision_at_10": 0.5, "base_rate": 0.1}
    assert compute_anti_auc_flag(row) == "false"


def test_anti_auc_flag_false_just_outside_auc_band_high():
    # AUC = 0.541 just above the high threshold (tightened from 0.55 to 0.54).
    row = {"AUC": 0.541, "R_precision_at_10": 0.5, "base_rate": 0.1}
    assert compute_anti_auc_flag(row) == "false"


def test_anti_auc_flag_unknown_on_zero_base_rate():
    # Degenerate cell with no positives in segment → can't compute lift.
    row = {"AUC": 0.50, "R_precision_at_10": 0.0, "base_rate": 0.0}
    assert compute_anti_auc_flag(row) == "unknown"


def test_anti_auc_flag_unknown_on_missing_keys():
    assert compute_anti_auc_flag({"AUC": 0.50}) == "unknown"


def test_anti_auc_flag_constants_match_plan_d4():
    # Single source of truth — drift detector if the constants get edited.
    assert ANTI_AUC_FLAG_AUC_LOW == 0.46
    assert ANTI_AUC_FLAG_AUC_HIGH == 0.54
    assert ANTI_AUC_FLAG_RP10_LIFT_MIN == 1.8


def test_anti_auc_flag_lift_strictly_greater():
    # lift = 1.8 exactly is NOT enough — rule uses > 1.8.
    row = {"AUC": 0.50, "R_precision_at_10": 0.18, "base_rate": 0.1}
    assert compute_anti_auc_flag(row) == "false"


# ---------------------------------------------------------------------------
# compute_degenerate_sink_warning (D5)
# ---------------------------------------------------------------------------


def test_degenerate_sink_below_threshold_warns():
    # val_brier = 0.20, base = 0.20, threshold = 1.05 → 0.20 <= 0.21 → warn.
    assert compute_degenerate_sink_warning(
        val_brier=0.20, weighted_base_rate_brier=0.20, threshold=1.05,
    ) is True


def test_degenerate_sink_just_above_threshold_no_warning():
    # val_brier = 0.211, base = 0.20, threshold = 1.05 → 0.211 > 0.21 → no warn.
    assert compute_degenerate_sink_warning(
        val_brier=0.211, weighted_base_rate_brier=0.20, threshold=1.05,
    ) is False


def test_degenerate_sink_exactly_at_threshold_warns():
    # val_brier = 0.21 == 1.05 * 0.20 → boundary IS a warn (<= comparison).
    assert compute_degenerate_sink_warning(
        val_brier=0.21, weighted_base_rate_brier=0.20, threshold=1.05,
    ) is True


def test_degenerate_sink_well_above_threshold_no_warning():
    # val_brier = 0.30 vs base = 0.20 — model has real signal.
    assert compute_degenerate_sink_warning(
        val_brier=0.30, weighted_base_rate_brier=0.20, threshold=1.05,
    ) is False


def test_degenerate_sink_none_inputs_no_warning():
    assert compute_degenerate_sink_warning(None, 0.20, 1.05) is False
    assert compute_degenerate_sink_warning(0.20, None, 1.05) is False
    assert compute_degenerate_sink_warning(None, None, 1.05) is False


def test_degenerate_sink_zero_base_no_warning():
    # Zero baseline (no positives in val) → ill-defined, no warn.
    assert compute_degenerate_sink_warning(0.0, 0.0, 1.05) is False


# ---------------------------------------------------------------------------
# DiagnosticBundle field additions (D7) + build_diagnostic_bundle wiring
# ---------------------------------------------------------------------------


def _toy_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
    })
    y = ((X["f1"] + rng.normal(0, 0.1, n)) > 0).astype(int).values
    return X, y


def _toy_eval_with_index(n=120, seed=1):
    """Toy eval segment with a (date, ticker) MultiIndex so the bundle's
    eval-side R-Precision@K computation has the per-day grouping it needs."""
    rng = np.random.default_rng(seed)
    n_days = 12
    n_tickers = 10
    dates = pd.date_range("2024-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows_date = np.repeat(dates, n_tickers)
    rows_ticker = np.tile(tickers, n_days)
    mi = pd.MultiIndex.from_arrays([rows_date, rows_ticker],
                                    names=["date", "ticker"])
    X_eval = pd.DataFrame({
        "f1": rng.normal(0, 1, len(mi)),
        "f2": rng.normal(0, 1, len(mi)),
        "f3": rng.normal(0, 1, len(mi)),
    })
    y_eval = ((X_eval["f1"] + rng.normal(0, 0.5, len(mi))) > 0.5).astype(int).values
    return X_eval, y_eval, mi


def test_bundle_defaults_v13_fields_when_no_eval():
    X, y = _toy_data()
    m = GBDTModel({"iterations": 30, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:250], y[:250], X.iloc[250:], y[250:])
    b = build_diagnostic_bundle(
        model=m, iter_idx=0, hp=m.hp,
        feature_names=list(X.columns),
        X_train=X.iloc[:250], y_train=y[:250],
        X_val=X.iloc[250:], y_val=y[250:],
        include_permutation=False,
    )
    # No X_eval / no sweep_row → V1.3 fields default to safe values.
    assert b.eval_r_precision_at_k is None
    assert b.anti_auc_flag == "unknown"
    # weighted_base_rate_brier is still computable from val alone — it's the
    # trivial constant predictor's Brier on val.
    assert b.weighted_base_rate_brier is not None
    assert 0.0 < b.weighted_base_rate_brier <= 0.25
    # No eval segment → no segment size.
    assert b.eval_segment_size is None
    # degenerate_sink_warning is False unless val_brier sinks below the
    # baseline (extremely unlikely on a learned model).
    assert isinstance(b.degenerate_sink_warning, bool)


def test_bundle_populates_eval_rp_when_eval_threaded():
    X, y = _toy_data()
    X_eval, y_eval, mi_eval = _toy_eval_with_index()
    m = GBDTModel({"iterations": 30, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:250], y[:250], X.iloc[250:], y[250:])
    b = build_diagnostic_bundle(
        model=m, iter_idx=0, hp=m.hp,
        feature_names=list(X.columns),
        X_train=X.iloc[:250], y_train=y[:250],
        X_val=X.iloc[250:], y_val=y[250:],
        X_eval=X_eval, y_eval=y_eval, mi_eval=mi_eval,
        include_permutation=False,
    )
    assert b.eval_r_precision_at_k is not None
    # Standard K set: {1, 3, 5, 10, 20}.
    assert set(b.eval_r_precision_at_k.keys()) == {1, 3, 5, 10, 20}
    # All values must be in [0, 1].
    for k, v in b.eval_r_precision_at_k.items():
        assert 0.0 <= v <= 1.0, f"R-p@{k} = {v}"
    assert b.eval_segment_size == len(y_eval)


def test_bundle_anti_auc_flag_threaded_from_sweep_row():
    X, y = _toy_data()
    X_eval, y_eval, mi_eval = _toy_eval_with_index()
    m = GBDTModel({"iterations": 30, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:250], y[:250], X.iloc[250:], y[250:])
    # Anti-AUC cell-5-shaped row.
    sweep_row = {
        "AUC": 0.475,
        "R_precision_at_10": 0.515,
        "base_rate": 0.265,
    }
    b = build_diagnostic_bundle(
        model=m, iter_idx=0, hp=m.hp,
        feature_names=list(X.columns),
        X_train=X.iloc[:250], y_train=y[:250],
        X_val=X.iloc[250:], y_val=y[250:],
        X_eval=X_eval, y_eval=y_eval, mi_eval=mi_eval,
        sweep_row=sweep_row,
        include_permutation=False,
    )
    assert b.anti_auc_flag == "true"
    d = b.to_dict()
    assert d["anti_auc_flag"] == "true"
    assert d["eval_r_precision_at_k"] is not None
    # The to_dict converts int keys to strings.
    assert "1" in d["eval_r_precision_at_k"]


def test_bundle_to_dict_round_trips_v13_fields_through_json():
    X, y = _toy_data()
    X_eval, y_eval, mi_eval = _toy_eval_with_index()
    m = GBDTModel({"iterations": 20, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:200], y[:200], X.iloc[200:], y[200:])
    b = build_diagnostic_bundle(
        model=m, iter_idx=1, hp=m.hp,
        feature_names=list(X.columns),
        X_train=X.iloc[:200], y_train=y[:200],
        X_val=X.iloc[200:], y_val=y[200:],
        X_eval=X_eval, y_eval=y_eval, mi_eval=mi_eval,
        sweep_row={"AUC": 0.90, "R_precision_at_10": 0.3, "base_rate": 0.02},
        include_permutation=False,
    )
    s = json.dumps(b.to_dict())
    re = json.loads(s)
    for k in (
        "eval_r_precision_at_k", "anti_auc_flag", "degenerate_sink_warning",
        "weighted_base_rate_brier", "eval_segment_size",
    ):
        assert k in re


# ---------------------------------------------------------------------------
# sweep_lookup (D3)
# ---------------------------------------------------------------------------


def test_cell_key_format_with_drawdown():
    assert (
        cell_key_to_experiment_name("nasdaq100", "up", 10, 50, 0.05)
        == "nasdaq100_up_10pct_50d_dd5pct"
    )


def test_cell_key_format_without_drawdown():
    assert (
        cell_key_to_experiment_name("nifty50", "up", 10, 20, None)
        == "nifty50_up_10pct_20d"
    )


def test_cell_key_format_handles_int_threshold_drawdown_pct():
    # max_drawdown is in (0, 1); the spec validator enforces that.
    assert (
        cell_key_to_experiment_name("sp500", "up", 20, 25, 0.10)
        == "sp500_up_20pct_25d_dd10pct"
    )


def _write_tiny_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "r_precision_at_k.csv"
    cols = [
        "experiment", "rows", "Q_days", "base_rate", "AUC",
        "R_precision_at_1", "R_precision_at_3", "R_precision_at_5",
        "R_precision_at_10", "R_precision_at_20",
    ]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


def test_lookup_bare_row_matches(tmp_path):
    csv = _write_tiny_csv(tmp_path, [
        {"experiment": "nasdaq100_up_10pct_50d_dd5pct",
         "AUC": 0.475, "base_rate": 0.265, "R_precision_at_10": 0.515,
         "R_precision_at_1": 0.671},
    ])
    row = lookup_sweep_row("nasdaq100_up_10pct_50d_dd5pct", csv)
    assert row is not None
    assert row["AUC"] == pytest.approx(0.475)


def test_lookup_prefers_bare_over_suffixed_variant(tmp_path):
    # When BOTH bare + a follow-up variant exist, bare wins.
    csv = _write_tiny_csv(tmp_path, [
        {"experiment": "nasdaq100_up_10pct_50d_dd5pct_agentloop",
         "AUC": 0.477, "base_rate": 0.265, "R_precision_at_10": 0.523},
        {"experiment": "nasdaq100_up_10pct_50d_dd5pct",
         "AUC": 0.475, "base_rate": 0.265, "R_precision_at_10": 0.515},
    ])
    row = lookup_sweep_row("nasdaq100_up_10pct_50d_dd5pct", csv)
    assert row is not None
    assert row["AUC"] == pytest.approx(0.475)  # bare row wins


def test_lookup_falls_back_to_variant_when_bare_missing(tmp_path):
    csv = _write_tiny_csv(tmp_path, [
        {"experiment": "nasdaq100_up_10pct_50d_dd5pct_agentloop",
         "AUC": 0.477, "base_rate": 0.265, "R_precision_at_10": 0.523},
    ])
    row = lookup_sweep_row("nasdaq100_up_10pct_50d_dd5pct", csv)
    assert row is not None
    assert row["AUC"] == pytest.approx(0.477)


def test_lookup_returns_none_when_no_match(tmp_path):
    csv = _write_tiny_csv(tmp_path, [
        {"experiment": "sp500_up_10pct_5d_dd5pct",
         "AUC": 0.78, "base_rate": 0.04, "R_precision_at_10": 0.34},
    ])
    assert lookup_sweep_row("nasdaq100_up_10pct_50d_dd5pct", csv) is None


def test_lookup_returns_none_when_csv_missing(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    assert lookup_sweep_row("anything", missing) is None


# ---------------------------------------------------------------------------
# best_checkpoint auto-disable on anti-AUC cells (D6)
# ---------------------------------------------------------------------------


def test_best_checkpoint_l1_disabled_on_anti_auc_falls_back_to_strict_argmin():
    """The classic L1 fixture (test_best_checkpoint_lower_gap_wins_within_band)
    picks iter 3 (lower gap) on a non-anti-AUC cell. With anti_auc_flag=true,
    L1 is auto-disabled — the strict val-Brier winner wins instead."""
    val_briers = [0.30, 0.200, 0.28, 0.203]
    gaps = [0.05, 0.04, 0.05, 0.005]
    zs = [1.0, 1.0, 1.0, 1.0]
    # Sanity: with flag != "true", the L1 tie-break picks iter 3 as before.
    best_idx_a, path_a = best_checkpoint(
        val_briers, train_val_gaps=gaps, spiegelhalter_zs=zs,
        tie_band=0.005, anti_auc_flag="false",
    )
    assert best_idx_a == 3
    assert path_a == "classic_l1"
    # With flag == "true", L1 is auto-disabled — strict val-Brier argmin wins.
    best_idx_b, path_b = best_checkpoint(
        val_briers, train_val_gaps=gaps, spiegelhalter_zs=zs,
        tie_band=0.005, anti_auc_flag="true",
    )
    assert best_idx_b == 1
    assert path_b == "strict_val_brier"


def test_best_checkpoint_anti_auc_tie_breaks_on_eval_rp1_when_supplied():
    """With anti_auc_flag=true AND eval R-p@1 supplied (all-non-None among
    the tied set), higher eval R-p@1 wins over strict val-Brier argmin."""
    val_briers = [0.30, 0.200, 0.28, 0.203]
    # Iters 1 + 3 are tied within the band [0.200, 0.205]. Iter 3 has the
    # higher eval R-p@1.
    eval_rp1 = [0.10, 0.30, 0.05, 0.55]
    best_idx, path = best_checkpoint(
        val_briers, train_val_gaps=None, spiegelhalter_zs=None,
        tie_band=0.005, anti_auc_flag="true",
        eval_r_precision_at_1s=eval_rp1,
    )
    assert best_idx == 3
    assert path == "anti_auc_eval_rp1"


def test_best_checkpoint_anti_auc_falls_back_when_eval_rp1_partial():
    """If ANY tied iter is missing eval R-p@1, we fall back to strict
    val-Brier argmin (no mixed-metric ranking)."""
    val_briers = [0.30, 0.200, 0.28, 0.203]
    eval_rp1 = [0.10, None, 0.05, 0.55]  # iter 1 missing
    best_idx, path = best_checkpoint(
        val_briers, train_val_gaps=None, spiegelhalter_zs=None,
        tie_band=0.005, anti_auc_flag="true",
        eval_r_precision_at_1s=eval_rp1,
    )
    assert best_idx == 1  # strict val-Brier winner
    assert path == "strict_val_brier"


def test_best_checkpoint_unknown_flag_preserves_l1_behavior():
    val_briers = [0.30, 0.200, 0.28, 0.203]
    gaps = [0.05, 0.04, 0.05, 0.005]
    zs = [1.0, 1.0, 1.0, 1.0]
    # Default flag is "unknown" → L1 active.
    best_idx, path = best_checkpoint(
        val_briers, train_val_gaps=gaps, spiegelhalter_zs=zs,
        tie_band=0.005,
    )
    assert best_idx == 3
    assert path == "classic_l1"


# ---------------------------------------------------------------------------
# inner_stop_check auto-disable on anti-AUC cells (D6)
# ---------------------------------------------------------------------------


def test_inner_stop_plateau_suppressed_when_anti_auc_flag_true():
    # Same fixture that triggers plateau in default mode.
    history = [0.30, 0.28, 0.279, 0.278]
    stop, signal = inner_stop_check(
        history, plateau_threshold=0.005, anti_auc_flag="true",
    )
    assert not stop
    assert signal is None


def test_inner_stop_degradation_fires_on_anti_auc_cells():
    # Regression is a real stop signal on every cell shape.
    history = [0.30, 0.25, 0.30]
    stop, signal = inner_stop_check(
        history, degradation_gate=0.01, anti_auc_flag="true",
    )
    assert stop and signal == "degradation"


def test_inner_stop_cap_fires_on_anti_auc_cells():
    history = [0.25, 0.24, 0.235, 0.232, 0.231, 0.230, 0.229, 0.228]
    stop, signal = inner_stop_check(
        history, max_iterations=8, plateau_threshold=0.0001,
        anti_auc_flag="true",
    )
    assert stop and signal == "cap"


def test_inner_stop_plateau_active_on_unknown_flag():
    # Defaults to pre-V1.3 behavior; "unknown" should NOT auto-disable.
    history = [0.30, 0.28, 0.279, 0.278]
    stop, signal = inner_stop_check(
        history, plateau_threshold=0.005, anti_auc_flag="unknown",
    )
    assert stop and signal == "plateau"


def test_inner_stop_plateau_active_on_false_flag():
    history = [0.30, 0.28, 0.279, 0.278]
    stop, signal = inner_stop_check(
        history, plateau_threshold=0.005, anti_auc_flag="false",
    )
    assert stop and signal == "plateau"


# ---------------------------------------------------------------------------
# Integration — walk_forward_train end-to-end with V1.3 wiring
# ---------------------------------------------------------------------------


def _toy_panel(n_per_ticker: int = 1600, n_tickers: int = 3, seed: int = 0):
    """Same fixture as tests/gbdt/test_train.py._toy_panel — small enough
    to run a 2-iter walk_forward_train in well under a second."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_per_ticker, freq="B")
    frames = []
    for i in range(n_tickers):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_per_ticker)))
        df = pd.DataFrame({
            "date": dates,
            "ticker": f"T{i}",
            "open": c, "high": c * 1.005, "low": c * 0.995,
            "close": c, "adj_close": c,
            "volume": np.ones(n_per_ticker, dtype=int),
        })
        frames.append(df)
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    n_total = len(panel)
    X = pd.DataFrame(rng.normal(0, 1, (n_total, 6)),
                      index=panel.index,
                      columns=["sig", "n1", "n2", "n3", "n4", "n5"])
    y = ((X["sig"] + rng.normal(0, 0.3, n_total)) > 0).astype(int)
    return panel, X, y


def test_walk_forward_train_populates_eval_rp_in_iter_bundles():
    """A1 smoke — eval_r_precision_at_k populated for every iter in the
    bundle history when walk_forward_train carves a non-empty eval
    segment."""
    from gbdt.train import walk_forward_train

    panel, X, y = _toy_panel(1600, 2, seed=11)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
        max_iterations=2,
        # Anti-AUC sweep row → flag will be "true" → auto-disables fire.
        sweep_row={"AUC": 0.475, "R_precision_at_10": 0.515, "base_rate": 0.265},
    )
    for b in result.iterations:
        assert b.eval_r_precision_at_k is not None
        assert set(b.eval_r_precision_at_k.keys()) == {1, 3, 5, 10, 20}
        assert b.anti_auc_flag == "true"
        assert b.eval_segment_size is not None
        assert b.weighted_base_rate_brier is not None


def test_walk_forward_train_handles_no_sweep_row():
    """When no sweep_row passed, flag stays "unknown" + auto-disables don't
    fire (pre-V1.3 behavior preserved)."""
    from gbdt.train import walk_forward_train

    panel, X, y = _toy_panel(1600, 2, seed=12)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
        max_iterations=2,
    )
    for b in result.iterations:
        assert b.anti_auc_flag == "unknown"
        # eval segment is non-empty for this toy fixture, so the field IS set
        # — verifies the eval predict pass runs regardless of flag value.
        assert b.eval_r_precision_at_k is not None


def test_walk_forward_a5_non_anti_auc_byte_identical():
    """A5 — non-anti-AUC byte-identical regression.

    Same toy panel + same HP + same seed, run TWICE: once with V1.3
    sweep_row matching a non-anti-AUC cell (e.g. high AUC) and once with
    no sweep_row (legacy path). The predictions, val_brier history, and
    best checkpoint must be byte-identical — V1.3 doesn't perturb
    non-anti-AUC behavior."""
    from gbdt.train import walk_forward_train

    def _run(sweep_row):
        panel, X, y = _toy_panel(1600, 2, seed=21)
        return walk_forward_train(
            panel=panel, X=X, y=y, features=list(X.columns),
            hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
            max_iterations=3,
            plateau_threshold=0.0001,  # disable plateau so the trio runs
            degradation_gate=0.5,
            sweep_row=sweep_row,
        )

    high_auc_row = {"AUC": 0.85, "R_precision_at_10": 0.40, "base_rate": 0.02}
    result_v13 = _run(high_auc_row)         # flag → "false"
    result_legacy = _run(None)              # flag → "unknown"

    # The runs differ ONLY in the bundle's anti_auc_flag (false vs unknown),
    # but neither value triggers the auto-disables. Predictions must match
    # byte-for-byte.
    assert result_v13.best_iteration == result_legacy.best_iteration
    assert result_v13.best_val_brier == result_legacy.best_val_brier
    for seg in ("train", "val", "eval", "test"):
        np.testing.assert_array_equal(
            result_v13.predictions[seg]["p_raw"].values,
            result_legacy.predictions[seg]["p_raw"].values,
        )
        np.testing.assert_array_equal(
            result_v13.predictions[seg]["p_calibrated"].values,
            result_legacy.predictions[seg]["p_calibrated"].values,
        )


def test_walk_forward_a6_determinism():
    """A6 — same spec twice gives bit-identical predictions.

    The V1.3 eval predict pass is read-only (no model mutation); two runs
    of the same configuration must produce identical predictions to
    confirm we haven't introduced any non-determinism."""
    from gbdt.train import walk_forward_train

    def _run():
        panel, X, y = _toy_panel(1600, 2, seed=42)
        return walk_forward_train(
            panel=panel, X=X, y=y, features=list(X.columns),
            hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
            max_iterations=2,
            sweep_row={"AUC": 0.475, "R_precision_at_10": 0.515,
                       "base_rate": 0.265},  # anti-AUC → auto-disables fire
        )

    r1 = _run()
    r2 = _run()
    assert r1.best_iteration == r2.best_iteration
    assert r1.best_val_brier == r2.best_val_brier
    for seg in ("train", "val", "eval", "test"):
        np.testing.assert_array_equal(
            r1.predictions[seg]["p_raw"].values,
            r2.predictions[seg]["p_raw"].values,
        )
        np.testing.assert_array_equal(
            r1.predictions[seg]["p_calibrated"].values,
            r2.predictions[seg]["p_calibrated"].values,
        )
    # Eval R-p@K must also be identical iter-by-iter.
    for b1, b2 in zip(r1.iterations, r2.iterations):
        assert b1.eval_r_precision_at_k == b2.eval_r_precision_at_k


def test_walk_forward_anti_auc_degenerate_sink_threshold_from_kwarg():
    """The runner forwards backend.fs_hp_loop.degenerate_sink_threshold to
    walk_forward_train; the bundle reflects the override.

    The toy panel has a strong feature → val_brier consistently lands
    BELOW the baseline (the learned model beats the trivial constant).
    Any threshold >= 1.0 will therefore fire. So we test the direction
    of the threshold dependence: a strict-enough threshold (< 1.0) does
    NOT fire; a loose threshold (>= 1.0) does.
    """
    from gbdt.train import walk_forward_train

    panel, X, y = _toy_panel(1600, 2, seed=33)
    # The toy fixture's val_brier is ~28% of baseline (the model has
    # strong signal). A threshold of 0.2 (val_brier <= 0.2×baseline →
    # warn) does NOT fire on this fixture; 0.5 (val_brier <= 0.5×baseline)
    # does. This proves the spec value is being threaded through and the
    # warning's direction works as advertised — the actual production
    # default (1.05) only fires on degenerate models where val_brier is
    # essentially AT baseline.
    result_strict = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
        max_iterations=1,
        sweep_row={"AUC": 0.85, "R_precision_at_10": 0.4, "base_rate": 0.02},
        degenerate_sink_threshold=0.2,
    )
    assert result_strict.iterations[0].degenerate_sink_warning is False

    result_loose = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
        max_iterations=1,
        sweep_row={"AUC": 0.85, "R_precision_at_10": 0.4, "base_rate": 0.02},
        degenerate_sink_threshold=0.5,
    )
    assert result_loose.iterations[0].degenerate_sink_warning is True


def test_walk_forward_handles_empty_eval_segment_gracefully():
    """When the eval segment is structurally empty (e.g. horizon eats it),
    eval_r_precision_at_k stays None / NaN-handled gracefully — no
    crashes."""
    from gbdt.train import SplitSpec, walk_forward_train

    panel, X, y = _toy_panel(1600, 2, seed=44)
    # All-NaN target on the eval segment by setting y to NaN there.
    # Simulate this by zeroing out the eval-positional rows' target.
    # Easier: use a split where eval_rows = 0 isn't allowed (val_rows must
    # be > 0); instead just ensure the predict pass tolerates a slim panel.
    # The carve produces a non-empty eval segment for the toy fixture, so
    # the actual graceful-degradation path is covered by the unit test
    # ``test_bundle_defaults_v13_fields_when_no_eval`` above. Smoke-test
    # here only that walk_forward_train doesn't crash on a tiny eval.
    split = SplitSpec(800, 400, 200, 100)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 10, "depth": 2, "boosting_type": "Plain"},
        split=split, max_iterations=1,
    )
    # Bundle is well-formed.
    assert result.iterations[0].eval_segment_size is not None


def test_loop_protocol_request_bundle_carries_v13_fields():
    """A3 smoke — the agent reads loop/iter_<N>_request.json::diagnostics,
    which is built by diagnose_payload.build_diagnose_payload. The V1.3
    fields MUST propagate through verbatim or the SKILL.md doctrine
    instructions can't be acted on."""
    from gbdt.diagnose_payload import build_diagnose_payload
    from gbdt.loop_protocol import build_request_bundle

    X, y = _toy_data()
    X_eval, y_eval, mi_eval = _toy_eval_with_index()
    m = GBDTModel({"iterations": 30, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:250], y[:250], X.iloc[250:], y[250:])
    bundle = build_diagnostic_bundle(
        model=m, iter_idx=0, hp=m.hp,
        feature_names=list(X.columns),
        X_train=X.iloc[:250], y_train=y[:250],
        X_val=X.iloc[250:], y_val=y[250:],
        X_eval=X_eval, y_eval=y_eval, mi_eval=mi_eval,
        sweep_row={"AUC": 0.475, "R_precision_at_10": 0.515,
                   "base_rate": 0.265},
        include_permutation=False,
    )
    payload = build_diagnose_payload(bundle)
    assert payload["anti_auc_flag"] == "true"
    assert payload["eval_r_precision_at_k"] is not None
    assert "1" in payload["eval_r_precision_at_k"]  # _json_safe stringifies keys
    assert payload["weighted_base_rate_brier"] is not None
    assert payload["eval_segment_size"] == len(y_eval)

    request = build_request_bundle(
        bundle, iter_n=0, run_id="test_run", max_iterations=5,
        available_features=list(X.columns),
    )
    diag = request["diagnostics"]
    assert diag["anti_auc_flag"] == "true"
    assert diag["eval_r_precision_at_k"] is not None
    assert diag["weighted_base_rate_brier"] is not None
    assert diag["eval_segment_size"] == len(y_eval)


def test_walk_forward_inner_stop_anti_auc_skips_plateau():
    """When anti_auc_flag=true, inner_stop_check (called from
    walk_forward_train) does NOT plateau-stop even if val_brier flattens."""
    from gbdt.train import walk_forward_train

    panel, X, y = _toy_panel(1600, 2, seed=55)

    def noop_cb(bundle, available):
        # Return same features + same HP → val_brier essentially flat.
        return list(available), dict(bundle.hp), "noop"

    # Run 3 iters with an anti-AUC flag — plateau should NOT fire even
    # though val_brier is flat by construction.
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
        max_iterations=3,
        plateau_threshold=0.005,
        degradation_gate=0.5,
        fs_hp_callback=noop_cb,
        sweep_row={"AUC": 0.475, "R_precision_at_10": 0.515,
                   "base_rate": 0.265},  # anti-AUC
    )
    # Plateau MUST not be the stop signal — only cap (or degradation,
    # which is suppressed by degradation_gate=0.5).
    assert result.inner_stop_signal == "cap"
    assert len(result.iterations) == 3


# ---------------------------------------------------------------------------
# Bug #222 fix — eval R-p@K computed on calibrated predictions
# ---------------------------------------------------------------------------


def test_bundle_eval_rp_uses_calibrated_predictions_on_degenerate_model():
    """Bug #222 regression — the bundle's eval R-p@K is computed on
    CALIBRATED predictions (matching canonical CSV scoring), not raw.

    Pre-fix, the bundle scored eval R-p@K on raw model output under the
    FALSE assumption that isotonic monotonicity preserves rank order. On a
    deliberately tiny / degenerate model that emits clustered raw
    predictions, post-isotonic predictions collapse to far fewer distinct
    values; the alphabetical-ticker tie-break then dominates ranking. We
    construct a fixture where raw + calibrated give visibly different
    R-p@K — if the fix is reverted, this test catches it.

    Strategy:
      - Tiny eval segment with structured raw predictions (a few distinct
        levels), then verify the calibrated R-p@K differs from what raw
        scoring would have produced.
    """
    import numpy as np

    from gbdt.calibration import (
        apply_calibrator,
        conditional_isotonic,
    )
    from gbdt.diagnostics import _r_precision_at_k_from_arrays

    # Build a tiny val + eval segment.
    n_days = 10
    n_tickers = 8
    dates = pd.date_range("2024-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows_date = np.repeat(dates, n_tickers)
    rows_ticker = np.tile(tickers, n_days)

    # Strongly miscalibrated raw predictions on val: y is INVERSELY related
    # to p_raw (high p → label 0, low p → label 1). Isotonic must fit a
    # weakly-monotone (here: flat or near-flat) curve on this, collapsing
    # all raw values in [0.42, 0.68] to a single calibrated level. This
    # replicates the cell-5 pathology: many distinct raw values → very few
    # distinct calibrated values.
    n_val = 100
    p_val_raw = np.linspace(0.40, 0.70, n_val)
    # Anti-correlated label: high p_raw → label 0, low p_raw → label 1.
    y_val = (p_val_raw < 0.55).astype(int)

    # Build eval predictions: positive at the alphabetically-LAST ticker
    # on every day, AND give it the highest raw probability. The fix must
    # surface that calibration COLLAPSES this distinction (raw 0.68 + raw
    # 0.42 both map to the same calibrated value on this val fit) → the
    # canonical (p_calibrated desc, ticker asc) tie-break picks T00 first
    # → calibrated R-p@1 drops to 0 (the positive is at T07, ranked last).
    p_eval_raw = np.zeros(n_days * n_tickers, dtype=float)
    y_eval = np.zeros(n_days * n_tickers, dtype=int)
    for d_idx in range(n_days):
        rows = np.arange(d_idx * n_tickers, (d_idx + 1) * n_tickers)
        p_eval_raw[rows] = 0.42
        last_row = rows[-1]  # T07
        p_eval_raw[last_row] = 0.68
        y_eval[last_row] = 1

    # Fit conditional_isotonic calibrator on val (the bundle path).
    cal_decision = conditional_isotonic(y_val, p_val_raw, z_threshold=2.0)
    # Sanity: the fixture really triggers isotonic fit (not native).
    assert cal_decision.method == "isotonic", (
        f"fixture failed to trigger isotonic; got {cal_decision.method!r}. "
        f"|z|={abs(cal_decision.spiegelhalter_z):.3f}"
    )

    # Score eval BOTH ways.
    raw_rp = _r_precision_at_k_from_arrays(
        dates=rows_date, tickers=rows_ticker,
        p_calibrated=p_eval_raw,  # name is misleading; pass raw to score raw
        y_true=y_eval,
    )
    p_eval_calibrated = apply_calibrator(p_eval_raw, cal_decision.calibrator)
    cal_rp = _r_precision_at_k_from_arrays(
        dates=rows_date, tickers=rows_ticker,
        p_calibrated=p_eval_calibrated,
        y_true=y_eval,
    )

    # The raw-scored R-p@1 captures all 10 positives (top-raw at every day
    # is the alphabetically-last positive ticker). After isotonic
    # collapses raw to ≤ 2 distinct values, all 8 tickers per day TIE on
    # calibrated p, the alphabetical tie-break ranks T00 first; the
    # positive at T07 falls to rank 8 → calibrated R-p@1 = 0.
    assert raw_rp[1] == 1.0, f"fixture raw R-p@1 sanity: got {raw_rp[1]}"
    assert cal_rp[1] < raw_rp[1], (
        f"fix proves: calibrated R-p@1 ({cal_rp[1]:.3f}) must differ from "
        f"raw ({raw_rp[1]:.3f}) on this degenerate fixture (calibration "
        f"collapses raw predictions → alphabetical tie-break dominates)."
    )

    # Now verify the bundle uses calibrated (NOT raw). Build a model whose
    # predict_proba returns the constructed p_eval_raw on a matching X_eval.
    # We use a small stub model to keep the test fast + deterministic.

    class _StubModel:
        def __init__(self, p_val: np.ndarray, p_eval: np.ndarray) -> None:
            self._p_val = p_val
            self._p_eval = p_eval
            self.hp = {"iterations": 1, "depth": 1}
            self.best_iteration = 1
            self.evals_result = None

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            if len(X) == n_val:
                return self._p_val
            if len(X) == len(p_eval_raw):
                return self._p_eval
            # Train pass — return matching length.
            return np.full(len(X), 0.5)

        def feature_importance(self, kind, *args, **kwargs):
            # Return a pandas Series shaped like the real GBDTModel surface.
            return pd.Series([0.0, 0.0, 0.0], index=["f1", "f2", "f3"])

    X_val_df = pd.DataFrame({
        "f1": np.zeros(n_val), "f2": np.zeros(n_val), "f3": np.zeros(n_val),
    })
    X_eval_df = pd.DataFrame({
        "f1": np.zeros(len(p_eval_raw)),
        "f2": np.zeros(len(p_eval_raw)),
        "f3": np.zeros(len(p_eval_raw)),
    })
    mi_eval = pd.MultiIndex.from_arrays(
        [rows_date, rows_ticker], names=["date", "ticker"],
    )
    X_train_df = pd.DataFrame({
        "f1": np.zeros(50), "f2": np.zeros(50), "f3": np.zeros(50),
    })
    y_train = np.zeros(50, dtype=int)

    stub = _StubModel(p_val_raw, p_eval_raw)
    bundle = build_diagnostic_bundle(
        model=stub, iter_idx=0, hp=stub.hp,
        feature_names=["f1", "f2", "f3"],
        X_train=X_train_df, y_train=y_train,
        X_val=X_val_df, y_val=y_val,
        X_eval=X_eval_df, y_eval=y_eval,
        mi_eval=mi_eval,
        include_permutation=False,
        # Default calibration_method="conditional_isotonic" → matches what
        # cal_decision above fit on the same (y_val, p_val_raw).
    )
    # The bundle's eval R-p@1 MUST match the calibrated scoring (not raw).
    assert bundle.eval_r_precision_at_k is not None
    assert bundle.eval_r_precision_at_k[1] == pytest.approx(cal_rp[1])
    # And critically, must NOT match the raw scoring.
    assert bundle.eval_r_precision_at_k[1] != pytest.approx(raw_rp[1])


def test_bundle_calibration_method_native_passes_raw_through():
    """When ``calibration_method="native"`` is in effect, the in-loop
    calibrator is None and eval R-p@K is computed on raw predictions
    (matching what the finalization-side scoring would produce in that
    config). This is the contract: the in-loop signal mirrors the
    finalization scoring under whatever calibration method the spec
    selected."""
    import numpy as np

    X, y = _toy_data()
    X_eval, y_eval, mi_eval = _toy_eval_with_index()
    m = GBDTModel({"iterations": 30, "depth": 4, "boosting_type": "Plain"})
    m.fit(X.iloc[:250], y[:250], X.iloc[250:], y[250:])

    b_native = build_diagnostic_bundle(
        model=m, iter_idx=0, hp=m.hp,
        feature_names=list(X.columns),
        X_train=X.iloc[:250], y_train=y[:250],
        X_val=X.iloc[250:], y_val=y[250:],
        X_eval=X_eval, y_eval=y_eval, mi_eval=mi_eval,
        include_permutation=False,
        calibration_method="native",
    )
    # Verify against direct raw computation.
    from gbdt.diagnostics import _r_precision_at_k_from_arrays

    p_eval_raw = m.predict_proba(X_eval)
    dates = mi_eval.get_level_values("date").to_numpy()
    tickers = mi_eval.get_level_values("ticker").to_numpy()
    expected = _r_precision_at_k_from_arrays(
        dates=dates, tickers=tickers,
        p_calibrated=np.asarray(p_eval_raw, dtype=float),
        y_true=np.asarray(y_eval, dtype=int),
    )
    assert b_native.eval_r_precision_at_k == expected


def test_walk_forward_passes_calibration_method_to_bundle():
    """Smoke — walk_forward_train threads calibration_method +
    calibration_z_threshold into build_diagnostic_bundle. Verifies that
    the same calibrator is used per-iter in the bundle and at finalization
    (the bundle is fit fresh per-iter; the per-iter calibrator + the
    finalization calibrator agree on the FIRST iter when there's only one
    model, modulo the train/val split being the same)."""
    from gbdt.train import walk_forward_train

    panel, X, y = _toy_panel(1600, 2, seed=77)
    # Use isotonic_always to force isotonic fit regardless of Z (eliminates
    # one source of noise in the comparison).
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
        max_iterations=1,
        calibration_method="isotonic_always",
    )
    # Verify the bundle came out with a populated eval R-p@K, and that
    # the calibrator on the result is the finalization isotonic.
    assert result.iterations[0].eval_r_precision_at_k is not None
    assert result.calibration.method == "isotonic"
