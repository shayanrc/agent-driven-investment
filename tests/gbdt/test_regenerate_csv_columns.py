"""2026-06-05 — training-regime descriptor columns (`mode`,
`n_iterations_run`, `backend`) on the canonical R-Precision@K CSV.

The CSV regenerator (`scripts/gbdt/regenerate_r_precision_at_k_csv`) reads
`spec.yaml` + `iterations.jsonl` from each artifact dir and falls back to a
static map + cell-name suffix rules for pruned `_agentloop*` cells whose
artifact dirs are gone.

The fallback must never override a primary (artifact-dir) classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.gbdt.regenerate_r_precision_at_k_csv import (
    _PRUNED_AGENTLOOP_FALLBACK,
    _REGIME_COLS,
    _backend_from_name,
    _classify_mode_from_name,
    _classify_mode_from_spec,
    _regime_for_artifact,
    _regime_for_pruned,
)


# ---------- _classify_mode_from_spec -------------------------------------


def test_mode_spec_agent_file_protocol_wins():
    """callback_mode=agent_file_protocol → agent_file_protocol regardless of max_iter."""
    assert _classify_mode_from_spec("agent_file_protocol", 1) == "agent_file_protocol"
    assert _classify_mode_from_spec("agent_file_protocol", 10) == "agent_file_protocol"
    assert _classify_mode_from_spec("agent_file_protocol", None) == "agent_file_protocol"


def test_mode_spec_default_with_max_iter_le_3_is_sweep():
    """callback_mode=default + max_iter <= 3 → sweep."""
    assert _classify_mode_from_spec("default", 1) == "sweep"
    assert _classify_mode_from_spec("default", 2) == "sweep"
    assert _classify_mode_from_spec("default", 3) == "sweep"


def test_mode_spec_default_with_max_iter_ge_8_is_full_loop():
    """callback_mode=default + max_iter >= 8 → default_full_loop."""
    assert _classify_mode_from_spec("default", 8) == "default_full_loop"
    assert _classify_mode_from_spec("default", 16) == "default_full_loop"


def test_mode_spec_default_with_max_iter_4_to_7_is_full_loop():
    """The 4-7 gap defaults to default_full_loop (it's not a sweep cap)."""
    for mi in (4, 5, 6, 7):
        assert _classify_mode_from_spec("default", mi) == "default_full_loop"


def test_mode_spec_missing_callback_mode_treated_as_default():
    """Older spec files omit callback_mode; runner treats absent as default."""
    assert _classify_mode_from_spec(None, 3) == "sweep"
    assert _classify_mode_from_spec(None, 8) == "default_full_loop"


# ---------- _classify_mode_from_name -------------------------------------


def test_mode_name_agentloop_legacy():
    assert _classify_mode_from_name("sp500_up_50pct_50d_dd25pct_agentloop") == "agentloop_legacy"
    assert _classify_mode_from_name("foo_agentloop_v1.3") == "agentloop_legacy"
    assert _classify_mode_from_name("foo_agentloop_mix_mcw3") == "agentloop_legacy"


def test_mode_name_suffix_dispatch():
    assert _classify_mode_from_name("foo_aligned") == "sweep"
    assert _classify_mode_from_name("foo_pilot") == "default_full_loop"
    assert _classify_mode_from_name("foo_xgb_acceptance") == "agent_file_protocol"
    assert _classify_mode_from_name("foo_acceptance") == "agent_file_protocol"
    assert _classify_mode_from_name("foo_phase8") == "default_full_loop"
    assert _classify_mode_from_name("foo_catboost_phase8") == "default_full_loop"


def test_mode_name_b_acceptance_variants():
    """P9 cells: `_b_acceptance_agent` is agent mode; `_b_acceptance` alone is default."""
    assert _classify_mode_from_name("foo_b_acceptance_agent") == "agent_file_protocol"
    assert _classify_mode_from_name("foo_b_acceptance") == "default_full_loop"


def test_mode_name_no_suffix_is_sweep():
    """Naked cell names (no suffix marker) were the original sweep cells."""
    assert _classify_mode_from_name("russell1000_up_50pct_25d_dd25pct") == "sweep"


# ---------- _backend_from_name -------------------------------------------


def test_backend_name_xgboost():
    assert _backend_from_name("foo_xgb_acceptance") == "xgboost"
    assert _backend_from_name("foo_xgboost_loop") == "xgboost"
    assert _backend_from_name("foo_agentloop_v1.3") == "xgboost"


def test_backend_name_catboost():
    assert _backend_from_name("foo_catboost_phase8") == "catboost"


def test_backend_name_blank_when_no_hint():
    assert _backend_from_name("russell1000_up_50pct_25d_dd25pct") == ""
    assert _backend_from_name("foo_aligned") == ""


# ---------- _regime_for_artifact (primary, via spec.yaml + iterations.jsonl)


def _make_art_dir(
    tmp_path: Path,
    *,
    library: str | None = "catboost",
    callback_mode: str | None = "default",
    max_iterations: int | None = 3,
    n_iter_lines: int = 0,
) -> Path:
    """Build a fake artifact dir with a spec.yaml + iterations.jsonl."""
    art = tmp_path / "results" / "gbdt" / "experiments" / "fake_cell"
    art.mkdir(parents=True, exist_ok=True)
    backend_block: dict = {}
    if library is not None:
        backend_block["library"] = library
    fs_hp_loop: dict = {}
    if callback_mode is not None:
        fs_hp_loop["callback_mode"] = callback_mode
    if max_iterations is not None:
        fs_hp_loop["max_iterations"] = max_iterations
    if fs_hp_loop:
        backend_block["fs_hp_loop"] = fs_hp_loop
    (art / "spec.yaml").write_text(yaml.safe_dump({"backend": backend_block}))
    iters = "\n".join(json.dumps({"iter": i}) for i in range(n_iter_lines))
    (art / "iterations.jsonl").write_text(iters + ("\n" if iters else ""))
    return art


def test_regime_artifact_sweep_case(tmp_path):
    art = _make_art_dir(
        tmp_path, library="catboost", callback_mode="default",
        max_iterations=3, n_iter_lines=2,
    )
    r = _regime_for_artifact(art)
    assert r == {"mode": "sweep", "n_iterations_run": "2", "backend": "catboost"}


def test_regime_artifact_default_full_loop_case(tmp_path):
    art = _make_art_dir(
        tmp_path, library="catboost", callback_mode="default",
        max_iterations=8, n_iter_lines=3,
    )
    r = _regime_for_artifact(art)
    assert r == {"mode": "default_full_loop", "n_iterations_run": "3", "backend": "catboost"}


def test_regime_artifact_agent_file_protocol_case(tmp_path):
    art = _make_art_dir(
        tmp_path, library="xgboost", callback_mode="agent_file_protocol",
        max_iterations=10, n_iter_lines=0,
    )
    r = _regime_for_artifact(art)
    assert r == {"mode": "agent_file_protocol", "n_iterations_run": "0", "backend": "xgboost"}


def test_regime_artifact_missing_library_defaults_to_catboost(tmp_path):
    """Specs that omit ``backend.library`` rely on the runner default = catboost."""
    art = _make_art_dir(
        tmp_path, library=None, callback_mode="default",
        max_iterations=3, n_iter_lines=2,
    )
    r = _regime_for_artifact(art)
    assert r["backend"] == "catboost"
    assert r["mode"] == "sweep"


def test_regime_artifact_no_spec_emits_blanks(tmp_path):
    """An artifact dir with no spec.yaml + no iterations.jsonl yields blanks."""
    art = tmp_path / "no_spec"
    art.mkdir()
    r = _regime_for_artifact(art)
    assert r == {"mode": "", "n_iterations_run": "", "backend": ""}


# ---------- _regime_for_pruned (fallback, via name + static map)


def test_pruned_fallback_static_map_hit():
    """Pruned _agentloop cells map to known (mode, n_iter, backend)."""
    r = _regime_for_pruned("sp500_up_50pct_50d_dd25pct_agentloop")
    assert r == _PRUNED_AGENTLOOP_FALLBACK["sp500_up_50pct_50d_dd25pct_agentloop"]
    assert r["mode"] == "agentloop_legacy"
    assert r["backend"] == "xgboost"


def test_pruned_fallback_unknown_agentloop_falls_through_to_suffix():
    """An _agentloop cell NOT in the static map still classifies as agentloop_legacy."""
    r = _regime_for_pruned("future_universe_up_10pct_50d_agentloop")
    assert r["mode"] == "agentloop_legacy"
    assert r["n_iterations_run"] == ""


def test_pruned_fallback_suffix_dispatch():
    assert _regime_for_pruned("foo_aligned")["mode"] == "sweep"
    assert _regime_for_pruned("foo_pilot")["mode"] == "default_full_loop"


# ---------- CSV column order lock


def test_regime_cols_constant_is_locked():
    """Schema lock — the 3 column names + order must not drift."""
    assert _REGIME_COLS == ("mode", "n_iterations_run", "backend")


def test_csv_output_column_order(tmp_path, monkeypatch):
    """End-to-end: an artifact dir under a tmp repo root + run main → CSV
    columns are in the expected order: metrics, regime, dates.
    """
    from scripts.gbdt import regenerate_r_precision_at_k_csv as m

    art = tmp_path / "results" / "gbdt" / "experiments" / "fake_cell"
    pred = art / "predictions"
    pred.mkdir(parents=True, exist_ok=True)
    # Minimal valid test.csv (per compute_row's required cols).
    pd.DataFrame({
        "date":          ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        "ticker":        ["AAA", "BBB", "AAA", "BBB"],
        "p_calibrated":  [0.8, 0.2, 0.6, 0.3],
        "y_true":        [1, 0, 0, 1],
    }).to_csv(pred / "test.csv", index=False)
    (art / "spec.yaml").write_text(yaml.safe_dump({
        "backend": {
            "library": "xgboost",
            "fs_hp_loop": {"callback_mode": "default", "max_iterations": 3},
        },
    }))
    (art / "iterations.jsonl").write_text(
        json.dumps({"iter": 0}) + "\n" + json.dumps({"iter": 1}) + "\n"
    )

    out_csv = tmp_path / "out.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "regen",
            "--repo-root", str(tmp_path),
            "--workspace-root", str(tmp_path / "nonexistent"),
            "--out", str(out_csv),
        ],
    )
    rc = m.main()
    assert rc == 0
    df = pd.read_csv(out_csv)
    expected_cols = [
        "experiment", "rows", "Q_days", "base_rate", "AUC",
        "R_precision_at_1", "R_precision_at_3", "R_precision_at_5",
        "R_precision_at_10", "R_precision_at_20",
        "mode", "n_iterations_run", "backend",
        "train_start", "train_end", "val_start", "val_end",
        "eval_start", "eval_end", "test_start", "test_end",
    ]
    assert list(df.columns) == expected_cols
    row = df.iloc[0]
    assert row["mode"] == "sweep"
    assert int(row["n_iterations_run"]) == 2
    assert row["backend"] == "xgboost"
