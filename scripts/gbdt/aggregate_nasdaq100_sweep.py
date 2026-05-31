"""Aggregate the nasdaq100 sweep (#192) into a single JSON data file.

Loads each completed cell's ``metrics.json`` + computes per-day weighted
R-precision on its ``predictions/eval.csv`` and ``predictions/test.csv`` via
:func:`gbdt.diagnose_core.per_day_r_precision` (the same code path the
ad-hoc ``scripts/gbdt/compute_r_precision.py`` re-exports — so this output is
methodology-identical to the per-cell CLI invocation).

Mirror of ``aggregate_russell1000_sweep.py`` (#188); only the universe
prefix, output filename, sweep/task labels, and branch label differ.

Output:
  ``results/gbdt/data/_192_nasdaq100_sweep_results_data.json``
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from gbdt.diagnose_core import per_day_r_precision

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "results/gbdt/experiments"
OUTPUT_FILE = REPO_ROOT / "results/gbdt/data/_192_nasdaq100_sweep_results_data.json"


def _verdict(auc: float | None, lift: float | None) -> str:
    if auc is None or lift is None:
        return "no-test-window"
    if 0.45 <= auc <= 0.55:
        if lift < 1.2:
            return "null"
        if lift > 1.5:
            return "hidden-top-tail"
        return "ambiguous"
    if auc > 0.55:
        return "discriminating"
    return "anti-predictive"  # AUC < 0.45


def _rprec(preds_csv: Path) -> dict | None:
    if not preds_csv.exists():
        return None
    preds = pd.read_csv(preds_csv, parse_dates=["date"])
    n_rows = len(preds)
    if n_rows == 0:
        return None
    return per_day_r_precision(preds)


def _parse_spec(name: str) -> dict:
    # e.g. nasdaq100_up_10pct_25d_dd5pct
    parts = name.split("_")
    threshold_pct = int(parts[2].rstrip("pct"))
    horizon_days = int(parts[3].rstrip("d"))
    dd_pct = int(parts[4][len("dd") :].rstrip("pct"))
    return {
        "universe": parts[0],
        "direction": parts[1],
        "threshold_pct": threshold_pct,
        "horizon_days": horizon_days,
        "max_drawdown_pct": dd_pct,
    }


def main() -> None:
    cells_meta: list[dict] = []
    wall_total = 0.0
    for cell_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("nasdaq100_"):
            continue
        # Filter out non-sweep variants by suffix.
        if cell_dir.name.endswith(("_xgb_repl", "_xgb_manual", "_xgb_manual_fsloop", "_acceptance")):
            continue
        m_path = cell_dir / "metrics.json"
        if not m_path.exists():
            print(f"[skip] {cell_dir.name} (no metrics.json)")
            continue
        m = json.loads(m_path.read_text())

        head_eval = m.get("headline_eval", {}) or {}
        head_test = m.get("headline_test", {}) or {}
        loop = m.get("loop", {}) or {}
        calib = m.get("calibration", {}) or {}

        rprec_eval = _rprec(cell_dir / "predictions/eval.csv")
        rprec_test = _rprec(cell_dir / "predictions/test.csv")

        # base rates: prefer weighted from headline blocks; fall back to rprec output
        base_eval = head_eval.get("weighted_prevalence") or (
            rprec_eval.get("base_rate_weighted") if rprec_eval else None
        )
        base_test = head_test.get("weighted_prevalence") if head_test else None
        if base_test is None and rprec_test:
            base_test = rprec_test.get("base_rate_weighted")

        rprec_e_w = rprec_eval.get("r_precision_weighted") if rprec_eval else None
        rprec_t_w = rprec_test.get("r_precision_weighted") if rprec_test else None

        lift_e = (rprec_e_w / base_eval) if (rprec_e_w and base_eval) else None
        lift_t = (rprec_t_w / base_test) if (rprec_t_w and base_test) else None

        auc_e = head_eval.get("roc_auc")
        auc_t = head_test.get("roc_auc") if head_test else None

        wall = float(m.get("wall_time_total_sec") or 0)
        wall_total += wall

        # n_features_final from features.yaml (final-iter selection)
        feats_yaml = cell_dir / "features.yaml"
        n_feat_final = None
        if feats_yaml.exists():
            import yaml

            fy = yaml.safe_load(feats_yaml.read_text()) or {}
            if isinstance(fy, dict):
                if "features" in fy and isinstance(fy["features"], list):
                    n_feat_final = len(fy["features"])
                elif "selected" in fy and isinstance(fy["selected"], list):
                    n_feat_final = len(fy["selected"])
                elif "kept" in fy and isinstance(fy["kept"], list):
                    n_feat_final = len(fy["kept"])

        cell = {
            "name": cell_dir.name,
            "spec": _parse_spec(cell_dir.name),
            "base_rate_eval_weighted": base_eval,
            "base_rate_test_weighted": base_test,
            "auc_eval": auc_e,
            "auc_test": auc_t,
            "rprec_eval_weighted": rprec_e_w,
            "rprec_test_weighted": rprec_t_w,
            "rprec_eval_baseline_weighted": (
                rprec_eval.get("base_rate_weighted") if rprec_eval else None
            ),
            "rprec_test_baseline_weighted": (
                rprec_test.get("base_rate_weighted") if rprec_test else None
            ),
            "lift_eval": lift_e,
            "lift_test": lift_t,
            "brier_eval": head_eval.get("brier"),
            "brier_eval_baseline": head_eval.get("brier_baseline_baserate"),
            "brier_test": head_test.get("brier") if head_test else None,
            "brier_test_baseline": head_test.get("brier_baseline_baserate")
            if head_test
            else None,
            "n_iterations": loop.get("n_iterations_run"),
            "best_iteration": loop.get("best_iteration"),
            "inner_stop_signal": loop.get("inner_stop_signal"),
            "n_features_final": n_feat_final,
            "calibration_decision": calib.get("decision"),
            "spiegelhalter_z": calib.get("spiegelhalter_z"),
            "spiegelhalter_p": calib.get("spiegelhalter_p"),
            "n_rows_eval": head_eval.get("n_rows"),
            "n_rows_test": head_test.get("n_rows") if head_test else None,
            "ess_kish_eval": head_eval.get("effective_sample_size_kish"),
            "ess_kish_test": head_test.get("effective_sample_size_kish")
            if head_test
            else None,
            "n_tickers_used": (m.get("data") or {}).get("n_tickers_used"),
            "n_tickers_in_universe": (m.get("data") or {}).get("n_tickers_in_universe"),
            "wall_time_sec": wall,
        }
        cell["verdict_basis"] = "test" if auc_t is not None else "eval"
        if auc_t is not None and lift_t is not None:
            cell["verdict"] = _verdict(auc_t, lift_t)
        else:
            cell["verdict"] = _verdict(auc_e, lift_e)
        cells_meta.append(cell)

    out = {
        "sweep": "nasdaq100",
        "task": "_192",
        "branch": "gbdt-nasdaq100-sweep-results",
        "ran_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_cells_total": len(cells_meta),
        "wall_time_cells_sum_sec": wall_total,
        "wall_time_cells_sum_hr": wall_total / 3600.0,
        "cells": cells_meta,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"[ok] wrote {OUTPUT_FILE} ({len(cells_meta)} cells, sum wall={wall_total/3600:.2f}h)")


if __name__ == "__main__":
    main()
