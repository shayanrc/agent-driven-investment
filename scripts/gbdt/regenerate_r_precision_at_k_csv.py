"""
Rebuild the canonical R-Precision@K registry CSV at
``results/gbdt/data/r_precision_at_k.csv``.

Scans every ``results/gbdt/experiments/*/predictions/test.csv`` under both the
current checkout and sibling worktrees (``/mnt/.../Workspace/wt-*/``), picks the
freshest test.csv per experiment name (by mtime), computes:

  base_rate              = df['y_true'].mean()
  AUC                    = sklearn.metrics.roc_auc_score
  R-Precision@K          = (1/Q) * sum r_q / min(K, R_q)   for K in {1,3,5,10,20}
  Q_days                 = number of days with R_q > 0

and writes the CSV sorted by AUC descending. See
``.claude/memories/project-r-precision-methodology.md`` for the definition.

Each row also carries 3 training-regime descriptors (added 2026-06-05):

  mode                   = sweep | default_full_loop | agent_file_protocol | agentloop_legacy
  n_iterations_run       = realized iteration count from iterations.jsonl
                           (or sidecar JSON for pruned _agentloop* cells)
  backend                = xgboost | catboost (or blank when untrackable)

Usage:
    uv run python -m scripts.gbdt.regenerate_r_precision_at_k_csv
    uv run python -m scripts.gbdt.regenerate_r_precision_at_k_csv --out custom.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

KS = (1, 3, 5, 10, 20)

# V1.4: 8 calendar-date columns appended to every row.
_DATE_COLS = (
    "train_start", "train_end",
    "val_start",   "val_end",
    "eval_start",  "eval_end",
    "test_start",  "test_end",
)
_SEGMENTS = ("train", "val", "eval", "test")

# 2026-06-05: training-regime descriptor columns. Inserted between the metric
# block (ends at R_precision_at_20) and the 8 calendar-date columns so they sit
# next to the metrics they describe while preserving the V1.4 date-cols-last
# convention.
_REGIME_COLS = ("mode", "n_iterations_run", "backend")

# Static fallback for pruned _agentloop* cells whose artifact dirs have been
# deleted from results/gbdt/experiments/. Values keyed to the canonical sidecar
# JSON each cell was reported in. See docs/gbdt/_185, _194, _195, _222, _223
# for the chain of custody.
#
# Per CLAUDE.md, _agentloop* runs were full FS+HP agent-driven loops (the
# pre-V1.3 generation). Backend convention: post-#185 the agentloop work was
# XGBoost (#149 invB onward); V1.3 Option A revalidation was XGBoost cell-5.
_PRUNED_AGENTLOOP_FALLBACK: dict[str, dict[str, str]] = {
    # _149 invB-derived; XGBoost agent-file-protocol loop on cell-5.
    "nasdaq100_up_10pct_50d_dd5pct_agentloop": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "",
        "backend": "xgboost",
    },
    # _222 V1.3 Option A validation; per_iter_trajectory len=11.
    "nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "11",
        "backend": "xgboost",
    },
    # _223 V1.3 Option A revalidation; loop_trajectory_calibrated_r_p_at_1 len=12.
    "nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "12",
        "backend": "xgboost",
    },
    # _195 cell-4 nasdaq100_up_40pct_50d_dd20pct variants.
    "nasdaq100_up_40pct_50d_dd20pct_agentloop": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "2",
        "backend": "xgboost",
    },
    "nasdaq100_up_40pct_50d_dd20pct_agentloop_colsample": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "3",
        "backend": "xgboost",
    },
    "nasdaq100_up_40pct_50d_dd20pct_agentloop_gamma": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "5",
        "backend": "xgboost",
    },
    "nasdaq100_up_40pct_50d_dd20pct_agentloop_mix": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "2",
        "backend": "xgboost",
    },
    "nasdaq100_up_40pct_50d_dd20pct_agentloop_mix_mcw3": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "2",
        "backend": "xgboost",
    },
    # _195 cell-3 sp500_up_20pct_25d_dd10pct variants.
    "sp500_up_20pct_25d_dd10pct_agentloop": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "2",
        "backend": "xgboost",
    },
    "sp500_up_20pct_25d_dd10pct_agentloop_mix": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "2",
        "backend": "xgboost",
    },
    # _195 cell-2 sp500_up_20pct_50d_dd10pct.
    "sp500_up_20pct_50d_dd10pct_agentloop": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "2",
        "backend": "xgboost",
    },
    # _195 cell-1 sp500_up_50pct_50d_dd25pct variants.
    "sp500_up_50pct_50d_dd25pct_agentloop": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "2",
        "backend": "xgboost",
    },
    "sp500_up_50pct_50d_dd25pct_agentloop_mix": {
        "mode": "agentloop_legacy",
        "n_iterations_run": "2",
        "backend": "xgboost",
    },
}


def _classify_mode_from_spec(
    callback_mode: str | None, max_iterations: int | None,
) -> str:
    """Apply the primary mode classifier — see module docstring + the regime
    columns introduction commit. ``callback_mode`` is None when the spec
    doesn't carry it (older spec files); treat as "default" (the runner's
    default). ``max_iterations`` is None when missing; treat as default_full_loop
    since we can't prove sweep without it.
    """
    cb = (callback_mode or "default").strip()
    if cb == "agent_file_protocol":
        return "agent_file_protocol"
    # default mode (or any non-agent value): split on max_iterations.
    if max_iterations is None:
        return "default_full_loop"
    if max_iterations <= 3:
        return "sweep"
    # max_iterations >= 4 (covers the "4-7" gap AND >= 8) → default_full_loop.
    return "default_full_loop"


def _classify_mode_from_name(name: str) -> str:
    """Fallback mode classifier from cell-name suffix. Only used when the
    artifact dir is gone. Never overrides a primary (artifact-dir) result.
    """
    if "_agentloop" in name:
        return "agentloop_legacy"
    if name.endswith("_aligned"):
        return "sweep"
    if name.endswith("_pilot"):
        return "default_full_loop"
    # _b_acceptance variants (P9): check these BEFORE the generic
    # `_acceptance` suffix since `_b_acceptance` is more specific. The
    # agent-mode variant ends in `_b_acceptance_agent`; the default-mode
    # variant ends in `_b_acceptance` alone.
    if name.endswith("_b_acceptance_agent"):
        return "agent_file_protocol"
    if name.endswith("_b_acceptance"):
        return "default_full_loop"
    if name.endswith("_xgb_acceptance") or name.endswith("_acceptance"):
        return "agent_file_protocol"
    if name.endswith("_phase8") or name.endswith("_catboost_phase8"):
        return "default_full_loop"
    return "sweep"


def _backend_from_name(name: str) -> str:
    """Fallback backend classifier from cell-name pattern."""
    if "_xgb_" in name or "_xgboost_" in name:
        return "xgboost"
    if "_catboost_" in name:
        return "catboost"
    if "_agentloop_v1.3" in name:
        return "xgboost"
    return ""


def _regime_for_artifact(art_dir: Path) -> dict[str, str]:
    """Read mode + n_iter + backend from spec.yaml + iterations.jsonl in
    the artifact dir. Returns empty strings for fields that can't be read.
    """
    out = {"mode": "", "n_iterations_run": "", "backend": ""}

    # spec.yaml: backend.library + backend.fs_hp_loop.{callback_mode,max_iterations}.
    spec_path = art_dir / "spec.yaml"
    callback_mode: str | None = None
    max_iterations: int | None = None
    if spec_path.is_file():
        try:
            spec = yaml.safe_load(spec_path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            spec = {}
        backend = spec.get("backend") or {}
        if isinstance(backend, dict):
            # Runner default when ``backend.library`` is omitted is catboost
            # (see src/gbdt/__main__.py: ``raw_library = ... or "catboost"``).
            # We materialize that default here so the CSV's ``backend`` column
            # is informative on every artifact-backed row.
            lib = backend.get("library")
            if isinstance(lib, str):
                out["backend"] = lib
            else:
                out["backend"] = "catboost"
            loop = backend.get("fs_hp_loop") or {}
            if isinstance(loop, dict):
                cb = loop.get("callback_mode")
                if isinstance(cb, str):
                    callback_mode = cb
                mi = loop.get("max_iterations")
                if isinstance(mi, (int, float)):
                    max_iterations = int(mi)
        out["mode"] = _classify_mode_from_spec(callback_mode, max_iterations)

    # iterations.jsonl: realized iter count = line count.
    iters_path = art_dir / "iterations.jsonl"
    if iters_path.is_file():
        try:
            with iters_path.open() as h:
                n_lines = sum(1 for _ in h)
            out["n_iterations_run"] = str(n_lines)
        except OSError:
            pass

    return out


def _regime_for_pruned(name: str) -> dict[str, str]:
    """Fallback regime descriptors for a pruned cell (no artifact dir).
    Static dispatch first; suffix-rule fallback for everything else.
    """
    if name in _PRUNED_AGENTLOOP_FALLBACK:
        return dict(_PRUNED_AGENTLOOP_FALLBACK[name])
    return {
        "mode": _classify_mode_from_name(name),
        "n_iterations_run": "",
        "backend": _backend_from_name(name),
    }


def find_test_csvs(roots: list[Path]) -> dict[str, Path]:
    """Return the freshest predictions/test.csv per experiment name across roots.

    Tolerates I/O errors on individual roots (stale/corrupted worktrees) by
    skipping the affected root with a warning.
    """
    freshest: dict[str, Path] = {}
    for root in roots:
        exp_root = root / "results" / "gbdt" / "experiments"
        try:
            if not exp_root.is_dir():
                continue
            csvs = list(exp_root.glob("*/predictions/test.csv"))
        except OSError as e:
            print(f"  SKIP root {root}: {e}", file=sys.stderr)
            continue
        for csv in csvs:
            try:
                mtime = csv.stat().st_mtime
            except OSError as e:
                print(f"  SKIP csv {csv}: {e}", file=sys.stderr)
                continue
            name = csv.parent.parent.name
            if name not in freshest or mtime > freshest[name].stat().st_mtime:
                freshest[name] = csv
    return freshest


# Dense-region date bounds — keep in sync with the twin in
# backfill_csv_segment_dates.py (identical logic).
DENSE_FRAC = 0.5  # a date is "dense" if it carries >= this fraction of the segment's PEAK daily row-count


def _dense_date_bounds(dts: pd.Series) -> tuple[str | None, str | None]:
    """(start, end) ISO dates of a segment's DENSE region, dropping sparse outliers.

    A trailing (row-based) split gives each ticker its own last-N-rows window, so a
    DELISTED ticker (whose history ends months before the cohort) lands segment rows
    dated well before the cohort's block — dragging a naive MIN(date) back (the
    nasdaq_40_50 ``test_start=2025-06-05`` artifact; true cohort start 2025-12-30). We
    count rows per calendar date and keep only dates carrying >= ``DENSE_FRAC`` × the
    PEAK daily count, then take MIN/MAX over those. Peak (not median) is the robust
    reference — strays sit far below it however many stray dates there are. No-op for
    clean segments; MAX is rarely affected (delisted tickers end early, not late).
    """
    daily = dts.dt.normalize().value_counts()
    if daily.empty:
        return None, None
    keep = daily[daily >= DENSE_FRAC * daily.max()].index
    if len(keep) == 0:
        keep = daily.index
    return min(keep).date().isoformat(), max(keep).date().isoformat()


def _segment_dates_for_artifact(art_dir: Path) -> dict[str, dict[str, str | None]]:
    """V1.4: prefer metrics.json::segment_dates; fall back to the per-segment DENSE
    date bounds across predictions/{train,val,eval,test}.csv (sparse delisted-ticker
    outliers dropped; see _dense_date_bounds).
    """
    metrics_path = art_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            m = json.loads(metrics_path.read_text())
            sd = m.get("segment_dates")
            if isinstance(sd, dict) and all(
                isinstance(sd.get(s), dict)
                and "start" in sd[s]
                and "end"   in sd[s]
                for s in _SEGMENTS
            ):
                return sd
        except (OSError, json.JSONDecodeError):
            pass
    # Calendar UNION fallback.
    out: dict[str, dict[str, str | None]] = {}
    pred_dir = art_dir / "predictions"
    for seg in _SEGMENTS:
        path = pred_dir / f"{seg}.csv"
        if not path.is_file():
            out[seg] = {"start": None, "end": None}
            continue
        try:
            df = pd.read_csv(path, usecols=["date"])
        except (OSError, ValueError):
            out[seg] = {"start": None, "end": None}
            continue
        if len(df) == 0:
            out[seg] = {"start": None, "end": None}
            continue
        dts = pd.to_datetime(df["date"], errors="coerce").dropna()
        if len(dts) == 0:
            out[seg] = {"start": None, "end": None}
            continue
        start, end = _dense_date_bounds(dts)
        out[seg] = {"start": start, "end": end}
    return out


def compute_row(name: str, path: Path) -> dict | None:
    df = pd.read_csv(path)
    if "y_pred" in df.columns and "p_calibrated" not in df.columns:
        df["p_calibrated"] = df["y_pred"]
    required = {"p_calibrated", "y_true", "date", "ticker"}
    if not required.issubset(df.columns) or len(df) == 0:
        return None
    out = {
        "experiment": name,
        "rows": len(df),
        "base_rate": float(df["y_true"].mean()),
    }
    try:
        out["AUC"] = float(roc_auc_score(df["y_true"], df["p_calibrated"]))
    except Exception:
        out["AUC"] = float("nan")
    # Tie-break: (p_calibrated desc, ticker asc) stable mergesort — matches
    # compute_r_precision.py + src/gbdt/topk_diagnostics.py + the methodology
    # memory's tie-break convention. Sorting by p_calibrated alone leaves order
    # of equal-p rows determined by row order in the CSV, which is data-dependent
    # and can shift R-Precision@1 by 3x on cells with many tied p values.
    by_day = [
        (d, g.sort_values(
            by=["p_calibrated", "ticker"],
            ascending=[False, True],
            kind="mergesort",
        ))
        for d, g in df.groupby("date")
    ]
    Q = None
    for K in KS:
        ratios = []
        for _d, g in by_day:
            R_q = int(g["y_true"].sum())
            if R_q == 0:
                continue
            r_q = int(g.head(K)["y_true"].sum())
            ratios.append(r_q / min(K, R_q))
        if Q is None:
            Q = len(ratios)
        out[f"R_precision_at_{K}"] = float(np.mean(ratios)) if ratios else float("nan")
    out["Q_days"] = Q
    # V1.4 (plan §5.1): 8 calendar-date columns. Artifact dir is the test
    # CSV's parent.parent. Empty string when neither metrics.json nor
    # predictions/*.csv carry usable dates.
    art_dir = path.parent.parent
    sd = _segment_dates_for_artifact(art_dir)
    for seg in _SEGMENTS:
        out[f"{seg}_start"] = sd[seg]["start"] or ""
        out[f"{seg}_end"]   = sd[seg]["end"]   or ""
    # 2026-06-05: training-regime descriptors. Primary classifier from
    # spec.yaml + iterations.jsonl in the artifact dir; never falls through
    # to the name-suffix fallback when the dir is present.
    regime = _regime_for_artifact(art_dir)
    for col in _REGIME_COLS:
        out[col] = regime[col]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    default_repo = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo,
        help="Repository root (default: derived from this script's path)",
    )
    p.add_argument(
        "--workspace-root",
        type=Path,
        # WORKSPACE_ROOT = parent dir where ``wt-*/`` worktrees live;
        # per-machine, see per-user memory ``scratch-cache-path``. Empty
        # default ("") fails the ``is_dir()`` check below and silently skips
        # wt-* siblings — so on a fresh machine with no env var set, the
        # script still works (just scans repo_root only). Pass --workspace-root
        # explicitly to override.
        default=Path(os.environ.get("WORKSPACE_ROOT", "")),
        help="Directory containing sibling wt-* worktrees to also scan "
             "(default: $WORKSPACE_ROOT env var, or empty = skip wt-* scan)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <repo_root>/results/gbdt/data/r_precision_at_k.csv)",
    )
    args = p.parse_args()

    roots = [args.repo_root]
    if args.workspace_root.is_dir():
        roots.extend(sorted(args.workspace_root.glob("wt-*")))

    freshest = find_test_csvs(roots)
    rows: list[dict] = []
    for name in sorted(freshest):
        try:
            row = compute_row(name, freshest[name])
        except Exception as e:
            print(f"  SKIP {name}: {e}", file=sys.stderr)
            continue
        if row is not None:
            rows.append(row)

    # Preserve pruned-cell rows from the existing CSV: any row whose
    # experiment name is NOT in the freshest-artifact-dir scan stays put.
    # We don't recompute metrics for those (no test.csv to compute against);
    # we only refresh the 3 training-regime descriptor columns via the
    # static fallback + suffix rules. Calendar-date columns persist as-is.
    out_path = args.out or (
        args.repo_root / "results" / "gbdt" / "data" / "r_precision_at_k.csv"
    )
    scanned_names = {r["experiment"] for r in rows}
    if out_path.is_file():
        try:
            existing = pd.read_csv(out_path, dtype=str)
        except (OSError, pd.errors.ParserError) as e:
            print(f"  WARN: could not read existing CSV for pruned-cell preservation: {e}", file=sys.stderr)
            existing = pd.DataFrame()
        for _, ex in existing.iterrows():
            name = ex.get("experiment")
            if not isinstance(name, str) or name in scanned_names:
                continue
            # Build a row dict preserving the existing values and adding/
            # refreshing the regime columns.
            preserved: dict = {}
            for col, val in ex.items():
                preserved[col] = val
            regime = _regime_for_pruned(name)
            for col in _REGIME_COLS:
                preserved[col] = regime[col]
            # Cast numeric columns back so the sort below works.
            for nc in ("rows", "Q_days", "base_rate", "AUC",
                       *(f"R_precision_at_{k}" for k in KS)):
                if nc in preserved and preserved[nc] not in (None, ""):
                    try:
                        preserved[nc] = float(preserved[nc])
                    except (TypeError, ValueError):
                        pass
            rows.append(preserved)

    df_out = pd.DataFrame(rows)
    df_out = df_out[
        ["experiment", "rows", "Q_days", "base_rate", "AUC"]
        + [f"R_precision_at_{k}" for k in KS]
        + list(_REGIME_COLS)
        + list(_DATE_COLS)
    ].sort_values("AUC", ascending=False, na_position="last")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Wrote {out_path}: {len(df_out)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
