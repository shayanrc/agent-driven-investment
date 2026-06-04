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

Usage:
    uv run python -m scripts.gbdt.regenerate_r_precision_at_k_csv
    uv run python -m scripts.gbdt.regenerate_r_precision_at_k_csv --out custom.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def _segment_dates_for_artifact(art_dir: Path) -> dict[str, dict[str, str | None]]:
    """V1.4: prefer metrics.json::segment_dates; fall back to calendar UNION
    across predictions/{train,val,eval,test}.csv (MIN start, MAX end).
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
        out[seg] = {
            "start": dts.min().date().isoformat(),
            "end":   dts.max().date().isoformat(),
        }
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
        default=Path("/mnt/122CEE982CEE765F/Workspace"),
        help="Directory containing sibling wt-* worktrees to also scan",
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

    df_out = pd.DataFrame(rows)
    df_out = df_out[
        ["experiment", "rows", "Q_days", "base_rate", "AUC"]
        + [f"R_precision_at_{k}" for k in KS]
        + list(_DATE_COLS)
    ].sort_values("AUC", ascending=False, na_position="last")

    out = args.out or (args.repo_root / "results" / "gbdt" / "data" / "r_precision_at_k.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False, float_format="%.6f")
    print(f"Wrote {out}: {len(df_out)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
