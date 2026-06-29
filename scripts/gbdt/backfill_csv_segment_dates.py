"""V1.4 P3 — backfill 8 calendar-date columns onto r_precision_at_k.csv.

For each existing CSV row, find the matching artifact dir (searching the
main checkout AND every active git worktree, per V1.4 plan §5.3 / D5).
When the artifact is found, compute calendar-UNION dates per segment
(MIN of starts, MAX of ends across the tickers in the segment) from the
predictions CSVs. Write the 8 columns; leave them empty for cells we
can't backfill, and log those to ``results/gbdt/data/v1.4_backfill_log.json``.

Usage:
    uv run python -m scripts.gbdt.backfill_csv_segment_dates
    uv run python -m scripts.gbdt.backfill_csv_segment_dates --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


_DATE_COLS = (
    "train_start", "train_end",
    "val_start",   "val_end",
    "eval_start",  "eval_end",
    "test_start",  "test_end",
)

_SEGMENTS = ("train", "val", "eval", "test")


def _list_worktrees(repo_root: Path) -> list[Path]:
    """Return absolute Paths of every git worktree visible from ``repo_root``.

    Includes the main checkout. Tolerates a non-git ``repo_root`` (treat as
    a single-root scan).
    """
    try:
        out = subprocess.check_output(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(repo_root), text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [repo_root]
    roots: list[Path] = []
    for line in out.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line[len("worktree "):]).resolve())
    return roots or [repo_root]


def _find_artifact_dir(cell_id: str, roots: list[Path]) -> Path | None:
    """Search roots for ``results/gbdt/experiments/<cell_id>`` and prefer
    the first hit that carries usable date evidence.

    Many cells exist in multiple worktrees as "shadow" directories
    (metrics.json + spec.yaml + report.md only, no predictions). The
    actual data-bearing dir lives in whichever worktree the sweep
    actually ran in. We prefer dirs with EITHER a populated
    metrics.json::segment_dates OR a non-empty predictions/ subdir;
    fall back to a shadow dir as a last resort so we can still log it.
    """
    shadow: Path | None = None
    for root in roots:
        cand = root / "results" / "gbdt" / "experiments" / cell_id
        try:
            if not cand.is_dir():
                continue
        except OSError:
            continue
        # Strong signal — metrics.json carries segment_dates (V1.4 P2+).
        if _segment_dates_from_metrics(cand) is not None:
            return cand
        # Strong signal — predictions dir with at least one CSV.
        pred_dir = cand / "predictions"
        try:
            if pred_dir.is_dir() and any(pred_dir.glob("*.csv")):
                return cand
        except OSError:
            pass
        # Otherwise remember the first shadow we saw and keep looking.
        if shadow is None:
            shadow = cand
    return shadow


# Dense-region date bounds — keep in sync with the twin in
# regenerate_r_precision_at_k_csv.py (identical logic).
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


def _segment_dates_from_predictions(art_dir: Path) -> dict[str, dict[str, str | None]]:
    """Compute the per-segment DENSE date bounds from the predictions CSVs.

    Returns the canonical 4-segment × {start, end} dict (ISO strings).
    Empty / missing segment files yield ``{"start": None, "end": None}``.
    """
    out: dict[str, dict[str, str | None]] = {}
    pred_dir = art_dir / "predictions"
    for seg in _SEGMENTS:
        path = pred_dir / f"{seg}.csv"
        if not path.is_file():
            out[seg] = {"start": None, "end": None}
            continue
        try:
            # Only need the date column — saves reading the whole 100k-row CSV.
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


def _segment_dates_from_metrics(art_dir: Path) -> dict[str, dict[str, str | None]] | None:
    """V1.4 P2+ artifacts: metrics.json already carries segment_dates.

    Prefer this over re-deriving from predictions CSVs (cheaper + canonical
    for date_aligned cells).
    """
    metrics_path = art_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        m = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    sd = m.get("segment_dates")
    if not isinstance(sd, dict):
        return None
    # Validate shape — bail out on partial structures.
    for seg in _SEGMENTS:
        if not isinstance(sd.get(seg), dict):
            return None
        if "start" not in sd[seg] or "end" not in sd[seg]:
            return None
    return sd


def _row_to_columns(sd: dict[str, dict[str, str | None]]) -> dict[str, str]:
    """Map a segment_dates dict to the 8 flat CSV columns. None → ''."""
    flat: dict[str, str] = {}
    for seg in _SEGMENTS:
        flat[f"{seg}_start"] = sd[seg]["start"] or ""
        flat[f"{seg}_end"]   = sd[seg]["end"]   or ""
    return flat


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    default_repo = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo,
        help="Repository root (default: derived from this script's path)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="r_precision_at_k.csv path (default: <repo>/results/gbdt/data/r_precision_at_k.csv)",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Backfill log path (default: <repo>/results/gbdt/data/v1.4_backfill_log.json)",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Compute + print; don't write the CSV or log.")
    args = p.parse_args(argv)

    csv_path = args.csv or (
        args.repo_root / "results" / "gbdt" / "data" / "r_precision_at_k.csv"
    )
    log_path = args.log or (
        args.repo_root / "results" / "gbdt" / "data" / "v1.4_backfill_log.json"
    )

    df = pd.read_csv(csv_path)
    # Add the 8 new columns (empty string defaults) if absent.
    for col in _DATE_COLS:
        if col not in df.columns:
            df[col] = ""

    roots = _list_worktrees(args.repo_root)
    print(f"Backfill: scanning {len(roots)} worktree root(s) for artifacts:",
          file=sys.stderr)
    for r in roots:
        print(f"  - {r}", file=sys.stderr)

    log_entries: list[dict] = []
    filled, skipped = 0, 0
    for i, row in df.iterrows():
        cell_id = str(row["experiment"])
        art_dir = _find_artifact_dir(cell_id, roots)
        if art_dir is None:
            skipped += 1
            log_entries.append({
                "cell": cell_id,
                "reason": "no artifact dir in any worktree root",
            })
            continue
        # Prefer metrics.json::segment_dates (V1.4 P2+); fall back to
        # computing the calendar UNION from predictions/*.csv (pre-V1.4).
        sd = _segment_dates_from_metrics(art_dir)
        source = "metrics.json"
        if sd is None:
            sd = _segment_dates_from_predictions(art_dir)
            source = "predictions"
        flat = _row_to_columns(sd)
        for col, val in flat.items():
            df.at[i, col] = val
        # Only count "filled" when at least one segment got a date — otherwise
        # the artifact dir exists but has no usable predictions either.
        any_filled = any(flat[c] for c in _DATE_COLS)
        if any_filled:
            filled += 1
        else:
            skipped += 1
            log_entries.append({
                "cell": cell_id,
                "reason": "artifact dir found but no per-segment dates extractable",
                "artifact_dir": str(art_dir),
                "source": source,
            })

    print(f"Backfill: filled {filled} / {len(df)} cells "
          f"({skipped} skipped / logged)", file=sys.stderr)

    if args.dry_run:
        print(df[["experiment"] + list(_DATE_COLS)].to_string(), file=sys.stderr)
        return 0

    df.to_csv(csv_path, index=False, float_format="%.6f")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "skipped_cells": log_entries,
        "n_filled": filled,
        "n_skipped": skipped,
        "n_total": len(df),
    }, indent=2))
    print(f"Wrote: {csv_path}", file=sys.stderr)
    print(f"Wrote: {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
