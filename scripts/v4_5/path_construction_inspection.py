"""V4.5.6 — A2.1 path-construction inspection at Mode-3 regression anchors.

V4.5.2 Mode 3 identified 8 anchors where A2.1 has diffuse top-K but still
regresses on CRPS — suggesting the regression is in *path construction*,
not *matcher selection*. A2.1 disables conditional block sampling (the
per-path re-match each block), so block-0 distances are reused for all
blocks.

This script inspects the existing forecasts at Mode-3 anchors:
- Per-day path dispersion (std across n_paths). Does it grow with horizon?
- Per-day path mean drift (vs realized).
- Compare to v2.4 at the same anchors.

Read-only on existing run dirs.

Outputs: results/analog_mc/data/v4_5_6_path_construction.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
A2_RUN = REPO / "runs/analog_mc/20260521T061730Z"
V24_RUN = REPO / "runs/analog_mc/20260520T045525Z"
OUT = REPO / "results/analog_mc/data/v4_5_6_path_construction.json"

# Mode-3 regression anchors from V4.5.2.
MODE3_ANCHORS = {
    "2012-03-14": (6607, 46),
    "2022-03-01": (9115, 67),
    "2026-02-19": (10111, 75),
    "2010-11-10": (6270, 43),
    "2001-04-04": (3856, 23),
    "1991-03-26": (1321, 2),
    "2017-06-01": (7919, 57),
    # WINS for comparison
    "2010-04-23": (6130, 42),
    "2001-10-02": (3977, 24),
    "2020-03-16": (8621, 63),
}
# v2.4 fold indices may differ from A2.1 (depends on run config). Compute on-the-fly.


def load_fold_summaries(run_dir: Path) -> list[dict]:
    folds_dir = run_dir / "folds"
    out = []
    for d in sorted(folds_dir.iterdir(), key=lambda p: int(p.name)):
        out.append(json.loads((d / "summary.json").read_text()))
    return out


def fold_for_origin(folds: list[dict], origin_idx: int) -> dict | None:
    for f in folds:
        if f["test_start"] <= origin_idx <= f["test_end"]:
            return f
    return None


def extract_anchor_paths(run_dir: Path, origin_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the anchor in the fold's forecasts.npz and extract its paths + realized.

    forecasts.npz has shape (n_test_origins, n_paths, horizon). We need to find
    which test_origin index corresponds to origin_idx.
    """
    folds = load_fold_summaries(run_dir)
    fold = fold_for_origin(folds, origin_idx)
    if fold is None:
        return None
    fold_dir = run_dir / "folds" / str(fold["fold_index"])
    data = np.load(fold_dir / "forecasts.npz")
    origin_ids = data["origin_idx"]
    # Find the row.
    match = np.flatnonzero(origin_ids == origin_idx)
    if match.size == 0:
        return None
    row = int(match[0])
    paths = data["paths"][row]      # (n_paths, H)
    realized = data["realized"][row]  # (H,)
    return paths, realized


def main() -> None:
    rows = []
    for anchor_date, (origin_idx, _) in MODE3_ANCHORS.items():
        a2_out = extract_anchor_paths(A2_RUN, origin_idx)
        v24_out = extract_anchor_paths(V24_RUN, origin_idx)
        if a2_out is None or v24_out is None:
            print(f"  {anchor_date} MISSING in run dirs; skip")
            continue
        a2_paths, a2_realized = a2_out
        v24_paths, v24_realized = v24_out
        # Sanity.
        assert np.allclose(a2_realized, v24_realized, atol=1e-8), f"realized mismatch at {anchor_date}"

        # Per-day mean and per-day cross-path std of CUMULATIVE log return.
        # Brownian baseline: cumulative-log std at day t ~ σ × √t.
        a2_cum = np.cumsum(a2_paths, axis=1)        # (n_paths, H)
        v24_cum = np.cumsum(v24_paths, axis=1)
        a2_cum_std = a2_cum.std(axis=0)              # (H,) cross-path std per day
        v24_cum_std = v24_cum.std(axis=0)
        a2_cum_mean_logret = a2_cum.mean(axis=0)
        v24_cum_mean_logret = v24_cum.mean(axis=0)
        a2_cum_mean = np.expm1(a2_cum_mean_logret) * 100.0
        v24_cum_mean = np.expm1(v24_cum_mean_logret) * 100.0
        realized_cum_pct = np.expm1(np.cumsum(a2_realized)) * 100.0

        # √t-growth ratio of cumulative-log std: std[59]/std[0] expected ≈ √60
        # for iid returns. A ratio much smaller means the paths converge over
        # time (tight scenarios); a ratio much larger means amplifying noise.
        a2_std_growth = float(a2_cum_std[-1] / a2_cum_std[0]) if a2_cum_std[0] > 0 else float("nan")
        v24_std_growth = float(v24_cum_std[-1] / v24_cum_std[0]) if v24_cum_std[0] > 0 else float("nan")
        # Also report terminal cum-std as percent (more intuitive than log).
        a2_terminal_cum_std_pct = float(a2_cum_std[-1] * 100.0)  # approx; small-r linear
        v24_terminal_cum_std_pct = float(v24_cum_std[-1] * 100.0)

        rows.append({
            "anchor_date": anchor_date,
            "origin_idx": origin_idx,
            "realized_60d_pct": float(realized_cum_pct[-1]),
            "a2_terminal_mean_pct": float(a2_cum_mean[-1]),
            "v24_terminal_mean_pct": float(v24_cum_mean[-1]),
            "cum_std_growth_sqrt_t_expected": float(np.sqrt(60)),
            "a2_cum_std_growth_actual": a2_std_growth,
            "v24_cum_std_growth_actual": v24_std_growth,
            "a2_terminal_cum_std_pct": a2_terminal_cum_std_pct,
            "v24_terminal_cum_std_pct": v24_terminal_cum_std_pct,
        })
        print(f"  {anchor_date:<12} real={realized_cum_pct[-1]:+6.1f}% "
              f"A2.mean={a2_cum_mean[-1]:+6.1f}% v24.mean={v24_cum_mean[-1]:+6.1f}% "
              f"A2.cum-σ-growth={a2_std_growth:.2f} v24.cum-σ-growth={v24_std_growth:.2f} "
              f"A2.term-σ={a2_terminal_cum_std_pct:.1f}% v24.term-σ={v24_terminal_cum_std_pct:.1f}% "
              f"(√60={np.sqrt(60):.2f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "method": {
            "description": "V4.5.6 path-construction inspection — per-anchor per-day "
                          "mean/std diagnostics from existing forecasts.npz.",
            "expected_std_growth_brownian": float(np.sqrt(60)),
        },
        "anchors": rows,
    }, indent=2))
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
