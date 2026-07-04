#!/usr/bin/env python
"""Render the F18 horizon×target sweep as a (threshold × horizon) delta heatmap.

Four panels — fund − base delta for AUC, test Brier, R-Precision@1, R-Precision@10
— laid out threshold (rows) × horizon (cols). Colormap is oriented so GREEN =
fundamentals help (higher AUC / R-p; lower Brier), RED = hurt; missing lattice
combos are gray. Cells annotated with the raw delta. Reads
results/gbdt/data/_274_fund_sweep_data.json; writes PNG+SVG to
results/gbdt/_274_fund_sweep/.

Usage: uv run python -m scripts.gbdt.plot_fund_sweep_heatmap
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "gbdt" / "data" / "_274_fund_sweep_data.json"
OUT = ROOT / "results" / "gbdt" / "_274_fund_sweep"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [10, 20, 40, 50]
HORIZONS = [5, 10, 25, 50, 100, 200]

PANELS = [
    ("d_auc", "ΔAUC", False, 0.03),           # higher = help
    ("d_test_brier", "Δtest Brier", True, 0.003),  # lower = help (invert)
    ("_rp1", "ΔR-Precision@1", False, 0.30),
    ("_rp10", "ΔR-Precision@10", False, 0.09),
]


def val(cell, key):
    if key == "_rp1":
        return cell["d_rpk"]["1"]
    if key == "_rp10":
        return cell["d_rpk"]["10"]
    return cell[key]


def main() -> None:
    d = json.load(open(DATA))
    by_th = {(c["threshold_pct"], c["horizon_days"]): c for c in d["cells"]}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "F18 fundamentals: fund − base delta across the sp500 horizon×target lattice\n"
        f"date_aligned window (test {d['cells'][0]['test_window'][0]} → "
        f"{d['cells'][0]['test_window'][1]}, Q=100 days) · matched single-fit HP · "
        "13-col F18 · green = fundamentals help",
        fontsize=11, y=0.98,
    )
    for ax, (key, title, invert, span) in zip(axes.flat, PANELS):
        M = np.full((len(THRESHOLDS), len(HORIZONS)), np.nan)
        for i, th in enumerate(THRESHOLDS):
            for j, h in enumerate(HORIZONS):
                c = by_th.get((th, h))
                if c is not None:
                    M[i, j] = val(c, key)
        # orient so green=help: for Brier (lower=better) plot the negated delta
        Mc = -M if invert else M
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad("#dddddd")
        norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
        ax.imshow(Mc, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(HORIZONS)), [f"{h}d" for h in HORIZONS])
        ax.set_yticks(range(len(THRESHOLDS)), [f"+{t}%" for t in THRESHOLDS])
        ax.set_xlabel("horizon"); ax.set_ylabel("target")
        ax.set_title(title, fontsize=10)
        for i in range(len(THRESHOLDS)):
            for j in range(len(HORIZONS)):
                if np.isnan(M[i, j]):
                    continue
                raw = M[i, j]
                txt = f"{raw:+.4f}" if key == "d_test_brier" else f"{raw:+.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"_274_fund_sweep_heatmap.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {OUT}/_274_fund_sweep_heatmap.png (+svg)")


if __name__ == "__main__":
    main()
