"""Spearman correlation heatmap for the nifty50 H=25 key features.

Covers the 17 constrained vol-estimator features (iter 5) + the index-regime
features they interact with + a few non-monotone references. In-sample rows
(test window excluded), so it matches the monotonicity/constraint analysis.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbdt import data as gbdt_data
from gbdt import features as gbdt_features

UNIVERSE = "nifty50"
TEST_TAIL_ROWS = 100

# Ordered by family so correlation blocks are visually coherent.
COLS = (
    [f"garman_klass_{w}" for w in (5, 10, 20, 50, 100, 200)]
    + [f"parkinson_{w}" for w in (5, 10, 20, 50, 100, 200)]
    + [f"realized_vol_{w}" for w in (10, 20, 50, 100, 200)]
    # --- index-regime interaction partners + non-monotone references ---
    + ["vol_of_vol_200", "index_vol_20", "index_vol_50", "index_vol_200",
       "index_return_50", "index_return_100", "index_return_200",
       "index_drawdown_100", "beta_50"]
)
# 17 constrained features are the first block; mark the boundary for the figure.
N_CONSTRAINED = 17


def main() -> None:
    panel_obj = gbdt_data.load_panel(UNIVERSE, min_rows=1600)
    X = gbdt_features.build_feature_matrix(
        panel_obj.panel, panel_obj.index_series,
        annualization=panel_obj.annualization_factor,
    ).dropna(axis=1, how="all")

    # in-sample: drop each ticker's trailing TEST_TAIL_ROWS
    keep = []
    for _, g in X.groupby(level="ticker"):
        keep.append(g.iloc[:-TEST_TAIL_ROWS] if len(g) > TEST_TAIL_ROWS else g)
    insample = pd.concat(keep)

    cols = [c for c in COLS if c in insample.columns]
    sub = insample[cols]
    corr = sub.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    # annotate cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.0, color="black" if abs(v) < 0.6 else "white")
    # boundary line separating the 17 constrained vol estimators from the rest
    b = N_CONSTRAINED - 0.5
    ax.axhline(b, color="lime", lw=2)
    ax.axvline(b, color="lime", lw=2)
    ax.set_title("nifty50 H=25 — Spearman correlation (in-sample)\n"
                 "green line: 17 constrained vol estimators (iter 5) | rest = "
                 "index-regime + non-monotone refs", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    fig.tight_layout()
    out = "results/gbdt/experiments/nifty50_feature_corr_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)

    # quick numeric summary: mean |corr| within the constrained block
    block = corr.iloc[:N_CONSTRAINED, :N_CONSTRAINED].values
    off = block[~np.eye(N_CONSTRAINED, dtype=bool)]
    print(f"constrained-block mean |off-diagonal corr| = {np.mean(np.abs(off)):.3f}")
    print(f"constrained-block min corr = {off.min():.3f}  max = {off.max():.3f}")


if __name__ == "__main__":
    main()
