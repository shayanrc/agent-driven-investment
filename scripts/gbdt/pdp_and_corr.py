"""Two figures for the monotone-constraint analysis (builds features once):

1. Spearman correlation heatmap of the key features (original ask).
2. 2D partial-dependence surfaces for a top vol x index interaction pair,
   BEFORE (unconstrained screening model) vs AFTER (iter5 constrained model).
   This shows the *functional form* — the regime-conditional sign-flip that the
   monotone constraint flattens, which the interaction-strength heatmap can't show.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from gbdt import data as gbdt_data
from gbdt import features as gbdt_features

UNIVERSE = "nifty50"
TEST_TAIL_ROWS = 100
BEFORE = "/mnt/122CEE982CEE765F/Workspace/wt-exp-nifty50-up10-25d/results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct/model.cbm"
AFTER = "results/gbdt/experiments/nifty50_manualloop_iter5/model.cbm"

# The PDP pair: stock vol (constrained +1) x index momentum (unconstrained).
F_VOL = "garman_klass_200"   # constrained monotone +1
F_IDX = "index_return_50"    # unconstrained; the regime axis

CORR_COLS = (
    [f"garman_klass_{w}" for w in (5, 10, 20, 50, 100, 200)]
    + [f"parkinson_{w}" for w in (5, 10, 20, 50, 100, 200)]
    + [f"realized_vol_{w}" for w in (10, 20, 50, 100, 200)]
    + ["vol_of_vol_200", "index_vol_20", "index_vol_50", "index_vol_200",
       "index_return_50", "index_return_100", "index_return_200",
       "index_drawdown_100", "beta_50"]
)
N_CONSTRAINED_CORR = 17
GRID = 12          # PDP grid resolution per axis
SUBSAMPLE = 4000   # rows to average the PDP over


def build_insample():
    po = gbdt_data.load_panel(UNIVERSE, min_rows=1600)
    X = gbdt_features.build_feature_matrix(
        po.panel, po.index_series, annualization=po.annualization_factor,
    ).dropna(axis=1, how="all")
    keep = []
    for _, g in X.groupby(level="ticker"):
        keep.append(g.iloc[:-TEST_TAIL_ROWS] if len(g) > TEST_TAIL_ROWS else g)
    return pd.concat(keep)


def corr_fig(ins: pd.DataFrame) -> None:
    cols = [c for c in CORR_COLS if c in ins.columns]
    corr = ins[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7); ax.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color="black" if abs(v) < 0.6 else "white")
    b = N_CONSTRAINED_CORR - 0.5
    ax.axhline(b, color="lime", lw=2); ax.axvline(b, color="lime", lw=2)
    ax.set_title("nifty50 H=25 — Spearman correlation (in-sample)\n"
                 "green line: 17 constrained vol estimators | rest = index-regime + refs", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig("results/gbdt/experiments/nifty50_feature_corr_heatmap.png", dpi=130, bbox_inches="tight")
    print("wrote corr heatmap")


def pdp_surface(model, Xsub: pd.DataFrame, names, vol_grid, idx_grid):
    """Average predicted P(pos) over Xsub while sweeping the vol x idx grid."""
    iv, ii = names.index(F_VOL), names.index(F_IDX)
    base = Xsub[names].values.copy()
    surf = np.zeros((len(idx_grid), len(vol_grid)))
    for a, gi in enumerate(idx_grid):
        for b, gv in enumerate(vol_grid):
            g = base.copy()
            g[:, iv] = gv
            g[:, ii] = gi
            surf[a, b] = model.predict_proba(g)[:, 1].mean()
    return surf


def pdp_fig(ins: pd.DataFrame) -> None:
    m_b = CatBoostClassifier(); m_b.load_model(BEFORE)
    m_a = CatBoostClassifier(); m_a.load_model(AFTER)
    names = list(m_b.feature_names_)
    sub = ins.sample(min(SUBSAMPLE, len(ins)), random_state=42)
    # grids over the central 5..95 pct so extreme tails don't dominate
    vol_grid = np.quantile(ins[F_VOL].dropna(), np.linspace(0.05, 0.95, GRID))
    idx_grid = np.quantile(ins[F_IDX].dropna(), np.linspace(0.05, 0.95, GRID))
    surf_b = pdp_surface(m_b, sub, names, vol_grid, idx_grid)
    surf_a = pdp_surface(m_a, sub, names, vol_grid, idx_grid)

    vmin = min(surf_b.min(), surf_a.min()); vmax = max(surf_b.max(), surf_a.max())
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, surf, title in [(axes[0], surf_b, "BEFORE (unconstrained)"),
                            (axes[1], surf_a, "AFTER (+1 monotone on garman_klass_200)")]:
        im = ax.imshow(surf, origin="lower", aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax,
                       extent=[0, GRID - 1, 0, GRID - 1])
        ax.set_xlabel(f"{F_VOL} decile (low -> high vol) ->  CONSTRAINED +1")
        ax.set_ylabel(f"{F_IDX} decile (low -> high index momentum)")
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean predicted P(+10% breakout)")
    fig.suptitle(f"2D partial dependence: {F_VOL} (x) x {F_IDX} (y)\n"
                 "BEFORE: P(breakout) vs vol can flip sign across index-momentum rows.  "
                 "AFTER: forced non-decreasing left->right at every row.", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/gbdt/experiments/nifty50_pdp_before_after.png", dpi=130, bbox_inches="tight")
    print("wrote pdp before/after")

    # numeric: per-index-row, does vol effect ever go DOWN (sign-flip)?
    def flip_rows(surf):
        return int(sum(1 for r in surf if np.any(np.diff(r) < -1e-4)))
    print(f"BEFORE rows with a vol sign-flip (vol effect decreasing somewhere): {flip_rows(surf_b)}/{GRID}")
    print(f"AFTER  rows with a vol sign-flip: {flip_rows(surf_a)}/{GRID}")


def main() -> None:
    ins = build_insample()
    print(f"in-sample rows={len(ins)}")
    corr_fig(ins)
    pdp_fig(ins)


if __name__ == "__main__":
    main()
