"""Pairwise interaction strength: unconstrained vs monotone-constrained.

before = screening model (all-279, depth6, NO constraints)
after  = iter5 model      (all-279, depth6, +17 monotone +1 on vol estimators)
3 panels: before | after | delta(after-before). Shows how the monotone
constraint redistributed/destroyed pairwise interaction structure.

**FROZEN ONE-SHOT.** The figures are committed under the nifty50 H=25
monotone-constraint memo. To re-run, set ``WORKSPACE_ROOT`` per per-user
memory ``scratch-cache-path``; the referenced wt-exp-nifty50-up10-25d/
worktree may have been pruned, in which case the BEFORE model load fails
loudly at first read.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier

# WORKSPACE_ROOT = parent dir where ``wt-*/`` worktrees live; per-machine,
# see per-user memory ``scratch-cache-path`` for the literal.
BEFORE = f"{os.environ.get('WORKSPACE_ROOT', '<SET-WORKSPACE_ROOT>')}/wt-exp-nifty50-up10-25d/results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct/model.cbm"
AFTER = "results/gbdt/experiments/nifty50_manualloop_iter5/model.cbm"

# Display set: 9 constrained vol estimators (left/top) then 7 unconstrained refs.
DISPLAY = [
    "garman_klass_50", "garman_klass_100", "garman_klass_200",
    "parkinson_50", "parkinson_100", "parkinson_200",
    "realized_vol_50", "realized_vol_100", "realized_vol_200",
    # --- unconstrained ---
    "vol_of_vol_200", "index_vol_50", "index_vol_200",
    "index_return_50", "index_return_100", "index_return_200", "index_drawdown_100",
]
N_CONSTRAINED = 9


def interaction_matrix(model_path: str, display: list[str]) -> np.ndarray:
    m = CatBoostClassifier(); m.load_model(model_path)
    names = m.feature_names_
    idx = {n: i for i, n in enumerate(names)}
    pos = {idx[d]: k for k, d in enumerate(display) if d in idx}
    mat = np.zeros((len(display), len(display)))
    for i1, i2, s in m.get_feature_importance(type="Interaction"):
        i1, i2 = int(i1), int(i2)
        if i1 in pos and i2 in pos:
            a, b = pos[i1], pos[i2]
            mat[a, b] = s
            mat[b, a] = s
    return mat


def main() -> None:
    before = interaction_matrix(BEFORE, DISPLAY)
    after = interaction_matrix(AFTER, DISPLAY)
    delta = after - before

    fig, axes = plt.subplots(1, 3, figsize=(26, 9))
    panels = [
        ("BEFORE (unconstrained)", before, "viridis", 0, max(before.max(), after.max())),
        ("AFTER (+1 on 9 vol est.)", after, "viridis", 0, max(before.max(), after.max())),
        ("DELTA (after - before)", delta, "RdBu_r", -np.abs(delta).max(), np.abs(delta).max()),
    ]
    for ax, (title, mat, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(DISPLAY)))
        ax.set_yticks(range(len(DISPLAY)))
        ax.set_xticklabels(DISPLAY, rotation=90, fontsize=8)
        ax.set_yticklabels(DISPLAY, fontsize=8)
        for i in range(len(DISPLAY)):
            for j in range(len(DISPLAY)):
                v = mat[i, j]
                if abs(v) > 1e-9:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if (cmap == "viridis" and v < vmax * 0.6) else "black")
        b = N_CONSTRAINED - 0.5
        ax.axhline(b, color="lime", lw=2); ax.axvline(b, color="lime", lw=2)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("nifty50 H=25 — pairwise interaction strength | green line separates "
                 "9 constrained vol estimators (top-left) from unconstrained features",
                 fontsize=13)
    fig.tight_layout()
    out = "results/gbdt/experiments/nifty50_interaction_before_after.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)

    # numeric: total interaction mass lost on constrained-involving pairs
    cc = slice(0, N_CONSTRAINED)
    constr_before = before[cc, :].sum() - before[cc, cc].sum() / 2 + before[cc, cc].sum() / 2
    print(f"sum interaction (display) before={before.sum()/2:.2f} after={after.sum()/2:.2f}")
    # mass involving >=1 constrained feature
    mask = np.zeros_like(before, dtype=bool)
    mask[:N_CONSTRAINED, :] = True; mask[:, :N_CONSTRAINED] = True
    print(f"constrained-involving mass: before={before[mask].sum()/2:.2f} "
          f"after={after[mask].sum()/2:.2f} "
          f"delta={(after[mask].sum()-before[mask].sum())/2:+.2f}")


if __name__ == "__main__":
    main()
