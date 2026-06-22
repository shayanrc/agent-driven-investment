#!/usr/bin/env python
"""Aggregate the macro-lattice sweep (base vs +macro per cell) into a delta table +
heatmap. Reads each <cell>_sw{base,macro}/predictions/test.csv, computes R-Precision@K
via scripts.gbdt.compute_r_precision, and renders a threshold x horizon heatmap of the
macro-minus-base delta at K in {1,5,10}. Memo _263.

Usage: uv run python -m scripts.gbdt.aggregate_macro_sweep [universe]   # default sp500
"""
import glob
import json
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UNI = sys.argv[1] if len(sys.argv) > 1 else "sp500"
KS = ["1", "3", "5", "10", "20"]


def rp(cell):
    p = f"results/gbdt/experiments/{cell}/predictions/test.csv"
    out = subprocess.run(
        [".venv/bin/python", "-m", "scripts.gbdt.compute_r_precision", p, "--json", "--no-legacy"],
        capture_output=True, text=True,
    ).stdout
    d = json.loads(out)["r_precision_at_k"]
    bk = d["by_k"]
    vals = {k: (bk[k]["r_precision_at_k"] if bk[k]["r_precision_at_k"] is not None else float("nan")) for k in KS}
    return d.get("base_rate", float("nan")), d.get("n_days_total", 0), vals


cells = sorted(set(
    re.match(rf"({UNI}_up_\d+pct_\d+d_dd\d+pct)_sw", f.split("/")[-3]).group(1)
    for f in glob.glob(f"results/gbdt/experiments/{UNI}_up_*pct_*d_dd*pct_sw*/predictions/test.csv")
))
rows = []
for c in cells:
    m = re.match(rf"{UNI}_up_(\d+)pct_(\d+)d_dd(\d+)pct", c)
    thr, hor, dd = int(m[1]), int(m[2]), int(m[3])
    br, Q, b = rp(c + "_swbase")
    _, _, mac = rp(c + "_swmacro")
    rows.append(dict(cell=c, thr=thr, hor=hor, dd=dd, base_rate=br, Q=Q, base=b, macro=mac))
    print(f"{c:34s} br={br:.4f} Q={Q:>3} "
          + "  ".join(f"@{k} {b[k]:.3f}->{mac[k]:.3f} ({mac[k]-b[k]:+.3f})" for k in ("1", "5", "10")))

json.dump(rows, open(f"results/gbdt/data/_263_macro_lattice_raw.json", "w"), indent=1)

# ---- heatmap: threshold (rows) x horizon (cols), one panel per K, colored by macro-base delta
THR = sorted({r["thr"] for r in rows})
HOR = sorted({r["hor"] for r in rows})
panels = ["1", "5", "10"]
fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), constrained_layout=True)
vmax = 0.18
for ax, K in zip(axes, panels):
    grid = np.full((len(THR), len(HOR)), np.nan)
    for r in rows:
        d = r["macro"][K] - r["base"][K]
        if not np.isnan(d):
            grid[THR.index(r["thr"]), HOR.index(r["hor"])] = d
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(HOR)), [f"{h}d" for h in HOR])
    ax.set_yticks(range(len(THR)), [f"+{t}%" for t in THR])
    ax.set_title(f"macro − base   R-Precision@{K}", fontsize=11, fontweight="bold")
    ax.set_xlabel("horizon"); ax.set_ylabel("threshold (dd = thr/2)")
    for i in range(len(THR)):
        for j in range(len(HOR)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="black" if abs(v) < vmax * 0.7 else "white")
            else:
                ax.text(j, i, "·", ha="center", va="center", fontsize=10, color="0.6")
fig.colorbar(im, ax=axes, shrink=0.8, label="Δ R-Precision (green = macro better)")
fig.suptitle(f"{UNI} macro-lattice sweep — does F17 macro help? (matched mcw=10 single-fit, snapshot 2026-06-20)",
             fontsize=12, fontweight="bold")
out = f"results/gbdt/data/_263_macro_lattice_heatmap.png"
fig.savefig(out, dpi=150)
fig.savefig(out.replace(".png", ".svg"))
print(f"\nheatmap -> {out}")
