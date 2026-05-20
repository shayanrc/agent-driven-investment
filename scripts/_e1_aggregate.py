"""E1 block-length sweep — aggregate ACF curves, per-regime CRPS, and rule verdicts.

Produces:
    docs/analog_mc/figs/e1_block_length_acf.png
    results/analog_mc/data/_e1_data.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analog_mc.diagnostics import (
    _acf,
    aggregate_crps_per_vol_regime,
    concatenate_oos,
    load_run,
)


def batch_acf_mean(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean sample ACF across rows of x (shape (N, H)), at lags 0..max_lag.

    Per-row ACF (centered, normalized by row sum-of-squares), then averaged
    across rows. Rows with zero variance are skipped.
    """
    x_c = x - x.mean(axis=1, keepdims=True)
    denom = (x_c * x_c).sum(axis=1)
    mask = denom > 0
    x_c = x_c[mask]
    denom = denom[mask]
    out = np.zeros(max_lag + 1)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        numer = (x_c[:, :-lag] * x_c[:, lag:]).sum(axis=1)
        out[lag] = (numer / denom).mean()
    return out
from analog_mc.data import load_returns

CELLS = [
    ("E1-bl5", 5, "runs/analog_mc/20260519T060335Z"),
    ("E1-bl10", 10, "runs/analog_mc/20260519T064102Z"),
    ("E1-bl20", 20, "runs/analog_mc/20260519T071520Z"),
]

LAGS_REPORTED = [1, 5, 10, 15, 20, 50]
MAX_LAG = 50


def cell_metrics(run_dir: str) -> dict:
    run = load_run(run_dir)
    returns = load_returns(run.config)

    paths, _, _ = concatenate_oos(run.folds)
    sim_sq = (paths**2).astype(np.float32, copy=False)
    O, P, H = sim_sq.shape
    sim_flat = sim_sq.reshape(O * P, H)
    sim_acf = batch_acf_mean(sim_flat, MAX_LAG)

    realized_sq = returns.to_numpy() ** 2
    realized_acf = _acf(realized_sq, MAX_LAG)

    per_regime = aggregate_crps_per_vol_regime(run, returns)
    report = json.loads(Path(run_dir, "diagnostic_report.json").read_text())

    return {
        "sim_acf": sim_acf,
        "realized_acf": realized_acf,
        "mean_crps": report["overall"]["mean_crps"],
        "median_crps": report["overall"]["median_crps"],
        "per_regime": per_regime.set_index("regime")["mean_crps"].to_dict(),
        "decision_rules": report["decision_rules"],
    }


def main() -> None:
    results = {}
    realized_acf = None
    for name, _bl, run_dir in CELLS:
        print(f"loading {name} from {run_dir} ...")
        m = cell_metrics(run_dir)
        results[name] = m
        if realized_acf is None:
            realized_acf = m["realized_acf"]

    lags = np.arange(MAX_LAG + 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(lags[1:], realized_acf[1:], marker="o", lw=2.5, color="black", label="realized")
    palette = {"E1-bl5": "tab:blue", "E1-bl10": "tab:orange", "E1-bl20": "tab:green"}
    for name, _bl, _ in CELLS:
        ax.plot(
            lags[1:],
            results[name]["sim_acf"][1:],
            marker="s",
            lw=1.6,
            color=palette[name],
            label=f"sim {name}",
        )
    for seam in (10, 20, 30, 40, 50):
        ax.axvline(seam, color="grey", linestyle=":", alpha=0.4)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF of squared returns")
    ax.set_title("E1 block-length sweep — squared-return ACF (simulated vs realized)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)
    fig_path = Path("docs/analog_mc/figs/e1_block_length_acf.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    print(f"saved {fig_path}")

    acf_rows = []
    for lag in LAGS_REPORTED:
        row = {"lag": lag, "realized": realized_acf[lag]}
        for name, _bl, _ in CELLS:
            row[name] = results[name]["sim_acf"][lag]
        acf_rows.append(row)
    acf_df = pd.DataFrame(acf_rows)
    print("\n=== ACF table ===")
    print(acf_df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n=== Headline metrics ===")
    for name, bl, _ in CELLS:
        m = results[name]
        rules = m["decision_rules"]
        print(
            f"{name} (bl={bl}): mean_crps={m['mean_crps']:.5f} "
            f"low={m['per_regime']['low_vol']:.4f} "
            f"mid={m['per_regime']['mid_vol']:.4f} "
            f"high={m['per_regime']['high_vol']:.4f}"
        )
        print(
            f"    rules: pit={rules['sloped_global_pit']['metric']:+.3f} "
            f"acf={rules['acf_seam_degradation']['metric']:+.3f} "
            f"uvol={rules['u_shaped_high_vol_pit']['metric']:+.3f} "
            f"clip={rules['clip_hit_excessive']['metric']:+.3f}"
        )

    out = {
        "lags_reported": LAGS_REPORTED,
        "realized_acf_at_lags": {str(l): float(realized_acf[l]) for l in LAGS_REPORTED},
        "cells": {},
    }
    for name, bl, run_dir in CELLS:
        m = results[name]
        out["cells"][name] = {
            "block_length": bl,
            "run_dir": run_dir,
            "mean_crps": m["mean_crps"],
            "median_crps": m["median_crps"],
            "per_regime_crps": m["per_regime"],
            "sim_acf_at_lags": {str(l): float(m["sim_acf"][l]) for l in LAGS_REPORTED},
            "rules": {
                k: {"metric": v["metric"], "fired": v.get("fired", False)}
                for k, v in m["decision_rules"].items()
            },
        }
    Path("results/analog_mc/data/_e1_data.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/analog_mc/data/_e1_data.json")


if __name__ == "__main__":
    main()
