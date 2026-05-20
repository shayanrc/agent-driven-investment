"""E10 (Cell D × bl=20) — aggregate metrics for _e10_celld_bl20.md.

Computes ACF curve, per-vol-regime CRPS, and produces a 4-way comparison plot:
    D-fast (Cell D, bl=10)
    E1-bl20 (zero drift, bl=20)
    E10    (Cell D, bl=20)
    realized

Outputs:
    docs/analog_mc/figs/e10_celld_bl20_acf.png
    docs/analog_mc/_e10_data.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analog_mc.diagnostics import (
    aggregate_crps_per_vol_regime,
    concatenate_oos,
    load_run,
)
from analog_mc.data import load_returns

# Reuse the vectorized ACF
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _e1_aggregate import batch_acf_mean  # type: ignore  # noqa: E402

CELLS = [
    ("D-fast", "runs/analog_mc/20260517T070003Z", "tab:purple"),       # Cell D, bl=10
    ("E1-bl20", "runs/analog_mc/20260519T071520Z", "tab:green"),        # zero drift, bl=20
    ("E10", "runs/analog_mc/20260519T083821Z", "tab:red"),              # Cell D, bl=20
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
    realized_acf_full = np.zeros(MAX_LAG + 1)
    realized_acf_full[0] = 1.0
    x = realized_sq - realized_sq.mean()
    denom = (x * x).sum()
    for lag in range(1, MAX_LAG + 1):
        realized_acf_full[lag] = (x[:-lag] * x[lag:]).sum() / denom

    per_regime = aggregate_crps_per_vol_regime(run, returns)
    report_path = Path(run_dir, "diagnostic_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        # Compute mean CRPS inline from per-fold parquet
        import pandas as pd
        summary = pd.read_parquet(Path(run_dir, "summary.parquet"))
        report = {
            "overall": {
                "mean_crps": float(summary["test_crps"].mean()),
                "median_crps": float(summary["test_crps"].median()),
            },
            "decision_rules": {},
        }

    return {
        "sim_acf": sim_acf,
        "realized_acf": realized_acf_full,
        "mean_crps": report["overall"]["mean_crps"],
        "median_crps": report["overall"]["median_crps"],
        "per_regime": per_regime.set_index("regime")["mean_crps"].to_dict(),
        "decision_rules": report["decision_rules"],
        "config": {
            "block_length": run.config.block_length,
            "n_blocks": run.config.n_blocks,
            "drift_mode": run.config.drift_mode,
            "conditional_block_sampling": run.config.conditional_block_sampling,
        },
    }


def main() -> None:
    results = {}
    realized_acf = None
    for name, run_dir, _ in CELLS:
        print(f"loading {name} from {run_dir} ...")
        m = cell_metrics(run_dir)
        results[name] = m
        if realized_acf is None:
            realized_acf = m["realized_acf"]

    lags = np.arange(MAX_LAG + 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(lags[1:], realized_acf[1:], marker="o", lw=2.5, color="black", label="realized")
    for name, _, color in CELLS:
        cfg = results[name]["config"]
        suffix = f" (bl={cfg['block_length']}, drift={cfg['drift_mode'][:4]}, cond={cfg['conditional_block_sampling']})"
        ax.plot(
            lags[1:],
            results[name]["sim_acf"][1:],
            marker="s",
            lw=1.6,
            color=color,
            label=f"sim {name}{suffix}",
        )
    for seam in (10, 20, 30, 40, 50):
        ax.axvline(seam, color="grey", linestyle=":", alpha=0.4)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF of squared returns")
    ax.set_title("E10 vs D-fast vs E1-bl20 — squared-return ACF")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)
    fig_path = Path("docs/analog_mc/figs/e10_celld_bl20_acf.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    print(f"saved {fig_path}")

    print("\n=== Headline metrics ===")
    for name, _, _ in CELLS:
        m = results[name]
        rules = m["decision_rules"]
        print(
            f"{name}: mean_crps={m['mean_crps']:.5f} "
            f"low={m['per_regime']['low_vol']:.4f} "
            f"mid={m['per_regime']['mid_vol']:.4f} "
            f"high={m['per_regime']['high_vol']:.4f}"
        )
        if rules:
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
    for name, run_dir, _ in CELLS:
        m = results[name]
        out["cells"][name] = {
            "run_dir": run_dir,
            "config": m["config"],
            "mean_crps": m["mean_crps"],
            "median_crps": m["median_crps"],
            "per_regime_crps": m["per_regime"],
            "sim_acf_at_lags": {str(l): float(m["sim_acf"][l]) for l in LAGS_REPORTED},
            "rules": {
                k: {"metric": v["metric"], "fired": v.get("fired", False)}
                for k, v in m["decision_rules"].items()
            },
        }
    Path("docs/analog_mc/_e10_data.json").write_text(json.dumps(out, indent=2))
    print("\nwrote docs/analog_mc/_e10_data.json")


if __name__ == "__main__":
    main()
