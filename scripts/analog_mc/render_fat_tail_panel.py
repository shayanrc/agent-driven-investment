"""Render the 15-anchor fat-tail panel for any walk-forward run.

Drop-in replacement for the per-anchor bash loop over plot_forecast_from_date.py.
Loads each fold's `forecasts.npz` once and renders all 15 charts in a single
process — faster than 15 subprocess invocations and supports per-experiment
title labels.

Usage:
    uv run python scripts/analog_mc/render_fat_tail_panel.py \\
        --run-dir runs/analog_mc/20260520T155220Z \\
        --label "B1 (Platzer local-linear)" \\
        --out-dir docs/analog_mc/experiments/figs/b1_local_linear_fat_tail \\
        --prefix b1_local_linear
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.data import load_close_series

ROOT = Path(__file__).resolve().parents[2]
ANCHORS_JSON = ROOT / "results" / "analog_mc" / "data" / "fat_tail_eval_anchors.json"


def all_anchor_dates() -> list[str]:
    payload = json.loads(ANCHORS_JSON.read_text())
    dates = []
    for sec in ("positive", "negative", "regime_coverage"):
        for e in payload.get(sec, []):
            dates.append(e["anchor_date"])
    return dates


def load_fold_summaries(run_dir: Path) -> list[dict]:
    return [
        json.loads((run_dir / "folds" / d.name / "summary.json").read_text())
        for d in sorted((run_dir / "folds").iterdir(), key=lambda p: int(p.name))
    ]


def find_origin(close: pd.Series, date_str: str) -> tuple[int, pd.Timestamp]:
    target = pd.Timestamp(date_str)
    pos = close.index.searchsorted(target)
    if pos >= len(close):
        raise SystemExit(f"date {date_str} past end of data ({close.index[-1].date()})")
    return pos - 1, close.index[pos]


def render_one(
    date_str: str, run_dir: Path, close: pd.Series, folds: list[dict],
    horizon: int, label: str, out_path: Path,
) -> dict | None:
    origin_idx, actual_date = find_origin(close, date_str)
    fold = next((f for f in folds if f["test_start"] <= origin_idx <= f["test_end"]), None)
    if fold is None:
        print(f"  SKIP {date_str}: no fold contains origin_idx={origin_idx}")
        return None
    arr = np.load(run_dir / "folds" / str(fold["fold_index"]) / "forecasts.npz")
    matches = np.where(arr["origin_idx"] == origin_idx)[0]
    if matches.size == 0:
        print(f"  SKIP {date_str}: origin not in fold's forecasts.npz")
        return None
    pos = int(matches[0])
    fc_returns = arr["paths"][pos]
    rl_returns = arr["realized"][pos]

    anchor_close_pos = origin_idx + 1
    p0 = float(close.iloc[anchor_close_pos])

    fc_prices = p0 * np.exp(np.cumsum(fc_returns, axis=1))
    rl_prices = p0 * np.exp(np.cumsum(rl_returns))

    forward_dates = close.index[anchor_close_pos + 1 : anchor_close_pos + 1 + horizon]

    q05 = np.quantile(fc_prices, 0.05, axis=0)
    q25 = np.quantile(fc_prices, 0.25, axis=0)
    q50 = np.quantile(fc_prices, 0.50, axis=0)
    q75 = np.quantile(fc_prices, 0.75, axis=0)
    q95 = np.quantile(fc_prices, 0.95, axis=0)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    hist_start = max(0, anchor_close_pos - 30)
    hist_dates = close.index[hist_start : anchor_close_pos + 1]
    hist_prices = close.iloc[hist_start : anchor_close_pos + 1].to_numpy()
    ax.plot(hist_dates, hist_prices, color="black", lw=1.5, label="historical")

    ax.fill_between(forward_dates, q05, q95, color="tab:red", alpha=0.15, label="forecast 90% band")
    ax.fill_between(forward_dates, q25, q75, color="tab:red", alpha=0.30, label="forecast 50% band")
    ax.plot(forward_dates, q50, color="tab:red", lw=1.6, label="forecast median")
    ax.plot(forward_dates, rl_prices, color="black", lw=1.8, label="realized")
    ax.axvline(actual_date, color="grey", lw=0.5, ls=":")
    ax.scatter([actual_date], [p0], color="black", s=20, zorder=5)

    w = fold["weights"]
    ax.set_title(
        f"NASDAQ100 — 60-day forecast vs realized (anchored {actual_date.date()})\n"
        f"Model: {label} · fold {fold['fold_index']} weights "
        f"[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}] n_eff={fold['n_eff']:.0f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("NASDAQ100 close")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    in_50 = int(((rl_prices >= q25) & (rl_prices <= q75)).sum())
    in_90 = int(((rl_prices >= q05) & (rl_prices <= q95)).sum())
    return {
        "anchor_date": date_str,
        "fold_index": fold["fold_index"],
        "in_50": in_50,
        "in_90": in_90,
        "out_path": str(out_path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="walk-forward run dir")
    p.add_argument("--label", required=True, help="model label for plot titles (e.g. 'B1')")
    p.add_argument("--out-dir", required=True, help="output dir for PNG panel")
    p.add_argument("--prefix", default="forecast", help="filename prefix (default: forecast)")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    cfg = Config.from_yaml(run_dir / "config.yaml")
    close = load_close_series(cfg.data_path, cfg.date_col, cfg.close_col)
    folds = load_fold_summaries(run_dir)

    results = []
    for date_str in all_anchor_dates():
        out_path = out_dir / f"{args.prefix}_{date_str.replace('-', '')}.png"
        r = render_one(date_str, run_dir, close, folds, cfg.forecast_horizon, args.label, out_path)
        if r is not None:
            results.append(r)
            print(f"  {date_str}: 50/60={r['in_50']:>2d}, 90/60={r['in_90']:>2d} -> {out_path}")
    print(f"\nRendered {len(results)}/15 panels into {out_dir}")


if __name__ == "__main__":
    main()
