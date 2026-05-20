"""Plot forecast vs realized for a given anchor date.

Looks up the cached forecast at the matching origin from any fold of the
canonical run. If the origin isn't in any fold's test window, errors out
(production would have produced a forecast there only if the walk-forward
schedule had reached it).

Usage:
    uv run python scripts/plot_forecast_from_date.py --date 2022-03-01
    uv run python scripts/plot_forecast_from_date.py --date 2020-03-16 --out custom.png
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


def find_origin_for_date(run_dir: Path, close, date_str: str) -> tuple[int, int, dict]:
    """Find (origin_idx, local_pos_in_fold, fold_summary) for the given date."""
    target = pd.Timestamp(date_str)
    pos = close.index.searchsorted(target)
    if pos >= len(close):
        raise SystemExit(f"date {date_str} is past the end of the data ({close.index[-1].date()})")
    actual_date = close.index[pos]
    origin_idx = pos - 1
    for fold_dir in sorted((run_dir / "folds").iterdir(), key=lambda p: int(p.name)):
        s = json.loads((fold_dir / "summary.json").read_text())
        if s["test_start"] <= origin_idx <= s["test_end"]:
            arr = np.load(fold_dir / "forecasts.npz")
            mask = arr["origin_idx"] == origin_idx
            if mask.any():
                return origin_idx, int(np.argmax(mask)), s, fold_dir, actual_date
    raise SystemExit(
        f"no fold's test window contains origin_idx={origin_idx} ({actual_date.date()}). "
        f"Try a date inside a walk-forward test window."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="anchor date (YYYY-MM-DD)")
    p.add_argument("--run", default="runs/analog_mc/20260520T045525Z")
    p.add_argument("--out", default=None, help="output path (auto-generated from date if omitted)")
    args = p.parse_args()

    run_dir = Path(args.run)
    cfg = Config.from_yaml(run_dir / "config.yaml")
    close = load_close_series(cfg.data_path, cfg.date_col, cfg.close_col)

    origin_idx, local_pos, fold_summary, fold_dir, actual_date = find_origin_for_date(
        run_dir, close, args.date
    )
    arr = np.load(fold_dir / "forecasts.npz")
    fc_returns = arr["paths"][local_pos]      # (n_paths, 60)
    rl_returns = arr["realized"][local_pos]   # (60,)

    anchor_close_pos = origin_idx + 1
    p0 = float(close.iloc[anchor_close_pos])

    fc_prices = p0 * np.exp(np.cumsum(fc_returns, axis=1))
    rl_prices = p0 * np.exp(np.cumsum(rl_returns))

    horizon = cfg.forecast_horizon
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

    ax.fill_between(forward_dates, q05, q95, color="tab:red", alpha=0.15,
                    label="forecast 90% band")
    ax.fill_between(forward_dates, q25, q75, color="tab:red", alpha=0.30,
                    label="forecast 50% band")
    ax.plot(forward_dates, q50, color="tab:red", lw=1.6, label="forecast median")
    ax.plot(forward_dates, rl_prices, color="black", lw=1.8, label="realized")
    ax.axvline(actual_date, color="grey", lw=0.5, ls=":")
    ax.scatter([actual_date], [p0], color="black", s=20, zorder=5)

    weights = fold_summary["weights"]
    n_eff = fold_summary["n_eff"]
    fold_idx = fold_summary["fold_index"]
    ax.set_title(
        f"NASDAQ100 — 60-day forecast vs realized (anchored {actual_date.date()})\n"
        f"Model: v2.4 Cell-D-s30 · fold {fold_idx} weights "
        f"[{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}] n_eff={n_eff:.0f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("NASDAQ100 close")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out) if args.out else Path(
        f"docs/analog_mc/figs/forecast_{actual_date.strftime('%Y%m%d')}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    inside_50 = ((rl_prices >= q25) & (rl_prices <= q75)).sum()
    inside_90 = ((rl_prices >= q05) & (rl_prices <= q95)).sum()

    print(f"anchor:           {actual_date.date()} close={p0:,.1f}")
    print(f"realized at t=59: {rl_prices[-1]:,.1f}   ({(rl_prices[-1]/p0-1)*100:+.2f}%)")
    print(f"forecast 50% at horizon: [{q25[-1]:,.1f}, {q75[-1]:,.1f}]")
    print(f"forecast 90% at horizon: [{q05[-1]:,.1f}, {q95[-1]:,.1f}]")
    print(f"realized in 50% band: {inside_50}/60 days")
    print(f"realized in 90% band: {inside_90}/60 days")
    print(f"weights: {weights}, n_eff={n_eff}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
