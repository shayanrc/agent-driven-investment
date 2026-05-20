"""Plot forecast vs realized for the most recent 60 days in the data.

Picks the latest origin in the canonical run (Cell-D-s30) with a complete
60-day realized comparison, integrates forecast log-returns into price
paths, and renders median + 50% / 90% credible bands against the realized
price track.

Usage:
    uv run python scripts/plot_last_60_days.py [--run RUN_DIR] [--out OUT_PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.data import load_close_series


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/analog_mc/20260520T045525Z")
    p.add_argument("--out", default="docs/analog_mc/figs/last_60_days_forecast.png")
    args = p.parse_args()

    run_dir = Path(args.run)
    cfg = Config.from_yaml(run_dir / "config.yaml")
    close = load_close_series(cfg.data_path, cfg.date_col, cfg.close_col)

    # Latest fold has the most recent origins; pick the latest origin with
    # complete 60-day realized horizon (origin_idx + horizon < len(close)).
    fold_dirs = sorted((run_dir / "folds").iterdir(), key=lambda p: int(p.name))
    last_fold = fold_dirs[-1]
    arr = np.load(last_fold / "forecasts.npz")
    origin_idx_arr = arr["origin_idx"]
    paths = arr["paths"]      # (n_origins, n_paths, horizon)
    realized = arr["realized"]  # (n_origins, horizon)

    # Pick the largest origin_idx (closest to the end of the data)
    best_local = int(np.argmax(origin_idx_arr))
    origin_idx = int(origin_idx_arr[best_local])
    fc_returns = paths[best_local]      # (n_paths, 60)
    rl_returns = realized[best_local]   # (60,)

    # The forecast/realized are log-returns. origin_idx is the index of the
    # last *observed* log return (so close.index[origin_idx + 1] is the
    # first close used as the integration anchor; the forecast then covers
    # close.index[origin_idx + 2 : origin_idx + 62]).
    # close has len(returns) + 1 prices; origin_idx-th log-return corresponds
    # to log(close[i+1]) - log(close[i]). So close[origin_idx + 1] is the
    # last "known" price (the anchor).
    anchor_close_pos = origin_idx + 1
    p0 = float(close.iloc[anchor_close_pos])
    anchor_date = close.index[anchor_close_pos]

    # Integrate log-returns into prices
    fc_prices = p0 * np.exp(np.cumsum(fc_returns, axis=1))  # (n_paths, 60)
    rl_prices = p0 * np.exp(np.cumsum(rl_returns))           # (60,)

    # Construct forward date axis: 60 trading days after anchor_close_pos
    forward_dates = close.index[anchor_close_pos + 1 : anchor_close_pos + 1 + 60]

    # Quantiles for bands
    q05 = np.quantile(fc_prices, 0.05, axis=0)
    q25 = np.quantile(fc_prices, 0.25, axis=0)
    q50 = np.quantile(fc_prices, 0.50, axis=0)
    q75 = np.quantile(fc_prices, 0.75, axis=0)
    q95 = np.quantile(fc_prices, 0.95, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Include a chunk of historical context — last 30 days before the anchor
    hist_start = max(0, anchor_close_pos - 30)
    hist_dates = close.index[hist_start : anchor_close_pos + 1]
    hist_prices = close.iloc[hist_start : anchor_close_pos + 1].to_numpy()
    ax.plot(hist_dates, hist_prices, color="black", lw=1.5, label="historical")

    # Forecast bands and median
    ax.fill_between(forward_dates, q05, q95, color="tab:red", alpha=0.15,
                    label="forecast 90% band")
    ax.fill_between(forward_dates, q25, q75, color="tab:red", alpha=0.30,
                    label="forecast 50% band")
    ax.plot(forward_dates, q50, color="tab:red", lw=1.6, label="forecast median")

    # Realized
    ax.plot(forward_dates, rl_prices, color="black", lw=1.8, label="realized")

    # Anchor marker
    ax.axvline(anchor_date, color="grey", lw=0.5, ls=":")
    ax.scatter([anchor_date], [p0], color="black", s=20, zorder=5)

    ax.set_title(
        f"NASDAQ100 — 60-day forecast vs realized (anchored {anchor_date.date()})\n"
        f"Model: v2.4 Cell-D-s30 (canonical run {run_dir.name})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("NASDAQ100 close")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    final_idx = -1
    print(f"anchor date:    {anchor_date.date()} (close = {p0:,.1f})")
    print(f"forecast end:   {forward_dates[final_idx].date()}")
    print(f"realized final: {rl_prices[final_idx]:,.1f}")
    print(f"forecast 50% interval at horizon: [{q25[final_idx]:,.1f}, {q75[final_idx]:,.1f}]")
    print(f"forecast 90% interval at horizon: [{q05[final_idx]:,.1f}, {q95[final_idx]:,.1f}]")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
