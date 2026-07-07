"""Plot forecast distribution vs realized series for a few representative origins.

Default: side-by-side comparison of v1 canonical (zero drift) and v2.1 acceptance
(trailing-momentum drift) forecasts at the SAME origin indices. Layout:

    [low-vol origin]   [mid-vol origin]   [high-vol origin]
    +-- v1 ----------+ +-- v1 ----------+ +-- v1 ----------+
    +-- v2.1 -------+ +-- v2.1 -------+ +-- v2.1 -------+

Each subplot shows: realized price path (solid black), forecast median (red),
50% credible band (red fill), 90% credible band (lighter red fill), and 30
sample paths (thin grey lines).

Usage:
    uv run python scripts/analog_mc/plot_forecast_vs_realized.py \\
        --v1-run runs/analog_mc/20260516T180000Z \\
        --v2-run runs/analog_mc/20260517T050831Z \\
        --out runs/analog_mc/_forecast_vs_realized.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analog_mc.data import load_close_series
from analog_mc.diagnostics import load_run


def _pick_representative_origins(run, returns: pd.Series, fold_index: int) -> list[tuple[int, int, str]]:
    """From the given fold, pick three test origins spanning the vol regime spectrum.

    Returns list of (within_fold_position, absolute_origin_idx, label).
    """
    fold = run.folds[fold_index]
    # Per-origin realized vol over the forecast horizon (proxy for difficulty).
    horizons = run.config.zscore_horizons
    halflife = run.config.ewma_halflife
    vol = returns.ewm(halflife=halflife, adjust=False).std().to_numpy()

    origin_vols = vol[fold.origin_idx]
    # Pick the origins at the 10th, 50th, and 90th percentile of forecast-window vol.
    order = np.argsort(origin_vols)
    n = len(order)
    picks = [
        (int(order[int(n * 0.10)]), "low vol"),
        (int(order[int(n * 0.50)]), "mid vol"),
        (int(order[int(n * 0.90)]), "high vol"),
    ]
    return [(pos, int(fold.origin_idx[pos]), label) for pos, label in picks]


def _paths_to_prices(log_return_paths: np.ndarray, anchor_price: float) -> np.ndarray:
    """Convert (n_paths, horizon) log-return paths to (n_paths, horizon+1) price paths.

    Prepends the anchor price at t=0 so the realized + forecast share the same
    starting point visually.
    """
    cum_log = np.cumsum(log_return_paths, axis=-1)
    # Shape (n_paths, horizon+1) with anchor_price at column 0.
    horizons = cum_log.shape[-1]
    prices = np.empty((cum_log.shape[0], horizons + 1))
    prices[:, 0] = anchor_price
    prices[:, 1:] = anchor_price * np.exp(cum_log)
    return prices


def _realized_to_prices(realized_log_returns: np.ndarray, anchor_price: float) -> np.ndarray:
    """(horizon,) log returns -> (horizon+1,) prices anchored at anchor_price."""
    cum = np.cumsum(realized_log_returns)
    prices = np.empty(realized_log_returns.size + 1)
    prices[0] = anchor_price
    prices[1:] = anchor_price * np.exp(cum)
    return prices


def _plot_one(ax, paths_log: np.ndarray, realized_log: np.ndarray, anchor_price: float,
              anchor_date: pd.Timestamp, horizon: int, title: str, color: str) -> None:
    """Render one forecast-vs-realized panel."""
    forecast_prices = _paths_to_prices(paths_log, anchor_price)  # (n_paths, h+1)
    realized_prices = _realized_to_prices(realized_log, anchor_price)  # (h+1,)
    days = np.arange(horizon + 1)

    p05 = np.percentile(forecast_prices, 5, axis=0)
    p25 = np.percentile(forecast_prices, 25, axis=0)
    p50 = np.percentile(forecast_prices, 50, axis=0)
    p75 = np.percentile(forecast_prices, 75, axis=0)
    p95 = np.percentile(forecast_prices, 95, axis=0)

    # 30 sample paths as a lightweight "shape of the distribution" hint.
    sample_idx = np.linspace(0, forecast_prices.shape[0] - 1, num=30, dtype=int)
    for i in sample_idx:
        ax.plot(days, forecast_prices[i], color=color, alpha=0.06, linewidth=0.6)

    ax.fill_between(days, p05, p95, color=color, alpha=0.15, label="90% band")
    ax.fill_between(days, p25, p75, color=color, alpha=0.30, label="50% band")
    ax.plot(days, p50, color=color, linewidth=1.8, label="forecast median")
    ax.plot(days, realized_prices, color="black", linewidth=1.8, label="realized")
    ax.scatter([0], [anchor_price], color="black", zorder=5, s=20)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("forecast day")
    ax.set_ylabel("NASDAQ100")
    ax.grid(alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-run", type=Path, required=True, help="v1 canonical run dir")
    parser.add_argument("--v2-run", type=Path, required=True, help="v2.1 acceptance run dir")
    parser.add_argument("--fold-index", type=int, default=50,
                        help="Which fold's test origins to draw from (default 50, mid-stream).")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    args = parser.parse_args()

    print(f"== loading v1 run {args.v1_run}")
    v1 = load_run(args.v1_run)
    print(f"== loading v2 run {args.v2_run}")
    v2 = load_run(args.v2_run)

    cfg = v1.config
    print(f"== loading prices from {cfg.data_path}")
    prices = load_close_series(cfg.data_path, date_col=cfg.date_col, close_col=cfg.close_col)
    returns = np.log(prices).diff().dropna()
    returns.name = "log_return"

    fold_idx = args.fold_index
    if fold_idx >= v1.n_folds or fold_idx >= v2.n_folds:
        fold_idx = min(v1.n_folds, v2.n_folds) - 1
        print(f"  (clamped fold_index to {fold_idx})")

    picks = _pick_representative_origins(v1, returns, fold_idx)
    print(f"== picks for fold {fold_idx}: {[(p, o, lbl) for p, o, lbl in picks]}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)

    for col, (within_pos, origin_idx, vol_label) in enumerate(picks):
        anchor_price = float(prices.iloc[origin_idx])
        anchor_date = prices.index[origin_idx]

        # v1 (top row)
        v1_fold = v1.folds[fold_idx]
        v1_paths = v1_fold.paths[within_pos]      # (n_paths, horizon)
        v1_real = v1_fold.realized[within_pos]    # (horizon,)
        _plot_one(
            axes[0, col], v1_paths, v1_real, anchor_price, anchor_date,
            cfg.forecast_horizon,
            f"v1 zero-drift — {vol_label}\n{anchor_date.date()} (origin idx {origin_idx})",
            color="tab:blue",
        )

        # v2.1 (bottom row) — same within_pos/origin_idx by construction (same fold layout)
        v2_fold = v2.folds[fold_idx]
        # Defensive: confirm the within-fold origin maps to the same absolute index.
        assert int(v2_fold.origin_idx[within_pos]) == origin_idx, (
            f"v1 and v2 disagree on fold {fold_idx} origin {within_pos}: "
            f"{origin_idx} vs {int(v2_fold.origin_idx[within_pos])}"
        )
        v2_paths = v2_fold.paths[within_pos]
        v2_real = v2_fold.realized[within_pos]
        _plot_one(
            axes[1, col], v2_paths, v2_real, anchor_price, anchor_date,
            cfg.forecast_horizon,
            f"v2.1 trailing-momentum — {vol_label}\n{anchor_date.date()} (origin idx {origin_idx})",
            color="tab:red",
        )

    # Single legend at the top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99))

    fig.suptitle(
        f"Forecast distribution vs realized — fold {fold_idx}, "
        f"{v1_fold.paths.shape[1]} paths per forecast",
        y=1.02, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"== wrote {args.out}")


if __name__ == "__main__":
    main()
