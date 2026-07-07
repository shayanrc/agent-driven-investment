"""Re-forecast from the date the realized first crossed below the original
forecast's 50% band, and plot the new forecast + partial realized.

Reuses fold 75's tuned weights (the most recent walk-forward state) so the
new forecast is what production would have produced at the dip date.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analog_mc.config import Config
from analog_mc.data import load_close_series, load_returns
from analog_mc.features import compute_features
from analog_mc.simulate import forecast


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/analog_mc/20260520T045525Z")
    p.add_argument("--out", default="docs/analog_mc/figs/reforecast_from_dip.png")
    args = p.parse_args()

    run_dir = Path(args.run)
    cfg = Config.from_yaml(run_dir / "config.yaml")
    close = load_close_series(cfg.data_path, cfg.date_col, cfg.close_col)
    returns_s = load_returns(cfg)
    returns = returns_s.to_numpy()

    # 1. Reconstruct the original forecast's 50% band, find first below-q25 crossing.
    arr = np.load(run_dir / "folds/75/forecasts.npz")
    best_local = int(np.argmax(arr["origin_idx"]))
    orig_origin = int(arr["origin_idx"][best_local])
    fc_returns = arr["paths"][best_local]
    rl_returns = arr["realized"][best_local]
    p0_orig = float(close.iloc[orig_origin + 1])
    fc_prices_orig = p0_orig * np.exp(np.cumsum(fc_returns, axis=1))
    rl_prices_orig = p0_orig * np.exp(np.cumsum(rl_returns))
    q25_orig = np.quantile(fc_prices_orig, 0.25, axis=0)
    below = rl_prices_orig < q25_orig
    if not below.any():
        raise SystemExit("realized never went below the original 50% band — check the original plot")
    t_dip = int(np.argmax(below))
    new_origin = orig_origin + t_dip
    new_anchor_close_pos = new_origin + 1
    new_anchor_date = close.index[new_anchor_close_pos]
    print(f"original anchor: {close.index[orig_origin+1].date()} (origin_idx={orig_origin})")
    print(f"first below-band crossing: t={t_dip}, date {new_anchor_date.date()} (new origin_idx={new_origin})")
    print(f"close at new origin: {close.iloc[new_anchor_close_pos]:,.1f}")

    # 2. Compute features causally through the whole history.
    # Use fold 75's tuned weights (production state at this point).
    summary = __import__("json").loads((run_dir / "folds/75/summary.json").read_text())
    weights = np.array(summary["weights"], dtype=float)
    n_eff = float(summary["n_eff"])
    print(f"weights (from fold 75): {weights.tolist()}, n_eff={n_eff}")

    features = compute_features(
        returns_s,
        halflife=cfg.ewma_halflife,
        horizons=cfg.zscore_horizons,
        momentum_lookback=cfg.momentum_lookback,
    )

    # Candidate pool: any past index (the matcher filters out forward-block
    # overlaps + NaN features). Use the same range fold 75 trained on, just
    # extended through the new origin's history.
    candidate_idx = np.arange(0, new_origin)

    # Run a fresh forecast.
    rng = np.random.default_rng(cfg.random_seed)
    new_paths = forecast(
        origin_idx=new_origin,
        returns=returns,
        candidate_idx=candidate_idx,
        features=features,
        weights=weights,
        n_eff=n_eff,
        config=cfg,
        rng=rng,
    )
    print(f"new forecast paths shape: {new_paths.shape}")

    # 3. Integrate to prices.
    p0_new = float(close.iloc[new_anchor_close_pos])
    new_fc_prices = p0_new * np.exp(np.cumsum(new_paths, axis=1))

    # Available realized after the new anchor: bounded by data end.
    horizon = cfg.forecast_horizon
    realized_avail = min(horizon, len(returns) - new_origin - 1)
    new_rl_returns = returns[new_origin + 1 : new_origin + 1 + realized_avail]
    new_rl_prices = p0_new * np.exp(np.cumsum(new_rl_returns))
    print(f"realized available: {realized_avail} days (of {horizon} forecast horizon)")

    # 4. Plot.
    # Forecast spans 60 trading days; data only covers `realized_avail` of those.
    # Build the full 60-day forward axis by extending via business-day range
    # past the last available close.
    import pandas as pd
    available_forward = close.index[new_anchor_close_pos + 1 : new_anchor_close_pos + 1 + horizon]
    if len(available_forward) < horizon:
        last_known = available_forward[-1] if len(available_forward) > 0 else new_anchor_date
        extra = pd.bdate_range(last_known + pd.Timedelta(days=1), periods=horizon - len(available_forward))
        forward_dates = available_forward.append(extra)
    else:
        forward_dates = available_forward
    realized_dates = available_forward[:realized_avail]

    q05 = np.quantile(new_fc_prices, 0.05, axis=0)
    q25 = np.quantile(new_fc_prices, 0.25, axis=0)
    q50 = np.quantile(new_fc_prices, 0.50, axis=0)
    q75 = np.quantile(new_fc_prices, 0.75, axis=0)
    q95 = np.quantile(new_fc_prices, 0.95, axis=0)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    hist_start = max(0, new_anchor_close_pos - 30)
    hist_dates = close.index[hist_start : new_anchor_close_pos + 1]
    hist_prices = close.iloc[hist_start : new_anchor_close_pos + 1].to_numpy()
    ax.plot(hist_dates, hist_prices, color="black", lw=1.5, label="historical")

    ax.fill_between(forward_dates, q05, q95, color="tab:blue", alpha=0.15,
                    label="forecast 90% band")
    ax.fill_between(forward_dates, q25, q75, color="tab:blue", alpha=0.30,
                    label="forecast 50% band")
    ax.plot(forward_dates, q50, color="tab:blue", lw=1.6, label="forecast median")

    ax.plot(realized_dates, new_rl_prices, color="black", lw=1.8, label="realized")

    ax.axvline(new_anchor_date, color="grey", lw=0.5, ls=":")
    ax.scatter([new_anchor_date], [p0_new], color="black", s=20, zorder=5)

    ax.set_title(
        f"NASDAQ100 — re-forecast from {new_anchor_date.date()} "
        f"(first below-band crossing of the 2026-02-19 forecast)\n"
        f"Model: v2.4 Cell-D-s30 (re-using fold-75 weights), "
        f"realized covers {realized_avail}/60 forecast days"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("NASDAQ100 close")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nNew forecast — anchor {new_anchor_date.date()} close={p0_new:,.1f}")
    if realized_avail > 0:
        print(f"  realized at t={realized_avail-1}: {new_rl_prices[-1]:,.1f}")
        print(f"  fc 50% interval there:           [{q25[realized_avail-1]:,.1f}, {q75[realized_avail-1]:,.1f}]")
        print(f"  fc 90% interval there:           [{q05[realized_avail-1]:,.1f}, {q95[realized_avail-1]:,.1f}]")
        inside_50 = (new_rl_prices >= q25[:realized_avail]) & (new_rl_prices <= q75[:realized_avail])
        inside_90 = (new_rl_prices >= q05[:realized_avail]) & (new_rl_prices <= q95[:realized_avail])
        print(f"  realized in 50% band: {inside_50.sum()}/{realized_avail} days")
        print(f"  realized in 90% band: {inside_90.sum()}/{realized_avail} days")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
