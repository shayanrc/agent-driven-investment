"""V0.1 — rolling-window opportunity scan.

For each NIFTY 50 stock and each horizon H in {10, 20, 50, 100} trading days,
count rolling-origin events where ``max(adj_close in (t, t+H]) >= 1.10 * adj_close[t]``.
Output: per-stock + pooled-across-stocks base rates + first-breach lag percentiles.

Reads directly from ``data/processed.db`` (the data_pipelines NSE cache). Reports
which tickers in the universe have no cached data so the gap is visible to the seed
follow-up. See ``docs/gbdt/V0_INVESTIGATION_PLAN.md``.

Note: this script writes its headline JSON in place at the path below; re-running
it against a drifted cache will silently overwrite the committed snapshot. Back up
or rename the prior JSON first if you want to compare across runs.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = REPO_ROOT / "configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml"
CACHE_DB = REPO_ROOT / "data/processed.db"
OUTPUT_JSON = REPO_ROOT / "results/gbdt/data/_v0_opportunity_scan_data.json"

HORIZONS = [10, 20, 50, 100]
THRESHOLD = 0.10


def load_tickers() -> list[str]:
    with UNIVERSE_PATH.open() as f:
        u = yaml.safe_load(f)
    return list(u["tickers"])


def load_close(con: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, adj_close FROM nse_equities_data "
        "WHERE ticker = ? AND adj_close IS NOT NULL "
        "ORDER BY date",
        con,
        params=(ticker,),
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def scan_one_horizon(close: np.ndarray, horizon: int, threshold: float) -> dict | None:
    n = len(close)
    if n < horizon + 1:
        return None
    # windows[t] = close[t+1 : t+horizon+1]; shape (n_origins, horizon)
    windows = sliding_window_view(close[1:], horizon)
    n_origins = len(windows)
    origin_close = close[:n_origins]
    breach = windows >= (origin_close[:, None] * (1.0 + threshold))
    events = breach.any(axis=1)
    n_events = int(events.sum())
    # first-breach lag (trading days from origin); argmax → 0 if all False, mask with events
    first_lag = breach.argmax(axis=1) + 1
    first_lag_event = first_lag[events]
    return {
        "horizon": horizon,
        "n_origins": int(n_origins),
        "n_events": n_events,
        "base_rate": n_events / n_origins,
        "lag_p25": float(np.percentile(first_lag_event, 25)) if n_events else None,
        "lag_p50": float(np.percentile(first_lag_event, 50)) if n_events else None,
        "lag_p75": float(np.percentile(first_lag_event, 75)) if n_events else None,
    }


def aggregate_horizon(per_stock: dict, horizon: int) -> dict | None:
    rates: list[float] = []
    pooled_origins = 0
    pooled_events = 0
    for info in per_stock.values():
        r = info["horizons"].get(str(horizon))
        if r is None:
            continue
        rates.append(r["base_rate"])
        pooled_origins += r["n_origins"]
        pooled_events += r["n_events"]
    if not rates:
        return None
    return {
        "horizon": horizon,
        "n_stocks": len(rates),
        "pooled_n_origins": pooled_origins,
        "pooled_n_events": pooled_events,
        "pooled_base_rate": pooled_events / pooled_origins,
        "per_stock_rate_min": float(min(rates)),
        "per_stock_rate_q25": float(np.percentile(rates, 25)),
        "per_stock_rate_median": float(np.percentile(rates, 50)),
        "per_stock_rate_q75": float(np.percentile(rates, 75)),
        "per_stock_rate_max": float(max(rates)),
    }


def main() -> int:
    tickers = load_tickers()
    con = sqlite3.connect(str(CACHE_DB))

    per_stock: dict[str, dict] = {}
    no_data: list[str] = []

    for ticker in tickers:
        df = load_close(con, ticker)
        if df.empty:
            no_data.append(ticker)
            continue
        close = df["adj_close"].to_numpy()
        per_stock[ticker] = {
            "n_rows": int(len(df)),
            "date_min": df["date"].min().strftime("%Y-%m-%d"),
            "date_max": df["date"].max().strftime("%Y-%m-%d"),
            "horizons": {str(h): scan_one_horizon(close, h, THRESHOLD) for h in HORIZONS},
        }

    con.close()

    aggregate = {str(h): aggregate_horizon(per_stock, h) for h in HORIZONS}

    headline = {
        "scan_spec": {
            "universe": "nifty50",
            "universe_path": str(UNIVERSE_PATH.relative_to(REPO_ROOT)),
            "n_tickers_in_universe": len(tickers),
            "horizons_trading_days": HORIZONS,
            "threshold": THRESHOLD,
            "direction": "up",
            "price_field": "adj_close",
            "event_def": "max(adj_close in (t, t+H]) >= (1+threshold) * adj_close[t]",
        },
        "n_tickers_with_data": len(per_stock),
        "tickers_no_data": no_data,
        "aggregate_per_horizon": aggregate,
        "per_stock": per_stock,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(headline, indent=2))

    print(
        f"NIFTY 50 — v0.1 opportunity scan: "
        f"≥{int(THRESHOLD * 100)}% up over horizons {HORIZONS} trading days "
        f"(price = adj_close, event = max in (t, t+H])"
    )
    print(f"  Tickers with cached data: {len(per_stock)}/{len(tickers)}")
    if no_data:
        print(f"  No cached data ({len(no_data)}): {no_data}")
    print()
    print(
        f"  {'H':>4} {'n_stocks':>9} {'pool_orig':>10} {'pool_event':>10} "
        f"{'pool_rate':>10} {'med_rate':>9} {'q25':>7} {'q75':>7} {'med_lag':>8}"
    )
    for h in HORIZONS:
        a = aggregate[str(h)]
        if a is None:
            print(f"  {h:>4}: no data")
            continue
        # median first-breach lag across all stocks (median of medians)
        lags = [
            info["horizons"][str(h)]["lag_p50"]
            for info in per_stock.values()
            if info["horizons"].get(str(h)) and info["horizons"][str(h)]["lag_p50"] is not None
        ]
        med_lag = f"{float(np.median(lags)):.1f}" if lags else "n/a"
        print(
            f"  {h:>4} {a['n_stocks']:>9} {a['pooled_n_origins']:>10} {a['pooled_n_events']:>10} "
            f"{a['pooled_base_rate']:>10.4f} {a['per_stock_rate_median']:>9.4f} "
            f"{a['per_stock_rate_q25']:>7.4f} {a['per_stock_rate_q75']:>7.4f} {med_lag:>8}"
        )
    print()
    print(f"  Output JSON: {OUTPUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
