"""V0.2 — full direction × threshold × horizon opportunity scan (NIFTY 50).

Extends v0.1: for each NIFTY 50 stock and each cell in the grid
``DIRECTIONS × THRESHOLDS × HORIZONS``, count rolling-origin events.

Event definition:
- ``up``    : max(adj_close in (t, t+H]) >= (1 + threshold) * adj_close[t]
- ``down``  : min(adj_close in (t, t+H]) <= (1 - threshold) * adj_close[t]

Output: per-stock cells + pooled-across-stocks per (direction, threshold, horizon).
Reads ``data/processed.db`` directly. See ``docs/gbdt/V0_INVESTIGATION_PLAN.md``.

Note: this script writes its headline JSON in place; re-running it against a drifted
cache will silently overwrite the committed snapshot. Back up or rename the prior
JSON first if you want to compare across runs.
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
OUTPUT_JSON = REPO_ROOT / "results/gbdt/data/_v0_opportunity_scan_full_data.json"

DIRECTIONS = ("up", "down")
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]
HORIZONS = [10, 20, 50, 100]


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


def scan_stock_horizon(close: np.ndarray, horizon: int) -> dict | None:
    """Scan one (stock, horizon) — fills all (direction, threshold) cells from
    one sliding-window pass.
    """
    n = len(close)
    if n < horizon + 1:
        return None

    windows = sliding_window_view(close[1:], horizon)  # (n_origins, horizon)
    n_origins = len(windows)
    origin_close = close[:n_origins]

    cells: dict[str, dict[str, dict]] = {d: {} for d in DIRECTIONS}

    for thr in THRESHOLDS:
        # up: event if any window value >= (1+thr) * origin
        breach_up = windows >= (origin_close[:, None] * (1.0 + thr))
        events_up = breach_up.any(axis=1)
        n_up = int(events_up.sum())
        lag_up = breach_up.argmax(axis=1) + 1
        lag_up_evt = lag_up[events_up]
        cells["up"][_thr_key(thr)] = {
            "horizon": horizon,
            "threshold": thr,
            "n_origins": int(n_origins),
            "n_events": n_up,
            "base_rate": n_up / n_origins,
            "lag_p25": float(np.percentile(lag_up_evt, 25)) if n_up else None,
            "lag_p50": float(np.percentile(lag_up_evt, 50)) if n_up else None,
            "lag_p75": float(np.percentile(lag_up_evt, 75)) if n_up else None,
        }

        # down: event if any window value <= (1-thr) * origin
        breach_dn = windows <= (origin_close[:, None] * (1.0 - thr))
        events_dn = breach_dn.any(axis=1)
        n_dn = int(events_dn.sum())
        lag_dn = breach_dn.argmax(axis=1) + 1
        lag_dn_evt = lag_dn[events_dn]
        cells["down"][_thr_key(thr)] = {
            "horizon": horizon,
            "threshold": thr,
            "n_origins": int(n_origins),
            "n_events": n_dn,
            "base_rate": n_dn / n_origins,
            "lag_p25": float(np.percentile(lag_dn_evt, 25)) if n_dn else None,
            "lag_p50": float(np.percentile(lag_dn_evt, 50)) if n_dn else None,
            "lag_p75": float(np.percentile(lag_dn_evt, 75)) if n_dn else None,
        }

    return cells


def _thr_key(thr: float) -> str:
    return f"{int(round(thr * 100)):02d}"


def aggregate_cell(per_stock: dict, direction: str, thr_key: str, horizon: int) -> dict | None:
    rates: list[float] = []
    lags: list[float] = []
    pooled_origins = 0
    pooled_events = 0
    for info in per_stock.values():
        cell = info["horizons"].get(str(horizon), {}).get(direction, {}).get(thr_key)
        if cell is None:
            continue
        rates.append(cell["base_rate"])
        pooled_origins += cell["n_origins"]
        pooled_events += cell["n_events"]
        if cell["lag_p50"] is not None:
            lags.append(cell["lag_p50"])
    if not rates:
        return None
    return {
        "direction": direction,
        "threshold": float(thr_key) / 100.0,
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
        "median_lag_across_stocks": float(np.median(lags)) if lags else None,
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
            "horizons": {str(h): scan_stock_horizon(close, h) for h in HORIZONS},
        }

    con.close()

    aggregate: dict = {}
    for direction in DIRECTIONS:
        aggregate[direction] = {}
        for thr in THRESHOLDS:
            key = _thr_key(thr)
            aggregate[direction][key] = {
                str(h): aggregate_cell(per_stock, direction, key, h) for h in HORIZONS
            }

    headline = {
        "scan_spec": {
            "universe": "nifty50",
            "universe_path": str(UNIVERSE_PATH.relative_to(REPO_ROOT)),
            "n_tickers_in_universe": len(tickers),
            "directions": list(DIRECTIONS),
            "thresholds": THRESHOLDS,
            "horizons_trading_days": HORIZONS,
            "price_field": "adj_close",
            "event_def": {
                "up": "max(adj_close in (t, t+H]) >= (1+threshold) * adj_close[t]",
                "down": "min(adj_close in (t, t+H]) <= (1-threshold) * adj_close[t]",
            },
        },
        "n_tickers_with_data": len(per_stock),
        "tickers_no_data": no_data,
        "aggregate": aggregate,
        "per_stock": per_stock,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(headline, indent=2))

    # Stdout: one grid per direction
    print(
        f"NIFTY 50 — v0.2 full opportunity scan: "
        f"{len(DIRECTIONS)} directions × {len(THRESHOLDS)} thresholds × {len(HORIZONS)} horizons"
    )
    print(f"  Tickers with cached data: {len(per_stock)}/{len(tickers)}")
    if no_data:
        print(f"  No cached data ({len(no_data)}): {no_data}")
    print()

    for direction in DIRECTIONS:
        print(f"  ─── Direction: {direction.upper()} — pooled base rate grid (rows=threshold, cols=horizon) ───")
        header = "  thr\\H  " + "  ".join(f"{h:>9}d" for h in HORIZONS)
        print(header)
        for thr in THRESHOLDS:
            key = _thr_key(thr)
            row = [f"{int(thr*100):>4}%  "]
            for h in HORIZONS:
                cell = aggregate[direction][key][str(h)]
                row.append(f"{cell['pooled_base_rate']:>9.4f} " if cell else f"{'n/a':>9} ")
            print("  " + "".join(row))
        print()

        print(f"  ─── Direction: {direction.upper()} — median first-breach lag (days, across stocks) ───")
        print(header)
        for thr in THRESHOLDS:
            key = _thr_key(thr)
            row = [f"{int(thr*100):>4}%  "]
            for h in HORIZONS:
                cell = aggregate[direction][key][str(h)]
                if cell and cell["median_lag_across_stocks"] is not None:
                    row.append(f"{cell['median_lag_across_stocks']:>9.1f} ")
                else:
                    row.append(f"{'n/a':>9} ")
            print("  " + "".join(row))
        print()

    print(f"  Output JSON: {OUTPUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
