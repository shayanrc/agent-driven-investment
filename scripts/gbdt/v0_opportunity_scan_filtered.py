"""V0.3 — drawdown-filtered opportunity scan (NIFTY 50).

Same grid as v0.2 ({up, down} × {5, 10, 20, 30, 50%} × {10, 20, 50, 100 days}),
but filters out events where the price moved *against* the target direction by
more than half the threshold BEFORE the target was reached.

Rule:
- UP   event clean iff min(adj_close in (t, t_breach]) > (1 - thr/2) * adj_close[t]
  i.e., the path to the +thr breach never dipped below -thr/2 first.
- DOWN event clean iff max(adj_close in (t, t_breach]) < (1 + thr/2) * adj_close[t]
  i.e., the path to the -thr breach never rallied above +thr/2 first.

This is a path-honesty metric on the price series — NOT a strategy backtest.
No positions, no PnL, no transaction costs (project-wide anti-rule). The
"clean rate" is the fraction of origins whose forward price path reached the
target without a half-target excursion in the wrong direction first.

Output: per-stock + pooled aggregates with both raw (= v0.2) and clean counts,
plus the filter ratio (= filtered / raw).

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
OUTPUT_JSON = REPO_ROOT / "results/gbdt/data/_v0_opportunity_scan_filtered_data.json"

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


def _thr_key(thr: float) -> str:
    return f"{int(round(thr * 100)):02d}"


def scan_stock_horizon(close: np.ndarray, horizon: int) -> dict | None:
    n = len(close)
    if n < horizon + 1:
        return None

    windows = sliding_window_view(close[1:], horizon)  # (n_origins, horizon)
    n_origins = len(windows)
    origin_close = close[:n_origins]
    sentinel = horizon + 1  # used as "never breached" position marker

    cells: dict[str, dict[str, dict]] = {d: {} for d in DIRECTIONS}

    for thr in THRESHOLDS:
        # ---- UP direction ----
        target_up = windows >= (origin_close[:, None] * (1.0 + thr))
        adverse_up = windows <= (origin_close[:, None] * (1.0 - thr / 2.0))
        ev_up = target_up.any(axis=1)
        adv_up = adverse_up.any(axis=1)
        # first-index per row (0..horizon-1); sentinel if never
        tgt_idx_up = np.where(ev_up, target_up.argmax(axis=1), sentinel)
        adv_idx_up = np.where(adv_up, adverse_up.argmax(axis=1), sentinel)
        # clean = target hit AND adverse never hit before target
        clean_up = ev_up & (adv_idx_up > tgt_idx_up)
        n_raw_up = int(ev_up.sum())
        n_clean_up = int(clean_up.sum())
        n_filtered_up = n_raw_up - n_clean_up
        cells["up"][_thr_key(thr)] = {
            "horizon": horizon,
            "threshold": thr,
            "n_origins": int(n_origins),
            "n_events_raw": n_raw_up,
            "n_events_clean": n_clean_up,
            "n_events_filtered": n_filtered_up,
            "raw_rate": n_raw_up / n_origins,
            "clean_rate": n_clean_up / n_origins,
            "filter_ratio": (n_filtered_up / n_raw_up) if n_raw_up else None,
        }

        # ---- DOWN direction ----
        target_dn = windows <= (origin_close[:, None] * (1.0 - thr))
        adverse_dn = windows >= (origin_close[:, None] * (1.0 + thr / 2.0))
        ev_dn = target_dn.any(axis=1)
        adv_dn = adverse_dn.any(axis=1)
        tgt_idx_dn = np.where(ev_dn, target_dn.argmax(axis=1), sentinel)
        adv_idx_dn = np.where(adv_dn, adverse_dn.argmax(axis=1), sentinel)
        clean_dn = ev_dn & (adv_idx_dn > tgt_idx_dn)
        n_raw_dn = int(ev_dn.sum())
        n_clean_dn = int(clean_dn.sum())
        n_filtered_dn = n_raw_dn - n_clean_dn
        cells["down"][_thr_key(thr)] = {
            "horizon": horizon,
            "threshold": thr,
            "n_origins": int(n_origins),
            "n_events_raw": n_raw_dn,
            "n_events_clean": n_clean_dn,
            "n_events_filtered": n_filtered_dn,
            "raw_rate": n_raw_dn / n_origins,
            "clean_rate": n_clean_dn / n_origins,
            "filter_ratio": (n_filtered_dn / n_raw_dn) if n_raw_dn else None,
        }

    return cells


def aggregate_cell(per_stock: dict, direction: str, thr_key: str, horizon: int) -> dict | None:
    raw_rates: list[float] = []
    clean_rates: list[float] = []
    filter_ratios: list[float] = []
    pooled_origins = 0
    pooled_raw = 0
    pooled_clean = 0
    for info in per_stock.values():
        cell = info["horizons"].get(str(horizon), {}).get(direction, {}).get(thr_key)
        if cell is None:
            continue
        raw_rates.append(cell["raw_rate"])
        clean_rates.append(cell["clean_rate"])
        if cell["filter_ratio"] is not None:
            filter_ratios.append(cell["filter_ratio"])
        pooled_origins += cell["n_origins"]
        pooled_raw += cell["n_events_raw"]
        pooled_clean += cell["n_events_clean"]
    if not raw_rates:
        return None
    return {
        "direction": direction,
        "threshold": float(thr_key) / 100.0,
        "horizon": horizon,
        "n_stocks": len(raw_rates),
        "pooled_n_origins": pooled_origins,
        "pooled_n_events_raw": pooled_raw,
        "pooled_n_events_clean": pooled_clean,
        "pooled_n_events_filtered": pooled_raw - pooled_clean,
        "pooled_raw_rate": pooled_raw / pooled_origins,
        "pooled_clean_rate": pooled_clean / pooled_origins,
        "pooled_filter_ratio": ((pooled_raw - pooled_clean) / pooled_raw) if pooled_raw else None,
        "per_stock_clean_rate_median": float(np.percentile(clean_rates, 50)),
        "per_stock_clean_rate_q25": float(np.percentile(clean_rates, 25)),
        "per_stock_clean_rate_q75": float(np.percentile(clean_rates, 75)),
        "per_stock_filter_ratio_median": (
            float(np.percentile(filter_ratios, 50)) if filter_ratios else None
        ),
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
            "n_tickers_in_universe": len(tickers),
            "directions": list(DIRECTIONS),
            "thresholds": THRESHOLDS,
            "horizons_trading_days": HORIZONS,
            "price_field": "adj_close",
            "filter_rule": {
                "up": "event clean iff min(adj_close in (t, t_breach]) > (1 - threshold/2) * adj_close[t]",
                "down": "event clean iff max(adj_close in (t, t_breach]) < (1 + threshold/2) * adj_close[t]",
            },
        },
        "n_tickers_with_data": len(per_stock),
        "tickers_no_data": no_data,
        "aggregate": aggregate,
        "per_stock": per_stock,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(headline, indent=2))

    print(
        f"NIFTY 50 — v0.3 drawdown-filtered opportunity scan: "
        f"{len(DIRECTIONS)} directions × {len(THRESHOLDS)} thresholds × {len(HORIZONS)} horizons"
    )
    print(f"  Filter rule: event clean iff price never moved > threshold/2 against the target before reaching it.")
    print(f"  Tickers with cached data: {len(per_stock)}/{len(tickers)}")
    if no_data:
        print(f"  No cached data ({len(no_data)}): {no_data}")
    print()

    for direction in DIRECTIONS:
        print(f"  ─── {direction.upper()} — raw → clean rate (pooled across stocks) ───")
        header = "  thr\\H " + "  ".join(f"{h:>14}d" for h in HORIZONS)
        print(header)
        for thr in THRESHOLDS:
            key = _thr_key(thr)
            row = [f"{int(thr*100):>3}%  "]
            for h in HORIZONS:
                cell = aggregate[direction][key][str(h)]
                if cell is None:
                    row.append(f"{'n/a':>15} ")
                    continue
                raw = cell["pooled_raw_rate"]
                clean = cell["pooled_clean_rate"]
                row.append(f"{raw:>6.4f}→{clean:>6.4f} ")
            print("  " + "".join(row))
        print()

        print(f"  ─── {direction.upper()} — filter ratio (= filtered/raw, pooled) ───")
        print(header)
        for thr in THRESHOLDS:
            key = _thr_key(thr)
            row = [f"{int(thr*100):>3}%  "]
            for h in HORIZONS:
                cell = aggregate[direction][key][str(h)]
                if cell is None or cell["pooled_filter_ratio"] is None:
                    row.append(f"{'n/a':>15} ")
                    continue
                row.append(f"{cell['pooled_filter_ratio']:>15.4f} ")
            print("  " + "".join(row))
        print()

    print(f"  Output JSON: {OUTPUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
