"""Post-seed validation report for the us_fundamentals cache.

Usage:
    uv run python -m scripts.data_pipelines.validate_us_fundamentals \
        [--universe all] [--data-root data]

Reports (read-only):
  - coverage: cached tickers vs the universe, per-provider breakdown
  - per-column NULL rates across all cached rows
  - rows-per-ticker distribution + history depth
  - spot-check rows (AAPL / WMT / GOOGL / BRK-B latest quarter)
  - tickers with zero cache (the failure table candidates)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from data_pipelines.domains.us_fundamentals.schema import (
    METRIC_COLUMNS,
    US_FUNDAMENTALS_SCHEMA,
)
from data_pipelines.domains.us_fundamentals.universe import load_universe

SPOT_TICKERS = ("FUND:AAPL", "FUND:WMT", "FUND:GOOGL", "FUND:BRK-B")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="all")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args(argv)

    db = Path(args.data_root) / "processed.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    universe = load_universe(args.universe)
    meta = pd.read_sql("SELECT * FROM us_fundamentals_meta", conn)
    data = pd.read_sql("SELECT * FROM us_fundamentals_data", conn)

    cached = set(meta["ticker"])
    missing = sorted(set(universe) - cached)
    extra = sorted(cached - set(universe))

    print(f"=== coverage ({args.universe}) ===")
    print(f"universe: {len(universe)}  cached: {len(cached)}  "
          f"missing: {len(missing)}  outside-universe cached: {len(extra)}")
    if missing:
        print("missing tickers:", ", ".join(m.split(":")[1] for m in missing))

    print("\n=== provider breakdown (per-ticker source set) ===")
    combos: dict[str, int] = {}
    for sj in meta["sources_json"]:
        provs = tuple(sorted({s["provider"] for s in json.loads(sj)}))
        combos["+".join(provs)] = combos.get("+".join(provs), 0) + 1
    for combo, n in sorted(combos.items(), key=lambda kv: -kv[1]):
        print(f"  {combo}: {n}")

    print("\n=== per-column NULL rates ===")
    n = len(data)
    print(f"total rows: {n}")
    for col in ("filed_date", *METRIC_COLUMNS):
        nulls = int(data[col].isna().sum())
        print(f"  {col}: {nulls} ({100.0 * nulls / max(n, 1):.1f}%)")

    print("\n=== rows per ticker ===")
    rc = meta["row_count"]
    print(f"  min={rc.min()}  p25={rc.quantile(0.25):.0f}  "
          f"median={rc.median():.0f}  p75={rc.quantile(0.75):.0f}  "
          f"max={rc.max()}")
    print("  earliest grid date:", data["date"].min(),
          " latest:", data["date"].max())

    print("\n=== spot checks (latest cached quarter) ===")
    cols = ["date", "fiscal_period_end", "filed_date",
            "revenue", "net_income", "ocf", "capex", "fcf", "eps_diluted"]
    for t in SPOT_TICKERS:
        rows = data.loc[data["ticker"] == t].sort_values("date")
        if rows.empty:
            print(f"  {t}: NOT CACHED")
            continue
        r = rows.iloc[-1]
        print(f"  {t}: " + "  ".join(f"{c}={r[c]}" for c in cols))

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
