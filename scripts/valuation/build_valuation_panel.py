"""Build the daily point-in-time valuation panel over the cached universe.

Reads fundamentals (us_fundamentals / in_fundamentals) + prices (us_equities /
nse_equities) + split factors (yfinance, cached to a local parquet so re-runs
skip the network), computes the point-in-time PE/PS/P-FCF panel, and writes
(US default):

  - results/valuation/data/valuation_panel.parquet   (full daily panel; gitignored)
  - results/valuation/data/valuation_latest.csv      (one row/ticker; checked in)

With ``--domain nse`` the NSE cache/table are used and the outputs are the
``*_nse`` variants (``valuation_panel_nse.parquet`` / ``valuation_latest_nse.csv``).

Usage:
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
      uv run python -m scripts.valuation.build_valuation_panel \
      [--domain us|nse] [--tickers FUND:AAPL ...] [--start 2018-01-01] \
      [--limit N] [--no-splits]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from data_pipelines.cache import list_cached_identifiers
from valuation.panel import build_panel, latest_snapshot
from valuation.prices import (
    fetch_splits,
    nse_equities_identifier,
    us_equities_identifier,
)

OUT_DIR = Path("results/valuation/data")
SPLIT_CACHE = OUT_DIR / "split_factors.parquet"


def _load_split_cache(path: Path = SPLIT_CACHE) -> dict[str, pd.Series]:
    if not path.is_file():
        return {}
    df = pd.read_parquet(path)
    return {
        sym: pd.Series(g.set_index("date")["ratio"])
        for sym, g in df.groupby("symbol")
    }


def _save_split_cache(cache: dict[str, pd.Series], path: Path = SPLIT_CACHE) -> None:
    rows = []
    for sym, s in cache.items():
        for d, r in s.items():
            rows.append({"symbol": sym, "date": pd.Timestamp(d), "ratio": float(r)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["symbol", "date", "ratio"]).to_parquet(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("us", "nse"), default="us",
                    help="us (default): us_fundamentals + us_equities; "
                         "nse: in_fundamentals + nse_equities")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-splits", action="store_true",
                    help="skip split adjustment (faster; splits treated as none)")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)

    root = Path(args.repo_root)
    avail_gb = shutil.disk_usage(root).free / 1e9
    if avail_gb < 10:
        print(f"ABORT: only {avail_gb:.1f} G free (<10 G)")
        return 1

    if args.domain == "nse":
        from data_pipelines.domains.in_fundamentals import get_domain
        id_mapper = nse_equities_identifier
        price_table = "nse_equities_data"
        split_suffix = ".NS"
        panel_out = OUT_DIR / "valuation_panel_nse.parquet"
        latest_out = OUT_DIR / "valuation_latest_nse.csv"
        split_cache_path = OUT_DIR / "split_factors_nse.parquet"
    else:
        from data_pipelines.domains.us_fundamentals import get_domain
        id_mapper = us_equities_identifier
        price_table = "us_equities_data"
        split_suffix = ""
        panel_out = OUT_DIR / "valuation_panel.parquet"
        latest_out = OUT_DIR / "valuation_latest.csv"
        split_cache_path = SPLIT_CACHE

    domain = get_domain()
    tickers = args.tickers or list_cached_identifiers(root / "data", domain)
    if args.limit:
        tickers = tickers[: args.limit]

    if args.no_splits:
        provider = lambda symbol: pd.Series(dtype="float64")
    else:
        split_cache = _load_split_cache(split_cache_path)

        def provider(symbol: str) -> pd.Series:
            if symbol not in split_cache:
                split_cache[symbol] = fetch_splits(symbol, suffix=split_suffix)
            return split_cache[symbol]

    print(f"building {args.domain} valuation panel for {len(tickers)} ticker(s) "
          f"from {args.start}…")

    def progress(i, n, t):
        if i % 100 == 0 or i == n:
            print(f"  [{i}/{n}] {t}", flush=True)

    panel = build_panel(
        tickers, start=args.start, end=args.end, repo_root=root,
        splits_provider=provider, on_progress=progress,
        domain=domain, id_mapper=id_mapper, price_table=price_table,
    )
    if not args.no_splits:
        _save_split_cache(split_cache, split_cache_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_out)
    snap = latest_snapshot(panel)
    snap.to_csv(latest_out, index=False)

    print(f"\n=== done ===")
    print(f"  panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers "
          f"→ {panel_out}")
    print(f"  latest snapshot: {len(snap)} tickers → {latest_out}")
    if not snap.empty:
        finite = snap[snap["pe"].notna()]
        print(f"  PE coverage in snapshot: {len(finite)}/{len(snap)} "
              f"(median PE {finite['pe'].median():.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
