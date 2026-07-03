"""Build the daily point-in-time valuation panel over the cached universe.

Reads fundamentals (us_fundamentals) + prices (us_equities) + split factors
(yfinance, cached to a local parquet so re-runs skip the network), computes the
point-in-time PE/PS/P-FCF panel, and writes:

  - results/valuation/data/valuation_panel.parquet   (full daily panel; gitignored)
  - results/valuation/data/valuation_latest.csv      (one row/ticker; checked in)

Usage:
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
      uv run python -m scripts.valuation.build_valuation_panel \
      [--tickers FUND:AAPL ...] [--start 2018-01-01] [--limit N] [--no-splits]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from data_pipelines.cache import list_cached_identifiers
from data_pipelines.domains.us_fundamentals import get_domain
from valuation.panel import build_panel, latest_snapshot
from valuation.prices import fetch_splits

OUT_DIR = Path("results/valuation/data")
SPLIT_CACHE = OUT_DIR / "split_factors.parquet"


def _split_provider(cache: dict[str, pd.Series]):
    def provider(symbol: str) -> pd.Series:
        if symbol not in cache:
            cache[symbol] = fetch_splits(symbol)
        return cache[symbol]
    return provider


def _load_split_cache() -> dict[str, pd.Series]:
    if not SPLIT_CACHE.is_file():
        return {}
    df = pd.read_parquet(SPLIT_CACHE)
    return {
        sym: pd.Series(g.set_index("date")["ratio"])
        for sym, g in df.groupby("symbol")
    }


def _save_split_cache(cache: dict[str, pd.Series]) -> None:
    rows = []
    for sym, s in cache.items():
        for d, r in s.items():
            rows.append({"symbol": sym, "date": pd.Timestamp(d), "ratio": float(r)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["symbol", "date", "ratio"]).to_parquet(SPLIT_CACHE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
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

    domain = get_domain()
    tickers = args.tickers or list_cached_identifiers(root / "data", domain)
    if args.limit:
        tickers = tickers[: args.limit]

    if args.no_splits:
        provider = lambda symbol: pd.Series(dtype="float64")
    else:
        split_cache = _load_split_cache()
        provider = _split_provider(split_cache)

    print(f"building valuation panel for {len(tickers)} ticker(s) "
          f"from {args.start}…")

    def progress(i, n, t):
        if i % 100 == 0 or i == n:
            print(f"  [{i}/{n}] {t}", flush=True)

    panel = build_panel(
        tickers, start=args.start, end=args.end, repo_root=root,
        splits_provider=provider, on_progress=progress,
    )
    if not args.no_splits:
        _save_split_cache(split_cache)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_DIR / "valuation_panel.parquet")
    snap = latest_snapshot(panel)
    snap.to_csv(OUT_DIR / "valuation_latest.csv", index=False)

    print(f"\n=== done ===")
    print(f"  panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers "
          f"→ {OUT_DIR/'valuation_panel.parquet'}")
    print(f"  latest snapshot: {len(snap)} tickers → {OUT_DIR/'valuation_latest.csv'}")
    if not snap.empty:
        finite = snap[snap["pe"].notna()]
        print(f"  PE coverage in snapshot: {len(finite)}/{len(snap)} "
              f"(median PE {finite['pe'].median():.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
