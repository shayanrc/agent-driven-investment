"""Cross-source parity: Tiingo vs yfinance on overlapping dates.

Goal: confirm both providers return the same numbers on the same dates (within
expected rounding) so the chain fall-through doesn't silently inject divergent
data into the cache.

Audits per ticker:
  - Overlap date count
  - close, adj_close, volume: mean rel diff, max rel diff, max abs diff
  - Flag any rel diff > 1% on adj_close (split/dividend adjustment mismatch)
  - Flag any rel diff > 0.1% on raw close (data error)

Each ticker is fetched DIRECTLY via each adapter into a tmp dir (NOT through
the cache) so we compare wire data, not merge-cache resolutions.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

import data_pipelines  # noqa: F401  — triggers .env load

from data_pipelines.domains.us_equities.adapters.tiingo import TiingoAdapter
from data_pipelines.domains.us_equities.adapters.yfinance import YFinanceAdapter
from data_pipelines.domains.us_equities.schema import OHLCV_SCHEMA

# Pick a mix: long-history mature, recent IPO, split-heavy, foreign-listed.
TICKERS = [
    "NASDAQ:AAPL",   # long, multiple splits (2014 7-for-1, 2020 4-for-1)
    "NASDAQ:MSFT",   # long, no recent splits
    "NASDAQ:NVDA",   # long, several splits incl. 10-for-1 in 2024
    "NASDAQ:META",   # medium, no splits
    "NASDAQ:TSLA",   # medium, splits (2020 5-for-1, 2022 3-for-1)
]

# Use a recent window where both providers definitely have data.
START = date(2020, 1, 2)
END = date(2024, 12, 31)


def fetch_via(adapter, identifier: str, tmp: Path) -> pd.DataFrame:
    raw = adapter.fetch(identifier, start=START, end=END, data_root=tmp)
    df = adapter.parse(raw)
    df = OHLCV_SCHEMA.normalize(
        df, source_column_map=getattr(adapter, "source_column_map", None),
        provider=adapter.name, identifier=identifier,
    )
    OHLCV_SCHEMA.validate(df, provider=adapter.name, identifier=identifier)
    return df


def compare(t_df: pd.DataFrame, y_df: pd.DataFrame) -> dict:
    """Inner-join on date and produce per-column divergence stats."""
    merged = t_df.merge(y_df, on="date", how="inner", suffixes=("_t", "_y"))
    if len(merged) == 0:
        return {"overlap_rows": 0}

    stats: dict = {"overlap_rows": int(len(merged))}
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        a = merged[f"{col}_t"].astype("float64")
        b = merged[f"{col}_y"].astype("float64")
        abs_diff = (a - b).abs()
        denom = ((a.abs() + b.abs()) / 2).replace(0, pd.NA)
        rel_diff = (abs_diff / denom).fillna(0.0)
        stats[col] = {
            "max_abs_diff": float(abs_diff.max()),
            "max_rel_diff_pct": float(rel_diff.max() * 100),
            "mean_rel_diff_pct": float(rel_diff.mean() * 100),
            "n_disagree_gt_1pct": int((rel_diff > 0.01).sum()),
        }
    return stats


def verdict(col_stats: dict, col: str, threshold_pct: float) -> str:
    if col_stats.get("max_rel_diff_pct", 0) > threshold_pct:
        return f"FAIL (max {col_stats['max_rel_diff_pct']:.3f}% > {threshold_pct}% threshold)"
    return f"ok (max {col_stats['max_rel_diff_pct']:.4f}%)"


def main() -> int:
    print(f"Parity check: Tiingo vs yfinance, window {START} → {END}")
    print(f"Tickers: {', '.join(TICKERS)}")
    print()

    tiingo = TiingoAdapter()
    yfin = YFinanceAdapter()

    overall_fails: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for ticker in TICKERS:
            print(f"=== {ticker} ===")
            try:
                t_df = fetch_via(tiingo, ticker, tmp / "tiingo")
                y_df = fetch_via(yfin, ticker, tmp / "yfinance")
            except Exception as e:
                print(f"  fetch error: {type(e).__name__}: {e}")
                overall_fails.append(f"{ticker}: fetch error")
                continue

            print(f"  tiingo:   {len(t_df):5d} rows, {t_df['date'].min().date()} .. {t_df['date'].max().date()}")
            print(f"  yfinance: {len(y_df):5d} rows, {y_df['date'].min().date()} .. {y_df['date'].max().date()}")

            stats = compare(t_df, y_df)
            print(f"  overlap rows: {stats['overlap_rows']}")
            if stats["overlap_rows"] == 0:
                print("  no overlap — skipping")
                overall_fails.append(f"{ticker}: no overlap")
                continue

            # Thresholds: raw OHLC should match to <0.1%; adj_close to <1%.
            for col, th in [
                ("close", 0.1), ("open", 0.1), ("high", 0.1), ("low", 0.1),
                ("adj_close", 1.0),
            ]:
                v = verdict(stats[col], col, th)
                marker = "❌" if "FAIL" in v else " "
                print(f"  {marker} {col:10s}: mean_rel={stats[col]['mean_rel_diff_pct']:.4f}%  "
                      f"max_rel={stats[col]['max_rel_diff_pct']:.4f}%  "
                      f"n_>1pct={stats[col]['n_disagree_gt_1pct']:4d}  {v}")
                if "FAIL" in v:
                    overall_fails.append(f"{ticker}/{col}: {v}")
            # Volume: large divergences are common (regular vs total). Report only.
            vs = stats["volume"]
            print(f"    volume:    mean_rel={vs['mean_rel_diff_pct']:.4f}%  "
                  f"max_rel={vs['max_rel_diff_pct']:.4f}%  (informational)")
            print()

    print("=" * 60)
    if not overall_fails:
        print("VERDICT: PARITY HOLDS — sources agree within tolerances on all tickers")
        return 0
    print("VERDICT: divergences found — review before treating sources as interchangeable")
    for f in overall_fails:
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
