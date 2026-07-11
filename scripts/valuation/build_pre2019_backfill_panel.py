"""Extend the NSE valuation panel back to ~2015 using screener.in annual
fundamentals (task #32).

The NSE XBRL fundamentals (in_fundamentals) only begin ~2019, so the NSE
valuation panel starts 2019-01-16 and F18-IN is 100% NaN before then. This script
builds a pre-2019 daily panel from the screener.in annual backfill and concatenates
it onto the existing 2019+ panel.

Basis / consistency (see _285 coverage-cliff investigation):
- revenue + net_income come from screener annual (validated 0% median discrepancy
  vs NSE XBRL; INR millions). These are period totals — basis-invariant.
- The annual figure IS the TTM at the fiscal year-end; it is stepped forward on
  `filed_date ~= fiscal_end + 75d` (SEBI 60-day audited-results rule + buffer, so
  strictly causal) via a backward merge_asof.
- `shares` = each ticker's EARLIEST split-adjusted `shares` from the existing
  panel, held CONSTANT. The existing panel's convention is `market_cap =
  as-traded adj_close x constant latest-basis shares` (verified: panel market_cap
  halves across RELIANCE's 2024 bonus). Holding the same latest-basis shares
  constant reproduces that convention exactly, so the concatenated series is
  CONTINUOUS at the 2019 boundary. The per-ticker split-basis quirk this inherits
  is pre-existing and identical on both sides.
- `adj_close` = as-traded (read_adj_close), same source as the 2019+ panel.
- screener EPS/shares are NOT used (bonus/split-adjusted basis, ~2x off on banks).
- ratios via the shared `compute_ratios` (earnings_yield, sales_yield, market_cap,
  pe/ps, eps_ttm, ...). fcf_ttm = NaN -> fcf_yield/p_fcf NaN (screener has no CF).

Caveat: cross-sectional yield ranks for tickers with many post-date splits carry
some basis error (inherited from the panel's convention). Acceptable + documented.

Run:  uv run python -m scripts.valuation.build_pre2019_backfill_panel
"""
from pathlib import Path

import numpy as np
import pandas as pd

from valuation.panel import nse_equities_identifier
from valuation.prices import read_adj_close
from valuation.ratios import compute_ratios

ROOT = Path(".")
OUT_DIR = ROOT / "results/valuation/data"
PANEL = OUT_DIR / "valuation_panel_nse.parquet"
SCREENER = OUT_DIR / "screener_annual_backfill.parquet"
BACKUP = OUT_DIR / "valuation_panel_nse.pre2019bak.parquet"
BACKFILL_START = "2014-06-01"   # FY2014 filings go active mid-2014 -> covers 2015 train start
FILING_LAG_DAYS = 75            # fiscal_end -> filed_date (SEBI 60d + buffer, causal)


def main() -> int:
    existing = pd.read_parquet(PANEL)
    existing["date"] = pd.to_datetime(existing["date"])
    boundary = existing["date"].min()
    cols = list(existing.columns)
    print(f"existing panel: {len(existing):,} rows, {existing['ticker'].nunique()} "
          f"tickers, starts {str(boundary)[:10]}")

    scr = pd.read_parquet(SCREENER)
    scr["fiscal_period_end"] = pd.to_datetime(scr["fiscal_period_end"]).astype("datetime64[ns]")
    scr["ticker_full"] = "INFUND:" + scr["ticker"].astype(str)
    scr["filed_date"] = scr["fiscal_period_end"] + pd.Timedelta(days=FILING_LAG_DAYS)
    scr = scr.dropna(subset=["revenue"])
    # only annual rows that can inform a pre-2019 date
    scr = scr[scr["filed_date"] < boundary]

    # back-cast shares: earliest split-adjusted `shares` per ticker (latest basis)
    shref = (existing.dropna(subset=["shares"]).sort_values("date")
             .groupby("ticker")["shares"].first())

    n_tk, n_skip_noshares, n_skip_noprice = 0, 0, 0
    out = []
    tickers = sorted(scr["ticker_full"].unique())
    for i, tk in enumerate(tickers):
        if tk not in shref.index:
            n_skip_noshares += 1
            continue
        shares_const = float(shref[tk])
        eid = nse_equities_identifier(tk)
        try:
            px = read_adj_close(eid, BACKFILL_START,
                                str((boundary - pd.Timedelta(days=1)).date()),
                                repo_root=ROOT, table="nse_equities_data")
        except Exception:
            px = None
        if px is None or len(px) == 0:
            n_skip_noprice += 1
            continue
        px = px.copy()
        px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
        px = px[px["date"] < boundary].sort_values("date")
        if px.empty:
            n_skip_noprice += 1
            continue

        ann = (scr[scr["ticker_full"] == tk]
               .sort_values("filed_date")[["filed_date", "fiscal_period_end",
                                           "revenue", "net_income"]])
        m = pd.merge_asof(px, ann.rename(columns={"filed_date": "date"}),
                          on="date", direction="backward")
        m = m.dropna(subset=["revenue"])   # drop dates before the first filing
        if m.empty:
            continue
        m["ticker"] = tk
        m["shares"] = shares_const
        m["revenue_ttm"] = m["revenue"]
        m["net_income_ttm"] = m["net_income"]
        m["fcf_ttm"] = np.nan
        m["revenue_q"] = np.nan
        m["asof_fiscal_period_end"] = m["fiscal_period_end"]
        m["asof_filed_date"] = m["fiscal_period_end"] + pd.Timedelta(days=FILING_LAG_DAYS)
        m = compute_ratios(m)
        for c in cols:
            if c not in m.columns:
                m[c] = np.nan
        out.append(m[cols])
        n_tk += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(tickers)}] built {n_tk} tickers", flush=True)

    pre = pd.concat(out, ignore_index=True)
    print(f"\npre-2019 rows built: {len(pre):,} across {pre['ticker'].nunique()} "
          f"tickers (skipped {n_skip_noshares} no-shares, {n_skip_noprice} no-price)")

    full = pd.concat([pre, existing], ignore_index=True)
    full = (full.drop_duplicates(subset=["ticker", "date"], keep="last")
            .sort_values(["ticker", "date"]).reset_index(drop=True))

    if not BACKUP.exists():
        existing.to_parquet(BACKUP)
        print(f"backed up original -> {BACKUP}")
    full.to_parquet(PANEL)
    print(f"extended panel: {len(full):,} rows -> {PANEL}")

    # coverage report
    d = pd.to_datetime(full["date"])
    ey = full["earnings_yield"]
    print("\nearnings_yield NaN fraction by year (extended panel):")
    by = ey.isna().groupby(d.dt.year).mean()
    print("  " + "  ".join(f"{int(y)}:{v:.2f}" for y, v in by.items() if 2014 <= y <= 2022))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
