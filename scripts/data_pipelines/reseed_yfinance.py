"""Force-re-seed an equities domain from yfinance so `close` is split-adjusted
(V5 fix, approach B — generalizes reseed_nse_yfinance.py to us_equities too).

yfinance's `Close` (auto_adjust=False) is split-adjusted but NOT dividend-adjusted
— exactly the price-return basis gbdt features/target have always used. This
forces the yfinance adapter, clears each ticker, and re-fetches, overwriting the
split-unadjusted bars (jugaad for NSE, Tiingo/Stooq raw close for US).

  uv run python -m scripts.data_pipelines.reseed_yfinance --domain us --ticker NASDAQ:AAPL
  uv run python -m scripts.data_pipelines.reseed_yfinance --domain us
"""
import argparse
import shutil
import sqlite3
import time
from datetime import date
from pathlib import Path

import pandas as pd

from data_pipelines.cache import processed_db_path
from data_pipelines.dispatch import fetch

DATA_ROOT = Path("data")
START, END = date(2010, 1, 1), date(2026, 7, 6)


def _yf_only(self, identifier, gap_size_trading_days, has_cache):
    return [self._adapters["yfinance"]] if "yfinance" in self._adapters else []


def _load_domain(name: str):
    if name == "nse":
        import data_pipelines.domains.nse_equities as m
        m.NSEDomain.chain_for_gap = _yf_only
        return m.get_domain(), "nse_equities_data"
    if name == "us":
        import data_pipelines.domains.us_equities as m
        m.USEquitiesDomain.chain_for_gap = _yf_only
        return m.get_domain(), "us_equities_data"
    raise SystemExit(f"unknown domain {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["nse", "us"])
    ap.add_argument("--ticker")
    args = ap.parse_args()
    domain, table = _load_domain(args.domain)

    def _db():
        return sqlite3.connect(processed_db_path(DATA_ROOT))

    def reseed_one(ident):
        con = _db()
        saved = pd.read_sql(f"SELECT * FROM {table} WHERE ticker = ?", con,
                            params=(ident,))
        con.execute(f"DELETE FROM {table} WHERE ticker = ?", (ident,))
        con.commit(); con.close()
        try:
            df = fetch(ident, START, END, data_root=str(DATA_ROOT))
        except Exception as e:
            con = _db(); saved.to_sql(table, con, if_exists="append", index=False)
            con.commit(); con.close()
            return f"FAIL ({str(e)[:45]}) — restored"
        if df is None or len(df) == 0:
            con = _db(); saved.to_sql(table, con, if_exists="append", index=False)
            con.commit(); con.close()
            return "FAIL empty — restored"
        mv = df.sort_values("date")["close"].pct_change().abs().dropna()
        return f"OK {len(df)} rows" + ("" if (mv <= 0.35).all() else " [spiky?]")

    if args.ticker:
        tickers = [args.ticker]
    else:
        from data_pipelines.cache import list_cached_identifiers
        tickers = list_cached_identifiers(DATA_ROOT, domain)
        print(f"re-seeding {len(tickers)} cached {args.domain}_equities tickers")

    backup = processed_db_path(DATA_ROOT).with_suffix(f".db.v5reseed_{args.domain}_bak")
    if not backup.exists():
        shutil.copy2(processed_db_path(DATA_ROOT), backup)
        print(f"backed up processed.db -> {backup}")

    ok = fails = spiky = 0
    for i, tk in enumerate(tickers):
        msg = reseed_one(tk)
        if msg.startswith("FAIL"):
            fails += 1
            print(f"  [{i+1}/{len(tickers)}] {tk}: {msg}", flush=True)
        else:
            ok += 1
            if "spiky" in msg:
                spiky += 1
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(tickers)} (ok {ok}, fail {fails}, spiky {spiky})",
                  flush=True)
        time.sleep(1.2)
    print(f"\n=== done {args.domain}: ok {ok}, fail {fails}, spiky {spiky} / {len(tickers)} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
