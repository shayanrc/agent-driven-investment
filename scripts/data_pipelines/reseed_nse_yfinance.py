"""Force-re-seed nse_equities from yfinance so `close`/`adj_close` are
split-adjusted (V5 fix, approach B).

The default chain is [jugaad, nselib, yfinance]; jugaad wins and supplies RAW
(split-unadjusted) OHLC, so every split injects a fake ~-50% return into the
gbdt features + target. yfinance back-applies splits to OHLC (verified: RELIANCE
close smooth across its 2024 2:1 split), so this script forces yfinance-only,
clears each ticker's cached rows, and re-fetches — overwriting the raw jugaad
bars with split-adjusted ones and setting adj_close to "full" quality.

Non-destructive guard: backs up processed.db once before the loop; a ticker whose
yfinance fetch fails is logged and its (now-cleared) rows are restored from the
backup so no data is lost.

Usage:
  uv run python -m scripts.data_pipelines.reseed_nse_yfinance --ticker NSE:RELIANCE   # validate one
  uv run python -m scripts.data_pipelines.reseed_nse_yfinance --universe nifty500     # full run
"""
import argparse
import shutil
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

import data_pipelines.domains.nse_equities as nsemod
from data_pipelines.cache import processed_db_path
from data_pipelines.dispatch import fetch

DATA_ROOT = Path("data")
START, END = date(2010, 1, 1), date(2026, 7, 6)
DATA_TABLE = "nse_equities_data"


# ---- force yfinance-only chain ---------------------------------------------
def _yf_only(self, identifier, gap_size_trading_days, has_cache):
    return [self._adapters["yfinance"]] if "yfinance" in self._adapters else []


nsemod.NSEDomain.chain_for_gap = _yf_only


def _db():
    return sqlite3.connect(processed_db_path(DATA_ROOT))


def _clear(identifier: str) -> pd.DataFrame:
    """Delete the ticker's processed rows; return them (for restore-on-failure)."""
    con = _db()
    saved = pd.read_sql(f"SELECT * FROM {DATA_TABLE} WHERE ticker = ?", con,
                        params=(identifier,))
    con.execute(f"DELETE FROM {DATA_TABLE} WHERE ticker = ?", (identifier,))
    con.commit()
    con.close()
    return saved


def _restore(saved: pd.DataFrame) -> None:
    if saved.empty:
        return
    con = _db()
    saved.to_sql(DATA_TABLE, con, if_exists="append", index=False)
    con.commit()
    con.close()


def _smooth_ok(df: pd.DataFrame) -> bool:
    """No |1-day close move| > 35% (split-spike proxy)."""
    if df is None or len(df) < 2:
        return False
    mv = df.sort_values("date")["close"].pct_change().abs().dropna()
    return bool((mv <= 0.35).all())


def reseed_one(identifier: str) -> tuple[str, str]:
    saved = _clear(identifier)
    try:
        df = fetch(identifier, START, END, data_root=str(DATA_ROOT))
    except Exception as e:
        _restore(saved)
        return identifier, f"FAIL fetch ({str(e)[:50]}) — restored"
    if df is None or len(df) == 0:
        _restore(saved)
        return identifier, "FAIL empty — restored"
    tag = "OK" if _smooth_ok(df) else "OK(still-spiky?)"
    return identifier, f"{tag} {len(df)} rows"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--universe", default=None)
    args = ap.parse_args()

    if args.ticker:
        tickers = [args.ticker]
    else:
        from data_pipelines.cache import list_cached_identifiers
        from data_pipelines.domains.nse_equities import get_domain
        tickers = list_cached_identifiers(DATA_ROOT, get_domain())
        print(f"re-seeding {len(tickers)} cached nse_equities tickers")

    backup = processed_db_path(DATA_ROOT).with_suffix(".db.v5reseed_bak")
    if not backup.exists():
        shutil.copy2(processed_db_path(DATA_ROOT), backup)
        print(f"backed up processed.db -> {backup}")

    ok = fails = spiky = 0
    for i, tk in enumerate(tickers):
        ident, msg = reseed_one(tk)
        if msg.startswith("FAIL"):
            fails += 1
            print(f"  [{i+1}/{len(tickers)}] {ident}: {msg}", flush=True)
        else:
            ok += 1
            if "spiky" in msg:
                spiky += 1
                print(f"  [{i+1}/{len(tickers)}] {ident}: {msg}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(tickers)} (ok {ok}, fail {fails}, spiky {spiky})",
                  flush=True)
        time.sleep(1.2)  # polite to yfinance

    print(f"\n=== done: ok {ok}, fail {fails}, still-spiky {spiky} / {len(tickers)} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
