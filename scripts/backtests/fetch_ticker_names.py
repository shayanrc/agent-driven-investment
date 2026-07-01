"""Fetch + refresh the ticker→company-name map in the DB (``us_equities_names`` table).

The us_equities cache stores no company names (its meta table is fetch bookkeeping only), so this
populates a dedicated ``us_equities_names(ticker, name, updated_utc)`` table in ``processed.db``
via yfinance ``longName``/``shortName``, for every distinct ticker in the forward-prediction log.
The ``/daily-predictions`` dashboard reads it cache-only (no runtime network).

Reliable + incremental — safe to re-run on any cadence:
  * default: fetch ONLY tickers missing a name (cheap; picks up new names as they enter the log)
  * ``--refresh-all``: re-fetch every logged ticker
  * ``--stale-days N``: also refresh names older than N days
  * retries transient failures; NEVER overwrites an existing good name with a blank/failed fetch
  * additive table — data_pipelines' seed/upsert never touches it, so it survives re-seeds

    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \\
      uv run python -m scripts.backtests.fetch_ticker_names [--refresh-all] [--stale-days N]
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "results/backtests/data/forward_predictions_log.csv"
DB = ROOT / "data/processed.db"
TABLE = "us_equities_names"


def _ensure_table(con: sqlite3.Connection) -> None:
    con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} "
                "(ticker TEXT PRIMARY KEY, name TEXT NOT NULL, updated_utc TEXT NOT NULL)")
    con.commit()


def _fetch_name(sym: str, retries: int = 3) -> str | None:
    """yfinance longName/shortName with bounded retries; None if all attempts fail/empty."""
    for attempt in range(retries):
        try:
            info = yf.Ticker(sym).info
            nm = info.get("longName") or info.get("shortName")
            if nm:
                return str(nm).strip()
        except Exception:  # noqa: BLE001 — transient network/parse; retry then give up
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def _is_stale(updated_utc: str, stale_days: int, now: datetime) -> bool:
    if stale_days <= 0 or not updated_utc:
        return False
    try:
        return (now - datetime.fromisoformat(updated_utc)).days >= stale_days
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-all", action="store_true", help="re-fetch every logged ticker")
    ap.add_argument("--stale-days", type=int, default=0, help="also refresh names older than N days")
    args = ap.parse_args()

    logged = sorted(pd.read_csv(LOG).ticker.unique())
    con = sqlite3.connect(DB)
    _ensure_table(con)
    existing = {t: (n, u) for t, n, u in con.execute(f"SELECT ticker, name, updated_utc FROM {TABLE}")}
    now = datetime.now(timezone.utc)
    todo = [t for t in logged if args.refresh_all or t not in existing
            or not existing[t][0] or _is_stale(existing[t][1], args.stale_days, now)]
    print(f"{len(logged)} logged tickers · {len(existing)} already in table · fetching {len(todo)}", flush=True)

    ok = 0
    for t in todo:
        nm = _fetch_name(t.split(":")[-1])
        if nm:
            con.execute(
                f"INSERT INTO {TABLE} (ticker, name, updated_utc) VALUES (?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET name=excluded.name, updated_utc=excluded.updated_utc",
                (t, nm, now.isoformat()))
            con.commit()
            ok += 1
            print(f"  {t} → {nm}", flush=True)
        else:
            kept = existing.get(t, ("", ""))[0]
            print(f"  {t} FAILED — kept existing: {kept or '—'}", flush=True)
    total = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    con.close()
    print(f"\nrefreshed {ok}/{len(todo)}; {TABLE} now holds {total} names", flush=True)


if __name__ == "__main__":
    main()
