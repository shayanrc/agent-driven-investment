"""Build the ticker→company-name map for the /daily-predictions dashboard.

The us_equities cache stores no company names, so this fetches longName/shortName via yfinance
for every distinct ticker in the forward log → ``results/backtests/data/ticker_names.csv`` — a
small committed map the dashboard reads cache-only (no runtime network). Re-run to refresh as
new tickers enter the log; existing names are re-fetched (cheap, ~100 tickers).

    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python -m scripts.backtests.fetch_ticker_names
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "results/backtests/data/forward_predictions_log.csv"
OUT = ROOT / "results/backtests/data/ticker_names.csv"


def main() -> None:
    tickers = sorted(pd.read_csv(LOG).ticker.unique())
    rows = []
    for t in tickers:
        sym = t.split(":")[-1]
        try:
            info = yf.Ticker(sym).info
            name = info.get("longName") or info.get("shortName") or ""
        except Exception as e:  # noqa: BLE001 — best-effort; blank name falls back to the symbol
            name = ""
            print(f"  {t} ERR {type(e).__name__}", flush=True)
        rows.append({"ticker": t, "name": name})
        print(f"{t} → {name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(df)} tickers, {(df.name != '').sum()} named)")


if __name__ == "__main__":
    main()
