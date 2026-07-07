"""Cross-source parity for nse_equities: jugaad vs nselib vs yfinance.

Mirrors scripts/data_pipelines/parity_check.py (us_equities) but for the
three NSE providers. Goal: confirm the chain doesn't silently inject
divergent data when it falls through, and document the known adjustment
divergence (yfinance "full" vs jugaad/nselib "none").

Per ticker:
  - overlap date count
  - close, volume: max rel diff jugaad↔nselib↔yfinance
  - adj_close: max rel diff jugaad↔yfinance (expected NON-zero on tickers
    with dividends since jugaad has no adjustment, yfinance does)

Each ticker is fetched DIRECTLY via each adapter into a tmp dir (NOT through
the cache) so the comparison is wire-data, not merge-cache resolutions.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import data_pipelines  # noqa: F401  — triggers .env load

from data_pipelines.domains.nse_equities.adapters.jugaad import JugaadAdapter
from data_pipelines.domains.nse_equities.adapters.nselib import NSElibAdapter
from data_pipelines.domains.nse_equities.adapters.yfinance import YFinanceNSEAdapter
from data_pipelines.errors import EmptyPayload
from data_pipelines.domains.nse_equities.schema import OHLCV_SCHEMA

# Mix: long-history mature, recent listing, multi-split, dividend-heavy.
TICKERS = [
    "NSE:RELIANCE",   # 2017 1:1 bonus issue (effective split); dividend-paying
    "NSE:TCS",        # no recent splits; large dividend payer
    "NSE:INFY",       # long history; dual-listed (NYSE + NSE)
    "NSE:HDFCBANK",   # large bank, regular dividends
    "NIFTY:50",       # index — only nselib + yfinance cover (jugaad broken)
]

START = date(2025, 1, 1)
END = date(2025, 4, 30)


def _fetch_one(adapter, ident: str, data_root: Path) -> pd.DataFrame | None:
    try:
        raw = adapter.fetch(ident, START, END, data_root=data_root)
    except EmptyPayload:
        return None
    df = adapter.parse(raw)
    return OHLCV_SCHEMA.normalize(
        df, source_column_map=adapter.source_column_map,
        provider=adapter.name, identifier=ident,
    ).set_index("date")


def _rel_diff(a: pd.Series, b: pd.Series) -> float:
    if len(a) == 0:
        return float("nan")
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-9)
    return float((a - b).abs().max() / scale)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nse_parity_"))
    print(f"data_root: {tmp}")
    print(f"range: {START} → {END}")
    print()

    print(f"{'identifier':14s}  {'common':>6s}  "
          f"{'close j↔n':>9s} {'close j↔y':>9s}  "
          f"{'vol j↔n':>8s} {'vol j↔y':>8s}  "
          f"{'adj j↔y':>8s}")
    print("-" * 80)

    j_a = JugaadAdapter()
    n_a = NSElibAdapter()
    y_a = YFinanceNSEAdapter()

    for ident in TICKERS:
        j = _fetch_one(j_a, ident, tmp)
        n = _fetch_one(n_a, ident, tmp)
        y = _fetch_one(y_a, ident, tmp)

        provs = {"j": j, "n": n, "y": y}
        avail = [k for k, v in provs.items() if v is not None and len(v) > 0]
        if not avail:
            print(f"{ident:14s}  no provider returned data")
            continue

        # Build common index across whichever providers returned data
        common = None
        for k in avail:
            common = provs[k].index if common is None else common.intersection(provs[k].index)
        n_common = len(common) if common is not None else 0

        def md(col, a_key, b_key):
            a, b = provs[a_key], provs[b_key]
            if a is None or b is None or n_common == 0:
                return float("nan")
            return _rel_diff(a[col].loc[common], b[col].loc[common])

        print(f"{ident:14s}  {n_common:>6d}  "
              f"{md('close','j','n'):>9.5f} {md('close','j','y'):>9.5f}  "
              f"{md('volume','j','n'):>8.4f} {md('volume','j','y'):>8.4f}  "
              f"{md('adj_close','j','y'):>8.4f}")


if __name__ == "__main__":
    main()
