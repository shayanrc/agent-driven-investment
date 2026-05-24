"""yfinance fallback adapter for us_equities (third tier).

Per docs/data_pipelines/goal.md, yfinance is known-unreliable (frequent
endpoint breakage, rate-limit instability, occasional adjusted-as-close
returns). It earns its slot ONLY as the last-resort fallback after Tiingo
fails on a specific ticker — never as a primary update source.

Wraps `yfinance.Ticker(symbol).history(start=..., end=..., auto_adjust=False)`.
auto_adjust=False keeps `Adj Close` as a separate column (split-AND-dividend
adjusted, matching us_equities D4 "full" semantics).

**Known yfinance limitation (v1 parity audit 2026-05-23):** despite
auto_adjust=False, yfinance's `Close` / `Open` / `High` / `Low` columns are
silently split-adjusted (the flag only controls dividend adjustment; splits
are always back-applied internally — there is no way to extract true raw
historical OHLC from yfinance). For tickers that have ever had a split,
this adapter's "raw" OHLC will differ from Tiingo's by the cumulative split
ratio. adj_close remains consistent across sources. Documented in the
us_equities schema docstring; consumers should prefer adj_close for
quantitative work.

Raw is parquet of the returned DataFrame — yfinance has no "native" wire
format we can preserve, so parquet captures dtypes + multi-index columns
losslessly for reprocess.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.us_equities.config import USEquitiesConfig
from data_pipelines.domains.us_equities.registry import parse_identifier
from data_pipelines.domains.us_equities.schema import QUALITY_FULL
from data_pipelines.errors import EmptyPayload, ProviderError
from data_pipelines.raw_store import write_raw_atomic

DOMAIN_NAME = "us_equities"

YFINANCE_COLUMN_MAP = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class YFinanceAdapter(Adapter):
    name = "yfinance"
    source_column_map = YFINANCE_COLUMN_MAP
    extra_meta = {"adjustment_quality": QUALITY_FULL}

    def __init__(self, config: USEquitiesConfig | None = None):
        self._config = config or USEquitiesConfig()

    def _yf_symbol(self, identifier: str) -> str:
        prefix, symbol = parse_identifier(identifier)
        if prefix == "INDEX":
            # yfinance uses ^SPX directly for indices.
            return symbol
        return symbol

    def fetch(
        self,
        identifier: str,
        start: date | None = None,
        end: date | None = None,
        *,
        data_root: Path,
    ) -> Path:
        if not self._config.yfinance_enabled:
            raise ProviderError(self.name, identifier, "yfinance disabled in config")
        if start is None or end is None:
            raise ProviderError(self.name, identifier, "yfinance requires explicit start and end")

        symbol = self._yf_symbol(identifier)
        try:
            import yfinance as yf
        except ImportError as e:
            raise ProviderError(self.name, identifier, "yfinance not installed") from e

        try:
            # yfinance end is exclusive; pad by 1 day to make our [start, end] inclusive.
            from datetime import timedelta
            df = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=False,
            )
        except Exception as e:  # yfinance raises a grab-bag of types
            raise ProviderError(self.name, identifier, f"yfinance error: {e}") from e

        if df is None or len(df) == 0:
            raise EmptyPayload(self.name, identifier)

        # Reset index so Date is a column (matches our canonical layout).
        df = df.reset_index()
        # Serialize to parquet bytes for the raw layer.
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)
        payload = buf.getvalue()

        prefix, ticker = parse_identifier(identifier)
        return write_raw_atomic(
            data_root,
            provider=self.name,
            domain=DOMAIN_NAME,
            exchange=prefix,
            ticker=ticker,
            payload=payload,
            range_start=start,
            range_end=end,
            ext="parquet",
            timestamp=datetime.now(timezone.utc),
        )

    def parse(self, raw_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(raw_path)
        # Flatten multi-index columns if present (yfinance >=0.2 returns
        # ("Open", "AAPL") tuples when given a list of tickers; single-ticker
        # mode is flat, but defend against the variant).
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
            df["Date"] = df["Date"].dt.normalize()
            df = df.sort_values("Date").reset_index(drop=True)
        return df

    def health_check(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return self._config.yfinance_enabled
        except ImportError:
            return False
