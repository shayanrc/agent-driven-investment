"""yfinance fallback adapter for nse_equities (third tier).

Mirror of us_equities/adapters/yfinance.py, retargeted for NSE: appends `.NS`
to equity symbols and uses the yfinance index slug for `NIFTY:` identifiers.
Per V1.7 plan's "what not to do" list, the two yfinance adapters are kept as
separate files until a third domain wants the same library — two copies is
cheaper than a premature shared abstraction.

Adjustment quality: yfinance is the only NSE-side provider that offers an
adjusted close (`Adj Close`), so this is the only adapter that supplies
`adjustment_quality = "full"`. The merge precedence in
nse_equities/schema.merge_overlap_nse_equities preserves yfinance-sourced
adj_close even when later jugaad/nselib fetches overwrite the same dates
with QUALITY_NONE rows.

**Same yfinance OHLC caveat as us_equities:** auto_adjust=False controls
dividends only — splits are always back-applied. For NSE tickers with splits
(e.g., RELIANCE 1:1 2017), this adapter's raw OHLC will differ from
jugaad/nselib's by the cumulative split ratio. adj_close stays consistent.
Use adj_close for quantitative work.

Symbol-translation:
    NSE:RELIANCE → "RELIANCE.NS"
    BSE:RELIANCE → "RELIANCE.BO"   (best-effort; yfinance has spotty BO coverage)
    NIFTY:50     → "^NSEI"          (via config.yfinance_index_slugs)
"""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.nse_equities.config import NSEEquitiesConfig
from data_pipelines.domains.nse_equities.registry import parse_identifier
from data_pipelines.domains.nse_equities.schema import QUALITY_FULL
from data_pipelines.errors import EmptyPayload, ProviderError
from data_pipelines.raw_store import write_raw_atomic
from data_pipelines.retry import RetryPolicy, call_with_retry

DOMAIN_NAME = "nse_equities"

YFINANCE_COLUMN_MAP = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class YFinanceNSEAdapter(Adapter):
    name = "yfinance"
    source_column_map = YFINANCE_COLUMN_MAP
    extra_meta = {"adjustment_quality": QUALITY_FULL, "currency": "INR"}

    def __init__(self, config: NSEEquitiesConfig | None = None):
        self._config = config or NSEEquitiesConfig()

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self._config.retry_max_retries,
            base_delay_sec=self._config.retry_base_delay_sec,
            max_delay_sec=self._config.retry_max_delay_sec,
            jitter=self._config.retry_jitter,
        )

    def _yf_symbol(self, identifier: str) -> str | None:
        prefix, symbol = parse_identifier(identifier)
        if prefix == "NSE":
            return f"{symbol}.NS"
        if prefix == "BSE":
            return f"{symbol}.BO"
        if prefix == "NIFTY":
            return self._config.yfinance_index_slugs.get(symbol)
        return None

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
        if symbol is None:
            raise EmptyPayload(self.name, identifier)

        try:
            import yfinance as yf
        except ImportError as e:
            raise ProviderError(self.name, identifier, "yfinance not installed") from e

        def _do_fetch() -> pd.DataFrame:
            try:
                return yf.Ticker(symbol).history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),  # yf end is exclusive
                    auto_adjust=False,
                    actions=False,
                )
            except Exception as e:
                raise ProviderError(self.name, identifier, f"yfinance error: {e}") from e

        df = call_with_retry(
            _do_fetch, self._retry_policy(),
            provider=self.name, identifier=identifier,
        )

        if df is None or len(df) == 0:
            raise EmptyPayload(self.name, identifier)

        df = df.reset_index()
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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "Date" in df.columns:
            # yfinance returns tz-aware (IST for .NS). Strip tz, take date.
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(
                "Asia/Kolkata"
            ).dt.tz_localize(None).astype("datetime64[ns]")
            df["Date"] = df["Date"].dt.normalize()
            df = df.sort_values("Date").reset_index(drop=True)
        return df

    def health_check(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return self._config.yfinance_enabled
        except ImportError:
            return False
