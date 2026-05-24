"""nselib adapter for nse_equities (secondary tier).

Wraps `nselib.capital_market.price_volume_data` for equities and
`nselib.capital_market.index_data` for indices. Both fetch raw NSE bhav data
— no split/dividend adjustment, so `adjustment_quality = "none"`.

Symbol-translation:
    NSE:RELIANCE → "RELIANCE"      (nselib takes bare NSE symbol)
    BSE:*        → EmptyPayload    (nselib is NSE-only; chain falls through)
    NIFTY:50     → "NIFTY 50"      (via registry.NIFTY_INDEX_SLUGS)

Key data-format quirks discovered in smoke test:
    - Equity numeric fields come back as strings with Indian lakh-crore
      comma format ("2,54,80,745" = 25,480,745). parse() strips commas
      before numeric coercion.
    - Equity Turnover column name carries a unicode rupee suffix
      ("Turnover₹"). parse() normalizes both that and the plain "Turnover"
      that older nselib versions used.
    - Equity Date is "30-Apr-2025" (dd-mmm-yyyy text).
    - Index Date is "30-APR-2025" (same shape, uppercase month).
    - Index columns are *_INDEX_VAL suffixed (OPEN_INDEX_VAL, etc.) and
      use TRADED_QTY for volume (jugaad's index endpoint lacks this).

Raw payload format:
    The DataFrame nselib returns, serialized to CSV. nselib doesn't expose
    a raw-bytes hook so we capture the post-cleanup frame; this is the most
    faithful raw representation available given the library's API. parse()
    re-reads the CSV and applies canonicalization.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.nse_equities.config import NSEEquitiesConfig
from data_pipelines.domains.nse_equities.registry import (
    parse_identifier,
    resolve_nifty_slug,
)
from data_pipelines.domains.nse_equities.schema import QUALITY_NONE
from data_pipelines.errors import EmptyPayload, ProviderError
from data_pipelines.raw_store import write_raw_atomic
from data_pipelines.retry import RetryPolicy, call_with_retry

DOMAIN_NAME = "nselib"  # used in raw path; actual domain is nse_equities

# We rename source columns to the canonical schema names directly in parse(),
# so source_column_map stays identity here (Schema.normalize is a no-op rename).
NSELIB_COLUMN_MAP = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj_close": "adj_close",
    "volume": "volume",
}


def _clean_indian_comma_number(s: pd.Series) -> pd.Series:
    """'2,54,80,745' → 25480745 (float). Handles already-numeric Series too."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


class NSElibAdapter(Adapter):
    name = "nselib"
    source_column_map = NSELIB_COLUMN_MAP
    extra_meta = {"adjustment_quality": QUALITY_NONE, "currency": "INR"}

    def __init__(self, config: NSEEquitiesConfig | None = None):
        self._config = config or NSEEquitiesConfig()

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self._config.retry_max_retries,
            base_delay_sec=self._config.retry_base_delay_sec,
            max_delay_sec=self._config.retry_max_delay_sec,
            jitter=self._config.retry_jitter,
        )

    def fetch(
        self,
        identifier: str,
        start: date | None = None,
        end: date | None = None,
        *,
        data_root: Path,
    ) -> Path:
        if start is None or end is None:
            raise ProviderError(self.name, identifier, "nselib requires explicit start and end")

        prefix, symbol = parse_identifier(identifier)
        if prefix == "BSE":
            raise EmptyPayload(self.name, identifier)

        try:
            from nselib import capital_market as cm
        except ImportError as e:
            raise ProviderError(self.name, identifier, "nselib not installed") from e

        from_str = start.strftime("%d-%m-%Y")
        to_str = end.strftime("%d-%m-%Y")

        def _do_fetch() -> pd.DataFrame:
            try:
                if prefix == "NIFTY":
                    upstream = resolve_nifty_slug(symbol)
                    if upstream is None:
                        raise EmptyPayload(self.name, identifier)
                    return cm.index_data(index=upstream, from_date=from_str, to_date=to_str)
                # NSE equity
                return cm.price_volume_data(symbol=symbol, from_date=from_str, to_date=to_str)
            except EmptyPayload:
                raise
            except Exception as e:
                raise ProviderError(self.name, identifier, f"nselib error: {e}") from e

        df = call_with_retry(
            _do_fetch, self._retry_policy(),
            provider=self.name, identifier=identifier,
        )

        if df is None or len(df) == 0:
            raise EmptyPayload(self.name, identifier)

        payload = df.to_csv(index=False).encode("utf-8")

        return write_raw_atomic(
            data_root,
            provider=self.name,
            domain="nse_equities",
            exchange=prefix,
            ticker=symbol,
            payload=payload,
            range_start=start,
            range_end=end,
            ext="csv",
            timestamp=datetime.now(timezone.utc),
        )

    def parse(self, raw_path: Path) -> pd.DataFrame:
        df = pd.read_csv(raw_path)
        cols = set(df.columns)

        if "INDEX_NAME" in cols:
            # Index payload.
            df = df.rename(columns={
                "TIMESTAMP": "date",
                "OPEN_INDEX_VAL": "open",
                "HIGH_INDEX_VAL": "high",
                "LOW_INDEX_VAL": "low",
                "CLOSE_INDEX_VAL": "close",
                "TRADED_QTY": "volume",
            })
            # Index date is "30-APR-2025" or "30-Apr-2025" (case-tolerant).
            df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        else:
            # Equity payload — has the Indian-comma + unicode rupee quirks.
            # Normalize Turnover column name (₹ suffix dropped).
            df = df.rename(columns={c: "Turnover" for c in df.columns if c.startswith("Turnover")})
            df = df.rename(columns={
                "Date": "date",
                "OpenPrice": "open",
                "HighPrice": "high",
                "LowPrice": "low",
                "ClosePrice": "close",
                "TotalTradedQuantity": "volume",
            })
            df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = _clean_indian_comma_number(df[c])

        # No adjusted close from nselib — mirror close (QUALITY_NONE).
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")
        # Volume → int64 (drop NaN by 0 — defensive; well-formed payloads
        # never have NaN here).
        df["volume"] = df["volume"].fillna(0).astype("int64")
        df["date"] = df["date"].dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

    def health_check(self) -> bool:
        try:
            import nselib  # noqa: F401
            return True
        except ImportError:
            return False
