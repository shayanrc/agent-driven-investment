"""jugaad-data adapter for nse_equities (primary tier).

Wraps `jugaad_data.nse.stock_raw` for equities. Raw OHLCV from NSE's
report-detail API — no split/dividend adjustment, no IPO predecessor
backfill. `adjustment_quality = "none"`.

Index endpoint scope:
    niftyindices.com /Backpage.aspx changed shape mid-2026 — it now
    requires a `cinfo` JSON-string parameter; unmodified upstream sent
    flat fields and got HTTP 500, which surfaced as KeyError: 'd'
    downstream. **Fixed in our vendored jugaad-data** at
    vendor/jugaad-data/jugaad_data/nse/history.py (search for
    "LOCAL PATCH"); `jugaad_data.nse.index_df` works again.

    Even with the fix, this adapter still short-circuits NIFTY:
    identifiers with EmptyPayload: jugaad's index payload has no VOLUME
    field (only OHLC), and the canonical schema requires non-null int64
    volume. Letting the chain fall through to nselib (which provides
    TRADED_QTY) keeps the schema honest without synthesizing fake zeros.
    Re-enable jugaad for NIFTY: only if (a) nselib breaks, or (b) volume
    is made nullable in the schema. See V1_IMPLEMENTATION_PLAN.md
    §"Implementation findings — Known upstream limitations".

Symbol-translation:
    NSE:RELIANCE → "RELIANCE"  (jugaad takes bare NSE symbol)
    BSE:*        → EmptyPayload (jugaad is NSE-only; chain falls through)
    NIFTY:*      → EmptyPayload (no volume in payload — see above)

Date discipline (open Q12):
    jugaad returns timestamps like "2025-04-29 18:30:00" (IST midnight
    expressed as naive UTC — see jugaad_data/util.py). The IST trade date
    is one calendar day forward. We extract the trade date by adding the
    IST offset and taking the date component. Stored as naive
    datetime64[ns] at midnight — matches us_equities.

Raw payload format:
    JSON list of the per-row dicts that jugaad returns from `stock_raw`.
    Preserved verbatim for D8 reprocess determinism. parse() reads that
    JSON and applies the same column projection/casting jugaad does in
    its stock_df, then drops time and applies the IST shift.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.nse_equities.config import NSEEquitiesConfig
from data_pipelines.domains.nse_equities.registry import parse_identifier
from data_pipelines.domains.nse_equities.schema import QUALITY_NONE
from data_pipelines.errors import EmptyPayload, ProviderError
from data_pipelines.raw_store import write_raw_atomic
from data_pipelines.retry import RetryPolicy, call_with_retry

DOMAIN_NAME = "nse_equities"

# Source-native column → canonical schema. close has no adjusted-version from
# jugaad; populated to == close in parse(), per QUALITY_NONE semantics.
JUGAAD_COLUMN_MAP = {
    "DATE": "date",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "ADJ_CLOSE": "adj_close",
    "VOLUME": "volume",
}

# IST is UTC+5:30 year-round (no DST).
_IST_OFFSET = timedelta(hours=5, minutes=30)


class JugaadAdapter(Adapter):
    name = "jugaad"
    source_column_map = JUGAAD_COLUMN_MAP
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
            raise ProviderError(self.name, identifier, "jugaad requires explicit start and end")

        prefix, symbol = parse_identifier(identifier)
        if prefix != "NSE":
            # BSE: not covered (NSE-only lib). NIFTY: covered by nselib/yf;
            # jugaad's index endpoint is currently broken (see module docstring).
            raise EmptyPayload(self.name, identifier)

        try:
            from jugaad_data.nse import stock_raw
        except ImportError as e:
            raise ProviderError(self.name, identifier, "jugaad-data not installed") from e

        def _do_fetch():
            try:
                return stock_raw(symbol=symbol, from_date=start, to_date=end, series="EQ")
            except Exception as e:
                raise ProviderError(self.name, identifier, f"jugaad-data error: {e}") from e

        raw = call_with_retry(
            _do_fetch, self._retry_policy(),
            provider=self.name, identifier=identifier,
        )

        if not raw:
            raise EmptyPayload(self.name, identifier)

        payload = json.dumps(raw, default=str, separators=(",", ":")).encode("utf-8")

        return write_raw_atomic(
            data_root,
            provider=self.name,
            domain=DOMAIN_NAME,
            exchange=prefix,
            ticker=symbol,
            payload=payload,
            range_start=start,
            range_end=end,
            ext="json",
            timestamp=datetime.now(timezone.utc),
        )

    def parse(self, raw_path: Path) -> pd.DataFrame:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(raw)

        # jugaad's _stock translates series="EQ" → "ALL" upstream, so the
        # response contains EQ rows PLUS any other series the security trades
        # on for the same date (commonly "BL" / "BZ" block-deal series). Those
        # collapse into duplicate (date, ticker) rows when persisted to the
        # cache's primary key. Keep only the EQ rows — bonds/block-deals are
        # out of scope for daily OHLCV.
        if "CH_SERIES" in df.columns:
            df = df[df["CH_SERIES"] == "EQ"].copy()

        # jugaad's source columns of interest (mirrors stock_select_headers).
        # Keep only what canonical schema needs to avoid carrying junk through
        # normalization.
        df = df.rename(columns={
            "CH_TIMESTAMP": "DATE",
            "CH_OPENING_PRICE": "OPEN",
            "CH_TRADE_HIGH_PRICE": "HIGH",
            "CH_TRADE_LOW_PRICE": "LOW",
            "CH_CLOSING_PRICE": "CLOSE",
            "CH_TOT_TRADED_QTY": "VOLUME",
        })

        # DATE arrives as ISO datetime string ("2025-04-29T18:30:00.000Z" or
        # similar) representing IST-midnight-as-naive-UTC. Convert to the IST
        # trade date.
        df["DATE"] = pd.to_datetime(df["DATE"], utc=False, errors="coerce")
        # If tz-aware, strip; either way the value is "IST-midnight expressed
        # as wall-clock". Adding the IST offset gives us back the actual IST
        # trade date.
        if hasattr(df["DATE"].dt, "tz") and df["DATE"].dt.tz is not None:
            df["DATE"] = df["DATE"].dt.tz_localize(None)
        df["DATE"] = (df["DATE"] + _IST_OFFSET).dt.normalize()

        # jugaad has no adjusted close → mirror close (QUALITY_NONE semantics).
        df["ADJ_CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")

        # Numeric casts; volume → int after coercion. NaNs in volume become 0
        # so the canonical int64 schema doesn't choke (no observed NaNs in
        # production data, but defensive).
        for c in ["OPEN", "HIGH", "LOW", "CLOSE"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce").fillna(0).astype("int64")

        df = df.sort_values("DATE").reset_index(drop=True)
        # Only keep canonical-bound columns; Schema.normalize will rename.
        return df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "ADJ_CLOSE", "VOLUME"]]

    def health_check(self) -> bool:
        try:
            import jugaad_data  # noqa: F401
            return True
        except ImportError:
            return False
