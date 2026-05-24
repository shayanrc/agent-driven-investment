"""Stooq seed adapter for us_equities.

Stooq's CSV endpoint:
    https://stooq.com/q/d/l/?s=<symbol>.us&i=d[&apikey=<key>]   (equities)
    https://stooq.com/q/d/l/?s=<index_slug>&i=d[&apikey=<key>]  (indices)

Returns full available history — perfect for bulk seed, wasteful for
incremental updates (hence the big_gap_threshold gating in chain_for_gap).
Response is a plain CSV with columns: Date, Open, High, Low, Close, Volume.
No Adj Close. We set adj_close = close and tag the source as split_only (D4).

**API key is OPTIONAL.** Investigation during v1 smoke (2026-05-23, see commit
log) showed Stooq gates *some* IPs/regions but not all — yfinance from the
same network returned valid data while Stooq returned a 'Get your apikey'
help page (HTTP 200, Content-disposition: error.txt). For gated callers,
registration is free at https://stooq.com/q/d/?get_apikey (captcha only).

Behavior:
  - If env var STOOQ_API_KEY (or config.stooq_api_key_env) is set, the
    apikey URL param is appended. Otherwise the request goes without — most
    callers from residential IPs don't need it.
  - The response is then checked for the apikey-required help page; if
    detected (gating triggered for this caller, or key invalid),
    ProviderError is raised with an actionable message. The help page is
    NEVER written to the raw layer.

Failure modes:
  - Stooq returns the 'Get your apikey:' help page → ProviderError (set
    STOOQ_API_KEY to bypass, or check rate-limit status)
  - HTTP non-200 → ProviderError
  - 200 but empty / 1-line body (Stooq's "no data" response, e.g. delisted)
    → EmptyPayload
  - Network timeout → ProviderError

Key is read from env at fetch time only (D6); never logged, never embedded
in raw filenames or _meta.json.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.us_equities.config import USEquitiesConfig
from data_pipelines.domains.us_equities.registry import parse_identifier
from data_pipelines.domains.us_equities.schema import QUALITY_SPLIT_ONLY
from data_pipelines.errors import EmptyPayload, ProviderError
from data_pipelines.raw_store import write_raw_atomic

DOMAIN_NAME = "us_equities"

# Stooq → canonical column rename. Applied via Schema.normalize().
STOOQ_COLUMN_MAP = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class StooqAdapter(Adapter):
    name = "stooq"
    source_column_map = None  # parse() returns canonical names already
    extra_meta = {"adjustment_quality": QUALITY_SPLIT_ONLY}

    def __init__(self, config: USEquitiesConfig | None = None):
        self._config = config or USEquitiesConfig()

    def _resolve_key(self) -> str | None:
        """Read the optional Stooq API key from env. Returns None if unset —
        Stooq does not gate every IP, so a missing key is not an error.
        """
        return os.environ.get(self._config.stooq_api_key_env) or None

    def _build_url(self, identifier: str, key: str | None) -> str:
        prefix, symbol = parse_identifier(identifier)
        if prefix == "INDEX":
            slug = self._config.stooq_index_slugs.get(symbol)
            if slug is None:
                # Permissive fallback: lowercase the symbol verbatim.
                slug = symbol.lower()
        else:
            slug = f"{symbol.lower()}.us"
        params: dict[str, str] = {"s": slug, "i": "d"}
        if key:
            params[self._config.stooq_api_key_param] = key
        return f"{self._config.stooq_base_url}?{urllib.parse.urlencode(params)}"

    def fetch(
        self,
        identifier: str,
        start: date | None = None,
        end: date | None = None,
        *,
        data_root: Path,
    ) -> Path:
        key = self._resolve_key()
        url = self._build_url(identifier, key)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "data_pipelines/0.1"})
            with urllib.request.urlopen(req, timeout=self._config.stooq_timeout_sec) as resp:
                status = resp.status
                if status != 200:
                    raise ProviderError(self.name, identifier, f"HTTP {status}")
                payload = resp.read()
        except urllib.error.HTTPError as e:
            raise ProviderError(self.name, identifier, f"HTTP {e.code}") from e
        except urllib.error.URLError as e:
            raise ProviderError(self.name, identifier, f"URL error: {e.reason}") from e
        except TimeoutError as e:
            raise ProviderError(self.name, identifier, "timeout") from e

        if _is_apikey_required(payload):
            hint = (
                "invalid STOOQ_API_KEY" if key
                else "Stooq is gating this IP/region — set STOOQ_API_KEY "
                     "(register free at https://stooq.com/q/d/?get_apikey) "
                     "or retry from a different network"
            )
            raise ProviderError(
                self.name, identifier,
                f"Stooq returned API-key-required page ({hint})",
            )
        if not _looks_like_data(payload):
            raise EmptyPayload(self.name, identifier)

        prefix, symbol = parse_identifier(identifier)
        rs = start or date(1900, 1, 1)
        re_ = end or date.today()
        return write_raw_atomic(
            data_root,
            provider=self.name,
            domain=DOMAIN_NAME,
            exchange=prefix,
            ticker=symbol,
            payload=payload,
            range_start=rs,
            range_end=re_,
            ext="csv",
            timestamp=datetime.now(timezone.utc),
        )

    def parse(self, raw_path: Path) -> pd.DataFrame:
        text = raw_path.read_text()
        df = pd.read_csv(StringIO(text))
        # Rename to canonical names (so dispatch's schema.normalize handles dtype/order).
        df = df.rename(columns=STOOQ_COLUMN_MAP)
        if "adj_close" not in df.columns:
            # Stooq doesn't supply adj_close — set to close, tagged split_only via extra_meta.
            if "close" not in df.columns:
                # Will surface as SchemaMismatch downstream; raise here would be premature.
                pass
            else:
                df["adj_close"] = df["close"]
        # Sort ascending (Stooq returns ascending already, but enforce D3).
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def health_check(self) -> bool:
        # Stooq doesn't require a key for most callers, so health is
        # "module loads and config is valid". The actual gating only
        # surfaces on a real fetch.
        return True


def _looks_like_data(payload: bytes) -> bool:
    """Stooq's 'no data' response is the literal string b'No data\\n' or a
    single header line. Real data has multiple lines, header + at least one row.
    """
    if not payload:
        return False
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return False
    if text.lower().startswith("no data"):
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) >= 2


# Signals Stooq returns when the request lacks a valid API key. Multiple
# strings because the page wording is stable but worth detecting via a few
# anchors in case Stooq tweaks the copy.
_APIKEY_REQUIRED_SIGNALS = (
    "get your apikey",
    "get_apikey",
    "enter the captcha",
)


def _is_apikey_required(payload: bytes) -> bool:
    """True iff payload is Stooq's 'register for an API key' help page rather
    than a CSV response. Guards against silently storing the help page as raw
    and then surfacing a confusing schema mismatch downstream.
    """
    if not payload:
        return False
    text = payload.decode("utf-8", errors="replace").lower()
    return any(signal in text for signal in _APIKEY_REQUIRED_SIGNALS)
