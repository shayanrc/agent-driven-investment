"""Tiingo update adapter for us_equities.

REST endpoint:
    GET https://api.tiingo.com/tiingo/daily/<symbol>/prices
        ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&token=<key>

Returns JSON array of bars with full split-AND-dividend adjustment baked in.
Generous free tier (500 req/day). Used for routine incremental updates per
the chain config; gap > big_gap_threshold_days delegates to Stooq seed.

D6 — API key safety:
  - Key is read from os.environ[config.tiingo_api_key_env] at fetch time only.
  - MissingAPIKey raised BEFORE any network call if env var is unset.
  - Key NEVER appears in error messages, log lines, or the raw filename.
  - Raw file is the JSON body only — Tiingo does not echo the token in the
    response.

Index symbols (^SPX etc.) are NOT supported by this endpoint — Tiingo's index
data is a separate paid product. Asking Tiingo for an INDEX:* identifier
raises ProviderError immediately so the chain falls through to yfinance (or
Stooq on next seed).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.us_equities.config import USEquitiesConfig
from data_pipelines.domains.us_equities.registry import parse_identifier
from data_pipelines.domains.us_equities.schema import QUALITY_FULL
from data_pipelines.errors import (
    EmptyPayload,
    MissingAPIKey,
    ProviderError,
)
from data_pipelines.raw_store import write_raw_atomic

DOMAIN_NAME = "us_equities"

TIINGO_COLUMN_MAP = {
    "adjClose": "adj_close",
    # date, open, high, low, close, volume already canonical
}


class TiingoAdapter(Adapter):
    name = "tiingo"
    source_column_map = TIINGO_COLUMN_MAP
    extra_meta = {"adjustment_quality": QUALITY_FULL}

    # Process-level circuit breaker. Tiingo's free tier has a 50-unique-symbols/
    # hour limit; once hit, every subsequent fetch returns 429 and the
    # exponential-backoff retry loop wastes ~7s per ticker. After
    # _CIRCUIT_BREAKER_THRESHOLD consecutive 429s within a single process
    # we treat Tiingo as exhausted for _CIRCUIT_BREAKER_COOLDOWN_SEC seconds
    # and fail fast — letting the chain drop straight to yfinance.
    _CIRCUIT_BREAKER_THRESHOLD = 3
    _CIRCUIT_BREAKER_COOLDOWN_SEC = 3600  # one hour matches Tiingo's window
    _consecutive_429s: int = 0
    _circuit_open_until: float = 0.0

    def __init__(self, config: USEquitiesConfig | None = None):
        self._config = config or USEquitiesConfig()

    def _resolve_key(self) -> str:
        env = self._config.tiingo_api_key_env
        key = os.environ.get(env)
        if not key:
            raise MissingAPIKey(self.name, env)
        return key

    def _build_url(self, identifier: str, start: date, end: date) -> str:
        prefix, symbol = parse_identifier(identifier)
        if prefix == "INDEX":
            raise ProviderError(
                self.name, identifier,
                "Tiingo does not support INDEX symbols on the free-tier daily endpoint",
            )
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "format": "json",
        }
        return (
            f"{self._config.tiingo_base_url}/tiingo/daily/{symbol.lower()}/prices"
            f"?{urllib.parse.urlencode(params)}"
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
            raise ProviderError(
                self.name, identifier,
                "Tiingo requires explicit start and end dates",
            )

        # Circuit breaker: if we're in cool-down, fail fast.
        if time.time() < TiingoAdapter._circuit_open_until:
            remaining = int(TiingoAdapter._circuit_open_until - time.time())
            raise ProviderError(
                self.name, identifier,
                f"circuit breaker open (rate-limit cool-down, "
                f"{remaining}s remaining)",
            )

        key = self._resolve_key()
        url = self._build_url(identifier, start, end)

        payload = self._request_with_retry(url, key, identifier)
        if not _has_rows(payload):
            raise EmptyPayload(self.name, identifier)

        prefix, symbol = parse_identifier(identifier)
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

    def _request_with_retry(
        self, url: str, key: str, identifier: str
    ) -> bytes:
        """GET with exponential backoff on 429. Key is in Authorization header
        — D6 forbids embedding it in URL query strings (which would log in
        access logs / process listings).
        """
        last_exc: Exception | None = None
        for attempt in range(self._config.tiingo_max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "Authorization": f"Token {key}",
                        "User-Agent": "data_pipelines/0.1",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(
                    req, timeout=self._config.tiingo_timeout_sec
                ) as resp:
                    if resp.status != 200:
                        raise ProviderError(self.name, identifier, f"HTTP {resp.status}")
                    # Successful response — reset 429 streak.
                    TiingoAdapter._consecutive_429s = 0
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    TiingoAdapter._consecutive_429s += 1
                    if (TiingoAdapter._consecutive_429s
                            >= TiingoAdapter._CIRCUIT_BREAKER_THRESHOLD):
                        TiingoAdapter._circuit_open_until = (
                            time.time() + TiingoAdapter._CIRCUIT_BREAKER_COOLDOWN_SEC
                        )
                    if attempt < self._config.tiingo_max_retries:
                        time.sleep(2 ** attempt)
                        last_exc = e
                        continue
                if e.code == 401:
                    # Sanitize: error includes status only, never key material.
                    raise ProviderError(self.name, identifier, "HTTP 401 unauthorized") from e
                raise ProviderError(self.name, identifier, f"HTTP {e.code}") from e
            except urllib.error.URLError as e:
                raise ProviderError(self.name, identifier, f"URL error: {e.reason}") from e
            except TimeoutError as e:
                raise ProviderError(self.name, identifier, "timeout") from e

        raise ProviderError(
            self.name, identifier,
            f"HTTP 429 after {self._config.tiingo_max_retries} retries",
        ) from last_exc

    def parse(self, raw_path: Path) -> pd.DataFrame:
        records = json.loads(raw_path.read_text())
        if not isinstance(records, list):
            # Tiingo error responses are dicts; let it raise SchemaMismatch downstream
            # via Schema.normalize missing-columns check rather than fabricating data.
            return pd.DataFrame()
        df = pd.DataFrame.from_records(records)
        if "date" in df.columns:
            # Tiingo dates are ISO with time/tz suffix; pin to UTC midnight ns.
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
            # Stamp at midnight for consistency.
            df["date"] = df["date"].dt.normalize()
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def health_check(self) -> bool:
        try:
            self._resolve_key()
            return True
        except MissingAPIKey:
            return False


def _has_rows(payload: bytes) -> bool:
    if not payload:
        return False
    try:
        records = json.loads(payload)
    except json.JSONDecodeError:
        return False
    return isinstance(records, list) and len(records) > 0
