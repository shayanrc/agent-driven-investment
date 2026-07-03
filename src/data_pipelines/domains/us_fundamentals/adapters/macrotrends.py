"""Macrotrends adapter (primary) for the us_fundamentals domain.

Two scraped statement pages per ticker carry everything the schema needs,
quarterly, ~2011→now, in USD millions:

  - ``/stocks/charts/<slug>/income-statement?freq=Q`` — Revenue, Net Income,
    weighted shares (basic/diluted), EPS (basic/diluted).
  - ``/stocks/charts/<slug>/cash-flow-statement?freq=Q`` — Operating cash
    flow, PP&E change (negative outflow → capex).

Each page embeds its full table as a ``var originalData = [...]`` JSON blob;
one row per line item, one key per fiscal period end (macrotrends dates
periods at fiscal *month*-ends — WMT shows Jan 31 / Apr 30 / ... — which is
why parse() snaps onto the calendar quarter-end grid via the shared util).

Slug resolution: page URLs need ``<TICKER>/<company-slug>``; a wrong slug
301-redirects but drops the ``?freq=Q`` query, so we never rely on redirects.
Instead one request to ``ticker_search_list.php`` returns the full ticker→slug
map (~6.6k entries), memoized per adapter instance and landed as a raw audit
file under the pseudo-ticker ``_META``. Class shares are dot-spelled there
(``BRK.B``) vs the universes' dash spelling (``BRK-B``) — resolution tries
both.

Politeness: a process-wide minimum interval + jitter between macrotrends
requests (config knobs), a browser User-Agent, and shared retry/backoff on
transient failures. HTTP 403/429 raise ``ProviderError`` (retried with
backoff); a clean 200 page without ``originalData`` raises ``EmptyPayload``
(NOT retried — that ticker simply has no macrotrends financials, so dispatch
falls through to EDGAR immediately).

Raw payload: one JSON envelope per fetch —
``{"slug": ..., "urls": {...}, "pages": {page: <html>}}`` — because the
Adapter contract is one raw Path per fetch and reprocess-from-raw needs both
pages together. The HTML is stored as-downloaded (immutable audit trail); if
extraction logic improves, reprocess re-derives from these bytes.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.us_fundamentals.config import USFundamentalsConfig
from data_pipelines.domains.us_fundamentals.registry import parse_identifier
from data_pipelines.domains.us_fundamentals.schema import (
    US_FUNDAMENTALS_SCHEMA,
    dedupe_grid_collisions,
    snap_to_quarter_end,
)
from data_pipelines.errors import EmptyPayload, ProviderError
from data_pipelines.raw_store import write_raw_atomic
from data_pipelines.retry import RetryPolicy, call_with_retry

_log = logging.getLogger(__name__)

DOMAIN_NAME = "us_fundamentals"

SLUG_LIST_PATH = "/assets/php/ticker_search_list.php"
PAGES: tuple[str, ...] = ("income-statement", "cash-flow-statement")

# line-item label (HTML tags stripped) → canonical column
INCOME_FIELD_MAP = {
    "Revenue": "revenue",
    "Net Income": "net_income",
    "Basic Shares Outstanding": "shares_basic",
    "Shares Outstanding": "shares_diluted",  # macrotrends' diluted line
    "Basic EPS": "eps_basic",
    "EPS - Earnings Per Share": "eps_diluted",
}
CASHFLOW_FIELD_MAP = {
    "Cash Flow From Operating Activities": "ocf",
    # negative = cash outflow; capex = -ppe_change (see parse()).
    "Net Change In Property, Plant, And Equipment": "ppe_change",
}

_ORIGINAL_DATA_RE = re.compile(r"var originalData = (\[.*?\]);", re.S)
_DATE_KEY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TAG_RE = re.compile(r"<[^>]+>")


class _Throttle:
    """Process-wide minimum interval + jitter between requests (politeness)."""

    def __init__(self, min_interval_sec: float, jitter_sec: float):
        self._min = min_interval_sec
        self._jitter = jitter_sec
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_ok:
                time.sleep(self._next_ok - now)
            self._next_ok = (
                time.monotonic() + self._min + random.uniform(0.0, self._jitter)
            )


class MacrotrendsAdapter(Adapter):
    name = "macrotrends"
    source_column_map = None  # parse() returns canonical columns
    extra_meta = {"source": "macrotrends", "units": "usd_millions"}

    def __init__(self, config: USFundamentalsConfig | None = None):
        self._config = config or USFundamentalsConfig()
        self._throttle = _Throttle(
            self._config.macrotrends_min_interval_sec,
            self._config.macrotrends_jitter_sec,
        )
        self._slug_map_cache: dict[str, str] | None = None

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self._config.retry_max_retries,
            base_delay_sec=self._config.retry_base_delay_sec,
            max_delay_sec=self._config.retry_max_delay_sec,
            jitter=self._config.retry_jitter,
        )

    # --- fetch ---------------------------------------------------------------

    def fetch(
        self,
        identifier: str,
        start: date | None = None,
        end: date | None = None,
        *,
        data_root: Path,
    ) -> Path:
        _, symbol = parse_identifier(identifier)
        slug = self._resolve_slug(symbol, data_root, identifier)
        if slug is None:
            # Not a macrotrends-covered ticker — no point retrying; let the
            # chain fall through to EDGAR.
            raise EmptyPayload(self.name, identifier)

        pages: dict[str, str] = {}
        urls: dict[str, str] = {}
        for page in PAGES:
            url = (
                f"{self._config.macrotrends_base_url}/stocks/charts/"
                f"{slug}/{page}?freq=Q"
            )
            body = self._get(url, identifier)
            html = body.decode("utf-8", errors="replace")
            if not _ORIGINAL_DATA_RE.search(html):
                # 200 but no data table: ticker has a slug but no financials
                # on macrotrends (some ADRs/funds). Non-retryable.
                raise EmptyPayload(self.name, identifier)
            pages[page] = html
            urls[page] = url

        envelope = json.dumps(
            {"slug": slug, "urls": urls, "pages": pages}
        ).encode("utf-8")
        return write_raw_atomic(
            data_root,
            provider=self.name,
            domain=DOMAIN_NAME,
            exchange="-",
            ticker=symbol,
            payload=envelope,
            range_start=start or date(1990, 1, 1),
            range_end=end or datetime.now(timezone.utc).date(),
            ext="json",
            timestamp=datetime.now(timezone.utc),
        )

    # --- slug map ------------------------------------------------------------

    def _resolve_slug(
        self, symbol: str, data_root: Path, identifier: str,
    ) -> str | None:
        slug_map = self._slug_map(data_root, identifier)
        # Universe symbols are dash-normalized (BRK-B); macrotrends spells
        # class shares inconsistently — BRK.B (dot) but CWENA (dash removed).
        # Try as-is, dot variant, then dash-removed.
        for candidate in (
            symbol, symbol.replace("-", "."), symbol.replace("-", ""),
        ):
            slug = slug_map.get(candidate)
            if slug is None:
                continue
            if not slug.isascii():
                # A few macrotrends map entries carry mojibake (a U+FFFD
                # replacement char from mangled bytes, e.g. CAI, MURGY); the
                # slug can't form a valid URL. Treat as unresolved so the
                # chain falls through to EDGAR/yfinance rather than crashing
                # on urllib's ASCII request-line encode.
                _log.warning(
                    "macrotrends: skipping non-URL-safe slug %r for %s; "
                    "falling through the chain", slug, identifier,
                )
                return None
            return slug
        return None

    def _slug_map(self, data_root: Path, identifier: str) -> dict[str, str]:
        if self._slug_map_cache is None:
            url = self._config.macrotrends_base_url + SLUG_LIST_PATH
            payload = self._get(url, identifier)
            try:
                entries = json.loads(payload)
            except json.JSONDecodeError:
                raise ProviderError(
                    self.name, identifier, "slug map is not valid JSON"
                ) from None
            today = datetime.now(timezone.utc).date()
            try:
                # Audit copy of the map actually used for this run's slugs.
                write_raw_atomic(
                    data_root,
                    provider=self.name,
                    domain=DOMAIN_NAME,
                    exchange="-",
                    ticker="_META",
                    payload=payload,
                    range_start=today,
                    range_end=today,
                    ext="json",
                    timestamp=datetime.now(timezone.utc),
                )
            except FileExistsError:
                pass  # same-second re-fetch in one process; the memo has it
            self._slug_map_cache = {
                str(e["s"]).split("/", 1)[0].upper(): str(e["s"])
                for e in entries
                if isinstance(e, dict) and e.get("s")
            }
        return self._slug_map_cache

    # --- transport -----------------------------------------------------------

    def _get(self, url: str, identifier: str) -> bytes:
        def _do() -> bytes:
            self._throttle.wait()
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self._config.browser_user_agent,
                        "Accept": "*/*",
                        "Connection": "close",
                    },
                )
                with urllib.request.urlopen(
                    req, timeout=self._config.timeout_sec
                ) as resp:
                    if resp.status != 200:
                        raise ProviderError(
                            self.name, identifier, f"HTTP {resp.status}"
                        )
                    return resp.read()
            except urllib.error.HTTPError as e:
                raise ProviderError(
                    self.name, identifier, f"HTTP {e.code}"
                ) from None
            except urllib.error.URLError as e:
                raise ProviderError(
                    self.name, identifier, f"URL error: {e.reason}"
                ) from None
            except TimeoutError:
                raise ProviderError(self.name, identifier, "timeout") from None
            except UnicodeEncodeError:
                # A non-ASCII char reached the request line (slugs are
                # pre-filtered, but guard the transport too). Not retryable.
                raise ProviderError(
                    self.name, identifier, "non-ASCII URL"
                ) from None

        return call_with_retry(
            _do, self._retry_policy(), provider=self.name, identifier=identifier
        )

    # --- parse ---------------------------------------------------------------

    def parse(self, raw_path: Path) -> pd.DataFrame:
        doc = json.loads(raw_path.read_text())
        income = _extract_table(
            doc["pages"]["income-statement"], INCOME_FIELD_MAP
        )
        cashflow = _extract_table(
            doc["pages"]["cash-flow-statement"], CASHFLOW_FIELD_MAP
        )
        df = income.merge(cashflow, on="fiscal_period_end", how="outer")

        # PP&E change is a (negative) cash outflow; store capex positive.
        # NaN propagates: no PP&E line → capex NULL → fcf NULL (schema policy).
        df["capex"] = -df["ppe_change"]
        df["fcf"] = df["ocf"] - df["capex"]
        df = df.drop(columns=["ppe_change"])

        df["filed_date"] = pd.NaT  # macrotrends has no filing dates
        df["date"] = pd.to_datetime(
            [snap_to_quarter_end(d.date()) for d in df["fiscal_period_end"]]
        ).astype("datetime64[ns]")

        for col in US_FUNDAMENTALS_SCHEMA.column_names:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[US_FUNDAMENTALS_SCHEMA.column_names]
        return dedupe_grid_collisions(df)


def _extract_table(html: str, field_map: dict[str, str]) -> pd.DataFrame:
    """Pull ``originalData`` out of one statement page and pivot the mapped
    line items to columns keyed by ``fiscal_period_end``.

    Cells are mixed int / numeric-string / empty-string — coerced to float64
    (empty → NaN). Values are USD millions (shares in millions, EPS in USD)
    as published.
    """
    m = _ORIGINAL_DATA_RE.search(html)
    if not m:
        # fetch() validated presence; absence here means corrupt raw bytes.
        raise ValueError("macrotrends raw envelope has no originalData table")
    rows = json.loads(m.group(1))

    by_column: dict[str, dict[str, object]] = {}
    for row in rows:
        label = _TAG_RE.sub("", str(row.get("field_name", ""))).strip()
        canonical = field_map.get(label)
        if canonical is None:
            continue
        cells = {
            k: v for k, v in row.items() if _DATE_KEY_RE.match(str(k))
        }
        by_column[canonical] = cells

    all_dates = sorted({d for cells in by_column.values() for d in cells})
    out = pd.DataFrame({
        "fiscal_period_end": pd.to_datetime(all_dates).astype("datetime64[ns]"),
    })
    for canonical in field_map.values():
        cells = by_column.get(canonical, {})
        out[canonical] = pd.to_numeric(
            pd.Series([cells.get(d, None) for d in all_dates])
            .replace("", None),
            errors="coerce",
        ).astype("float64")
    return out
