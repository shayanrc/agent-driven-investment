"""NSE corporate-filings adapter (primary and only v1 provider).

Two-stage fetch, one immutable envelope:

1. **Metadata** — the ``corporates-financial-results`` JSON on the bot-guarded
   main site (one request per ticker): every quarterly filing since ~2005 with
   exchange-timestamped ``filingDate``/``broadCastDate`` (native point-in-time
   truth — the US domain needed a separate EDGAR enrichment pass for this),
   ``consolidated``/standalone and ``audited`` flags, and the XBRL attachment
   URL.
2. **XBRL** — the Ind-AS results instances on ``nsearchives.nseindia.com``
   (the reliable static host), gap-bounded: only filings whose quarter end
   falls inside the requested ``[start, end]`` (and ≥ ``config.min_year``)
   are downloaded. Per-filing files make incremental fetches natural — unlike
   the US providers' full-history pages.

``parse()`` is a pure function of the envelope (D8): context-driven quarter
selection (duration 70–115 d ending on the filing's ``toDate`` — rejects YTD
contexts structurally, not by name), namespace-agnostic tag priority (the
taxonomy prefix drifts across eras), exact-unit scaling (absolute INR → INR
millions), consolidated-preferred / earliest-filed-wins basis selection.

Cookie handling: the API needs a session cookie from a warmup request to the
main site. The warmup itself may 403 — its ``Set-Cookie`` headers still
count, so cookies are extracted from error responses too (verified live from
this host, 2026-07-09).
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.in_fundamentals.config import InFundamentalsConfig
from data_pipelines.domains.in_fundamentals.registry import parse_identifier
from data_pipelines.domains.in_fundamentals.schema import (
    IN_FUNDAMENTALS_SCHEMA,
    dedupe_grid_collisions,
    snap_to_quarter_end,
)
from data_pipelines.errors import EmptyPayload, ProviderError
from data_pipelines.raw_store import write_raw_atomic
from data_pipelines.retry import RetryPolicy, call_with_retry

_log = logging.getLogger(__name__)

DOMAIN_NAME = "in_fundamentals"

# Only the quarterly stream carries the per-quarter P&L this domain stores.
_PERIOD_KEEP = "Quarterly"

# A quarter-duration context: 70-115 days (rejects half-year/YTD/annual).
_QUARTER_MIN_DAYS, _QUARTER_MAX_DAYS = 70, 115

# Tag priority per metric — namespace-agnostic LOCAL names ordered
# most-specific-first. Bank taxonomy fallbacks (interest earned / total
# income) are documented approximations; see V4 plan D9.
_TAG_PRIORITY: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromOperations",
        "TotalRevenueFromOperations",
        "RevenueFromInterestAndDividendOperations",
        "InterestEarned",
        "TotalIncome",
    ),
    "net_income": (
        "ProfitLossForPeriod",
        "ProfitLossForPeriodFromContinuingOperations",
        # Bank results taxonomy (live finding, HDFCBANK pilot): banks file
        # "...ForThePeriod" (with "The") and the minority-adjusted bottom
        # line. Owners-attributable preferred where both exist.
        "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
        "ProfitLossForThePeriod",
        "ProfitLossFromOrdinaryActivitiesAfterTax",
    ),
    "eps_basic": (
        "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        "BasicEarningsLossPerShare",
        "BasicEarningsLossPerShareFromContinuingOperations",
        # Bank results taxonomy (HDFCBANK pilot): "EarningsPerShare" without
        # "Loss"; headline EPS = after extraordinary items.
        "BasicEarningsPerShareAfterExtraordinaryItems",
        "BasicEarningsPerShareBeforeExtraordinaryItems",
    ),
    "eps_diluted": (
        "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        "DilutedEarningsLossPerShare",
        "DilutedEarningsLossPerShareFromContinuingOperations",
        "DilutedEarningsPerShareAfterExtraordinaryItems",
        "DilutedEarningsPerShareBeforeExtraordinaryItems",
    ),
}

# Instant facts used for the fallback share count (actual, not weighted —
# see parse(); weighted-consistent derivation from EPS is preferred).
_PAIDUP_TAG = "PaidUpValueOfEquityShareCapital"
_FACE_VALUE_TAG = "FaceValueOfEquityShareCapital"

# INR absolute → INR millions.
_INR_SCALE = 1e-6

_DATE_FMT = "%d-%b-%Y"           # "31-Dec-2024"
_FILED_FMTS = ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y")


def _parse_nse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), _DATE_FMT).date()
    except (ValueError, AttributeError):
        return None


def _parse_filed(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in _FILED_FMTS:
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _normalize_integrated_record(rec: dict) -> dict | None:
    """Map an integrated-filing record onto the classic record shape.

    SEBI's Integrated Filing regime replaced the classic quarterly stream
    from Q4 FY25; its records use different field names (``qe_Date``,
    ``broadcast_Date``, ``seq_Id``, ``consolidated`` = "Standalone" instead
    of "Non-Consolidated") and interleave governance filings with the
    financials. Pure function so ``parse()`` can re-derive it from the raw
    envelope (D8).
    """
    if rec.get("type") != "Integrated Filing- Financials":
        return None
    qe = rec.get("qe_Date")
    if not qe:
        return None
    return {
        "period": _PERIOD_KEEP,
        "toDate": qe,
        "consolidated": "Consolidated"
        if rec.get("consolidated") == "Consolidated" else "Non-Consolidated",
        "filingDate": rec.get("broadcast_Date"),
        "broadCastDate": rec.get("broadcast_Date"),
        "seqNumber": f"INT_{rec.get('seq_Id')}",
        "xbrl": rec.get("xbrl"),
        "audited": rec.get("audited"),
        "relatingTo": None,
    }


def _iter_quarterly_records(doc: dict):
    """Yield classic-shaped quarterly records from both metadata streams."""
    for rec in doc.get("metadata", []):
        if rec.get("period") == _PERIOD_KEEP:
            yield rec
    for raw in doc.get("integrated_metadata", []):
        rec = _normalize_integrated_record(raw)
        if rec is not None:
            yield rec


class _Throttle:
    """Process-wide min-interval + jitter between requests (thread-safe)."""

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


class NSEXbrlAdapter(Adapter):
    name = "nse_xbrl"
    source_column_map = None  # parse() returns canonical columns
    extra_meta = {"source": "nse_xbrl", "units": "inr_millions"}

    def __init__(self, config: InFundamentalsConfig | None = None):
        self._config = config or InFundamentalsConfig()
        self._api_throttle = _Throttle(
            self._config.api_min_interval_sec, self._config.api_jitter_sec
        )
        self._archive_throttle = _Throttle(
            self._config.archive_min_interval_sec,
            self._config.archive_jitter_sec,
        )
        self._cookiejar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookiejar)
        )
        self._session_warm = False

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self._config.retry_max_retries,
            base_delay_sec=self._config.retry_base_delay_sec,
            max_delay_sec=self._config.retry_max_delay_sec,
            jitter=self._config.retry_jitter,
        )

    # --- fetch ----------------------------------------------------------------

    def fetch(
        self,
        identifier: str,
        start: date | None = None,
        end: date | None = None,
        *,
        data_root: Path,
    ) -> Path:
        _, symbol = parse_identifier(identifier)
        records = self._fetch_metadata(symbol, identifier)
        integrated = self._fetch_integrated(symbol, identifier)
        if not records and not integrated:
            raise EmptyPayload(self.name, identifier)

        lo = start or date(self._config.min_year, 1, 1)
        hi = end or datetime.now(timezone.utc).date()
        doc_view = {"metadata": records, "integrated_metadata": integrated}
        wanted: list[dict] = []
        for rec in _iter_quarterly_records(doc_view):
            qe = _parse_nse_date(rec.get("toDate", ""))
            if qe is None or qe.year < self._config.min_year:
                continue
            # Older records carry an "-" placeholder instead of an XBRL URL
            # (pre-XBRL-era filings) — only real attachment links are
            # downloadable (live finding, 2026-07-09 pilot).
            link = str(rec.get("xbrl") or "")
            if lo <= qe <= hi and link.startswith("http"):
                wanted.append(rec)

        xbrl: dict[str, str] = {}
        for rec in wanted:
            key = str(rec.get("seqNumber", rec["xbrl"]))
            try:
                xbrl[key] = self._get_archive(rec["xbrl"], identifier).decode(
                    "utf-8", errors="replace"
                )
            except ProviderError as e:
                # Per-filing isolation: one broken attachment must not sink
                # the ticker. The metadata row still lands in the envelope
                # (audit trail); parse() skips records without XML.
                _log.warning(
                    "nse_xbrl: %s filing %s attachment failed (%s); skipping",
                    identifier, key, e,
                )
        if wanted and not xbrl:
            # Every attachment in this window is gone from the archives
            # (e.g. HDFCBANK's Sep-2018 filing) — that's an EmptyPayload,
            # not a provider outage: the dispatcher soft-fails the gap and
            # continues to the next one. A hard ProviderError here aborted
            # the whole multi-gap fetch (live finding, 2026-07-09 pilot).
            raise EmptyPayload(self.name, identifier)

        envelope = json.dumps({
            "symbol": symbol,
            "metadata": records,
            "integrated_metadata": integrated,
            "xbrl": xbrl,
        }).encode("utf-8")
        return write_raw_atomic(
            data_root,
            provider=self.name,
            domain=DOMAIN_NAME,
            exchange="-",
            ticker=symbol,
            payload=envelope,
            range_start=lo,
            range_end=hi,
            ext="json",
            timestamp=datetime.now(timezone.utc),
        )

    # --- transport ------------------------------------------------------------

    def _warm_session(self) -> None:
        """Prime the cookie jar from the main site. A 403 is fine — its
        Set-Cookie headers still establish the session (verified live)."""
        if self._session_warm:
            return
        req = urllib.request.Request(
            self._config.nse_base_url,
            headers={
                "User-Agent": self._config.browser_user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        try:
            with self._opener.open(req, timeout=self._config.timeout_sec):
                pass
        except urllib.error.HTTPError as e:
            # Error responses carry cookies too.
            self._cookiejar.extract_cookies(e, req)
        except urllib.error.URLError:
            pass  # the API request itself will surface a real outage
        self._session_warm = True

    def _fetch_metadata(self, symbol: str, identifier: str) -> list[dict]:
        url = (
            self._config.nse_base_url + self._config.results_api_path
            + "?" + urllib.parse.urlencode({
                "index": "equities", "symbol": symbol, "period": _PERIOD_KEEP,
            })
        )

        def _do() -> list[dict]:
            self._warm_session()
            self._api_throttle.wait()
            req = urllib.request.Request(url, headers={
                "User-Agent": self._config.browser_user_agent,
                "Accept": "application/json",
                "Referer": self._config.results_referer,
            })
            try:
                with self._opener.open(
                    req, timeout=self._config.timeout_sec
                ) as resp:
                    body = resp.read()
            except urllib.error.HTTPError as e:
                # 401/403 usually means a stale session — re-warm on retry.
                self._session_warm = False
                raise ProviderError(
                    self.name, identifier, f"HTTP {e.code}"
                ) from None
            except urllib.error.URLError as e:
                raise ProviderError(
                    self.name, identifier, f"URL error: {e.reason}"
                ) from None
            except TimeoutError:
                raise ProviderError(self.name, identifier, "timeout") from None
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._session_warm = False
                raise ProviderError(
                    self.name, identifier, "non-JSON response (bot guard?)"
                ) from None
            if not isinstance(payload, list):
                raise ProviderError(
                    self.name, identifier,
                    f"unexpected payload shape: {type(payload).__name__}",
                )
            return payload

        return call_with_retry(
            _do, self._retry_policy(), provider=self.name, identifier=identifier
        )

    def _fetch_integrated(self, symbol: str, identifier: str) -> list[dict]:
        """Integrated-filing metadata (the post-Q4-FY25 quarterly stream).

        Soft-fails to [] on provider errors: the classic stream still covers
        pre-2025 history, and an unfilled recent gap is reported honestly by
        the dispatcher's gap accounting rather than sinking the whole fetch.
        """
        url = (
            self._config.nse_base_url + self._config.integrated_api_path
            + "?" + urllib.parse.urlencode({
                "index": "equities", "symbol": symbol, "period": _PERIOD_KEEP,
            })
        )

        def _do() -> list[dict]:
            self._warm_session()
            self._api_throttle.wait()
            req = urllib.request.Request(url, headers={
                "User-Agent": self._config.browser_user_agent,
                "Accept": "application/json",
                "Referer": self._config.results_referer,
            })
            try:
                with self._opener.open(
                    req, timeout=self._config.timeout_sec
                ) as resp:
                    body = resp.read()
            except urllib.error.HTTPError as e:
                self._session_warm = False
                raise ProviderError(
                    self.name, identifier, f"HTTP {e.code} (integrated)"
                ) from None
            except urllib.error.URLError as e:
                raise ProviderError(
                    self.name, identifier, f"URL error: {e.reason}"
                ) from None
            except TimeoutError:
                raise ProviderError(self.name, identifier, "timeout") from None
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._session_warm = False
                raise ProviderError(
                    self.name, identifier,
                    "non-JSON integrated response (bot guard?)",
                ) from None
            data = payload.get("data") if isinstance(payload, dict) else None
            return data if isinstance(data, list) else []

        try:
            return call_with_retry(
                _do, self._retry_policy(),
                provider=self.name, identifier=identifier,
            )
        except ProviderError as e:
            _log.warning(
                "nse_xbrl: integrated-filing metadata failed for %s (%s); "
                "continuing with the classic stream only", identifier, e,
            )
            return []

    def _get_archive(self, url: str, identifier: str) -> bytes:
        def _do() -> bytes:
            self._archive_throttle.wait()
            req = urllib.request.Request(url, headers={
                "User-Agent": self._config.browser_user_agent,
                "Accept": "*/*",
                "Connection": "close",
            })
            try:
                with urllib.request.urlopen(
                    req, timeout=self._config.timeout_sec
                ) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # A permanently-missing attachment — retrying can't help
                    # (EmptyPayload is in the policy's do_not_retry set).
                    _log.info("nse_xbrl: 404 attachment %s", url)
                    raise EmptyPayload(self.name, identifier) from None
                raise ProviderError(
                    self.name, identifier, f"HTTP {e.code} ({url})"
                ) from None
            except urllib.error.URLError as e:
                raise ProviderError(
                    self.name, identifier, f"URL error: {e.reason}"
                ) from None
            except TimeoutError:
                raise ProviderError(self.name, identifier, "timeout") from None

        return call_with_retry(
            _do, self._retry_policy(), provider=self.name, identifier=identifier
        )

    def health_check(self) -> bool:
        try:
            self._session_warm = False
            self._warm_session()
            return True
        except Exception:
            return False

    # --- parse ----------------------------------------------------------------

    def parse(self, raw_path: Path) -> pd.DataFrame:
        doc = json.loads(raw_path.read_text())
        xbrl: dict[str, str] = doc.get("xbrl", {})

        rows: list[dict] = []
        for rec in _iter_quarterly_records(doc):
            key = str(rec.get("seqNumber", rec.get("xbrl", "")))
            xml_text = xbrl.get(key)
            if not xml_text:
                continue
            qe = _parse_nse_date(rec.get("toDate", ""))
            if qe is None:
                continue
            facts = _extract_quarter_facts(xml_text, qe)
            if facts is None:
                continue
            filed = _parse_filed(
                rec.get("filingDate") or rec.get("broadCastDate")
            )
            rows.append({
                "fiscal_period_end": qe,
                "filed_date": filed,
                "consolidated": 1.0
                if rec.get("consolidated") == "Consolidated" else 0.0,
                **facts,
            })
        if not rows:
            return _empty_frame()

        df = pd.DataFrame(rows)

        # Basis + revision selection per grid date: consolidated preferred
        # over standalone; within a basis the EARLIEST-FILED record wins
        # (as-first-published point-in-time; revisions/re-filings lose).
        # NaT filed dates sort last, so a dated record always beats an
        # undated duplicate.
        df["date"] = df["fiscal_period_end"].map(snap_to_quarter_end)
        df = df.sort_values(
            ["date", "consolidated", "filed_date"],
            ascending=[True, False, True],
            na_position="last",
            kind="mergesort",
        ).drop_duplicates(subset="date", keep="first")

        for c in ("date", "fiscal_period_end", "filed_date"):
            df[c] = pd.to_datetime(df[c])
        df = dedupe_grid_collisions(df)

        # Quarterly cash flow does not exist in India (SEBI: half-yearly).
        df["ocf"] = float("nan")
        df["capex"] = float("nan")
        df["fcf"] = float("nan")

        return (
            df[list(IN_FUNDAMENTALS_SCHEMA.column_names)]
            .sort_values("date")
            .reset_index(drop=True)
        )


# --- XBRL extraction (pure helpers) -------------------------------------------

_LOCAL_RE = re.compile(r"^\{.*\}")


def _local(tag: str) -> str:
    return _LOCAL_RE.sub("", tag)


def _extract_quarter_facts(xml_text: str, quarter_end: date) -> dict | None:
    """Extract the canonical metrics for the quarter ending ``quarter_end``.

    Returns None when the instance has no usable quarter context (e.g. an
    old-taxonomy filing this parser doesn't speak) — the caller skips the
    record honestly.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    # contextRef → (start, end) for durations, → (None, instant) for instants.
    contexts: dict[str, tuple[date | None, date | None]] = {}
    for ctx in root.iter():
        if _local(ctx.tag) != "context":
            continue
        cid = ctx.get("id")
        if not cid:
            continue
        s = e = inst = None
        for el in ctx.iter():
            ln = _local(el.tag)
            if ln == "startDate":
                s = _iso_date(el.text)
            elif ln == "endDate":
                e = _iso_date(el.text)
            elif ln == "instant":
                inst = _iso_date(el.text)
        contexts[cid] = (None, inst) if inst else (s, e)

    def _is_quarter_ctx(cid: str) -> bool:
        if cid not in contexts:
            # 2018-2021-era instances reference the headline P&L facts with
            # contextRef="OneD" WITHOUT defining that context (era filing-tool
            # quirk, live finding 2026-07-09). The results-format column
            # convention is positional and stable: OneD = the current
            # quarter. Fall back to it only when the id is undefined —
            # defined contexts are always validated structurally.
            return cid == "OneD"
        s, e = contexts[cid]
        if s is None or e is None or e != quarter_end:
            return False
        return _QUARTER_MIN_DAYS <= (e - s).days <= _QUARTER_MAX_DAYS

    def _is_instant_at_end(cid: str) -> bool:
        if cid not in contexts:
            # Same convention fallback, instant flavor (OneI = at quarter end).
            return cid == "OneI"
        s, e = contexts[cid]
        return s is None and e == quarter_end

    # local tag name → list[(contextRef, value)]
    facts: dict[str, list[tuple[str, float]]] = {}
    for el in root.iter():
        ln = _local(el.tag)
        cid = el.get("contextRef")
        if not cid or el.text is None:
            continue
        try:
            v = float(el.text)
        except ValueError:
            continue
        facts.setdefault(ln, []).append((cid, v))

    def _pick(priority: tuple[str, ...], ctx_filter) -> float | None:
        for tag in priority:
            for cid, v in facts.get(tag, []):
                if ctx_filter(cid):
                    return v
        return None

    if not any(
        _is_quarter_ctx(cid)
        for entries in facts.values() for cid, _ in entries
    ):
        return None

    revenue = _pick(_TAG_PRIORITY["revenue"], _is_quarter_ctx)
    net_income = _pick(_TAG_PRIORITY["net_income"], _is_quarter_ctx)
    eps_basic = _pick(_TAG_PRIORITY["eps_basic"], _is_quarter_ctx)
    eps_diluted = _pick(_TAG_PRIORITY["eps_diluted"], _is_quarter_ctx)

    if revenue is None and net_income is None:
        return None

    # Shares: weighted-consistent derivation from EPS preferred; paid-up /
    # face-value fallback is actual-not-weighted (documented approximation).
    ni_inr_m = None if net_income is None else net_income * _INR_SCALE

    def _derived_shares(eps: float | None) -> float | None:
        if ni_inr_m is None or eps is None or eps == 0.0:
            return None
        return ni_inr_m / eps

    shares_basic = _derived_shares(eps_basic)
    shares_diluted = _derived_shares(eps_diluted)
    if shares_basic is None:
        paidup = _pick((_PAIDUP_TAG,), _is_instant_at_end)
        face = _pick((_FACE_VALUE_TAG,), _is_instant_at_end)
        if paidup and face:
            shares_basic = (paidup / face) * _INR_SCALE

    return {
        "revenue": None if revenue is None else revenue * _INR_SCALE,
        "net_income": ni_inr_m,
        "shares_basic": shares_basic,
        "shares_diluted": shares_diluted,
        "eps_basic": eps_basic,
        "eps_diluted": eps_diluted,
    }


def _iso_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text.strip())
    except ValueError:
        return None


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        c.name: pd.Series(dtype=c.dtype)
        for c in IN_FUNDAMENTALS_SCHEMA.columns
    })
