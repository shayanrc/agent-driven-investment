"""SEC EDGAR adapter (secondary) for the us_fundamentals domain.

One request per ticker to the official XBRL companyfacts API —
``GET data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json`` — carries the
company's full filed-fact history (~2008→now, the XBRL-mandate era). Free,
official, and the ONLY provider of ``filed_date``: every fact carries the
date its filing hit EDGAR, which is the point-in-time truth the modeling
phase needs for causal lags.

Normalizing XBRL into clean quarters is the hard part (design validated
against the AAPL fixture, see ``V3_US_FUNDAMENTALS_PLAN.md``):

- **Tag drift** — companies switch tags over the years (AAPL: SalesRevenueNet
  → RevenueFromContractWithCustomer... at ASC 606). Each metric pools points
  from an ordered tag-priority list.
- **Point-in-time dedup** — the same (start, end) period appears in many
  filings (original 10-Q, next year's comparatives, 10-K/A restatements).
  Keep the candidate minimizing ``(filed, tag_rank, accn)``: earliest-filed
  wins, so restatements and comparatives lose automatically and the stored
  value is what the market knew first.
- **``fy``/``fp`` are filing-relative** (they describe the filing a fact
  appears in, not the fact's own period) — never used for period identity;
  only ``(start, end)`` is.
- **Quarter recovery** — direct quarters are 70–115-day durations. Flow
  metrics (revenue, NI, OCF, capex) additionally recover quarters by
  differencing consecutive year-to-date durations that share a fiscal-year
  ``start`` (10-Q cash-flow statements are YTD-only for many filers, and
  Q4 = FY − 3Q-YTD falls out of the same rule). Weighted-average shares and
  EPS must NOT be differenced (averages don't difference); shares Q4 uses
  ``4×FY − ΣQ1..3`` and missing EPS is filled from ``net_income / shares``
  behind ``DERIVE_EPS``.
- **Row ``filed_date`` = max over the row's metrics** — the date by which
  every value in the row was publicly derivable (a differenced/derived value
  is knowable only when its latest component was filed). Conservative by
  construction for causal joins.

Known limitations (documented punts): pre-2008 history is the primary
adapter's job; banks/insurers may have NULL revenue (no tag in the priority
list); non-USD filers yield NULL metrics (exact unit-key match only);
sub-70-day fiscal-transition stubs are dropped.
"""

from __future__ import annotations

import json
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

DOMAIN_NAME = "us_fundamentals"

CIK_MAP_PATH = "/files/company_tickers.json"
COMPANYFACTS_PATH = "/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_PATH = "/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_PATH = "/submissions/{name}"

# Periodic-report forms whose filing date is the "this quarter became public"
# date. Amendments are included but lose to the original on earliest-filed.
FILING_FORMS = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A"})

# Ordered tag-priority lists per metric (first = preferred at equal filed).
TAG_PRIORITY: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "Revenues",
    ),
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "ocf": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "shares_basic": (
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfSharesOutstandingBasicAndDiluted",
    ),
    "shares_diluted": (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ),
    "eps_basic": (
        "EarningsPerShareBasic",
        "EarningsPerShareBasicAndDiluted",
    ),
    "eps_diluted": (
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
    ),
}

UNIT_KEY: dict[str, str] = {
    "revenue": "USD", "net_income": "USD", "ocf": "USD", "capex": "USD",
    "shares_basic": "shares", "shares_diluted": "shares",
    "eps_basic": "USD/shares", "eps_diluted": "USD/shares",
}

# raw USD / share counts → millions; EPS stays per-share.
SCALE: dict[str, float] = {
    "revenue": 1e-6, "net_income": 1e-6, "ocf": 1e-6, "capex": 1e-6,
    "shares_basic": 1e-6, "shares_diluted": 1e-6,
    "eps_basic": 1.0, "eps_diluted": 1.0,
}

# Metrics whose YTD durations difference cleanly (sums over time). Weighted
# averages (shares) and ratios (EPS) do NOT belong here.
ADDITIVE_METRICS = ("revenue", "net_income", "ocf", "capex")
SHARES_METRICS = ("shares_basic", "shares_diluted")

Q_MIN_DAYS, Q_MAX_DAYS = 70, 115     # single quarter (13w=90 / 14w=97)
FY_MIN_DAYS, FY_MAX_DAYS = 330, 380  # full fiscal year (shares Q4 rule)

# Fill missing EPS from net_income / shares (±$0.01 vs as-reported rounding).
DERIVE_EPS = True


class EdgarAdapter(Adapter):
    name = "edgar"
    source_column_map = None  # parse() returns canonical columns
    extra_meta = {"source": "sec_edgar_companyfacts", "units": "usd_millions"}

    def __init__(self, config: USFundamentalsConfig | None = None):
        self._config = config or USFundamentalsConfig()
        self._cik_map_cache: dict[str, int] | None = None

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self._config.retry_max_retries,
            base_delay_sec=self._config.retry_base_delay_sec,
            max_delay_sec=self._config.retry_max_delay_sec,
            jitter=self._config.retry_jitter,
        )

    def _user_agent(self) -> str:
        # SEC fair-access policy: identify the caller with contact info.
        import os
        contact = (
            os.environ.get(self._config.edgar_contact_env)
            or self._config.edgar_contact_default
        )
        return f"data_pipelines/0.1 ({contact})"

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
        cik = self._resolve_cik(symbol, data_root, identifier)
        if cik is None:
            raise EmptyPayload(self.name, identifier)  # not an SEC filer

        url = self._config.edgar_base_url + COMPANYFACTS_PATH.format(cik=cik)
        payload = self._get(url, identifier)
        return write_raw_atomic(
            data_root,
            provider=self.name,
            domain=DOMAIN_NAME,
            exchange="-",
            ticker=symbol,
            payload=payload,
            range_start=start or date(1990, 1, 1),
            range_end=end or datetime.now(timezone.utc).date(),
            ext="json",
            timestamp=datetime.now(timezone.utc),
        )

    # --- filing dates (SEC submissions API) ----------------------------------

    def filing_dates(
        self, identifier: str, *, data_root: Path,
    ) -> dict[date, date]:
        """Map each fiscal period end (``reportDate``) → the **earliest**
        filing date of the 10-K/10-Q that reported it, from the SEC
        submissions API.

        This is the authoritative "when did this quarter's numbers become
        public" date, and unlike the companyfacts-derived dates in ``parse()``
        it is exact for derived-Q4 quarters: the fiscal-year 10-K's own
        ``reportDate`` is the Q4 end, so its filing date lands on Q4 directly
        (no differencing-max approximation).

        Reads the ``recent`` block plus any older paginated files so full
        history is covered. Returns ``{}`` for non-SEC filers (no CIK).
        Raw submissions JSON is landed under the ``edgar_submissions`` provider
        for the audit trail. Pure w.r.t. its inputs modulo the network fetch;
        the returned mapping is deterministic for a given filing history.
        """
        _, symbol = parse_identifier(identifier)
        cik = self._resolve_cik(symbol, data_root, identifier)
        if cik is None:
            return {}

        main = self._get_submissions_doc(
            self._config.edgar_base_url + SUBMISSIONS_PATH.format(cik=cik),
            symbol, identifier, data_root,
        )
        blocks = [main.get("filings", {}).get("recent", {})]
        for f in (main.get("filings", {}).get("files") or []):
            name = f.get("name")
            if not name:
                continue
            older = self._get_submissions_doc(
                self._config.edgar_base_url
                + SUBMISSIONS_FILE_PATH.format(name=name),
                symbol, identifier, data_root,
            )
            # paginated files carry the arrays at the top level
            blocks.append(older if "form" in older else older.get("recent", {}))

        out: dict[date, date] = {}
        for block in blocks:
            forms = block.get("form", [])
            fdates = block.get("filingDate", [])
            rdates = block.get("reportDate", [])
            for form, fd, rd in zip(forms, fdates, rdates):
                if form not in FILING_FORMS or not fd or not rd:
                    continue
                try:
                    r = date.fromisoformat(rd)
                    f_date = date.fromisoformat(fd)
                except (ValueError, TypeError):
                    continue
                if r not in out or f_date < out[r]:
                    out[r] = f_date
        return out

    def _get_submissions_doc(
        self, url: str, symbol: str, identifier: str, data_root: Path,
    ) -> dict:
        payload = self._get(url, identifier)
        try:
            write_raw_atomic(
                data_root,
                provider="edgar_submissions",
                domain=DOMAIN_NAME,
                exchange="-",
                ticker=symbol,
                payload=payload,
                range_start=date(1990, 1, 1),
                range_end=datetime.now(timezone.utc).date(),
                ext="json",
                timestamp=datetime.now(timezone.utc),
            )
        except FileExistsError:
            pass  # same-second re-fetch; audit copy already present
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            raise ProviderError(
                self.name, identifier, "submissions is not valid JSON"
            ) from None

    # --- CIK map -------------------------------------------------------------

    def _resolve_cik(
        self, symbol: str, data_root: Path, identifier: str,
    ) -> int | None:
        cik_map = self._cik_map(data_root, identifier)
        # SEC spells class shares with dashes (BRK-B) like our universes;
        # try the dot variant anyway for robustness.
        for candidate in (symbol, symbol.replace(".", "-"),
                          symbol.replace("-", ".")):
            if candidate in cik_map:
                return cik_map[candidate]
        return None

    def _cik_map(self, data_root: Path, identifier: str) -> dict[str, int]:
        if self._cik_map_cache is None:
            url = self._config.sec_www_base_url + CIK_MAP_PATH
            payload = self._get(url, identifier)
            try:
                entries = json.loads(payload)
            except json.JSONDecodeError:
                raise ProviderError(
                    self.name, identifier, "CIK map is not valid JSON"
                ) from None
            today = datetime.now(timezone.utc).date()
            try:
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
                pass
            self._cik_map_cache = {
                str(v["ticker"]).upper(): int(v["cik_str"])
                for v in entries.values()
                if isinstance(v, dict) and v.get("ticker")
            }
        return self._cik_map_cache

    # --- transport -----------------------------------------------------------

    def _get(self, url: str, identifier: str) -> bytes:
        def _do() -> bytes:
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": self._user_agent(),
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
                if e.code == 404:
                    # no companyfacts for this CIK — not retryable; fall
                    # through the chain.
                    raise EmptyPayload(self.name, identifier) from None
                raise ProviderError(
                    self.name, identifier, f"HTTP {e.code}"
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

    # --- parse ---------------------------------------------------------------

    def parse(self, raw_path: Path) -> pd.DataFrame:
        doc = json.loads(raw_path.read_text())
        gaap = (doc.get("facts") or {}).get("us-gaap") or {}

        # metric → {fiscal_end(date) → (value, filed(date))}
        quarters: dict[str, dict[date, tuple[float, date]]] = {}
        fy_periods: dict[str, list[tuple[date, date, float, date]]] = {}
        for metric, tags in TAG_PRIORITY.items():
            periods = _dedupe_periods(_pool_points(gaap, metric, tags))
            q = _direct_quarters(periods)
            if metric in ADDITIVE_METRICS:
                _add_ytd_differenced(periods, q)
            quarters[metric] = q
            if metric in SHARES_METRICS:
                fy_periods[metric] = [
                    (s, e, v, f) for (s, e), (v, f) in periods.items()
                    if FY_MIN_DAYS <= (e - s).days <= FY_MAX_DAYS
                ]

        for metric in SHARES_METRICS:
            _add_q4_shares(quarters[metric], fy_periods.get(metric, []))

        all_ends = sorted({e for q in quarters.values() for e in q})
        if not all_ends:
            return _empty()

        rows: list[dict] = []
        for end in all_ends:
            row: dict = {
                "fiscal_period_end": pd.Timestamp(end),
                "date": pd.Timestamp(snap_to_quarter_end(end)),
            }
            filed_dates: list[date] = []
            for metric in TAG_PRIORITY:
                hit = quarters[metric].get(end)
                if hit is None:
                    row[metric] = float("nan")
                else:
                    row[metric] = hit[0] * SCALE[metric]
                    filed_dates.append(hit[1])
            # max: the date by which EVERY value in this row was derivable.
            row["filed_date"] = (
                pd.Timestamp(max(filed_dates)) if filed_dates else pd.NaT
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        df["fcf"] = df["ocf"] - df["capex"]
        if DERIVE_EPS:
            for side in ("basic", "diluted"):
                eps, sh = f"eps_{side}", f"shares_{side}"
                fill = df["net_income"] / df[sh]
                df[eps] = df[eps].fillna(fill)

        for c in ("date", "fiscal_period_end", "filed_date"):
            df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
        df = df[US_FUNDAMENTALS_SCHEMA.column_names]
        return dedupe_grid_collisions(df)


# ---------------------------------------------------------------------------
# Module helpers (pure)
# ---------------------------------------------------------------------------

def _pool_points(
    gaap: dict, metric: str, tags: tuple[str, ...],
) -> list[tuple[date, date, float, date, int, str]]:
    """All duration facts for a metric across its tag list, exact unit key
    only: (start, end, val, filed, tag_rank, accn)."""
    unit_key = UNIT_KEY[metric]
    pool = []
    for rank, tag in enumerate(tags):
        points = ((gaap.get(tag) or {}).get("units") or {}).get(unit_key, [])
        for p in points:
            if "start" not in p or p.get("val") is None:
                continue  # instant facts (balance sheet) are not durations
            try:
                s = date.fromisoformat(p["start"])
                e = date.fromisoformat(p["end"])
                f = date.fromisoformat(p["filed"])
            except (KeyError, ValueError, TypeError):
                continue
            pool.append((s, e, float(p["val"]), f, rank, str(p.get("accn", ""))))
    return pool


def _dedupe_periods(
    pool: list[tuple[date, date, float, date, int, str]],
) -> dict[tuple[date, date], tuple[float, date]]:
    """Per (start, end) period keep the candidate minimizing
    (filed, tag_rank, accn) — earliest-filed wins (point-in-time), tag rank
    breaks same-filing ties, accession number makes the order total (D8)."""
    best: dict[tuple[date, date], tuple[date, int, str, float]] = {}
    for s, e, val, filed, rank, accn in pool:
        key = (s, e)
        cand = (filed, rank, accn, val)
        if key not in best or cand[:3] < best[key][:3]:
            best[key] = cand
    return {k: (v[3], v[0]) for k, v in best.items()}


def _direct_quarters(
    periods: dict[tuple[date, date], tuple[float, date]],
) -> dict[date, tuple[float, date]]:
    return {
        e: (val, filed)
        for (s, e), (val, filed) in periods.items()
        if Q_MIN_DAYS <= (e - s).days <= Q_MAX_DAYS
    }


def _add_ytd_differenced(
    periods: dict[tuple[date, date], tuple[float, date]],
    quarters: dict[date, tuple[float, date]],
) -> None:
    """Recover quarters by differencing consecutive YTD durations that share
    a fiscal-year start. Because FY − 3Q-YTD is ~91 days, Q4 falls out of the
    same rule. Mutates ``quarters`` in place (direct facts win)."""
    by_start: dict[date, list[tuple[date, float, date]]] = {}
    for (s, e), (val, filed) in periods.items():
        by_start.setdefault(s, []).append((e, val, filed))
    for s, group in by_start.items():
        group.sort(key=lambda t: t[0])  # by period end == by duration
        for (e_prev, v_prev, f_prev), (e_cur, v_cur, f_cur) in zip(
            group, group[1:]
        ):
            if e_cur in quarters:
                continue
            if Q_MIN_DAYS <= (e_cur - e_prev).days <= Q_MAX_DAYS:
                quarters[e_cur] = (v_cur - v_prev, max(f_prev, f_cur))


def _add_q4_shares(
    quarters: dict[date, tuple[float, date]],
    fy_periods: list[tuple[date, date, float, date]],
) -> None:
    """Weighted-average shares are never reported for fiscal Q4; when a fiscal
    year has an FY-duration value plus exactly 3 direct quarters inside it,
    ``shares_q4 = 4×FY − (Q1+Q2+Q3)`` (validated: implies AAPL's reported Q4
    EPS exactly). Mutates ``quarters`` in place."""
    for s, e, fy_val, fy_filed in fy_periods:
        if e in quarters:
            continue
        inner = [
            (qe, val, filed) for qe, (val, filed) in quarters.items()
            if s <= qe < e
        ]
        if len(inner) != 3:
            continue
        q4 = 4.0 * fy_val - sum(v for _, v, _ in inner)
        filed = max([fy_filed] + [f for _, _, f in inner])
        quarters[e] = (q4, filed)


def _empty() -> pd.DataFrame:
    cols = {}
    for c in US_FUNDAMENTALS_SCHEMA.columns:
        cols[c.name] = pd.Series([], dtype=c.dtype)
    return pd.DataFrame(cols)
