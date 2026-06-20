"""FRED adapter for the fred_macro domain (auto transport).

One logical source (the St. Louis Fed), two transports:

  - **keyless** — ``GET fredgraph.csv?id=<id>&cosd=<start>&coed=<end>``
    (no key, CSV body: ``observation_date,<SERIES_ID>``).
  - **keyed**   — ``GET <api>/series/observations?series_id=<id>&api_key=<k>``
    ``&file_type=json&observation_start=<start>&observation_end=<end>``.

Transport is chosen at fetch time: the keyed JSON API when ``FRED_API_KEY`` is
set (more official/documented/stable; supports ALFRED vintages later), else the
keyless CSV download (zero setup). Both yield the same canonical
``(date, value)`` frame. ``parse()`` dispatches on the raw file extension
(``.json`` vs ``.csv``) so reprocess-from-raw (D8) works regardless of which
transport produced the file.

Single-source domain — no fallback chain (``FREDMacroDomain.chain_for_gap``
returns just this adapter).

D6 — API key safety:
  - Key read from ``os.environ[config.fred_api_key_env]`` at fetch time only.
  - Sent only as the documented ``api_key`` query param to api.stlouisfed.org.
  - NEVER written to the raw filename, a log line, or an error message
    (``urllib.error.HTTPError`` can echo the request URL — we raise our own
    ``ProviderError`` carrying just the status code, with ``from None`` so the
    original URL-bearing exception is not chained into logs).

Daily-series densification: a daily series is reindexed in ``parse()`` to every
weekday in its covered span (``NaN`` on no-data days) so federal-holiday gaps
don't read as perpetual 1-day gaps that re-fetch on every refresh. Monthly /
quarterly series are left as FRED dates them (period-start), matching the
domain's ``MonthlyCalendar`` / ``QuarterlyCalendar`` grids.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.fred_macro.calendar import BusinessDayCalendar
from data_pipelines.domains.fred_macro.config import FredMacroConfig
from data_pipelines.domains.fred_macro.registry import parse_identifier
from data_pipelines.errors import EmptyPayload, ProviderError
from data_pipelines.raw_store import write_raw_atomic
from data_pipelines.retry import RetryPolicy, call_with_retry

DOMAIN_NAME = "fred_macro"


class FredAdapter(Adapter):
    name = "fred"
    source_column_map = None  # parse() returns canonical ("date", "value")
    extra_meta = {"source": "fred"}

    def __init__(
        self,
        config: FredMacroConfig | None = None,
        frequency_map: dict[str, str] | None = None,
    ):
        self._config = config or FredMacroConfig()
        # series_id (UPPER) → frequency; used only to decide the daily densify.
        # Unknown/out-of-curated-universe series → treated as non-daily (no
        # densify); their gap behaviour is best-effort (documented).
        self._frequency_map = {
            str(k).upper(): v for k, v in (frequency_map or {}).items()
        }
        self._weekday_cal = BusinessDayCalendar()

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self._config.retry_max_retries,
            base_delay_sec=self._config.retry_base_delay_sec,
            max_delay_sec=self._config.retry_max_delay_sec,
            jitter=self._config.retry_jitter,
        )

    def _api_key(self) -> str | None:
        return os.environ.get(self._config.fred_api_key_env) or None

    # --- fetch ---------------------------------------------------------------

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
                self.name, identifier, "FRED requires explicit start and end dates"
            )
        _, series_id = parse_identifier(identifier)

        if self._api_key():
            payload, ext = self._fetch_keyed(identifier, series_id, start, end)
        else:
            payload, ext = self._fetch_keyless(identifier, series_id, start, end)

        return write_raw_atomic(
            data_root,
            provider=self.name,
            domain=DOMAIN_NAME,
            exchange="-",
            ticker=series_id,
            payload=payload,
            range_start=start,
            range_end=end,
            ext=ext,
            timestamp=datetime.now(timezone.utc),
        )

    def _fetch_keyless(
        self, identifier: str, series_id: str, start: date, end: date,
    ) -> tuple[bytes, str]:
        params = {
            "id": series_id,
            "cosd": start.isoformat(),
            "coed": end.isoformat(),
        }
        url = f"{self._config.fredgraph_base_url}?{urllib.parse.urlencode(params)}"
        payload = self._get(url, identifier)
        if not _csv_has_values(payload):
            raise EmptyPayload(self.name, identifier)
        return payload, "csv"

    def _fetch_keyed(
        self, identifier: str, series_id: str, start: date, end: date,
    ) -> tuple[bytes, str]:
        key = self._api_key()
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }
        url = (
            f"{self._config.fred_api_base_url}/series/observations"
            f"?{urllib.parse.urlencode(params)}"
        )
        payload = self._get(url, identifier)
        if not _json_has_values(payload):
            raise EmptyPayload(self.name, identifier)
        return payload, "json"

    def _get(self, url: str, identifier: str) -> bytes:
        """GET ``url`` with shared retry/backoff. Errors are sanitized to a
        status code only — the URL (which carries the api_key for the keyed
        transport) never reaches an exception message or log line (D6).
        """

        def _do() -> bytes:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "data_pipelines/0.1"}
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
                # `from None`: do NOT chain `e` (its .url / str echoes the key).
                raise ProviderError(self.name, identifier, f"HTTP {e.code}") from None
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
        ext = raw_path.suffix.lower().lstrip(".")
        if ext == "json":
            df = self._parse_json(raw_path)
        else:
            df = self._parse_csv(raw_path)

        # Daily series: densify the covered span to weekdays (NaN on no-data
        # days) so holiday gaps don't re-fetch forever. The series id is the
        # raw path's parent dir: data/raw/fred/fred_macro/-/<SERIES_ID>/<file>.
        series_id = raw_path.parent.name.upper()
        if self._frequency_map.get(series_id) == "daily" and len(df) > 0:
            df = self._densify_weekdays(df)
        return df

    def _parse_csv(self, raw_path: Path) -> pd.DataFrame:
        raw = pd.read_csv(raw_path)
        if raw.shape[1] < 2:
            return _empty()
        # fredgraph CSV: first column is the date, second the value, whatever
        # they're named ("DATE"/"observation_date", "<SERIES_ID>").
        date_col, val_col = raw.columns[0], raw.columns[1]
        return _finalize(raw[date_col], raw[val_col])

    def _parse_json(self, raw_path: Path) -> pd.DataFrame:
        doc = json.loads(raw_path.read_text())
        obs = doc.get("observations", []) if isinstance(doc, dict) else []
        if not obs:
            return _empty()
        dates = [o.get("date") for o in obs]
        values = [o.get("value") for o in obs]
        return _finalize(pd.Series(dates), pd.Series(values))

    def _densify_weekdays(self, df: pd.DataFrame) -> pd.DataFrame:
        first = df["date"].iloc[0].date()
        last = df["date"].iloc[-1].date()
        weekdays = self._weekday_cal.trading_days(first, last)
        grid = pd.DataFrame(
            {"date": pd.to_datetime(weekdays).astype("datetime64[ns]")}
        )
        merged = grid.merge(df, on="date", how="left")
        return merged[["date", "value"]].reset_index(drop=True)

    def health_check(self) -> bool:
        # Keyless transport needs no credential, so the adapter is always
        # reachable in principle. Never raises.
        return True


# ---------------------------------------------------------------------------
# Module helpers (pure)
# ---------------------------------------------------------------------------

def _to_float(s: pd.Series) -> pd.Series:
    """Coerce a FRED value column to float64. FRED encodes missing as ``"."``
    (and occasionally an empty string), both → ``NaN``.
    """
    cleaned = s.astype(str).str.strip().replace({".": None, "": None})
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _finalize(dates: pd.Series, values: pd.Series) -> pd.DataFrame:
    """Build the canonical ``(date, value)`` frame: normalized midnight dates,
    float64 value with NaN for missing, sorted, de-duped on date.
    """
    out = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce"),
        "value": _to_float(values),
    })
    out = out.dropna(subset=["date"])  # rows whose date failed to parse
    out["date"] = out["date"].dt.normalize().astype("datetime64[ns]")
    out = (
        out.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    return out[["date", "value"]]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series([], dtype="datetime64[ns]"),
        "value": pd.Series([], dtype="float64"),
    })


def _csv_has_values(payload: bytes) -> bool:
    """True iff a fredgraph CSV body has at least one real (non-".") value."""
    try:
        df = pd.read_csv(io.BytesIO(payload))
    except Exception:
        return False
    if df.shape[1] < 2 or len(df) == 0:
        return False
    return bool(_to_float(df.iloc[:, 1]).notna().any())


def _json_has_values(payload: bytes) -> bool:
    """True iff a FRED-API JSON body has at least one real (non-".") value."""
    try:
        doc = json.loads(payload)
    except Exception:
        return False
    obs = doc.get("observations", []) if isinstance(doc, dict) else []
    return any(o.get("value") not in (".", "", None) for o in obs)
