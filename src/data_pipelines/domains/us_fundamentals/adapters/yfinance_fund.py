"""yfinance fundamentals adapter (tertiary) for the us_fundamentals domain.

Last-resort provider, mirroring yfinance's third-tier role in the us_equities
chain: ``Ticker.quarterly_income_stmt`` + ``Ticker.quarterly_cashflow`` cover
only the **last ~5 quarters**, so this adapter can never seed history — its
job is the newest quarter when both macrotrends and EDGAR fail on a ticker.

Raw is a JSON envelope of the two frames serialized with
``to_json(orient="split")`` — yfinance has no native wire format to preserve
(same rationale as the us_equities yfinance adapter's parquet), and the
split orientation keeps line-item labels + period columns losslessly for
deterministic reprocess.

Sign convention: yfinance's ``Capital Expenditure`` is a negative outflow →
``capex = -CapitalExpenditure`` (stored positive), ``fcf = ocf − capex`` —
identical derivation to the other adapters. No filing dates → ``filed_date``
NaT.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

from data_pipelines.adapter import Adapter
from data_pipelines.domains.us_fundamentals.config import USFundamentalsConfig
from data_pipelines.domains.us_fundamentals.registry import parse_identifier
from data_pipelines.domains.us_fundamentals.schema import (
    METRIC_COLUMNS,
    US_FUNDAMENTALS_SCHEMA,
    dedupe_grid_collisions,
    snap_to_quarter_end,
)
from data_pipelines.errors import EmptyPayload, ProviderError
from data_pipelines.raw_store import write_raw_atomic

DOMAIN_NAME = "us_fundamentals"

# yfinance line-item label → canonical column
INCOME_FIELD_MAP = {
    "Total Revenue": "revenue",
    "Net Income": "net_income",
    "Basic Average Shares": "shares_basic",
    "Diluted Average Shares": "shares_diluted",
    "Basic EPS": "eps_basic",
    "Diluted EPS": "eps_diluted",
}
CASHFLOW_FIELD_MAP = {
    "Operating Cash Flow": "ocf",
    "Capital Expenditure": "capex_signed",  # negative outflow → negated
}

# raw USD / share counts → millions; EPS stays per-share.
_UNSCALED = {"eps_basic", "eps_diluted"}


class YFinanceFundamentalsAdapter(Adapter):
    name = "yfinance"
    source_column_map = None  # parse() returns canonical columns
    extra_meta = {"source": "yfinance_quarterly_statements",
                  "units": "usd_millions"}

    def __init__(self, config: USFundamentalsConfig | None = None):
        self._config = config or USFundamentalsConfig()

    def fetch(
        self,
        identifier: str,
        start: date | None = None,
        end: date | None = None,
        *,
        data_root: Path,
    ) -> Path:
        _, symbol = parse_identifier(identifier)
        try:
            import yfinance as yf
        except ImportError as e:
            raise ProviderError(
                self.name, identifier, "yfinance not installed"
            ) from e

        try:
            ticker = yf.Ticker(symbol.replace(".", "-"))  # yahoo dash-spells
            income = ticker.quarterly_income_stmt
            cashflow = ticker.quarterly_cashflow
        except Exception as e:  # yfinance raises a zoo of exception types
            raise ProviderError(
                self.name, identifier, f"yfinance error: {type(e).__name__}: {e}"
            ) from None

        if (income is None or income.empty) and (
            cashflow is None or cashflow.empty
        ):
            raise EmptyPayload(self.name, identifier)

        envelope = json.dumps({
            "income_statement": _frame_to_json(income),
            "cash_flow": _frame_to_json(cashflow),
        }).encode("utf-8")
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

    def parse(self, raw_path: Path) -> pd.DataFrame:
        doc = json.loads(raw_path.read_text())
        income = _extract(
            _frame_from_json(doc.get("income_statement")), INCOME_FIELD_MAP
        )
        cashflow = _extract(
            _frame_from_json(doc.get("cash_flow")), CASHFLOW_FIELD_MAP
        )
        df = income.merge(cashflow, on="fiscal_period_end", how="outer")

        df["capex"] = -df["capex_signed"]  # negative outflow → positive capex
        df["fcf"] = df["ocf"] - df["capex"]
        df = df.drop(columns=["capex_signed"])

        df["filed_date"] = pd.NaT
        df["date"] = pd.to_datetime(
            [snap_to_quarter_end(d.date()) for d in df["fiscal_period_end"]]
        ).astype("datetime64[ns]")

        for col in US_FUNDAMENTALS_SCHEMA.column_names:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[US_FUNDAMENTALS_SCHEMA.column_names]
        # yfinance pads with empty trailing quarters; an all-NaN row would
        # claim grid coverage and mask the date from the better providers.
        df = df.dropna(subset=list(METRIC_COLUMNS), how="all")
        return dedupe_grid_collisions(df)


def _frame_to_json(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    return df.to_json(orient="split", date_format="iso")


def _frame_from_json(payload: str | None) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    df = pd.read_json(StringIO(payload), orient="split")
    df.columns = pd.to_datetime(df.columns)
    return df


def _extract(frame: pd.DataFrame, field_map: dict[str, str]) -> pd.DataFrame:
    """Pivot a yfinance statement frame (line items × quarter-end columns)
    to canonical columns keyed by ``fiscal_period_end``, scaled to millions
    (except EPS)."""
    periods = sorted(frame.columns) if not frame.empty else []
    out = pd.DataFrame({
        "fiscal_period_end": pd.to_datetime(periods).astype("datetime64[ns]"),
    })
    for label, canonical in field_map.items():
        if not frame.empty and label in frame.index:
            values = pd.to_numeric(
                frame.loc[label].reindex(periods), errors="coerce"
            ).to_numpy(dtype="float64")
        else:
            values = float("nan")
        scale = 1.0 if canonical in _UNSCALED else 1e-6
        out[canonical] = values
        if canonical not in _UNSCALED:
            out[canonical] = out[canonical] * scale
    return out
