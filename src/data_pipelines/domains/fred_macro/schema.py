"""fred_macro canonical processed-layer schema.

A single observation value per date — ``(date, value)``. ``value`` is nullable:
FRED encodes "no data" for a day as ``"."`` (federal holidays on a daily
series, suppressed/unreleased observations), which the adapter maps to ``NaN``.
The SQLite cache stores ``NaN`` as SQL ``NULL`` and restores it on read;
downstream consumers forward-fill or ``dropna`` per their needs.

Deliberately NOT shoehorned into the equities OHLCV shape — different domain,
different schema, per ``docs/data_pipelines/adding_a_domain.md``. The framework
knows nothing about OHLCV vs (date, value); it just reads ``domain.schema``.
"""

from __future__ import annotations

from data_pipelines.schema import ColumnSpec, Schema

FRED_SCHEMA = Schema(columns=(
    ColumnSpec("date", "datetime64[ns]"),
    ColumnSpec("value", "float64", nullable=True),
))
