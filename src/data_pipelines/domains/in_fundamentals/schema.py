"""in_fundamentals canonical processed-layer schema.

One row per ticker per **grid date** (calendar quarter-end) — the same grid as
us_fundamentals, because Indian fiscal quarters end on calendar quarter-ends
natively (fiscal year Apr–Mar; Q1 FY = Apr–Jun ends Jun 30). The grid-snap
util is imported from us_fundamentals (shared by construction, not copied):
Indian ``toDate`` values are exact quarter-ends, so the snap is a no-op
safety net rather than a normalizer.

Columns are the 12 us_fundamentals columns **plus ``consolidated``** (1.0 =
consolidated basis served the row, 0.0 = standalone), appended last so the
shared prefix stays positionally identical for cross-domain consumers.

Units: money in **INR millions** (XBRL values are absolute INR → ÷1e6),
shares in millions, EPS in INR/share.

Expected-NaN columns: ``ocf``/``capex``/``fcf`` — SEBI LODR mandates cash-flow
statements only half-yearly, so quarterly cash flow does not exist in India.
The columns are kept for schema symmetry with us_fundamentals (the valuation
panel's code path reads the same names); fcf-derived ratios will honestly be
NaN. Half-yearly CF fills are a ``V4_TBD.md`` follow-up.

``filed_date`` is NON-nullable in practice for this domain (every NSE filing
record carries its exchange-timestamped ``filingDate``) but stays nullable in
the schema for symmetry and for hypothetical future fallback providers that
lack it.
"""

from __future__ import annotations

from data_pipelines.schema import ColumnSpec, Schema

# Shared grid utilities — deliberately imported, not copied (see docstring).
from data_pipelines.domains.us_fundamentals.schema import (  # noqa: F401
    SNAP_BACK_TOLERANCE_DAYS,
    dedupe_grid_collisions,
    quarter_ends,
    snap_to_quarter_end,
)

IN_FUNDAMENTALS_SCHEMA = Schema(columns=(
    ColumnSpec("date", "datetime64[ns]"),
    ColumnSpec("fiscal_period_end", "datetime64[ns]"),
    ColumnSpec("filed_date", "datetime64[ns]", nullable=True),
    ColumnSpec("revenue", "float64", nullable=True),
    ColumnSpec("net_income", "float64", nullable=True),
    ColumnSpec("ocf", "float64", nullable=True),
    ColumnSpec("capex", "float64", nullable=True),
    ColumnSpec("fcf", "float64", nullable=True),
    ColumnSpec("shares_basic", "float64", nullable=True),
    ColumnSpec("shares_diluted", "float64", nullable=True),
    ColumnSpec("eps_basic", "float64", nullable=True),
    ColumnSpec("eps_diluted", "float64", nullable=True),
    ColumnSpec("consolidated", "float64", nullable=True),
))

# Metric columns in canonical order (everything except the three date columns).
METRIC_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in IN_FUNDAMENTALS_SCHEMA.columns
    if c.dtype == "float64"
)
