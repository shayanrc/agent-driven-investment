"""us_fundamentals canonical processed-layer schema + the shared grid-snap util.

One row per ticker per **grid date** (calendar quarter-end). The grid exists
because providers date quarters differently — macrotrends uses fiscal
month-ends (WMT: Jan 31 / Apr 30 / ...), EDGAR uses exact fiscal period ends
(AAPL: 2026-03-28) — and gap detection needs every adapter to land rows on one
deterministic, convergent grid. The true period end is preserved in
``fiscal_period_end``; ``date`` is the join/grid key.

Snap rule (shared by every adapter — divergence here would make the same
quarter land on different grid dates depending on which provider served it):
snap ``fiscal_period_end`` FORWARD to the next calendar quarter-end, with a
small backward tolerance for the 52/53-week-fiscal-year wobble (AAPL's fiscal
Q3 FY2017 ended 2017-07-01 — one day past Jun 30; snapping it forward to
Sep 30 would collide with fiscal Q4). Forward-by-default is the conservative,
look-ahead-free choice: the quarter ended on or before its grid date.

Money columns are **USD millions** (macrotrends' native unit; EDGAR raw USD is
scaled down), shares are **millions**, EPS is USD/share. All metric columns
are nullable — providers legitimately lack values (banks have no product
revenue analog, non-USD filers are skipped, small caps miss line items).
``filed_date`` is nullable because only EDGAR knows it; it is the causal-lag
hook for the modeling phase (a quarter's numbers are not public knowledge
until filed).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from data_pipelines.schema import ColumnSpec, Schema

US_FUNDAMENTALS_SCHEMA = Schema(columns=(
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
))

# Metric columns in canonical order (everything except the three date columns).
METRIC_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in US_FUNDAMENTALS_SCHEMA.columns
    if c.dtype == "float64"
)

# Fiscal ends up to this many days AFTER a grid date snap back to it (52/53-week
# wobble); anything later snaps forward to the next grid date.
SNAP_BACK_TOLERANCE_DAYS = 7

_QE_MONTH_DAY = ((3, 31), (6, 30), (9, 30), (12, 31))


def quarter_ends(start: date, end: date) -> list[date]:
    """Calendar quarter-end dates (Mar 31 / Jun 30 / Sep 30 / Dec 31) in
    ``[start, end]``, ascending."""
    if start > end:
        return []
    out: list[date] = []
    for y in range(start.year, end.year + 1):
        for m, d in _QE_MONTH_DAY:
            g = date(y, m, d)
            if start <= g <= end:
                out.append(g)
    return out


def snap_to_quarter_end(
    fiscal_end: date, back_tolerance_days: int = SNAP_BACK_TOLERANCE_DAYS,
) -> date:
    """Snap a fiscal period end onto the calendar quarter-end grid.

    Within ``back_tolerance_days`` after a grid date → that grid date
    (2017-07-01 → 2017-06-30); otherwise the next grid date at or after
    ``fiscal_end`` (WMT 2025-01-31 → 2025-03-31; exact grid dates map to
    themselves).
    """
    grid = quarter_ends(
        date(fiscal_end.year - 1, 12, 1), date(fiscal_end.year + 1, 4, 1)
    )
    prev = max(g for g in grid if g <= fiscal_end)
    if (fiscal_end - prev).days <= back_tolerance_days:
        return prev
    return min(g for g in grid if g >= fiscal_end)


def dedupe_grid_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve two fiscal periods snapping to one grid date (fiscal-year-change
    stubs): keep the row whose ``fiscal_period_end`` is closest to the grid
    date (ties → later fiscal_period_end, deterministically).

    Expects canonical columns; returns rows sorted by ``date`` ascending.
    """
    if df.empty:
        return df.reset_index(drop=True)
    gap = (df["date"] - df["fiscal_period_end"]).abs()
    out = (
        df.assign(_gap=gap)
        .sort_values(["date", "_gap", "fiscal_period_end"],
                     ascending=[True, True, False])
        .drop_duplicates(subset=["date"], keep="first")
        .drop(columns="_gap")
        .reset_index(drop=True)
    )
    return out
