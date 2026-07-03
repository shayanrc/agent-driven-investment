"""Point-in-time trailing-twelve-month (TTM) engine.

Turns a ticker's quarterly fundamentals into a sequence of TTM snapshots, each
stamped with the calendar date it became **public** — so a downstream daily
join can, for any day ``t``, use only what was filed by ``t``.

A TTM snapshot at quarter ``q`` sums the four consecutive quarters
``{q-3, q-2, q-1, q}`` (flow metrics) and takes ``q``'s share count. It is
valid — and emitted — only when:
  - all four quarters carry a non-NaT ``filed_date`` (else the window can't be
    placed in time), and
  - the four are genuinely consecutive (no missing quarter: every adjacent
    ``fiscal_period_end`` gap ≤ ``MAX_QUARTER_GAP_DAYS``).

Its ``effective_date`` is the **latest** ``filed_date`` in the window — the day
by which every component was public (normally the newest quarter's filing).
Per-metric: a metric's TTM is NaN if any of its four quarters is NaN, so a
sparse metric doesn't silently sum three quarters as if it were four.

Pure: no I/O, deterministic. Split-adjustment of ``shares`` is the caller's
job (this engine just carries the column it's given).
"""

from __future__ import annotations

import pandas as pd

TTM_QUARTERS = 4
MAX_QUARTER_GAP_DAYS = 110  # 13-week quarters ≈ 91d, 14-week ≈ 98d; >110 ⇒ gap
FLOW_METRICS = ("revenue", "net_income", "fcf")

_OUT_COLUMNS = (
    "effective_date", "asof_fiscal_period_end", "asof_filed_date",
    "revenue_ttm", "net_income_ttm", "fcf_ttm", "shares",
)


def build_ttm_timeline(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Quarterly fundamentals → TTM snapshots (one row per valid trailing-4Q
    window), sorted by ``effective_date``.

    ``quarterly`` needs columns ``fiscal_period_end``, ``filed_date`` (datetime,
    NaT allowed), ``revenue``, ``net_income``, ``fcf``, ``shares`` (float). Rows
    are sorted by ``fiscal_period_end`` internally.
    """
    q = quarterly.sort_values("fiscal_period_end").reset_index(drop=True)
    fe = pd.to_datetime(q["fiscal_period_end"])
    filed = pd.to_datetime(q["filed_date"])

    rows: list[dict] = []
    for i in range(TTM_QUARTERS - 1, len(q)):
        window = slice(i - TTM_QUARTERS + 1, i + 1)
        w_filed = filed.iloc[window]
        w_fe = fe.iloc[window]
        # causal: every quarter in the window must be dated
        if w_filed.isna().any():
            continue
        # consecutive: no missing quarter inside the window
        if (w_fe.diff().dropna().dt.days > MAX_QUARTER_GAP_DAYS).any():
            continue
        row = {
            "effective_date": w_filed.max(),
            "asof_fiscal_period_end": fe.iloc[i],
            "asof_filed_date": filed.iloc[i],
            "shares": float(q["shares"].iloc[i]),
        }
        for m in FLOW_METRICS:
            vals = q[m].iloc[window]
            row[f"{m}_ttm"] = (
                float(vals.sum()) if vals.notna().all() else float("nan")
            )
        rows.append(row)

    out = pd.DataFrame(rows, columns=list(_OUT_COLUMNS))
    if out.empty:
        for c in ("effective_date", "asof_fiscal_period_end", "asof_filed_date"):
            out[c] = pd.Series(dtype="datetime64[ns]")
        for c in ("revenue_ttm", "net_income_ttm", "fcf_ttm", "shares"):
            out[c] = pd.Series(dtype="float64")
        return out
    for c in ("effective_date", "asof_fiscal_period_end", "asof_filed_date"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ns]")
    # a later filing can never be effective before an earlier one; sort + keep
    # the ordering stable for the downstream asof-merge.
    return out.sort_values("effective_date").reset_index(drop=True)


def asof_daily(
    ttm_timeline: pd.DataFrame, trading_days: pd.Series,
) -> pd.DataFrame:
    """Forward-fill the TTM timeline onto ``trading_days``: each day gets the
    most recent snapshot whose ``effective_date ≤ day`` (point-in-time). Days
    before the first snapshot get NaN/NaT. Returns one row per trading day.
    """
    days = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(trading_days))
        .dt.normalize().astype("datetime64[ns]")
    }).sort_values("date").reset_index(drop=True)
    if ttm_timeline.empty:
        for c in _OUT_COLUMNS:
            days[c] = pd.NaT if "date" in c or c.startswith("asof") else float("nan")
        return days
    tl = ttm_timeline.sort_values("effective_date").reset_index(drop=True)
    return pd.merge_asof(
        days, tl, left_on="date", right_on="effective_date", direction="backward",
    )
