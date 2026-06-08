"""Top-K + per-ticker + per-quarter + prediction-range diagnostics.

These analyses operate on the **per-segment prediction DataFrame** that the
runner already persists at ``results/gbdt/experiments/<name>/predictions/
<segment>.csv`` — schema ``(date, ticker, p_raw, p_calibrated, y_true,
sample_weight)``.

They live in their own module (not ``diagnostics.py``) because that file is
scoped to per-iteration FS+HP bundles; the four diagnostics here are
post-prediction segment-level analyses that the report layer consumes in
addition to the headline metrics.

Design notes
------------
- All ranking uses ``p_calibrated`` (what downstream consumers actually
  receive), NOT ``p_raw``.
- Tie-break order is deterministic: ``p_calibrated`` desc then ``ticker``
  asc. Same inputs always yield the same outputs.
- Base rate is **unweighted prevalence on the segment** — simpler than the
  weighted variant, and on segments with uniform per-ticker weights (the
  current uniqueness-weighting scheme) the two are numerically identical.
- Empty segments return well-formed empty structures, not exceptions.
- Every returned dict is JSON-serializable (no ``np.float64``, no
  ``Timestamp``, no ``Period``).

Public surface
--------------
- ``compute_top_k_metrics(df, k_values=(1, 5, 10))``
- ``compute_r_precision_at_k(df, k_values=(1, 3, 5, 10, 20))`` — canonical
  macro form (matches ``scripts/gbdt/regenerate_r_precision_at_k_csv.py``
  and the canonical CSV ``results/gbdt/data/r_precision_at_k.csv``)
- ``compute_per_ticker_hit_rate(df, k=5)``
- ``compute_per_quarter_p_k(df, k=5)``
- ``compute_prediction_range(df, low_separation_threshold=0.05)``
- ``compute_all(df, ...)`` — bundle helper that calls all of the above.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_top_k(k_values: Iterable[int]) -> dict[str, Any]:
    return {
        "formula_version": "v2_min_R_d_k",
        "base_rate": None,
        "n_rows": 0,
        "per_day": {str(k): _empty_per_k() for k in k_values},
        "global": {str(k): _empty_global_k() for k in k_values},
    }


def _empty_per_k() -> dict[str, Any]:
    return {
        "p_at_k": None,
        "n_picks_total": 0,
        "n_positives_in_picks": 0,
        "n_denom": 0,
        "n_days_R_lt_k": 0,
        "n_days_full_k": 0,
        "n_days_total": 0,
        "lift": None,
    }


def _empty_global_k() -> dict[str, Any]:
    return {
        "p_at_k": None,
        "n_picks": 0,
        "n_positives_in_picks": 0,
        "n_denom": 0,
        "lift": None,
    }


def _sorted_by_score(df: pd.DataFrame) -> pd.DataFrame:
    # Deterministic tie-break: p_calibrated desc, ticker asc.
    return df.sort_values(
        ["p_calibrated", "ticker"], ascending=[False, True], kind="mergesort"
    )


# ---------------------------------------------------------------------------
# 1. Top-K (per-day + global)
# ---------------------------------------------------------------------------


def compute_top_k_metrics(
    df: pd.DataFrame, k_values: Iterable[int] = (1, 5, 10)
) -> dict[str, Any]:
    """Per-day top-K precision + global top-K precision.

    For each ``k``:
      ``per_day`` — group by date, take the top-``k`` rows per day, pool
        across days. Weighted aggregate
        ``p_at_k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))``
        where ``R(d)`` is the count of positives on day ``d``. The
        ``min(R(d), k)`` denominator (achievable positives) is mandatory —
        using the picks-made count (``min(k, n_tickers(d))``) silently
        mis-normalizes on staggered panels where ``R(d) < k`` for many
        days. ``lift = p_at_k / base_rate``. ``n_days_full_k`` counts days
        that had at least ``k`` tickers (= days where the pick size
        equals ``k``); ``n_days_R_lt_k`` counts days where ``R(d) < k``
        (the days for which the new denominator differs from the old).
      ``global`` — flat top-``k`` across the entire segment, with the
        analogous denominator: ``min(k, total_positives_in_segment)``
        (achievable positives across the segment). Chosen for consistency
        with the per-day block; segments with ``R_total < k`` are rare in
        practice but the recall-style semantics are well-defined.

    ``formula_version: "v2_min_R_d_k"`` (introduced 2026-05-28, fixes
    issue #45). Pre-fix artifacts have no such field — absence = v1
    (``"v1_picks_made"``, the buggy denominator). See
    ``.claude/memories/project-r-precision-methodology.md`` for the full
    spec and the bug history.
    """
    k_values = tuple(sorted(int(k) for k in k_values))
    if df is None or df.empty:
        return _empty_top_k(k_values)

    n_rows = int(len(df))
    y = df["y_true"].values.astype(int)
    base_rate = float(y.mean()) if n_rows else None
    total_positives = int(y.sum())

    out: dict[str, Any] = {
        "formula_version": "v2_min_R_d_k",
        "base_rate": base_rate,
        "n_rows": n_rows,
        "per_day": {},
        "global": {},
    }

    # ---- per-day blocks ----
    # Stable sort by date, then by score-desc / ticker-asc inside each date.
    sorted_df = df.sort_values(
        ["date", "p_calibrated", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    n_days_total = int(sorted_df["date"].nunique())
    grouped = sorted_df.groupby("date", sort=False)
    # Per-day R(d) = positives on day d.
    per_day_r = grouped["y_true"].sum().astype(int)

    for k in k_values:
        picks = grouped.head(k)
        n_picks_total = int(len(picks))
        n_pos = int(picks["y_true"].sum())
        # Day pick size = number of rows per date (capped at k by .head).
        # n_days_full_k = days where pick size == k (i.e., at least k tickers
        # were available that day).
        sizes = picks.groupby("date", sort=False).size()
        n_days_full_k = int((sizes >= k).sum())
        # Denominator under the corrected formula: sum_d(min(R(d), k))
        # = achievable positives. n_days_R_lt_k counts days where the
        # denominator is R(d) rather than k.
        clipped_r = per_day_r.clip(upper=k)
        n_denom = int(clipped_r.sum())
        n_days_R_lt_k = int((per_day_r < k).sum())
        p_at_k = float(n_pos / n_denom) if n_denom else None
        lift = (
            float(p_at_k / base_rate)
            if (p_at_k is not None and base_rate not in (None, 0.0))
            else None
        )
        out["per_day"][str(k)] = {
            "p_at_k": p_at_k,
            "n_picks_total": n_picks_total,
            "n_positives_in_picks": n_pos,
            "n_denom": n_denom,
            "n_days_R_lt_k": n_days_R_lt_k,
            "n_days_full_k": n_days_full_k,
            "n_days_total": n_days_total,
            "lift": lift,
        }

    # ---- global blocks ----
    flat_sorted = _sorted_by_score(df)
    for k in k_values:
        n_picks = int(min(k, n_rows))
        picks = flat_sorted.head(n_picks)
        n_pos = int(picks["y_true"].sum())
        # Achievable-positives denominator for the global block:
        # min(k, total_positives_in_segment). Matches the per-day
        # convention. Reduces to k whenever there are at least k positives
        # in the segment (the usual case).
        n_denom = int(min(k, total_positives))
        p_at_k = float(n_pos / n_denom) if n_denom else None
        lift = (
            float(p_at_k / base_rate)
            if (p_at_k is not None and base_rate not in (None, 0.0))
            else None
        )
        out["global"][str(k)] = {
            "p_at_k": p_at_k,
            "n_picks": n_picks,
            "n_positives_in_picks": n_pos,
            "n_denom": n_denom,
            "lift": lift,
        }

    return out


# ---------------------------------------------------------------------------
# 1b. Canonical R-Precision@K (macro) — matches the registry CSV
# ---------------------------------------------------------------------------


def compute_r_precision_at_k(
    df: pd.DataFrame, k_values: Iterable[int] = (1, 3, 5, 10, 20)
) -> dict[str, Any]:
    """Canonical R-Precision@K — per-day fixed K, **macro-averaged**.

    Reconciles the runner's segment_diagnostics surface with the
    project-canonical R-Precision@K registry produced by
    ``scripts/gbdt/regenerate_r_precision_at_k_csv.py`` (and the
    ``results/gbdt/data/r_precision_at_k.csv`` registry of record).

    Formula (per ``.claude/memories/project-r-precision-methodology.md``)::

        R-Precision@K = (1 / Q) · Σ_{q in days with R_q > 0}  r_q / min(K, R_q)

    where:
      - ``R_q`` = number of positives on day ``q``
      - ``r_q`` = positives caught in the top-``K`` picks on day ``q``,
        sorted by ``(p_calibrated desc, ticker asc)`` with stable
        mergesort (same tie-break as
        :func:`compute_top_k_metrics` and the regenerate script)
      - ``Q`` = count of days with ``R_q > 0``; days with no positives
        are skipped (``min(K, 0)`` is ill-defined and contributes no
        information).

    Distinct from :func:`compute_top_k_metrics`'s ``per_day.p_at_k``,
    which uses **micro** aggregation
    ``sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))``. Both forms
    are mathematically valid; macro is the cross-cell headline (the
    canonical CSV stores it) and matches what memos must report.
    See the methodology memory for the relationship between the two.

    Returns a dict::

        {
          "formula_version": "macro_per_day_fixed_k",
          "tie_break": "(p_calibrated desc, ticker asc) mergesort",
          "base_rate": float | None,
          "n_rows": int,
          "Q_days": int,     # days with R_q > 0 (the macro denominator)
          "by_k": {
             "1": {"r_precision_at_k": float | None,
                    "n_qualifying_days": int},
             "3": ...,
             "5": ...,
             "10": ...,
             "20": ...,
          },
        }

    Empty segment / no day with ``R_q > 0`` → all ``r_precision_at_k``
    values are ``None``.
    """
    k_values = tuple(sorted(int(k) for k in k_values))
    empty_by_k = {
        str(k): {"r_precision_at_k": None, "n_qualifying_days": 0}
        for k in k_values
    }
    if df is None or df.empty:
        return {
            "formula_version": "macro_per_day_fixed_k",
            "tie_break": "(p_calibrated desc, ticker asc) mergesort",
            "base_rate": None,
            "n_rows": 0,
            "Q_days": 0,
            "by_k": empty_by_k,
        }

    n_rows = int(len(df))
    base_rate = float(df["y_true"].mean()) if n_rows else None

    # Canonical tie-break, identical to compute_top_k_metrics + the
    # regenerate script: (date asc, p_calibrated desc, ticker asc)
    # stable mergesort. Sorting by date upfront lets the head(k) groupby
    # honor the within-day ranking deterministically.
    sorted_df = df.sort_values(
        ["date", "p_calibrated", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    grouped = sorted_df.groupby("date", sort=False)
    per_day_r = grouped["y_true"].sum().astype(int)
    qualifying_days = per_day_r[per_day_r > 0].index
    Q = int(len(qualifying_days))

    by_k: dict[str, Any] = {}
    if Q == 0:
        by_k = empty_by_k
    else:
        per_day_R_qual = per_day_r.reindex(qualifying_days)
        for k in k_values:
            picks = grouped.head(int(k))
            per_day_caught = (
                picks.groupby("date", sort=False)["y_true"].sum().astype(int)
            )
            per_day_caught = per_day_caught.reindex(
                qualifying_days, fill_value=0
            )
            per_day_denom = per_day_R_qual.clip(upper=int(k))
            per_day_ratio = (
                per_day_caught.astype(float) / per_day_denom.astype(float)
            )
            by_k[str(k)] = {
                "r_precision_at_k": float(per_day_ratio.mean()),
                "n_qualifying_days": Q,
            }

    return {
        "formula_version": "macro_per_day_fixed_k",
        "tie_break": "(p_calibrated desc, ticker asc) mergesort",
        "base_rate": base_rate,
        "n_rows": n_rows,
        "Q_days": Q,
        "by_k": by_k,
    }


# ---------------------------------------------------------------------------
# 2. Per-ticker hit-rate when picked
# ---------------------------------------------------------------------------


def compute_per_ticker_hit_rate(
    df: pd.DataFrame, k: int = 5
) -> dict[str, Any]:
    """Aggregate per-ticker hit-rate over per-day top-``k`` picks.

    Returns a dict with:
      ``k`` (int) — the per-day pick budget used.
      ``rows`` (list[dict]) — full table, one row per ticker that was
        picked at least once. Sorted by ``n_picks`` desc then ``ticker``
        asc. Each row: ``(ticker, n_picks, n_positives, hit_rate)``.

    ``hit_rate = n_positives / n_picks``.
    """
    if df is None or df.empty:
        return {"k": int(k), "rows": []}

    sorted_df = df.sort_values(
        ["date", "p_calibrated", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    picks = sorted_df.groupby("date", sort=False).head(int(k))
    if picks.empty:
        return {"k": int(k), "rows": []}

    agg = (
        picks.groupby("ticker")
        .agg(n_picks=("y_true", "size"), n_positives=("y_true", "sum"))
        .reset_index()
    )
    agg["hit_rate"] = agg["n_positives"] / agg["n_picks"]
    agg = agg.sort_values(
        ["n_picks", "ticker"], ascending=[False, True], kind="mergesort"
    )

    rows = [
        {
            "ticker": str(r.ticker),
            "n_picks": int(r.n_picks),
            "n_positives": int(r.n_positives),
            "hit_rate": float(r.hit_rate),
        }
        for r in agg.itertuples(index=False)
    ]
    return {"k": int(k), "rows": rows}


# ---------------------------------------------------------------------------
# 3. Per-quarter P@k stability
# ---------------------------------------------------------------------------


def compute_per_quarter_p_k(
    df: pd.DataFrame, k: int = 5
) -> dict[str, Any]:
    """Per-calendar-quarter P@k stability.

    Returns:
      ``k`` (int) — per-day pick budget used.
      ``rows`` (list[dict]) — one row per quarter that has any picks,
        sorted chronologically. Each row:
        ``(quarter, n_picks, n_positives, p_at_k, base_rate, lift)``.
        ``base_rate`` is the *segment-wide* base rate (not per-quarter);
        ``lift = p_at_k / base_rate``.
    """
    if df is None or df.empty:
        return {"k": int(k), "rows": []}

    base_rate = float(df["y_true"].mean())
    sorted_df = df.sort_values(
        ["date", "p_calibrated", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    # Ensure date is datetime for quarter extraction.
    dates = pd.to_datetime(sorted_df["date"])
    sorted_df = sorted_df.assign(quarter=dates.dt.to_period("Q").astype(str))

    picks = sorted_df.groupby("date", sort=False).head(int(k))
    if picks.empty:
        return {"k": int(k), "rows": []}

    agg = (
        picks.groupby("quarter", sort=True)
        .agg(n_picks=("y_true", "size"), n_positives=("y_true", "sum"))
        .reset_index()
    )
    agg["p_at_k"] = agg["n_positives"] / agg["n_picks"]
    agg["lift"] = agg["p_at_k"] / base_rate if base_rate > 0 else np.nan

    rows = [
        {
            "quarter": str(r.quarter),
            "n_picks": int(r.n_picks),
            "n_positives": int(r.n_positives),
            "p_at_k": float(r.p_at_k),
            "base_rate": base_rate,
            "lift": (float(r.lift) if pd.notna(r.lift) else None),
        }
        for r in agg.itertuples(index=False)
    ]
    return {"k": int(k), "rows": rows}


# ---------------------------------------------------------------------------
# 4. Prediction-range diagnostics
# ---------------------------------------------------------------------------


def compute_prediction_range(
    df: pd.DataFrame, low_separation_threshold: float = 0.05
) -> dict[str, Any]:
    """Min / max / std / mean of ``p_calibrated`` + a low-separation flag.

    The flag fires when ``std < low_separation_threshold``: the model's
    predictions cluster so tightly that ranking is essentially noise — the
    Sweep #1 H=100 pathology (std ≈ 0.04).
    """
    if df is None or df.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "n_rows": 0,
            "low_separation_threshold": float(low_separation_threshold),
            "flag_low_separation": False,
        }
    p = df["p_calibrated"].values.astype(float)
    std = float(np.std(p, ddof=0))
    return {
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "mean": float(np.mean(p)),
        "std": std,
        "n_rows": int(len(p)),
        "low_separation_threshold": float(low_separation_threshold),
        "flag_low_separation": bool(std < low_separation_threshold),
    }


# ---------------------------------------------------------------------------
# Bundle helper
# ---------------------------------------------------------------------------


def compute_all(
    df: pd.DataFrame,
    k_values: Iterable[int] = (1, 5, 10),
    per_ticker_k: int = 5,
    per_quarter_k: int = 5,
    low_separation_threshold: float = 0.05,
    r_precision_k_values: Iterable[int] = (1, 3, 5, 10, 20),
) -> dict[str, Any]:
    """Compute all segment diagnostics. Returns the bundle dict
    the report layer threads into ``metrics.json``.

    The ``r_precision_at_k`` block carries the **macro-aggregated**
    R-Precision@K values that match the canonical CSV at
    ``results/gbdt/data/r_precision_at_k.csv`` — distinct from the
    ``top_k_metrics.per_day.p_at_k`` field, which is the
    micro-aggregated form (back-compat, ``formula_version: v2_min_R_d_k``).
    Memo authors should read ``r_precision_at_k`` for cross-cell
    headlines; see ``.claude/memories/project-r-precision-methodology.md``.
    """
    return {
        "top_k_metrics": compute_top_k_metrics(df, k_values=k_values),
        "r_precision_at_k": compute_r_precision_at_k(
            df, k_values=r_precision_k_values
        ),
        "per_ticker_hit_rate": compute_per_ticker_hit_rate(df, k=per_ticker_k),
        "per_quarter_p_k": compute_per_quarter_p_k(df, k=per_quarter_k),
        "prediction_range": compute_prediction_range(
            df, low_separation_threshold=low_separation_threshold
        ),
    }


__all__ = [
    "compute_top_k_metrics",
    "compute_r_precision_at_k",
    "compute_per_ticker_hit_rate",
    "compute_per_quarter_p_k",
    "compute_prediction_range",
    "compute_all",
]
