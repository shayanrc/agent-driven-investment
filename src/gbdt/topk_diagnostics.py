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
- ``compute_per_ticker_hit_rate(df, k=5)``
- ``compute_per_quarter_p_k(df, k=5)``
- ``compute_prediction_range(df, low_separation_threshold=0.05)``
- ``compute_all(df, ...)`` — bundle helper that calls the four above.
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
        "n_days_full_k": 0,
        "n_days_total": 0,
        "lift": None,
    }


def _empty_global_k() -> dict[str, Any]:
    return {
        "p_at_k": None,
        "n_picks": 0,
        "n_positives_in_picks": 0,
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
        across days. ``p_at_k`` = positives in picks / picks; ``lift`` =
        ``p_at_k / base_rate``. ``n_days_full_k`` counts days that had at
        least ``k`` tickers (= days where the pick size equals ``k``).
      ``global`` — flat top-``k`` across the entire segment.
    """
    k_values = tuple(sorted(int(k) for k in k_values))
    if df is None or df.empty:
        return _empty_top_k(k_values)

    n_rows = int(len(df))
    y = df["y_true"].values.astype(int)
    base_rate = float(y.mean()) if n_rows else None

    out: dict[str, Any] = {
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

    for k in k_values:
        picks = grouped.head(k)
        n_picks_total = int(len(picks))
        n_pos = int(picks["y_true"].sum())
        # Day pick size = number of rows per date (capped at k by .head).
        # n_days_full_k = days where pick size == k (i.e., at least k tickers
        # were available that day).
        sizes = picks.groupby("date", sort=False).size()
        n_days_full_k = int((sizes >= k).sum())
        p_at_k = float(n_pos / n_picks_total) if n_picks_total else None
        lift = (
            float(p_at_k / base_rate)
            if (p_at_k is not None and base_rate not in (None, 0.0))
            else None
        )
        out["per_day"][str(k)] = {
            "p_at_k": p_at_k,
            "n_picks_total": n_picks_total,
            "n_positives_in_picks": n_pos,
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
        p_at_k = float(n_pos / n_picks) if n_picks else None
        lift = (
            float(p_at_k / base_rate)
            if (p_at_k is not None and base_rate not in (None, 0.0))
            else None
        )
        out["global"][str(k)] = {
            "p_at_k": p_at_k,
            "n_picks": n_picks,
            "n_positives_in_picks": n_pos,
            "lift": lift,
        }

    return out


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
) -> dict[str, Any]:
    """Compute all four diagnostics on one segment. Returns the bundle dict
    the report layer threads into ``metrics.json``."""
    return {
        "top_k_metrics": compute_top_k_metrics(df, k_values=k_values),
        "per_ticker_hit_rate": compute_per_ticker_hit_rate(df, k=per_ticker_k),
        "per_quarter_p_k": compute_per_quarter_p_k(df, k=per_quarter_k),
        "prediction_range": compute_prediction_range(
            df, low_separation_threshold=low_separation_threshold
        ),
    }


__all__ = [
    "compute_top_k_metrics",
    "compute_per_ticker_hit_rate",
    "compute_per_quarter_p_k",
    "compute_prediction_range",
    "compute_all",
]
