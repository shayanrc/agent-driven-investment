"""Pure, package-level statistical helpers shared by gbdt diagnostics.

These functions are pure (numpy/pandas only, no I/O, no model object, no
feature-matrix rebuild) and are needed in **two** places:

- ``scripts/gbdt/diagnose.py`` (the on-disk ``/gbdt-diagnose`` verb) and
  ``scripts/gbdt/compute_r_precision.py`` (the R-precision CLI), which is where
  they originated.
- ``src/gbdt/diagnose_payload.py`` (the V1.1 Phase-3 in-loop diagnose-shaped
  payload), which is **package** code.

They live here in ``src/gbdt/`` (the installed package) rather than under
``scripts/`` so the dependency flows the right way: package code never imports
from the ad-hoc ``scripts/`` tree (which is not part of the wheel and only
resolves when the CWD is the repo root). ``scripts/gbdt/diagnose.py`` and
``scripts/gbdt/compute_r_precision.py`` re-export these names so their public
API + CLI are unchanged.

No metric is re-derived: the in-loop payload and the on-disk diagnose call the
*same* function objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Overfit threshold on train_val_gap (= val_brier - train_brier); matches the
# fallback regularization trigger in src/gbdt/train.py.
_OVERFIT_GAP_THR = 0.02


def assess_overfit(
    train_val_gap: float | None, *, threshold: float = _OVERFIT_GAP_THR
) -> bool | None:
    """Is the cell free of overfit? Based on the train/val gap ALONE.

    ``gap = val_brier - train_brier``; POSITIVE = val worse than train = overfit.
    No-overfit when ``gap <= threshold`` (default 0.02, matching the fallback
    regularization trigger in src/gbdt/train.py). Early-stopping firing is
    deliberately NOT a factor: it's the healthy mechanism that selects the tree
    count, not an overfit signal — a model can early-stop at tree 67 with a
    deeply negative gap (val better than train), which is the opposite of
    overfit. Returns None when the gap is unavailable.
    """
    if train_val_gap is None:
        return None
    return bool(train_val_gap <= threshold)


def prevalence_drift(seg_prevalence: dict[str, float]) -> dict:
    """Given per-segment positive prevalence, flag non-stationarity."""
    vals = [v for v in seg_prevalence.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return {"spread": float("nan"), "drift_flag": False, "monotone_decline": False}
    spread = float(max(vals) - min(vals))
    order = [seg_prevalence.get(s) for s in ("train", "val", "eval", "test")
             if seg_prevalence.get(s) is not None]
    monotone_decline = bool(len(order) >= 3 and all(
        order[i] >= order[i + 1] for i in range(len(order) - 1)))
    # flag if the spread is a large fraction of the mean prevalence
    drift_flag = bool(spread > 0.5 * (np.mean(vals) if np.mean(vals) > 0 else 1))
    return {"spread": spread, "drift_flag": drift_flag,
            "monotone_decline": monotone_decline}


def per_day_r_precision(preds: pd.DataFrame) -> dict:
    """Compute per-day R-precision.

    Args:
        preds: dataframe with columns date, ticker, p_calibrated, y_true.

    Returns:
        dict with keys:
            r_precision_mean_unweighted: mean over days (R>0) of per-day R-precision
            r_precision_weighted:        sum(correct@R) / sum(R) — global recall@R
            base_rate_mean_unweighted:   mean per-day R(d)/n(d) (random-picker baseline)
            base_rate_weighted:          total positives / total rows
            lift_mean:                   r_precision_mean / base_rate_mean
            lift_weighted:               r_precision_weighted / base_rate_weighted
            n_days_total
            n_days_with_positives
            per_day_rprec_quantiles:     {p10, p25, p50, p75, p90}
            r_distribution:              {min, mean, max} of R(d) for days with R>0
    """
    per_day = []
    for date, day_df in preds.groupby("date"):
        n = len(day_df)
        r = int(day_df["y_true"].sum())
        if r == 0 or r > n:
            # skip degenerate days for the rprec mean; still count for stats
            per_day.append({"date": date, "n": n, "R": r, "correct_at_R": None, "rprec": None})
            continue

        # canonical tie-break: (p_calibrated desc, ticker asc), stable mergesort
        ordered = day_df.sort_values(
            by=["p_calibrated", "ticker"],
            ascending=[False, True],
            kind="mergesort",
        )
        top_r = ordered.head(r)
        correct = int(top_r["y_true"].sum())
        per_day.append({"date": date, "n": n, "R": r, "correct_at_R": correct, "rprec": correct / r})

    df = pd.DataFrame(per_day)
    valid = df[df["rprec"].notna()].copy()

    if len(valid) == 0:
        return {
            "r_precision_mean_unweighted": None,
            "r_precision_weighted": None,
            "base_rate_mean_unweighted": None,
            "base_rate_weighted": None,
            "lift_mean": None,
            "lift_weighted": None,
            "n_days_total": int(len(df)),
            "n_days_with_positives": 0,
            "per_day_rprec_quantiles": None,
            "r_distribution": None,
        }

    rprec_mean = float(valid["rprec"].mean())
    total_correct = int(valid["correct_at_R"].sum())
    total_r = int(valid["R"].sum())
    rprec_weighted = total_correct / total_r if total_r > 0 else None

    base_rates = (valid["R"] / valid["n"]).astype(float)
    base_rate_mean = float(base_rates.mean())
    total_rows = int(valid["n"].sum())
    base_rate_weighted = total_r / total_rows if total_rows > 0 else None

    quantiles = valid["rprec"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict()
    quantiles = {f"p{int(q*100)}": float(v) for q, v in quantiles.items()}

    return {
        "r_precision_mean_unweighted": rprec_mean,
        "r_precision_weighted": float(rprec_weighted) if rprec_weighted is not None else None,
        "base_rate_mean_unweighted": base_rate_mean,
        "base_rate_weighted": float(base_rate_weighted) if base_rate_weighted is not None else None,
        "lift_mean": float(rprec_mean / base_rate_mean) if base_rate_mean > 0 else None,
        "lift_weighted": float(rprec_weighted / base_rate_weighted) if rprec_weighted and base_rate_weighted else None,
        "n_days_total": int(len(df)),
        "n_days_with_positives": int(len(valid)),
        "per_day_rprec_quantiles": quantiles,
        "r_distribution": {
            "min": int(valid["R"].min()),
            "mean": float(valid["R"].mean()),
            "max": int(valid["R"].max()),
        },
    }


__all__ = [
    "_OVERFIT_GAP_THR",
    "assess_overfit",
    "prevalence_drift",
    "per_day_r_precision",
]
