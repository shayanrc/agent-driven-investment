"""Sample-uniqueness weighting for overlapping-label correction.

Reference: López de Prado, *Advances in Financial Machine Learning* §4.4–4.5.

**Why this exists.** gbdt experiments label every ``(ticker, date)`` row with
the question "did this ticker breach +X% within H days while keeping
drawdown ≤ Y%?". Because the H-day forward window overlaps across
consecutive entry dates, the SAME outcome event labels many adjacent rows
the same way. The result is severe inflation of effective sample size:
sweep exp #1 (nasdaq100 +10%/100d/dd5%) observed prevalence 42.4% in the
overlap-naive training panel vs 19.7% in non-overlapping EDA — a 2.15×
bias.

**The fix.** Down-weight each row by the count of other rows whose
forward windows overlap with it. After down-weighting, ``Σ weights`` is
approximately the number of *independent* forward events the panel
covers, and weighted loss/metrics report what the model has actually
learned about independent outcomes (not the same outcome counted 2H+1
times).

**Algorithm choice.** This module implements the simpler closed-form
"forward-window overlap" approximation. The forward window matches the
target builder's convention — `(t, t+H]` (exclusive at entry, inclusive
at the horizon), length ``H``. Two rows ``i, j`` (same ticker, ``i < j``)
share a future bar iff ``j - i ≤ H - 1``. For each ticker independently:

- An interior row at position ``i`` with both ``i ≥ H - 1`` and
  ``N - 1 - i ≥ H - 1`` overlaps with exactly ``2(H - 1)`` other rows,
  giving weight ``1 / (2H - 1)``.
- Edge rows overlap with fewer neighbors; weight is
  ``1 / (overlap_count + 1)`` with
  ``overlap_count = min(i, H - 1) + min(N - 1 - i, H - 1)``.
- ``horizon = 1`` is a no-op: each window is a single future bar, no two
  windows share a bar, and every weight is exactly ``1.0``.

For a single ticker with ``N`` rows ≥ ``2H - 1``,
``Σ weights ≈ N / (2H - 1)``. For the task's H=100 sweep cell that's
``N / 199`` — close enough to the LdP §4.5 closed-form ``N / (2H + 1)``
(≈ N / 201) that the documented "within 5%" guard from the test plan
holds.

The exact LdP indicator-co-occurrence algorithm differs from this
approximation only in the presence of non-uniform date spacing within a
ticker — for the contiguous business-day panels this codebase ships, the
two agree to within rounding. The forward-window-only convention used
here is the principled choice for matching :func:`gbdt.targets.build_target`,
whose label is defined over ``(t, t+H]`` exclusive of the entry bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_uniqueness_weights(
    panel: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """Compute per-row sample-uniqueness weights for an overlapping-label panel.

    Parameters
    ----------
    panel : pd.DataFrame
        MultiIndex ``(date, ticker)`` panel. Only the index is used; the
        columns are ignored. Rows are assumed sorted within each ticker by
        date (the gbdt loader produces this layout).
    horizon : int
        Forward window length ``H`` in trading rows. Must be ``>= 1``.

    Returns
    -------
    pd.Series
        Weight per row, aligned to ``panel.index``. Weights are in
        ``(0, 1]``: an isolated row gets weight 1.0, an interior row in a
        densely-overlapping region gets weight ``1 / (2H - 1)``.

    Notes
    -----
    Weights are computed independently per ticker — two tickers' rows
    never count as "overlapping" even on the same date.

    For ``horizon == 1`` every row's forward window is a single future
    bar disjoint from every other row's — so all weights are exactly 1.0
    (a no-op). To skip uniqueness weighting entirely (e.g. for legacy
    reproduction of the overlap-naive baseline) the caller sets
    ``target.uniqueness_weighting: false`` in the spec.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if "ticker" not in panel.index.names or "date" not in panel.index.names:
        raise ValueError(
            "panel must have a MultiIndex with names ('date', 'ticker')"
        )

    H = int(horizon)
    weights = pd.Series(np.nan, index=panel.index, name="_sample_weight")

    # Iterate tickers. Sort each ticker's slice by date so positional
    # neighbor logic is correct.
    for ticker in panel.index.get_level_values("ticker").unique():
        sub_idx = panel.xs(ticker, level="ticker").sort_index().index
        N = len(sub_idx)
        if N == 0:
            continue
        w = _per_ticker_weights(N, H)
        # Place back into the (date, ticker) MultiIndex
        mi = pd.MultiIndex.from_arrays(
            [sub_idx, [ticker] * N], names=["date", "ticker"],
        )
        weights.loc[mi] = w

    # Defensive: any row not assigned (shouldn't happen with the loop above)
    # falls back to 1.0 rather than NaN so downstream sample_weight= calls
    # don't crash.
    weights = weights.fillna(1.0)
    return weights


def _per_ticker_weights(n: int, horizon: int) -> np.ndarray:
    """Per-row weights for a single ticker with ``n`` rows + horizon ``H``.

    Vectorized — no Python loops over rows. For positions
    ``i ∈ {0, ..., n-1}``:

        radius           = H - 1                    (overlap reach on each side)
        overlap_count(i) = min(i, radius) + min(n - 1 - i, radius)
        weight(i)        = 1 / (overlap_count(i) + 1)

    For ``n ≥ 2H - 1`` interior rows have weight ``1 / (2H - 1)``; for
    ``H == 1`` the radius is ``0`` and every weight is ``1.0``.
    """
    H = int(horizon)
    radius = H - 1
    i = np.arange(n)
    if radius <= 0:
        return np.ones(n, dtype=float)
    left = np.minimum(i, radius)
    right = np.minimum(n - 1 - i, radius)
    overlap = left + right
    return 1.0 / (overlap + 1.0)


def effective_sample_size(weights: np.ndarray | pd.Series) -> float:
    """Kish's effective sample size: ``(Σ w)² / Σ w²``.

    Reduces to ``n`` when all weights are equal (or all 1.0). For a panel
    with uniform interior weight ``1 / (2H + 1)``, ESS approaches
    ``n / (2H + 1)`` from below as edges' larger weights pull ``Σ w²``
    up — i.e. ESS is a tight lower bound on the number of independent
    events the panel encodes.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        return 0.0
    s = float(w.sum())
    sq = float((w * w).sum())
    if sq <= 0:
        return 0.0
    return (s * s) / sq


# ---------------------------------------------------------------------------
# Weighted metrics
# ---------------------------------------------------------------------------


def weighted_brier(
    y_true: np.ndarray | pd.Series,
    p_pred: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
) -> float:
    """Weighted Brier score: ``Σ w (p - y)² / Σ w``.

    With ``weights=None`` or uniform weights this reduces exactly to
    :func:`sklearn.metrics.brier_score_loss`.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    if y.size != p.size:
        raise ValueError(f"length mismatch: y={y.size}, p={p.size}")
    if y.size == 0:
        return float("nan")
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != y.size:
            raise ValueError(
                f"weights length {w.size} != y/p length {y.size}"
            )
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    return float(np.sum(w * (p - y) ** 2) / total)


def weighted_auc(
    y_true: np.ndarray | pd.Series,
    p_pred: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
) -> float | None:
    """Weighted ROC-AUC via :func:`sklearn.metrics.roc_auc_score`.

    Returns ``None`` if AUC is undefined (single-class ``y_true``).
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    if weights is None:
        return float(roc_auc_score(y, p))
    w = np.asarray(weights, dtype=float).ravel()
    return float(roc_auc_score(y, p, sample_weight=w))


def weighted_spiegelhalter_z(
    y_true: np.ndarray | pd.Series,
    p_pred: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
) -> tuple[float, float]:
    """Weighted Spiegelhalter Z + two-sided p-value.

    The unweighted statistic is
    ``Z = Σ (y - p)(1 - 2p) / sqrt(Σ (1 - 2p)² p (1 - p))``. The weighted
    variant down-weights both the numerator's per-row residual contribution
    AND the variance estimate by ``w_i`` — equivalent to repeating each
    row ``w_i`` times and taking the standard Z under exchangeable noise.

    With ``weights=None`` this matches :func:`gbdt.calibration.spiegelhalter_z`.
    """
    from scipy import stats

    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(p_pred, dtype=float).ravel(), 1e-7, 1.0 - 1e-7)
    if y.size != p.size:
        raise ValueError(f"length mismatch: y={y.size}, p={p.size}")
    if y.size == 0:
        return 0.0, 1.0
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != y.size:
            raise ValueError(
                f"weights length {w.size} != y/p length {y.size}"
            )
    num = float(np.sum(w * (y - p) * (1.0 - 2.0 * p)))
    var = float(np.sum(w * (1.0 - 2.0 * p) ** 2 * p * (1.0 - p)))
    if var <= 0:
        return 0.0, 1.0
    z = num / np.sqrt(var)
    pval = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(pval)


__all__ = [
    "compute_uniqueness_weights",
    "effective_sample_size",
    "weighted_brier",
    "weighted_auc",
    "weighted_spiegelhalter_z",
]
