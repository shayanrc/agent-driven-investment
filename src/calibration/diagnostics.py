"""Calibration diagnostics — ECE + reliability diagram (plan §5.1).

Calibration-specific metrics live here, not in a generic metric library
(goal.md: "Diagnostics are shipped, not optional"). Both functions operate
on plain arrays — no predictor or strategy coupling.
"""

from __future__ import annotations

import numpy as np

# matplotlib is imported lazily inside reliability_diagram so importing this
# module (e.g. for ECE only) never pulls in the plotting stack.


def expected_calibration_error(
    p_calibrated: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Weighted expected calibration error over equal-width probability bins.

    ECE = Σ_b (n_b / N) · |acc_b − conf_b|, where ``conf_b`` is the mean
    predicted probability in bin ``b`` and ``acc_b`` the (weighted) observed
    positive rate. Bins are equal-WIDTH on [0, 1] (the standard ECE
    convention), independent of the calibrator's quantile binning.
    """
    p = np.asarray(p_calibrated, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"p shape {p.shape} != y shape {y.shape}")
    if p.size == 0:
        raise ValueError("expected_calibration_error: empty input")
    w = (
        np.ones_like(p)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index in [0, n_bins-1]; clip the right edge into the last bin.
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, n_bins - 1)
    total_w = w.sum()
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        wb = w[mask].sum()
        if wb == 0.0:
            continue
        conf_b = np.average(p[mask], weights=w[mask])
        acc_b = np.average(y[mask], weights=w[mask])
        ece += (wb / total_w) * abs(acc_b - conf_b)
    return float(ece)


def reliability_diagram(
    p_calibrated: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    *,
    p_low: np.ndarray | None = None,
    p_high: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot a reliability diagram with optional 95% credible bands.

    Returns the matplotlib ``Axes``. When ``p_low`` / ``p_high`` are passed
    (the per-row credible bounds from a Bayesian calibrator), the per-bin
    mean band is drawn as a shaded region so band width is visible per bin
    (the plan's Stage-5 checkpoint reads ``max_band_width`` off this).
    """
    import matplotlib.pyplot as plt

    p = np.asarray(p_calibrated, dtype=float)
    y = np.asarray(y_true, dtype=float)
    w = (
        np.ones_like(p)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, n_bins - 1)

    conf, acc, lo_band, hi_band = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        if w[mask].sum() == 0.0:
            continue
        conf.append(np.average(p[mask], weights=w[mask]))
        acc.append(np.average(y[mask], weights=w[mask]))
        if p_low is not None and p_high is not None:
            lo_band.append(np.average(np.asarray(p_low)[mask], weights=w[mask]))
            hi_band.append(np.average(np.asarray(p_high)[mask], weights=w[mask]))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(conf, acc, "o-", color="C0", label="observed")
    if lo_band:
        ax.fill_between(
            conf, lo_band, hi_band, alpha=0.2, color="C0",
            label="95% credible band",
        )
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or "Reliability diagram")
    ax.legend(loc="upper left", fontsize=8)
    return ax
