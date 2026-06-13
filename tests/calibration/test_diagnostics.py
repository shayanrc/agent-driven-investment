"""Tests for calibration diagnostics (ECE + reliability diagram)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np

from calibration.diagnostics import (
    expected_calibration_error,
    reliability_diagram,
)


def test_ece_near_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    ece = expected_calibration_error(p, y, n_bins=10)
    assert ece < 0.02


def test_ece_large_for_overconfident():
    rng = np.random.default_rng(0)
    # Model says 0.9 everywhere but true rate is 0.3.
    p = np.full(10000, 0.9)
    y = (rng.uniform(size=p.size) < 0.3).astype(float)
    ece = expected_calibration_error(p, y, n_bins=10)
    assert ece > 0.5


def test_ece_weighted_path():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    w = rng.uniform(0.5, 2.0, p.size)
    ece = expected_calibration_error(p, y, n_bins=10, sample_weight=w)
    assert 0.0 <= ece < 0.05


def test_reliability_diagram_returns_axes_with_bands():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    lo = np.clip(p - 0.1, 0, 1)
    hi = np.clip(p + 0.1, 0, 1)
    ax = reliability_diagram(p, y, n_bins=8, p_low=lo, p_high=hi, title="t")
    assert ax.get_title() == "t"
    # The credible-band fill_between adds a PolyCollection.
    assert len(ax.collections) >= 1
