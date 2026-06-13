"""Tests for BetaBinomialBucketed (plan §5.1 / §6.1, R8)."""

from __future__ import annotations

import numpy as np
import pytest

from calibration import CalibrationOutput
from calibration.bayesian import BetaBinomialBucketed


def _well_separated(n=5000, seed=0):
    """p_raw spread across [0,1]; y drawn so the true rate ~ p_raw."""
    rng = np.random.default_rng(seed)
    p_raw = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < p_raw).astype(float)
    return p_raw, y


def test_fit_transform_shapes_and_ordering():
    p_raw, y = _well_separated()
    cal = BetaBinomialBucketed(n_bins=10).fit(p_raw, y)
    out = cal.transform(p_raw)
    assert isinstance(out, CalibrationOutput)
    assert out.p_mean.shape == p_raw.shape
    # Credible interval brackets the mean everywhere.
    assert np.all(out.p_low <= out.p_mean + 1e-9)
    assert np.all(out.p_mean <= out.p_high + 1e-9)
    assert np.all((out.p_low >= 0) & (out.p_high <= 1))


def test_posterior_recovers_monotone_rate():
    """On well-calibrated data, posterior mean should track p_raw upward."""
    p_raw, y = _well_separated(n=20000)
    cal = BetaBinomialBucketed(n_bins=10).fit(p_raw, y)
    # Transform the bin midpoints and check monotone-ish increase.
    grid = np.linspace(0.05, 0.95, 10)
    means = cal.transform(grid).p_mean
    # Spearman-ish: mostly increasing.
    assert means[0] < means[-1]
    diffs = np.diff(means)
    assert (diffs >= -0.05).all()  # allow tiny non-monotone wobble


def test_credible_band_narrows_with_more_data():
    p_raw, y = _well_separated(n=200)
    p_raw2, y2 = _well_separated(n=20000)
    cal_small = BetaBinomialBucketed(n_bins=5).fit(p_raw, y)
    cal_big = BetaBinomialBucketed(n_bins=5).fit(p_raw2, y2)
    mid = np.array([0.5])
    w_small = (cal_small.transform(mid).p_high - cal_small.transform(mid).p_low)[0]
    w_big = (cal_big.transform(mid).p_high - cal_big.transform(mid).p_low)[0]
    assert w_big < w_small


def test_tie_handling_drops_duplicate_edges():
    """Tiny model: few distinct p_raw → duplicate quantile edges dropped (R8)."""
    rng = np.random.default_rng(1)
    # Only 4 distinct p_raw values, one of them >40% of mass (forces a
    # duplicate quantile edge under naive np.quantile).
    base = np.array([0.30, 0.30, 0.30, 0.30, 0.38, 0.42, 0.50])
    p_raw = rng.choice(base, size=4000, p=[0.18, 0.18, 0.18, 0.16, 0.12, 0.10, 0.08])
    y = (rng.uniform(size=p_raw.size) < p_raw).astype(float)
    cal = BetaBinomialBucketed(n_bins=10, min_effective_bins=3).fit(p_raw, y)
    # Survived with >=3 effective bins despite the 10 requested.
    assert cal.fit_diagnostics_["effective_n_bins"] >= 3
    assert cal.fit_diagnostics_["effective_n_bins"] <= 4  # only 4 distinct vals
    out = cal.transform(p_raw)
    assert out.p_mean.shape == p_raw.shape


def test_raises_when_too_few_effective_bins():
    """Single distinct p_raw → can't form min_effective_bins → ValueError."""
    p_raw = np.full(1000, 0.4)
    y = (np.arange(1000) % 3 == 0).astype(float)
    with pytest.raises(ValueError, match="bins survived"):
        BetaBinomialBucketed(n_bins=10, min_effective_bins=3).fit(p_raw, y)


def test_min_bin_size_merges_sparse_tail():
    """Sparse upper region merges so each surviving bin has >= min_bin_size."""
    rng = np.random.default_rng(2)
    # Dense low region, very sparse high region.
    low = rng.uniform(0.0, 0.4, 4000)
    high = rng.uniform(0.8, 1.0, 30)
    p_raw = np.concatenate([low, high])
    y = (rng.uniform(size=p_raw.size) < p_raw).astype(float)
    cal = BetaBinomialBucketed(n_bins=10, min_bin_size=50).fit(p_raw, y)
    bin_n = np.array(cal.fit_diagnostics_["bin_n"])
    # Every surviving bin (except possibly a folded last) clears min_bin_size.
    assert (bin_n >= 50).sum() >= len(bin_n) - 1


def test_sample_weight_affects_posterior():
    p_raw, y = _well_separated(n=4000)
    cal_unw = BetaBinomialBucketed(n_bins=5).fit(p_raw, y)
    w = np.where(y == 1, 5.0, 1.0)  # upweight positives
    cal_w = BetaBinomialBucketed(n_bins=5).fit(p_raw, y, sample_weight=w)
    mid = np.array([0.5])
    assert cal_w.transform(mid).p_mean[0] > cal_unw.transform(mid).p_mean[0]


def test_transform_clips_out_of_range():
    p_raw, y = _well_separated()
    cal = BetaBinomialBucketed(n_bins=10).fit(p_raw, y)
    # Values outside the fitted range map to the edge bins, no crash.
    out = cal.transform(np.array([-0.5, 0.0, 1.0, 2.0]))
    assert out.p_mean.shape == (4,)
    assert np.all(np.isfinite(out.p_mean))


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        BetaBinomialBucketed().transform(np.array([0.5]))
