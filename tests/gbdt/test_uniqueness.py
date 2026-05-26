"""Tests for sample-uniqueness weighting (LdP §4.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt.uniqueness import (
    compute_uniqueness_weights,
    effective_sample_size,
    weighted_auc,
    weighted_brier,
    weighted_spiegelhalter_z,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(n: int, ticker: str = "A") -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "close": np.arange(n, dtype=float) + 1.0,
    })
    return df.set_index(["date", "ticker"]).sort_index()


def _multi_ticker_panel(n_per: dict[str, int]) -> pd.DataFrame:
    frames = []
    for t, n in n_per.items():
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        frames.append(pd.DataFrame({
            "date": dates,
            "ticker": t,
            "close": np.arange(n, dtype=float) + 1.0,
        }))
    df = pd.concat(frames, ignore_index=True)
    return df.set_index(["date", "ticker"]).sort_index()


# ---------------------------------------------------------------------------
# Core weight properties
# ---------------------------------------------------------------------------


def test_isolated_row_weight_is_one():
    """A single row in isolation has nothing to overlap with → weight 1.0."""
    panel = _make_panel(1)
    w = compute_uniqueness_weights(panel, horizon=1)
    assert len(w) == 1
    assert float(w.iloc[0]) == pytest.approx(1.0)


def test_horizon_1_is_no_op():
    """horizon=1: each forward window covers one bar, no two windows
    share a bar, all weights should be 1.0."""
    panel = _make_panel(50)
    w = compute_uniqueness_weights(panel, horizon=1)
    assert np.allclose(w.values, 1.0)


def test_dense_overlap_weights_sum_to_n_over_h():
    """For a single ticker with N rows and horizon H,
    Σ weights ≈ N / (2H - 1) under the forward-window convention.

    Spec from the implementation plan tolerates a 5% slack vs the LdP
    closed-form N/(2H+1); our forward-window-only convention sits at
    N/(2H-1). Both are within the 5% band for H=100 / N=2000.
    """
    N, H = 2000, 100
    panel = _make_panel(N)
    w = compute_uniqueness_weights(panel, horizon=H)
    s = float(w.sum())
    # Exact closed form for our convention:
    # Σ = (N - 2(H-1)) / (2H-1) + 2 * Σ_{k=0..H-2} 1/(k + H)
    expected = (N - 2 * (H - 1)) / (2 * H - 1) + 2 * sum(
        1.0 / (k + H) for k in range(H - 1)
    )
    assert s == pytest.approx(expected, rel=1e-9)
    # And it sits within 5% of the LdP closed-form N/(2H+1):
    ldp_approx = N / (2 * H + 1)
    assert abs(s - ldp_approx) / ldp_approx < 0.05


def test_interior_weight_is_one_over_two_h_minus_one():
    """Interior rows (positions in [H-1, N-1-(H-1)]) have weight 1/(2H-1)."""
    N, H = 200, 20
    panel = _make_panel(N)
    w = compute_uniqueness_weights(panel, horizon=H).values
    interior = w[H - 1: N - (H - 1)]
    assert np.allclose(interior, 1.0 / (2 * H - 1))


def test_disjoint_tickers_independent():
    """Two tickers' weights are computed independently — overlap is
    intra-ticker only, never cross-ticker."""
    panel = _multi_ticker_panel({"A": 100, "B": 50})
    w = compute_uniqueness_weights(panel, horizon=10)
    # Subset A: should equal what we get for A alone
    w_a = compute_uniqueness_weights(_make_panel(100, "A"), horizon=10)
    w_b = compute_uniqueness_weights(_make_panel(50, "B"), horizon=10)
    a_from_joint = w.xs("A", level="ticker").sort_index().values
    b_from_joint = w.xs("B", level="ticker").sort_index().values
    assert np.allclose(a_from_joint, w_a.values)
    assert np.allclose(b_from_joint, w_b.values)


def test_weights_aligned_to_panel_index():
    """Returned Series has the same index as the input panel — no shuffling."""
    panel = _multi_ticker_panel({"A": 30, "B": 30, "C": 30})
    w = compute_uniqueness_weights(panel, horizon=5)
    assert w.index.equals(panel.index)


def test_weights_in_open_interval_zero_one():
    """All weights are in (0, 1]."""
    panel = _make_panel(50)
    for H in (1, 2, 5, 10, 50):
        w = compute_uniqueness_weights(panel, horizon=H)
        assert (w > 0).all()
        assert (w <= 1.0).all()


def test_short_ticker_below_horizon():
    """Ticker with fewer rows than the horizon still produces valid weights."""
    panel = _make_panel(5)
    w = compute_uniqueness_weights(panel, horizon=10)
    # Every row overlaps with every other row → weights = 1/5 each
    assert np.allclose(w.values, 1.0 / 5.0)


def test_horizon_validation():
    panel = _make_panel(10)
    with pytest.raises(ValueError):
        compute_uniqueness_weights(panel, horizon=0)
    with pytest.raises(ValueError):
        compute_uniqueness_weights(panel, horizon=-3)


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------


def test_effective_sample_size_kish():
    """Kish's ESS reduces to n for uniform weights and to specific values
    for skewed weights."""
    # Uniform weights
    w = np.ones(100)
    assert effective_sample_size(w) == pytest.approx(100.0)

    # All weight on one row → ESS == 1
    w = np.zeros(100)
    w[0] = 1.0
    assert effective_sample_size(w) == pytest.approx(1.0)

    # Half-half: 50 rows at weight 1, 50 at weight 0 → ESS == 50
    w = np.array([1.0] * 50 + [0.0] * 50)
    assert effective_sample_size(w) == pytest.approx(50.0)

    # Uniformly-rescaled weights have same ESS as unscaled (Kish invariance)
    w1 = np.array([2.0, 2.0, 2.0])
    w2 = np.array([0.5, 0.5, 0.5])
    assert effective_sample_size(w1) == pytest.approx(effective_sample_size(w2))


def test_sum_of_weights_approximates_independent_events():
    """Σ(weights) is the natural measure of "number of independent forward
    events" the panel encodes — close to ``N / (2H - 1)`` for a
    contiguous single-ticker series of length N >> H."""
    N, H = 2000, 100
    panel = _make_panel(N)
    w = compute_uniqueness_weights(panel, horizon=H)
    s = float(w.sum())
    base = N / (2 * H - 1)
    # Σw is interior contribution + edge tail; the edge adds <2 to the
    # sum so within ~20% of base.
    assert base * 0.95 < s < base * 1.25


def test_kish_ess_recovers_n_for_uniform_uniqueness_weights():
    """Kish's ESS = (Σw)²/Σw² is invariant to uniform rescaling of
    weights — for an interior-uniform 1/(2H-1) weighting it recovers
    something close to N (not N/(2H-1)). Σw is the right "independent
    events" measure; Kish ESS measures the *variance penalty* from
    non-uniform weights, which is small when most rows sit in the
    interior."""
    N, H = 2000, 100
    panel = _make_panel(N)
    w = compute_uniqueness_weights(panel, horizon=H)
    ess = effective_sample_size(w.values)
    # Should be within ~5% of N — the edges' larger weights are the
    # only source of variance and there are only ~2(H-1) of them.
    assert 0.95 * N <= ess <= N


# ---------------------------------------------------------------------------
# Weighted metric sanity
# ---------------------------------------------------------------------------


def test_weighted_brier_matches_unweighted_for_uniform_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0, 1, size=200)
    from sklearn.metrics import brier_score_loss
    unw = brier_score_loss(y, p)
    wt = weighted_brier(y, p, np.ones(200))
    assert wt == pytest.approx(unw, rel=1e-9)
    # None argument also collapses to unweighted
    wt_none = weighted_brier(y, p, None)
    assert wt_none == pytest.approx(unw, rel=1e-9)


def test_weighted_auc_matches_unweighted_for_uniform_weights():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0, 1, size=200)
    from sklearn.metrics import roc_auc_score
    unw = roc_auc_score(y, p)
    wt = weighted_auc(y, p, np.ones(200))
    assert wt == pytest.approx(unw, rel=1e-9)


def test_weighted_auc_returns_none_for_single_class():
    y = np.ones(50, dtype=int)
    p = np.random.uniform(0, 1, size=50)
    assert weighted_auc(y, p) is None


def test_weighted_spiegelhalter_matches_unweighted_for_uniform_weights():
    from gbdt.calibration import spiegelhalter_z as unweighted_z
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=300)
    p = rng.uniform(0.05, 0.95, size=300)
    z_unw, _ = unweighted_z(y, p)
    z_wt, _ = weighted_spiegelhalter_z(y, p, np.ones(300))
    assert z_wt == pytest.approx(z_unw, rel=1e-9)


def test_weighted_brier_zero_for_perfect_prediction():
    y = np.array([0, 1, 1, 0, 1])
    p = y.astype(float)
    w = np.array([0.1, 0.5, 0.2, 0.3, 1.0])
    assert weighted_brier(y, p, w) == pytest.approx(0.0)


def test_weighted_brier_handles_zero_weight_total():
    y = np.array([0, 1])
    p = np.array([0.5, 0.5])
    w = np.zeros(2)
    # Σw = 0 → undefined; we return NaN rather than divide-by-zero
    val = weighted_brier(y, p, w)
    assert np.isnan(val)
