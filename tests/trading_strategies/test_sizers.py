"""Tests for the v1 sizers (plan §6.3 / §6.4 / §7 baseline)."""

from __future__ import annotations

import numpy as np
import pytest

from trading_strategies import PerPredictionSizer, PortfolioSizer
from trading_strategies.sizing import (
    DiscreteBoundedLossKelly,
    FixedFraction,
    VinceOptimalF,
)


# -- DiscreteBoundedLossKelly -----------------------------------------------
def test_kelly_breakeven_is_zero():
    k = DiscreteBoundedLossKelly()
    # breakeven p = q/(1+b) = 1/(1 + win/loss); win/loss=2 → p=1/3.
    assert k.fraction_at_risk(1 / 3, payoff_win=0.10, payoff_loss=0.05) == pytest.approx(0.0, abs=1e-12)


def test_kelly_below_breakeven_clips_to_zero():
    k = DiscreteBoundedLossKelly()
    assert k.fraction_at_risk(0.2, payoff_win=0.10, payoff_loss=0.05) == 0.0


def test_kelly_known_value():
    k = DiscreteBoundedLossKelly()
    # p=0.385, b=2 → (2*0.385 - 0.615)/2 = 0.0775
    assert k.fraction_at_risk(0.385, payoff_win=0.10, payoff_loss=0.05) == pytest.approx(0.0775, abs=1e-9)


def test_kelly_monotone_in_p():
    k = DiscreteBoundedLossKelly()
    ps = np.linspace(0.34, 0.9, 20)
    fs = [k.fraction_at_risk(p, payoff_win=0.10, payoff_loss=0.05) for p in ps]
    assert all(b >= a for a, b in zip(fs, fs[1:]))


def test_kelly_validates_inputs():
    k = DiscreteBoundedLossKelly()
    with pytest.raises(ValueError):
        k.fraction_at_risk(1.5, payoff_win=0.1, payoff_loss=0.05)
    with pytest.raises(ValueError):
        k.fraction_at_risk(0.5, payoff_win=0.0, payoff_loss=0.05)


def test_kelly_satisfies_protocol():
    assert isinstance(DiscreteBoundedLossKelly(), PerPredictionSizer)


# -- VinceOptimalF -----------------------------------------------------------
def test_vince_fit_finds_interior_optimum():
    rng = np.random.default_rng(0)
    # Positive-edge series with a bounded worst loss: clip then pin so the
    # worst realized loss is exactly -0.10.
    r = np.clip(rng.normal(0.02, 0.05, 500), -0.099, None)
    r[0] = -0.10  # pin the worst loss
    v = VinceOptimalF().fit(r)
    f = v.per_position_fraction_at_risk
    assert 0.0 < f < 1.0
    assert v.diagnostics["n_trades"] == 500
    assert v.diagnostics["max_loss"] == pytest.approx(0.10)


def test_vince_requires_a_loss():
    v = VinceOptimalF()
    with pytest.raises(ValueError, match="no losing trade"):
        v.fit(np.array([0.01, 0.02, 0.03]))


def test_vince_unfit_raises():
    with pytest.raises(RuntimeError):
        _ = VinceOptimalF().per_position_fraction_at_risk


def test_vince_satisfies_protocol():
    assert isinstance(VinceOptimalF(), PortfolioSizer)


# -- FixedFraction -----------------------------------------------------------
def test_fixed_returns_constant():
    f = FixedFraction(0.2)
    assert f.fraction_at_risk(0.9, payoff_win=0.1, payoff_loss=0.05) == 0.2
    assert f.fraction_at_risk(0.4, payoff_win=0.3, payoff_loss=0.1) == 0.2


def test_fixed_validates_range():
    with pytest.raises(ValueError):
        FixedFraction(1.5)


def test_fixed_satisfies_protocol():
    assert isinstance(FixedFraction(0.1), PerPredictionSizer)
