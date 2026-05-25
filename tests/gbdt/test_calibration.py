"""Stage 5 — calibration tests."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from gbdt.calibration import (
    apply_calibrator,
    conditional_isotonic,
    fit_isotonic,
    isotonic_always,
    spiegelhalter_z,
)


def _wellcal(n: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return y, p


def _miscal_overconfident(n: int = 5000, seed: int = 0):
    """Predictions stretched away from 0.5 — overconfident.

    Realized y matches true_p, not the stretched p. Z should fire.
    """
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.2, 0.8, n)
    stretched = np.clip(true_p + np.sign(true_p - 0.5) * 0.15, 0.01, 0.99)
    y = (rng.uniform(0, 1, n) < true_p).astype(int)
    return y, stretched


def test_spiegelhalter_z_well_calibrated():
    y, p = _wellcal(8000, seed=1)
    z, pval = spiegelhalter_z(y, p)
    assert abs(z) < 2.0
    assert pval > 0.05


def test_spiegelhalter_z_miscalibrated_fires():
    y, p = _miscal_overconfident(8000, seed=2)
    z, pval = spiegelhalter_z(y, p)
    assert abs(z) >= 2.0
    assert pval < 0.05


def test_conditional_isotonic_native_branch_on_wellcal():
    y, p = _wellcal(5000, seed=3)
    dec = conditional_isotonic(y, p, z_threshold=2.0)
    assert dec.method == "native"
    assert dec.calibrator is None
    assert "native" in dec.rationale.lower()


def test_conditional_isotonic_isotonic_branch_on_miscal():
    y, p = _miscal_overconfident(6000, seed=4)
    dec = conditional_isotonic(y, p, z_threshold=2.0)
    assert dec.method == "isotonic"
    assert dec.calibrator is not None
    # After calibration the val Z should be much closer to 0
    p_cal = apply_calibrator(p, dec.calibrator)
    z_after, _ = spiegelhalter_z(y, p_cal)
    assert abs(z_after) < abs(dec.spiegelhalter_z)


def test_isotonic_always_returns_isotonic():
    y, p = _wellcal(2000, seed=5)
    dec = isotonic_always(y, p)
    assert dec.method == "isotonic"
    assert dec.calibrator is not None


def test_apply_calibrator_none_is_passthrough():
    p = np.array([0.1, 0.5, 0.9])
    out = apply_calibrator(p, None)
    assert np.allclose(out, p)


def test_apply_calibrator_isotonic_clips_to_unit_interval():
    iso = fit_isotonic([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])
    p = np.array([-0.5, 0.0, 0.5, 1.0, 2.0])
    out = apply_calibrator(p, iso)
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_threshold_boundary_just_above():
    """When |z| equals the threshold (borderline), we expect isotonic to fire
    on |z| >= threshold per the spec wording."""
    y, p = _miscal_overconfident(8000, seed=6)
    z, _ = spiegelhalter_z(y, p)
    # Pick a threshold just below |z|, so it should pick isotonic.
    dec = conditional_isotonic(y, p, z_threshold=abs(z) - 0.001)
    assert dec.method == "isotonic"
    # And just above → native
    dec2 = conditional_isotonic(y, p, z_threshold=abs(z) + 0.5)
    assert dec2.method == "native"


def test_spiegelhalter_z_empty_inputs():
    z, pval = spiegelhalter_z(np.array([]), np.array([]))
    assert z == 0.0 and pval == 1.0


def test_spiegelhalter_z_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        spiegelhalter_z(np.array([0, 1]), np.array([0.5]))


# ---------------------------------------------------------------------------
# Calibration artifact round-trip (PR #8 review, Minor 2)
# ---------------------------------------------------------------------------


def test_isotonic_pickle_roundtrip_preserves_outputs(tmp_path):
    """Fit isotonic, pickle.dump → pickle.load, assert outputs match.

    Guards against the plaintext-vs-pickle bug surfaced in PR #8 review:
    when ``calibration.pkl`` is sometimes a placeholder text file and
    sometimes a real pickle, ``pickle.load`` raises ``UnpicklingError``.
    """
    y, p = _miscal_overconfident(4000, seed=11)
    iso = fit_isotonic(y, p)
    out_before = apply_calibrator(p, iso)

    path = tmp_path / "calibration.pkl"
    with open(path, "wb") as f:
        pickle.dump(iso, f)
    with open(path, "rb") as f:
        loaded = pickle.load(f)

    out_after = apply_calibrator(p, loaded)
    assert np.allclose(out_before, out_after)


def test_none_calibrator_pickle_roundtrip(tmp_path):
    """Persist ``None`` via pickle (native-pass case) and load it back.

    The artifact emitter unconditionally writes a pickle so downstream
    ``pickle.load`` works whether or not calibration fired.
    """
    path = tmp_path / "calibration.pkl"
    with open(path, "wb") as f:
        pickle.dump(None, f)
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded is None
    # And apply_calibrator passes p through unchanged.
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(apply_calibrator(p, loaded), p)
