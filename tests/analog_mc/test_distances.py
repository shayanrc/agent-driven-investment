"""Tests for analog_mc.distances (constraint C2)."""

from __future__ import annotations

import numpy as np
import pytest

from analog_mc.distances import (
    composite_distance,
    composite_distance_batched,
    distances_to_probs,
    distances_to_probs_batched,
)


# ---------------------------------------------------------------------------
# composite_distance
# ---------------------------------------------------------------------------


def test_composite_distance_basic() -> None:
    z_target = np.array([0.0, 0.0, 0.0])
    z_cand = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    weights = np.array([1.0, 1.0, 1.0])
    d = composite_distance(z_target, z_cand, weights)
    np.testing.assert_allclose(d, [1.0, 2.0, 3.0])


def test_composite_distance_weights_affect_metric() -> None:
    z_target = np.array([0.0, 0.0, 0.0])
    z_cand = np.array([[1.0, 1.0, 1.0]])
    # Heavier weight on first axis -> distance dominated by that diff.
    d_uniform = composite_distance(z_target, z_cand, np.array([1.0, 1.0, 1.0]))
    d_first = composite_distance(z_target, z_cand, np.array([10.0, 0.1, 0.1]))
    # Heavier first-axis weight + uniform diff -> larger total distance.
    assert d_first[0] > d_uniform[0]


def test_composite_distance_zero_when_identical() -> None:
    z_target = np.array([0.5, -0.2, 1.1])
    z_cand = z_target.reshape(1, -1)
    d = composite_distance(z_target, z_cand, np.array([1.0, 1.0, 1.0]))
    assert d[0] == pytest.approx(0.0, abs=1e-12)


def test_composite_distance_rejects_bad_shapes() -> None:
    z_target = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        composite_distance(z_target, np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        composite_distance(z_target, np.array([[0.0, 0.0]]), np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        composite_distance(z_target, np.array([[0.0, 0.0, 0.0]]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        composite_distance(z_target, np.array([[0.0, 0.0, 0.0]]), np.array([-1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# distances_to_probs
# ---------------------------------------------------------------------------


@pytest.fixture
def random_distances() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(0.0, 5.0, size=500)


@pytest.mark.parametrize("target", [15.0, 30.0, 50.0, 80.0, 150.0])
def test_distances_to_probs_hits_target_n_eff(random_distances, target) -> None:
    p = distances_to_probs(random_distances, target_n_eff=target)
    assert p.shape == random_distances.shape
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    # n_eff within 5% of target — the spec's tolerance.
    n_eff = np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))
    assert n_eff == pytest.approx(target, rel=0.05)


def test_distances_to_probs_uniform_when_all_equal() -> None:
    d = np.full(10, 2.5)
    p = distances_to_probs(d, target_n_eff=10.0)
    np.testing.assert_allclose(p, np.full(10, 0.1))


def test_distances_to_probs_rejects_impossible_uniform_target() -> None:
    d = np.full(10, 2.5)
    with pytest.raises(ValueError, match="All distances equal"):
        distances_to_probs(d, target_n_eff=5.0)


def test_distances_to_probs_rejects_out_of_range_target(random_distances) -> None:
    with pytest.raises(ValueError):
        distances_to_probs(random_distances, target_n_eff=1.0)
    with pytest.raises(ValueError):
        distances_to_probs(random_distances, target_n_eff=0.5)
    with pytest.raises(ValueError):
        distances_to_probs(random_distances, target_n_eff=len(random_distances) + 1)


def test_distances_to_probs_rejects_negative_distances() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        distances_to_probs(np.array([1.0, -0.1, 2.0]), target_n_eff=2.0)


def test_distances_to_probs_concentrates_on_minimum() -> None:
    """As target_n_eff -> small, probability concentrates on the nearest analog."""
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    p_sharp = distances_to_probs(d, target_n_eff=1.1)
    p_diffuse = distances_to_probs(d, target_n_eff=9.0)
    assert p_sharp[0] > p_diffuse[0]
    assert np.argmax(p_sharp) == 0


def test_distances_to_probs_is_monotone_in_target() -> None:
    """Larger target_n_eff -> entropy strictly larger -> more diffuse."""
    rng = np.random.default_rng(11)
    d = rng.uniform(0.0, 3.0, size=200)
    targets = [10.0, 30.0, 60.0, 120.0, 180.0]
    n_effs = []
    for t in targets:
        p = distances_to_probs(d, target_n_eff=t)
        n_effs.append(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    # Strictly increasing.
    assert all(b > a for a, b in zip(n_effs, n_effs[1:]))


# ---------------------------------------------------------------------------
# composite_distance_batched (v2.2)
# ---------------------------------------------------------------------------


def test_composite_distance_batched_matches_scalar() -> None:
    rng = np.random.default_rng(0)
    z_targets = rng.normal(size=(7, 3))
    z_candidates = rng.normal(size=(40, 3))
    weights = np.array([1.0, 0.5, 2.0])
    batched = composite_distance_batched(z_targets, z_candidates, weights)
    assert batched.shape == (7, 40)
    for i in range(7):
        expected = composite_distance(z_targets[i], z_candidates, weights)
        np.testing.assert_allclose(batched[i], expected, rtol=1e-12, atol=1e-12)


def test_composite_distance_batched_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        composite_distance_batched(np.zeros(3), np.zeros((5, 3)), np.ones(3))
    with pytest.raises(ValueError):
        composite_distance_batched(np.zeros((2, 3)), np.zeros((5, 4)), np.ones(3))
    with pytest.raises(ValueError):
        composite_distance_batched(np.zeros((2, 3)), np.zeros((5, 3)), np.ones(2))
    with pytest.raises(ValueError):
        composite_distance_batched(np.zeros((2, 3)), np.zeros((5, 3)), np.array([-1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# distances_to_probs_batched (v2.2)
# ---------------------------------------------------------------------------


def test_distances_to_probs_batched_matches_scalar_row_by_row() -> None:
    rng = np.random.default_rng(3)
    distances = rng.uniform(0.0, 4.0, size=(8, 200))
    batched = distances_to_probs_batched(distances, target_n_eff=50.0)
    assert batched.shape == (8, 200)
    np.testing.assert_allclose(batched.sum(axis=1), 1.0, atol=1e-9)
    for i in range(8):
        expected = distances_to_probs(distances[i], target_n_eff=50.0)
        np.testing.assert_allclose(batched[i], expected, rtol=1e-9, atol=1e-12)


def test_distances_to_probs_batched_rejects_1d() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        distances_to_probs_batched(np.array([1.0, 2.0, 3.0]), target_n_eff=2.0)
