"""Tests for scripts/v5/ensemble_paths.py — V5.A.2 path mixing helpers.

Loaded by adding ``scripts/`` to ``sys.path`` so the module is importable as
``v5.ensemble_paths`` from pytest, matching how the script is invoked at
runtime (``uv run python scripts/v5/ensemble_paths.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from v5.ensemble_paths import (  # noqa: E402
    ensemble_one_fold,
    mix_paths,
    write_run_dir,
)


# --- Pure-function tests for mix_paths ------------------------------------


def _make(n: int, h: int, tag: float) -> np.ndarray:
    """Build a 2D array of shape (n, h). All rows have first column == tag so
    the *source* of each row can be identified after mixing; remaining columns
    hold row-specific values so the rows are themselves distinguishable."""
    arr = np.zeros((n, h), dtype=np.float64)
    arr[:, 0] = tag
    arr[:, 1:] = (np.arange(n)[:, None] + 0.001 * np.arange(h - 1)[None, :])
    return arr


def test_mix_paths_preserves_shape() -> None:
    a = _make(500, 60, tag=1.0)
    b = _make(500, 60, tag=2.0)
    rng = np.random.default_rng(0)
    out = mix_paths(a, b, alpha=0.5, n_target=1000, rng=rng)
    assert out.shape == (1000, 60)
    assert out.dtype == np.float64


def test_mix_paths_alpha_zero_uses_only_a() -> None:
    a = _make(500, 60, tag=1.0)
    b = _make(500, 60, tag=2.0)
    rng = np.random.default_rng(0)
    out = mix_paths(a, b, alpha=0.0, n_target=500, rng=rng)
    # Every output row's tag (column 0) is A's tag.
    assert (out[:, 0] == 1.0).all()


def test_mix_paths_alpha_one_uses_only_b() -> None:
    a = _make(500, 60, tag=1.0)
    b = _make(500, 60, tag=2.0)
    rng = np.random.default_rng(0)
    out = mix_paths(a, b, alpha=1.0, n_target=500, rng=rng)
    assert (out[:, 0] == 2.0).all()


def test_mix_paths_alpha_half_is_500_500_split() -> None:
    a = _make(500, 60, tag=1.0)
    b = _make(500, 60, tag=2.0)
    rng = np.random.default_rng(0)
    out = mix_paths(a, b, alpha=0.5, n_target=1000, rng=rng)
    n_from_a = (out[:, 0] == 1.0).sum()
    n_from_b = (out[:, 0] == 2.0).sum()
    assert n_from_a == 500
    assert n_from_b == 500


def test_mix_paths_determinism_same_seed() -> None:
    a = _make(500, 60, tag=1.0)
    b = _make(500, 60, tag=2.0)
    out1 = mix_paths(a, b, 0.5, 1000, np.random.default_rng(123))
    out2 = mix_paths(a, b, 0.5, 1000, np.random.default_rng(123))
    assert np.array_equal(out1, out2)


def test_mix_paths_different_seeds_differ() -> None:
    a = _make(500, 60, tag=1.0)
    b = _make(500, 60, tag=2.0)
    out1 = mix_paths(a, b, 0.5, 1000, np.random.default_rng(1))
    out2 = mix_paths(a, b, 0.5, 1000, np.random.default_rng(2))
    # Row 1 (the per-row identifier) should differ between the two draws.
    assert not np.array_equal(out1[:, 1], out2[:, 1])


def test_mix_paths_rejects_bad_alpha() -> None:
    a = _make(10, 5, 0.0)
    b = _make(10, 5, 0.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        mix_paths(a, b, alpha=-0.1, n_target=10, rng=rng)
    with pytest.raises(ValueError):
        mix_paths(a, b, alpha=1.1, n_target=10, rng=rng)


def test_mix_paths_rejects_horizon_mismatch() -> None:
    a = _make(10, 5, 0.0)
    b = _make(10, 6, 0.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        mix_paths(a, b, alpha=0.5, n_target=10, rng=rng)


def test_mix_paths_rejects_insufficient_a() -> None:
    a = _make(3, 5, 0.0)
    b = _make(50, 5, 0.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        mix_paths(a, b, alpha=0.5, n_target=20, rng=rng)


# --- End-to-end test on a synthetic mini run dir --------------------------


def _make_mini_run(
    run_dir: Path, fold_indices: list[int], n_origins: int, n_paths: int,
    horizon: int, paths_offset: float, seed: int,
) -> None:
    """Write a minimal canonical-shaped run dir to ``run_dir`` for tests."""
    rng = np.random.default_rng(seed)
    (run_dir / "folds").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        "data_path: data/NASDAQ100.csv\n"
        "date_col: observation_date\n"
        "close_col: NASDAQ100\n"
        f"forecast_horizon: {horizon}\n"
    )
    for fi in fold_indices:
        fd = run_dir / "folds" / str(fi)
        fd.mkdir(parents=True)
        # Origin indices monotonically increasing per fold.
        origins = np.arange(1000 + fi * 100, 1000 + fi * 100 + n_origins, dtype=np.int64)
        # Tag each path's first column with paths_offset + row index so we can
        # tell which source produced it.
        paths = np.empty((n_origins, n_paths, horizon), dtype=np.float32)
        for o in range(n_origins):
            for p in range(n_paths):
                paths[o, p, :] = paths_offset + p + 0.0001 * np.arange(horizon)
        # Use a fixed sub-seed so the realized array is identical across v24
        # and a2 mini-runs (both runs forecast the same test origins, so their
        # realized arrays must match — this mirrors the canonical invariant
        # checked in ensemble_one_fold).
        realized = np.random.default_rng((9999, fi)).normal(
            scale=0.01, size=(n_origins, horizon)
        ).astype(np.float64)
        ratios = rng.normal(size=(n_origins, n_paths, 3)).astype(np.float32)
        np.savez_compressed(
            fd / "forecasts.npz",
            paths=paths, realized=realized, origin_idx=origins, ratios=ratios,
        )
        (fd / "summary.json").write_text(json.dumps({
            "fold_index": fi,
            "train_start": 0,
            "train_end": 999,
            "val_start": 1000,
            "val_end": 1059,
            "test_start": int(origins[0]),
            "test_end": int(origins[-1]),
            "weights": [0.5, 0.3, 0.2],
            "n_eff": 50.0,
            "val_crps": 0.05,
            "test_crps": 0.06,
            "n_test_origins": n_origins,
            "n_search_forecasts": 100,
        }))


def test_ensemble_one_fold_shapes(tmp_path: Path) -> None:
    v24 = tmp_path / "v24"
    a2 = tmp_path / "a2"
    _make_mini_run(v24, [0], n_origins=3, n_paths=500, horizon=60, paths_offset=0.0, seed=1)
    _make_mini_run(a2, [0], n_origins=3, n_paths=500, horizon=60, paths_offset=1000.0, seed=2)
    summary, paths, ratios, realized, origins = ensemble_one_fold(
        v24, a2, fold_idx=0, alpha=0.5, n_target=1000, seed=42,
    )
    assert paths.shape == (3, 1000, 60)
    assert ratios is not None and ratios.shape == (3, 1000, 3)
    assert realized.shape == (3, 60)
    assert origins.shape == (3,)
    # 50/50 split: half the rows from v24 (first column < 1000), half from a2.
    for o in range(3):
        n_from_v24 = (paths[o, :, 0] < 1000.0).sum()
        n_from_a2 = (paths[o, :, 0] >= 1000.0).sum()
        assert n_from_v24 == 500
        assert n_from_a2 == 500
    assert summary["ensemble_source"]["alpha"] == 0.5
    assert summary["ensemble_source"]["v24_weights"] == [0.5, 0.3, 0.2]
    # search-stage diagnostics should be stripped from synthesized summary.
    assert "val_crps" not in summary
    assert "test_crps" not in summary


def test_write_run_dir_determinism(tmp_path: Path) -> None:
    """Same inputs + same seed => bit-identical forecasts.npz."""
    v24 = tmp_path / "v24"
    a2 = tmp_path / "a2"
    _make_mini_run(v24, [0, 1], n_origins=2, n_paths=200, horizon=10, paths_offset=0.0, seed=1)
    _make_mini_run(a2, [0, 1], n_origins=2, n_paths=200, horizon=10, paths_offset=1000.0, seed=2)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    write_run_dir(out_a, v24, a2, alpha=0.5, seed=42, n_target=400)
    write_run_dir(out_b, v24, a2, alpha=0.5, seed=42, n_target=400)
    for fi in (0, 1):
        npz_a = np.load(out_a / "folds" / str(fi) / "forecasts.npz")
        npz_b = np.load(out_b / "folds" / str(fi) / "forecasts.npz")
        assert np.array_equal(npz_a["paths"], npz_b["paths"])
        assert np.array_equal(npz_a["ratios"], npz_b["ratios"])
        assert np.array_equal(npz_a["realized"], npz_b["realized"])
        assert np.array_equal(npz_a["origin_idx"], npz_b["origin_idx"])


def test_write_run_dir_alpha_boundaries(tmp_path: Path) -> None:
    v24 = tmp_path / "v24"
    a2 = tmp_path / "a2"
    _make_mini_run(v24, [0], n_origins=2, n_paths=200, horizon=10, paths_offset=0.0, seed=1)
    _make_mini_run(a2, [0], n_origins=2, n_paths=200, horizon=10, paths_offset=1000.0, seed=2)

    out_zero = tmp_path / "out_zero"
    write_run_dir(out_zero, v24, a2, alpha=0.0, seed=42, n_target=200)
    npz = np.load(out_zero / "folds/0/forecasts.npz")
    # All rows should come from v24 (first column < 1000).
    assert (npz["paths"][:, :, 0] < 1000.0).all()

    out_one = tmp_path / "out_one"
    write_run_dir(out_one, v24, a2, alpha=1.0, seed=42, n_target=200)
    npz = np.load(out_one / "folds/0/forecasts.npz")
    # All rows from a2.
    assert (npz["paths"][:, :, 0] >= 1000.0).all()


def test_write_run_dir_refuses_existing_without_overwrite(tmp_path: Path) -> None:
    v24 = tmp_path / "v24"
    a2 = tmp_path / "a2"
    _make_mini_run(v24, [0], n_origins=1, n_paths=10, horizon=5, paths_offset=0.0, seed=1)
    _make_mini_run(a2, [0], n_origins=1, n_paths=10, horizon=5, paths_offset=1000.0, seed=2)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit):
        write_run_dir(out, v24, a2, alpha=0.5, seed=42, n_target=10)
    # With overwrite, succeeds.
    write_run_dir(out, v24, a2, alpha=0.5, seed=42, n_target=10, overwrite=True)
    assert (out / "folds/0/forecasts.npz").exists()


def test_write_run_dir_refuses_symlink_destination(tmp_path: Path) -> None:
    v24 = tmp_path / "v24"
    a2 = tmp_path / "a2"
    _make_mini_run(v24, [0], n_origins=1, n_paths=10, horizon=5, paths_offset=0.0, seed=1)
    _make_mini_run(a2, [0], n_origins=1, n_paths=10, horizon=5, paths_offset=1000.0, seed=2)
    real_target = tmp_path / "real_canonical"
    real_target.mkdir()
    symlinked = tmp_path / "shared_link"
    symlinked.symlink_to(real_target)
    with pytest.raises(SystemExit, match="symlinked"):
        write_run_dir(symlinked, v24, a2, alpha=0.5, seed=42, n_target=10)
