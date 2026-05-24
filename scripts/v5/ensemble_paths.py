"""V5.A.2 — path-level ensemble of two canonical runs.

Builds a synthetic "v5_a2_ensemble" run directory under
``runs/analog_mc/v5_a2_ensemble/`` by mixing two existing walk-forward runs'
cached ``forecasts.npz`` at a path-level ratio ``alpha`` (the fraction of paths
taken from the *second* run, A2.1, with the remainder taken from the *first*
run, v2.4).

Per V5_EXPERIMENTS_PLAN.md §V5.A.2, this is NOT a new walk-forward — both
inputs and outputs share the exact same fold/origin grid, the same realized
arrays, and the same horizon. The ensemble run dir is a virtual artifact whose
folds re-use the upstream realized/origin metadata. The synthesized
``summary.json`` per fold carries the v2.4 fold's weights/n_eff for downstream
plotting (the ensemble does not have an optimized weight tuple of its own).

The path mixing is fully deterministic given ``--seed`` so the resulting
forecasts.npz is reproducible bit-for-bit across reruns.

CRITICAL: the script writes to a *worktree-local* directory only. It refuses to
overwrite any existing symlinked timestamped run dir — those are shared
canonical artifacts.

Usage:
    uv run python scripts/v5/ensemble_paths.py \\
        --v24-run runs/analog_mc/20260520T045525Z \\
        --a2-run  runs/analog_mc/20260521T061730Z \\
        --out-dir runs/analog_mc/v5_a2_ensemble \\
        --alpha 0.5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load_fold_summary(run_dir: Path, fold_idx: int) -> dict:
    return json.loads((run_dir / "folds" / str(fold_idx) / "summary.json").read_text())


def list_folds(run_dir: Path) -> list[int]:
    return sorted(int(d.name) for d in (run_dir / "folds").iterdir())


def mix_paths(
    paths_a: np.ndarray,
    paths_b: np.ndarray,
    alpha: float,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Concatenate ``(1-alpha) * n_target`` paths from ``paths_a`` and
    ``alpha * n_target`` paths from ``paths_b``.

    Boundary semantics:
      - ``alpha = 0.0`` => returns rows entirely from ``paths_a``.
      - ``alpha = 1.0`` => returns rows entirely from ``paths_b``.
      - ``alpha = 0.5`` => 50/50 split, prefers ``paths_a`` when ``n_target`` is
        odd (``n_a_take = n_target - round(alpha * n_target)``).

    Shapes are preserved: output has shape ``(n_target, H)`` where ``H`` is the
    horizon (axis 1 of both inputs).

    The sampling is via ``rng.choice(..., replace=False)`` which makes the
    output bit-identical when given the same inputs + seed (the only RNG draw
    is the index permutation, identical to the V4.5.8 preview's approach).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if paths_a.ndim != 2 or paths_b.ndim != 2:
        raise ValueError(
            f"paths must be 2D (n_paths, horizon); got "
            f"{paths_a.shape} and {paths_b.shape}"
        )
    if paths_a.shape[1] != paths_b.shape[1]:
        raise ValueError(
            f"horizon mismatch: paths_a.shape[1]={paths_a.shape[1]} "
            f"vs paths_b.shape[1]={paths_b.shape[1]}"
        )

    n_b_take = int(round(alpha * n_target))
    n_a_take = n_target - n_b_take
    if n_a_take > paths_a.shape[0]:
        raise ValueError(
            f"need {n_a_take} paths from A, only {paths_a.shape[0]} available"
        )
    if n_b_take > paths_b.shape[0]:
        raise ValueError(
            f"need {n_b_take} paths from B, only {paths_b.shape[0]} available"
        )

    parts: list[np.ndarray] = []
    if n_a_take > 0:
        idx_a = rng.choice(paths_a.shape[0], size=n_a_take, replace=False)
        parts.append(paths_a[idx_a])
    if n_b_take > 0:
        idx_b = rng.choice(paths_b.shape[0], size=n_b_take, replace=False)
        parts.append(paths_b[idx_b])
    if not parts:
        # alpha in {0.0, 1.0} can't reach here because n_target > 0 + integer
        # math, but guard explicitly.
        raise ValueError("no paths selected (n_target must be > 0)")
    return np.concatenate(parts, axis=0)


def mix_ratios(
    ratios_a: np.ndarray | None,
    ratios_b: np.ndarray | None,
    alpha: float,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Apply the same row-selection logic to the per-path ``ratios`` array, so
    that downstream consumers (PIT, sigma-scaling diagnostics) see paths and
    ratios consistently. Both inputs use the *same* RNG sequence as
    ``mix_paths`` would for the same call: we re-derive the indices here from
    a *separate* RNG passed in (the caller must structure RNG draws to align).

    Returns None if either input is None (i.e., one of the upstream runs does
    not carry ratios).
    """
    if ratios_a is None or ratios_b is None:
        return None
    n_b_take = int(round(alpha * n_target))
    n_a_take = n_target - n_b_take
    parts: list[np.ndarray] = []
    if n_a_take > 0:
        idx_a = rng.choice(ratios_a.shape[0], size=n_a_take, replace=False)
        parts.append(ratios_a[idx_a])
    if n_b_take > 0:
        idx_b = rng.choice(ratios_b.shape[0], size=n_b_take, replace=False)
        parts.append(ratios_b[idx_b])
    return np.concatenate(parts, axis=0)


def ensemble_one_fold(
    v24_run: Path,
    a2_run: Path,
    fold_idx: int,
    alpha: float,
    n_target: int,
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Build the ensemble forecasts.npz arrays for one fold.

    Returns (summary_dict, paths_mixed, ratios_mixed_or_None,
             realized, origin_idx).
    """
    s24 = load_fold_summary(v24_run, fold_idx)
    sa2 = load_fold_summary(a2_run, fold_idx)
    if (s24["test_start"], s24["test_end"]) != (sa2["test_start"], sa2["test_end"]):
        raise SystemExit(
            f"fold {fold_idx}: test range mismatch v24={s24['test_start']}-{s24['test_end']} "
            f"vs a2={sa2['test_start']}-{sa2['test_end']}"
        )

    npz24 = np.load(v24_run / "folds" / str(fold_idx) / "forecasts.npz")
    npza2 = np.load(a2_run / "folds" / str(fold_idx) / "forecasts.npz")

    origins24 = npz24["origin_idx"]
    originsa2 = npza2["origin_idx"]
    if not np.array_equal(origins24, originsa2):
        raise SystemExit(f"fold {fold_idx}: origin_idx arrays differ")
    if not np.allclose(npz24["realized"], npza2["realized"], atol=1e-9):
        raise SystemExit(f"fold {fold_idx}: realized arrays differ")

    n_origins, n_paths_24, horizon = npz24["paths"].shape
    n_paths_a2 = npza2["paths"].shape[1]

    n_b_take = int(round(alpha * n_target))
    n_a_take = n_target - n_b_take

    # Pre-load full arrays into memory once (they're at most ~14MB per array
    # at canonical scale). Avoid per-origin .astype(float64)/.astype(float32)
    # round-trips — preserve the original float32 dtype.
    paths24_all = npz24["paths"]  # (n_origins, n_paths_24, horizon)
    paths_a2_all = npza2["paths"]
    has_ratios = "ratios" in npz24.files and "ratios" in npza2.files
    if has_ratios:
        ratios24_all = npz24["ratios"]
        ratios_a2_all = npza2["ratios"]
        n_ratio_dim = ratios24_all.shape[2]
        ratios_mixed: np.ndarray | None = np.empty(
            (n_origins, n_target, n_ratio_dim), dtype=ratios24_all.dtype
        )
    else:
        ratios_mixed = None

    paths_mixed = np.empty((n_origins, n_target, horizon), dtype=paths24_all.dtype)

    for o in range(n_origins):
        # One independent RNG per origin so seeding stays compositional: if a
        # downstream consumer wants to inspect a single origin, the RNG state
        # is localized to (fold_idx, origin_idx).
        path_rng = np.random.default_rng((seed, fold_idx, int(origins24[o]), 0))
        if n_a_take > 0:
            idx_a = path_rng.choice(n_paths_24, size=n_a_take, replace=False)
        if n_b_take > 0:
            idx_b = path_rng.choice(n_paths_a2, size=n_b_take, replace=False)
        if n_a_take > 0 and n_b_take > 0:
            paths_mixed[o, :n_a_take] = paths24_all[o][idx_a]
            paths_mixed[o, n_a_take:] = paths_a2_all[o][idx_b]
            if ratios_mixed is not None:
                ratios_mixed[o, :n_a_take] = ratios24_all[o][idx_a]
                ratios_mixed[o, n_a_take:] = ratios_a2_all[o][idx_b]
        elif n_a_take > 0:
            paths_mixed[o] = paths24_all[o][idx_a]
            if ratios_mixed is not None:
                ratios_mixed[o] = ratios24_all[o][idx_a]
        else:  # n_b_take > 0
            paths_mixed[o] = paths_a2_all[o][idx_b]
            if ratios_mixed is not None:
                ratios_mixed[o] = ratios_a2_all[o][idx_b]

    # Synthesize fold summary. Weight tuple + n_eff carry v24's values for
    # downstream plot titles; we tag the ensemble metadata under
    # ``ensemble_source``.
    synth = dict(s24)
    synth["ensemble_source"] = {
        "v24_run": str(v24_run),
        "a2_run": str(a2_run),
        "alpha": alpha,
        "seed": seed,
        "n_paths_total": n_target,
        "n_paths_from_v24": n_target - int(round(alpha * n_target)),
        "n_paths_from_a2": int(round(alpha * n_target)),
        "v24_weights": s24["weights"],
        "v24_n_eff": s24["n_eff"],
        "a2_weights": sa2["weights"],
        "a2_n_eff": sa2["n_eff"],
    }
    synth["n_test_origins"] = int(n_origins)
    # Drop search-stage knobs that are misleading for an ensemble run.
    synth.pop("val_crps", None)
    synth.pop("test_crps", None)
    synth.pop("n_search_forecasts", None)

    return synth, paths_mixed, ratios_mixed, npz24["realized"], origins24


def write_run_dir(
    out_dir: Path,
    v24_run: Path,
    a2_run: Path,
    alpha: float,
    seed: int,
    n_target: int,
    overwrite: bool = False,
) -> None:
    # SAFETY: refuse to write into a symlinked dir under runs/analog_mc/.
    # The shared canonical artifacts live behind symlinks; only worktree-local
    # directories are writable.
    if out_dir.is_symlink():
        raise SystemExit(
            f"refusing to write into symlinked path {out_dir} — pick a "
            f"worktree-local destination"
        )
    if out_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"{out_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(out_dir)
    (out_dir / "folds").mkdir(parents=True)

    # Copy v24's config.yaml verbatim (data settings, horizon, block_length etc.
    # are inherited; the ensemble doesn't change any pipeline knobs).
    src_cfg = (v24_run / "config.yaml").resolve()
    shutil.copy2(src_cfg, out_dir / "config.yaml")

    fold_indices = list_folds(v24_run)
    if list_folds(a2_run) != fold_indices:
        raise SystemExit("v24 and a2 runs have non-matching fold sets")

    for fi in fold_indices:
        synth, paths, ratios, realized, origin_idx = ensemble_one_fold(
            v24_run, a2_run, fi, alpha, n_target, seed
        )
        fold_dir = out_dir / "folds" / str(fi)
        fold_dir.mkdir(parents=True)
        save_kwargs = {
            "paths": paths,
            "realized": realized,
            "origin_idx": origin_idx,
        }
        if ratios is not None:
            save_kwargs["ratios"] = ratios
        # Uncompressed: compression saves only ~15% (paths are nearly random
        # float32) but doubles the write cost. The worktree-local ensemble
        # dir doesn't need the space optimization.
        np.savez(fold_dir / "forecasts.npz", **save_kwargs)
        (fold_dir / "summary.json").write_text(json.dumps(synth, indent=2))
        print(f"  fold {fi}: {paths.shape[0]} origins × {paths.shape[1]} paths × {paths.shape[2]}d")

    meta = {
        "ensemble": True,
        "method": "V5.A.2 path-level concatenation",
        "alpha": alpha,
        "seed": seed,
        "n_paths_total": n_target,
        "v24_run": str(v24_run),
        "a2_run": str(a2_run),
        "n_folds": len(fold_indices),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--v24-run", default="runs/analog_mc/20260520T045525Z",
        help="path to the v2.4 (baseline) canonical run dir",
    )
    p.add_argument(
        "--a2-run", default="runs/analog_mc/20260521T061730Z",
        help="path to the A2.1v1 canonical run dir",
    )
    p.add_argument(
        "--out-dir", default="runs/analog_mc/v5_a2_ensemble",
        help="output (worktree-local) ensemble run dir",
    )
    p.add_argument("--alpha", type=float, default=0.5,
                   help="fraction of paths from A2.1 (default 0.5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-paths", type=int, default=1000,
                   help="total paths per origin in the ensemble (default 1000)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace --out-dir if it already exists")
    args = p.parse_args()

    v24_run = Path(args.v24_run)
    a2_run = Path(args.a2_run)
    out_dir = Path(args.out_dir)

    if not v24_run.exists():
        raise SystemExit(f"missing v24 run dir: {v24_run}")
    if not a2_run.exists():
        raise SystemExit(f"missing a2 run dir: {a2_run}")

    print(
        f"V5.A.2 ensemble: alpha={args.alpha} seed={args.seed} "
        f"n_paths={args.n_paths}\n  v24: {v24_run}\n  a2:  {a2_run}\n  out: {out_dir}"
    )
    write_run_dir(
        out_dir, v24_run, a2_run, args.alpha, args.seed, args.n_paths, args.overwrite
    )
    print(f"\nWrote ensemble run dir: {out_dir}")


if __name__ == "__main__":
    main()
