"""Walk-forward orchestration (Stage 8).

Drives the full pipeline across folds: per-fold hyperparameter search →
lock weights → evaluate on test → persist artifacts → append to global OOS
record. Crash-resumable: re-running on an existing run directory skips
folds whose summary.json already exists.

Per-run directory layout (rooted at config.runs_dir/<timestamp>/):

    config.yaml             frozen config used for the run
    meta.json               git commit hash, config hash, timestamps
    lock                    presence file while the run is in progress
    folds/
      0/
        search.parquet      full grid evaluation table
        forecasts.npz       test paths, realized, origin_idx (float32)
        summary.json        locked (weights, n_eff, val_crps, test_crps)
      1/
        ...

Multi-run safety (per project-streamlit memory): each run gets its own
timestamped directory; the ``lock`` file lets a dashboard detect in-progress
runs without coordinating IPC. No global mutex needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.data import Fold, generate_folds
from analog_mc.features import compute_features
from analog_mc.scoring import crps_sample
from analog_mc.search import SearchResult, run_search
from analog_mc.simulate import forecast


log = logging.getLogger("analog_mc.walk_forward")


# ---------------------------------------------------------------------------
# Run directory setup
# ---------------------------------------------------------------------------


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _config_hash(config: Config) -> str:
    """Stable hash of the config's YAML representation."""
    import yaml
    s = yaml.safe_dump(config.to_dict(), sort_keys=True)
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def create_run_dir(config: Config, root: str | Path | None = None) -> Path:
    """Create and return a fresh timestamped run directory.

    Writes ``config.yaml`` and a ``meta.json`` with git + config hashes and a
    ``lock`` sentinel file. The lock is removed by ``run_walk_forward`` on
    successful completion (and left in place on crash).
    """
    base = Path(root) if root is not None else Path(config.runs_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / _utc_timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "folds").mkdir()

    config.to_yaml(run_dir / "config.yaml")
    meta = {
        "git_commit": _git_commit_hash(),
        "config_hash": _config_hash(config),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "lock").touch()
    return run_dir


# ---------------------------------------------------------------------------
# Per-fold evaluation and persistence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldOutcome:
    """Per-fold results after search + test evaluation."""
    fold_index: int
    weights: np.ndarray
    n_eff: float
    val_crps: float
    test_crps: float
    n_test_origins: int


def _test_origins_with_full_horizon(fold: Fold, returns: np.ndarray, horizon: int) -> np.ndarray:
    """Test origin indices that have a full realized horizon available."""
    return fold.test_idx[fold.test_idx + horizon < returns.size]


def _evaluate_on_test(
    fold: Fold,
    weights: np.ndarray,
    n_eff: float,
    returns: np.ndarray,
    features: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Forecast at every eligible test origin; return paths, ratios, realized, origin_idx, mean CRPS.

    Returns:
        paths_arr:   (n_origins, n_paths, horizon) — float32 to halve disk
        ratios_arr:  (n_origins, n_paths, n_blocks) — pre-clip σ ratios for clip-hit
        realized:    (n_origins, horizon)
        origin_idx:  (n_origins,)
        test_crps:   scalar mean per-origin CRPS
    """
    from analog_mc.search import _seed_for  # reuse the deterministic seeding

    origins = _test_origins_with_full_horizon(fold, returns, config.forecast_horizon)
    paths_list: list[np.ndarray] = []
    ratios_list: list[np.ndarray] = []
    realized_list: list[np.ndarray] = []
    used_origins: list[int] = []
    crpss: list[float] = []

    for origin in origins:
        origin_i = int(origin)
        rng = np.random.default_rng(_seed_for(config.random_seed, weights, n_eff, origin_i))
        try:
            paths, ratios = forecast(
                origin_idx=origin_i,
                returns=returns,
                candidate_idx=fold.train_idx,
                features=features,
                weights=weights,
                n_eff=n_eff,
                config=config,
                rng=rng,
                record_ratios=True,
            )
        except ValueError:
            continue
        rl = returns[origin_i + 1 : origin_i + 1 + config.forecast_horizon]
        paths_list.append(paths.astype(np.float32))
        ratios_list.append(ratios.astype(np.float32))
        realized_list.append(rl)
        used_origins.append(origin_i)
        crpss.append(crps_sample(paths, rl))

    if not paths_list:
        return (
            np.empty((0, config.n_paths, config.forecast_horizon), dtype=np.float32),
            np.empty((0, config.n_paths, config.n_blocks), dtype=np.float32),
            np.empty((0, config.forecast_horizon)),
            np.empty((0,), dtype=np.int64),
            float("inf"),
        )

    return (
        np.stack(paths_list),
        np.stack(ratios_list),
        np.stack(realized_list),
        np.array(used_origins, dtype=np.int64),
        float(np.mean(crpss)),
    )


def _persist_fold(
    fold_dir: Path,
    fold: Fold,
    result: SearchResult,
    paths: np.ndarray,
    ratios: np.ndarray,
    realized: np.ndarray,
    test_origin_idx: np.ndarray,
    test_crps: float,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    result.grid_df.to_parquet(fold_dir / "search.parquet")
    np.savez_compressed(
        fold_dir / "forecasts.npz",
        paths=paths,
        ratios=ratios,
        realized=realized,
        origin_idx=test_origin_idx,
    )
    summary = {
        "fold_index": fold.index,
        "train_start": int(fold.train_idx[0]),
        "train_end": int(fold.train_idx[-1]),
        "val_start": int(fold.val_idx[0]),
        "val_end": int(fold.val_idx[-1]),
        "test_start": int(fold.test_idx[0]),
        "test_end": int(fold.test_idx[-1]),
        "weights": [float(w) for w in result.weights],
        "n_eff": float(result.n_eff),
        "val_crps": float(result.val_crps),
        "test_crps": float(test_crps),
        "n_test_origins": int(test_origin_idx.size),
        "n_search_forecasts": int(result.n_forecasts),
    }
    (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _load_fold_summary(fold_dir: Path) -> dict | None:
    p = fold_dir / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_walk_forward(
    returns: pd.Series,
    config: Config,
    run_dir: Path | None = None,
    resume: bool = True,
    progress_callback=None,
) -> Path:
    """Run the full walk-forward pipeline; return the run directory path.

    Args:
        returns:           the log-return series (datetime-indexed).
        config:            pipeline config.
        run_dir:           target run directory. If None, a fresh timestamped
                           dir is created under ``config.runs_dir``.
        resume:            if True and ``run_dir`` exists with completed folds,
                           skip them.
        progress_callback: optional callable(fold_index, n_folds, outcome)
                           invoked after each fold completes.

    Returns:
        The run directory path.
    """
    if run_dir is None:
        run_dir = create_run_dir(config)
    else:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            (run_dir / "folds").mkdir(exist_ok=True)
            config.to_yaml(run_dir / "config.yaml")
            (run_dir / "lock").touch()

    log.info("Run directory: %s", run_dir)
    log.info("Computing features over %d returns", len(returns))
    features = compute_features(returns, halflife=config.ewma_halflife, horizons=config.zscore_horizons)
    folds = generate_folds(returns, config)
    log.info("Generated %d folds", len(folds))

    returns_arr = returns.to_numpy()
    outcomes: list[FoldOutcome] = []
    started = time.perf_counter()

    for fold in folds:
        fold_dir = run_dir / "folds" / str(fold.index)
        if resume and _load_fold_summary(fold_dir) is not None:
            summary = _load_fold_summary(fold_dir)
            log.info(
                "fold %d/%d: SKIP (already completed) val_crps=%.5f test_crps=%.5f",
                fold.index, len(folds), summary["val_crps"], summary["test_crps"],
            )
            outcomes.append(FoldOutcome(
                fold_index=fold.index,
                weights=np.array(summary["weights"]),
                n_eff=summary["n_eff"],
                val_crps=summary["val_crps"],
                test_crps=summary["test_crps"],
                n_test_origins=summary["n_test_origins"],
            ))
            if progress_callback:
                progress_callback(fold.index, len(folds), outcomes[-1])
            continue

        t0 = time.perf_counter()
        log.info("fold %d/%d: searching...", fold.index, len(folds))
        result = run_search(fold, returns_arr, features, config)
        log.info(
            "fold %d/%d: search done in %.1fs — best weights=%s n_eff=%g val_crps=%.5f",
            fold.index, len(folds), time.perf_counter() - t0,
            [f"{w:.2f}" for w in result.weights], result.n_eff, result.val_crps,
        )

        t1 = time.perf_counter()
        paths, ratios, realized, test_origin_idx, test_crps = _evaluate_on_test(
            fold, result.weights, result.n_eff, returns_arr, features, config,
        )
        log.info(
            "fold %d/%d: test eval done in %.1fs — test_crps=%.5f over %d origins",
            fold.index, len(folds), time.perf_counter() - t1, test_crps, test_origin_idx.size,
        )

        _persist_fold(fold_dir, fold, result, paths, ratios, realized, test_origin_idx, test_crps)
        outcome = FoldOutcome(
            fold_index=fold.index,
            weights=result.weights,
            n_eff=result.n_eff,
            val_crps=result.val_crps,
            test_crps=test_crps,
            n_test_origins=int(test_origin_idx.size),
        )
        outcomes.append(outcome)
        if progress_callback:
            progress_callback(fold.index, len(folds), outcome)

    # Roll up a flat summary table across folds.
    summary_df = pd.DataFrame([
        {
            "fold_index": o.fold_index,
            "w0": o.weights[0], "w1": o.weights[1], "w2": o.weights[2],
            "n_eff": o.n_eff,
            "val_crps": o.val_crps,
            "test_crps": o.test_crps,
            "n_test_origins": o.n_test_origins,
        }
        for o in outcomes
    ])
    summary_df.to_parquet(run_dir / "summary.parquet")

    # Update meta and clear the lock.
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"git_commit": _git_commit_hash(), "config_hash": _config_hash(config)}
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["wall_seconds"] = time.perf_counter() - started
    meta["n_folds_total"] = len(folds)
    meta["n_folds_completed"] = len(outcomes)
    meta_path.write_text(json.dumps(meta, indent=2))
    (run_dir / "lock").unlink(missing_ok=True)

    log.info(
        "Walk-forward complete: %d folds in %.1fs, mean test_crps=%.5f",
        len(outcomes), meta["wall_seconds"], summary_df["test_crps"].mean(),
    )
    return run_dir


# ---------------------------------------------------------------------------
# CLI entry point (used by the Streamlit run_experiment view via subprocess)
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    from analog_mc.data import load_returns

    parser = argparse.ArgumentParser(prog="python -m analog_mc.walk_forward")
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--run-dir", default=None, help="resume an existing run dir")
    parser.add_argument("--no-resume", action="store_true", help="restart all folds even if completed")
    parser.add_argument("--ticker", default=None, help="override config ticker (for logging/labels)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = Config.from_yaml(args.config)
    if args.ticker:
        # Permit a runtime ticker override without rewriting the YAML.
        cfg = Config(**{**cfg.to_dict(), "ticker": args.ticker})

    returns = load_returns(cfg)
    run_dir = run_walk_forward(
        returns, cfg,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        resume=not args.no_resume,
    )
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
