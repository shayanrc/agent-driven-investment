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

Test-eval parallelism (env-controlled):

    ANALOG_MC_TEST_WORKERS=N   override pool size for the per-origin test-eval
                               loop. Default: max(1, cpu_count() − 2). 1 falls
                               back to the original serial path. Each worker
                               clamps its BLAS thread count to 1 to avoid
                               oversubscribing the host (N processes × multi-
                               threaded BLAS will thrash on small core counts).
                               The per-origin RNG is derived deterministically
                               from (random_seed, weights, n_eff, origin_idx)
                               via ``_seed_for``, so parallelism does not
                               affect bit-identity of forecasts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.data import Fold, generate_folds
from analog_mc.features import compute_features
from analog_mc.scoring import crps_sample
from analog_mc.search import SearchResult, _seed_for, run_search
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


# ---------------------------------------------------------------------------
# Per-origin worker for the test-eval pool
# ---------------------------------------------------------------------------
#
# Workers must be importable (module-level) for ProcessPoolExecutor. They
# receive immutable inputs (returns_arr, features, candidate_idx, config) via
# the pool initializer; per-task arguments are the small (origin_idx, weights,
# n_eff) triple plus the horizon, which is also pickled per-call but trivially
# small. Results come back as (origin_i, paths_or_None, ratios_or_None,
# realized_or_None, crps_or_None) — the Nones flag origins that ``forecast``
# refused (NaN features, empty candidate pool, etc.) so the caller can drop
# them without sentinel arrays crossing the IPC boundary.

_worker_state: dict[str, object] = {}


def _init_test_eval_worker(returns_arr: np.ndarray, features: pd.DataFrame) -> None:
    """Pool initializer: pin BLAS to 1 thread and stash shared arrays.

    Each NumPy op inside the conditional sampler typically spins up several
    BLAS threads. With N worker processes that compounds to N × T threads on
    an 8-core host — a context-switch storm that erases the parallelism win.
    threadpool_limits clamps every loaded BLAS/OMP runtime to a single
    thread for the lifetime of the worker.
    """
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        # threadpoolctl is a declared dependency; absence is unexpected but
        # not fatal. Without it, workers will use default BLAS threading and
        # may oversubscribe, but correctness is unaffected.
        pass
    _worker_state["returns"] = returns_arr
    _worker_state["features"] = features


def _forecast_one_origin(
    args: tuple[int, np.ndarray, float, int, np.ndarray, int, Config],
) -> tuple[int, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    """Run a single test-origin forecast inside a pool worker.

    Reads ``returns`` and ``features`` from the worker's module-level state
    (populated by ``_init_test_eval_worker``) to avoid re-pickling them on
    every task — they're constant across all 60 origins in a fold.
    """
    origin_i, weights, n_eff, random_seed, candidate_idx, horizon, config = args
    returns = _worker_state["returns"]
    features = _worker_state["features"]
    rng = np.random.default_rng(_seed_for(random_seed, weights, n_eff, origin_i))
    try:
        paths, ratios = forecast(
            origin_idx=origin_i,
            returns=returns,
            candidate_idx=candidate_idx,
            features=features,
            weights=weights,
            n_eff=n_eff,
            config=config,
            rng=rng,
            record_ratios=True,
        )
    except ValueError:
        return origin_i, None, None, None, None
    rl = returns[origin_i + 1 : origin_i + 1 + horizon]
    crps = crps_sample(paths, rl)
    return origin_i, paths.astype(np.float32), ratios.astype(np.float32), rl, float(crps)


def _resolve_worker_count() -> int:
    """Resolve the test-eval pool size from env, falling back to cpu_count − 2."""
    raw = os.environ.get("ANALOG_MC_TEST_WORKERS")
    if raw is not None:
        try:
            n = int(raw)
            return max(1, n)
        except ValueError:
            log.warning("ANALOG_MC_TEST_WORKERS=%r is not an int; using default", raw)
    return max(1, (os.cpu_count() or 1) - 2)


def _evaluate_on_test(
    fold: Fold,
    weights: np.ndarray,
    n_eff: float,
    returns: np.ndarray,
    features: pd.DataFrame,
    config: Config,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Forecast at every eligible test origin; return paths, ratios, realized, origin_idx, mean CRPS.

    If ``executor`` is provided, per-origin forecasts run in parallel across
    its workers. Per-origin RNGs are seeded deterministically from
    ``(config.random_seed, weights, n_eff, origin_idx)``, so the parallel
    path produces bit-identical results to the serial path. Workers receive
    ``returns`` and ``features`` via the pool's initializer (one pickle per
    worker at pool startup) rather than per-task.

    Returns:
        paths_arr:   (n_origins, n_paths, horizon) — float32 to halve disk
        ratios_arr:  (n_origins, n_paths, n_blocks) — pre-clip σ ratios for clip-hit
        realized:    (n_origins, horizon)
        origin_idx:  (n_origins,)
        test_crps:   scalar mean per-origin CRPS
    """
    origins = _test_origins_with_full_horizon(fold, returns, config.forecast_horizon)
    horizon = config.forecast_horizon

    tasks = [
        (int(o), weights, n_eff, config.random_seed, fold.train_idx, horizon, config)
        for o in origins
    ]

    if executor is None:
        # Serial fallback — identical behavior to the pre-parallel path.
        results = [_forecast_one_origin_local(t, returns, features) for t in tasks]
    else:
        # executor.map preserves submission order, so results align with `origins`.
        results = list(executor.map(_forecast_one_origin, tasks))

    paths_list: list[np.ndarray] = []
    ratios_list: list[np.ndarray] = []
    realized_list: list[np.ndarray] = []
    used_origins: list[int] = []
    crpss: list[float] = []
    for origin_i, paths, ratios, rl, crps in results:
        if paths is None:
            continue
        paths_list.append(paths)
        ratios_list.append(ratios)
        realized_list.append(rl)
        used_origins.append(origin_i)
        crpss.append(crps)

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


def _forecast_one_origin_local(
    args: tuple[int, np.ndarray, float, int, np.ndarray, int, Config],
    returns: np.ndarray,
    features: pd.DataFrame,
) -> tuple[int, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    """Serial-fallback equivalent of _forecast_one_origin (no pool state)."""
    origin_i, weights, n_eff, random_seed, candidate_idx, horizon, config = args
    rng = np.random.default_rng(_seed_for(random_seed, weights, n_eff, origin_i))
    try:
        paths, ratios = forecast(
            origin_idx=origin_i,
            returns=returns,
            candidate_idx=candidate_idx,
            features=features,
            weights=weights,
            n_eff=n_eff,
            config=config,
            rng=rng,
            record_ratios=True,
        )
    except ValueError:
        return origin_i, None, None, None, None
    rl = returns[origin_i + 1 : origin_i + 1 + horizon]
    crps = crps_sample(paths, rl)
    return origin_i, paths.astype(np.float32), ratios.astype(np.float32), rl, float(crps)


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
    features = compute_features(
        returns,
        halflife=config.ewma_halflife,
        horizons=config.zscore_horizons,
        momentum_lookback=config.momentum_lookback if config.drift_mode != "zero" else None,
    )
    folds = generate_folds(returns, config)
    log.info("Generated %d folds", len(folds))

    returns_arr = returns.to_numpy()
    outcomes: list[FoldOutcome] = []
    started = time.perf_counter()

    # Spin up the test-eval process pool once per run. Worker count defaults
    # to cpu_count − 2 (so the main process and the OS each retain headroom);
    # override via ANALOG_MC_TEST_WORKERS. n_workers == 1 keeps the serial
    # fallback path (no pool overhead) so test parity with the pre-parallel
    # implementation is straightforward.
    n_workers = _resolve_worker_count()
    if n_workers > 1:
        log.info("Test-eval pool: %d workers (BLAS clamped to 1 thread each)", n_workers)
        # forkserver: child workers fork from a clean ancestor (not from the
        # main process), so BLAS-thread state in the parent can't deadlock in
        # the child. Tiny one-time startup cost (~300 ms total at 6 workers),
        # eliminates the Python 3.12+ fork-from-multithreaded-process warning.
        mp_ctx = multiprocessing.get_context("forkserver")
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
            initializer=_init_test_eval_worker,
            initargs=(returns_arr, features),
        )
    else:
        log.info("Test-eval pool: serial (n_workers=1)")
        executor = None

    try:
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
            # Test-only conditional sampling contingency (V2_PLAN open question 7):
            # if the user opted to disable conditional sampling during search, build a
            # search-time config copy with conditional_block_sampling=False. Test eval
            # below uses the original config (so the test forecasts get the v2.2 path).
            if config.conditional_block_sampling and not config.conditional_block_sampling_in_search:
                from dataclasses import replace
                search_config = replace(config, conditional_block_sampling=False)
                log.info("fold %d/%d: searching (search uses v1 sampling; test uses conditional)...",
                         fold.index, len(folds))
            else:
                search_config = config
                log.info("fold %d/%d: searching...", fold.index, len(folds))
            result = run_search(fold, returns_arr, features, search_config)
            log.info(
                "fold %d/%d: search done in %.1fs — best weights=%s n_eff=%g val_crps=%.5f",
                fold.index, len(folds), time.perf_counter() - t0,
                [f"{w:.2f}" for w in result.weights], result.n_eff, result.val_crps,
            )

            t1 = time.perf_counter()
            paths, ratios, realized, test_origin_idx, test_crps = _evaluate_on_test(
                fold, result.weights, result.n_eff, returns_arr, features, config,
                executor=executor,
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
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
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
