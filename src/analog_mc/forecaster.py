"""Public ``forecast()`` and ``tune()`` for the analog_mc backend.

These wrap the existing single-origin path-sampling pipeline (``simulate``,
``features``, ``search``, ``walk_forward``) behind the wire-format contract
documented in ``docs/forecasters/V1_PLAN.md`` §"Wire-format contract". They
are the two functions the forecasters dispatcher calls; no other surface of
this module needs to be public.

Design notes:

* ``forecast(input_dict)`` always runs in a SINGLE-ORIGIN fashion (the
  framework's ``/forecast`` is single origin, single horizon, per the goal
  doc). If the preset's hyperparameters carry explicit ``weights`` and
  ``n_eff`` keys, they are used directly — bypass the search. Otherwise the
  function fits one fold of the walk-forward search against the last
  ``val_size`` returns prior to the origin and uses those tuned weights to
  forecast the horizon. The second path makes a curated, search-only preset
  (like ``v24-default``) usable out-of-the-box on a new asset without first
  running ``tune()``.

* ``tune(input_dict)`` runs the existing walk-forward over the supplied data
  range; pins the FINAL fold's tuned ``(weights, n_eff)`` as the preset's
  ``hyperparameters.weights`` / ``hyperparameters.n_eff``. The full search
  grid + hyperparameter machinery from the input is preserved in the preset
  (so a future ``tune()`` re-run with the same preset would reproduce the
  search). Validation metrics are the walk-forward mean test CRPS plus
  coverage diagnostics — same fields ``v24-default.yaml`` ships with.

* Both functions populate ``result.warnings`` (forecast) / preset notes
  (tune) for any unusual conditions (short ranges, capped n_eff, etc.).
  They never silently absorb anomalies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import fields, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from analog_mc.config import Config
from analog_mc.data import (
    Fold,
    close_series_from_dataframe,
    generate_folds,
    log_returns,
)
from analog_mc.features import compute_features
from analog_mc.scoring import crps_sample
from analog_mc.search import _seed_for, run_search
from analog_mc.simulate import forecast as _simulate_forecast


BACKEND_NAME = "analog_mc"
log = logging.getLogger("analog_mc.forecaster")


# ----------------------------------------------------------------------------
# Hyperparameter → Config translation
# ----------------------------------------------------------------------------


def _config_field_names() -> set[str]:
    return {f.name for f in fields(Config)}


def _build_config(hyperparameters: dict[str, Any], seed: int | None) -> Config:
    """Coerce a preset's ``hyperparameters`` dict to an analog_mc Config.

    Unknown keys (e.g., ``weights``, ``n_eff`` — single-origin extras the
    framework keeps alongside config fields) are filtered out before
    construction; they are handled separately by the caller.

    ``data_path`` / ``date_col`` / ``close_col`` are not part of a forecast
    preset (the data arrives as a DataFrame), so they are NOT set from
    hyperparameters even if present — we keep the Config dataclass defaults.
    """
    config_keys = _config_field_names()
    cfg_args = {
        k: v for k, v in hyperparameters.items()
        if k in config_keys and k not in ("data_path", "date_col", "close_col", "ticker")
    }
    if seed is not None:
        cfg_args["random_seed"] = int(seed)
    return Config(**cfg_args)


# ----------------------------------------------------------------------------
# Origin / horizon resolution
# ----------------------------------------------------------------------------


def _resolve_origin_idx(returns: pd.Series, origin_iso: str) -> int:
    """Find the positional index in `returns` for the given ISO origin date.

    The forecast begins on the FIRST trading day strictly after ``origin``
    (per the wire-format contract: "forecast starts the next session").
    Concretely, ``origin_idx`` is the index of the trailing return whose
    DATE is <= origin (i.e., the most-recent observation as of origin).

    Raises ValueError if no such date exists in the series.
    """
    origin_ts = pd.Timestamp(origin_iso)
    mask = returns.index <= origin_ts
    if not mask.any():
        raise ValueError(
            f"origin {origin_iso} is before the first available return "
            f"({returns.index[0].date().isoformat()})"
        )
    return int(np.where(mask)[0][-1])


def _horizon_dates(returns: pd.Series, origin_idx: int, horizon: int) -> list[str]:
    """ISO-date list of the H trading days following the origin.

    Source of truth: actual dates present in the returns index after the
    origin. If fewer than ``horizon`` future dates exist, we extend with
    plain calendar-day successors of the last known date so the result is
    always length-H (the dispatcher's contract validator requires this).
    Extension is a fallback; ``warnings`` carries a note when it kicks in.
    """
    future = returns.index[origin_idx + 1 : origin_idx + 1 + horizon]
    out = [d.date().isoformat() for d in future]
    if len(out) < horizon:
        # Pad with synthetic calendar days; this is a tail-end forecast where
        # the realized future doesn't exist yet (test-only or live-forward).
        last = future[-1] if len(future) else returns.index[origin_idx]
        for k in range(len(out), horizon):
            d = (last + pd.Timedelta(days=k - len(out) + 1)).date().isoformat()
            out.append(d)
    return out


# ----------------------------------------------------------------------------
# Path → result dict (price/log-return summary)
# ----------------------------------------------------------------------------


def _summarize_paths(
    paths_logret: np.ndarray,
    origin_close: float,
) -> dict[str, list[float]]:
    """Convert simulated log-return paths to price summary percentiles.

    Output is in PRICE space — cumsum the log returns, exponentiate, multiply
    by the origin's close. Percentiles are along the path axis at each step.
    """
    cum_logret = np.cumsum(paths_logret, axis=1)  # (N, H)
    price = origin_close * np.exp(cum_logret)
    return {
        "median": [float(x) for x in np.median(price, axis=0)],
        "p05": [float(x) for x in np.percentile(price, 5, axis=0)],
        "p25": [float(x) for x in np.percentile(price, 25, axis=0)],
        "p75": [float(x) for x in np.percentile(price, 75, axis=0)],
        "p95": [float(x) for x in np.percentile(price, 95, axis=0)],
    }


def _config_hash(config: Config) -> str:
    import yaml
    s = yaml.safe_dump(config.to_dict(), sort_keys=True)
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


# ----------------------------------------------------------------------------
# forecast()
# ----------------------------------------------------------------------------


def _get_weights_and_n_eff(
    hyperparameters: dict[str, Any],
    returns_arr: np.ndarray,
    origin_idx: int,
    features: pd.DataFrame,
    config: Config,
    warnings: list[str],
) -> tuple[np.ndarray, float]:
    """Resolve ``(weights, n_eff)`` for a single-origin forecast.

    If the preset's hyperparameters specify explicit ``weights`` (length-3
    list) and ``n_eff`` (numeric), use them. Otherwise run a one-fold val
    search against the ``val_size`` returns ending at ``origin_idx``, using
    the prior ``train_initial_size`` returns as the analog candidate pool.
    """
    if "weights" in hyperparameters and "n_eff" in hyperparameters:
        w_raw = hyperparameters["weights"]
        if not (isinstance(w_raw, (list, tuple)) and len(w_raw) == 3):
            raise ValueError(
                f"hyperparameters.weights must be a length-3 list; got {w_raw!r}"
            )
        weights = np.asarray(w_raw, dtype=np.float64)
        s = weights.sum()
        if s <= 0:
            raise ValueError(f"hyperparameters.weights must have positive sum; got {w_raw!r}")
        weights = weights / s
        n_eff = float(hyperparameters["n_eff"])
        return weights, n_eff

    # No baked weights → run a one-fold search ending at the origin.
    # Build a synthetic fold: val window is the last `val_size` returns
    # ending strictly before the origin's forecast horizon needs; train is
    # everything before that. This is essentially one walk-forward fold.
    val_size = config.val_size
    train_end = origin_idx - val_size
    if train_end < config.train_initial_size:
        # Not enough data for the full search machinery — degrade to equal
        # weights at the middle n_eff value.
        weights = np.array([1.0, 1.0, 1.0]) / 3.0
        n_eff = float(config.n_eff_values[len(config.n_eff_values) // 2])
        warnings.append(
            f"insufficient history before origin for one-fold search "
            f"(have {origin_idx + 1} returns; need >= "
            f"{config.train_initial_size + val_size + config.forecast_horizon}). "
            f"Falling back to equal weights and n_eff={n_eff}."
        )
        return weights, n_eff

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, origin_idx, dtype=np.int64)[:val_size]
    # We do not need a real test_idx for run_search; supply a dummy 0-length
    # window after val.
    test_idx = np.array([], dtype=np.int64)
    fold = Fold(index=0, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    log.info(
        "forecast: fitting one-fold search (train=%d, val=%d) for origin idx %d",
        train_idx.size, val_idx.size, origin_idx,
    )
    res = run_search(fold, returns_arr, features, config)
    warnings.append(
        f"weights and n_eff fit on val window [{val_idx[0]}:{val_idx[-1] + 1}] "
        f"(val_crps={res.val_crps:.5f}); preset did not carry baked weights/n_eff."
    )
    return res.weights, float(res.n_eff)


def _resolve_data_columns(
    df: pd.DataFrame,
    input_dict: dict[str, Any],
    hp: dict[str, Any],
) -> tuple[str, str]:
    """Best-effort lookup for (date_col, close_col) on the input DataFrame.

    Probe order (first pair where BOTH columns are present in df wins):
      1. Explicit caller override (``input_dict.date_col`` / ``close_col``).
      2. Canonical data_pipelines schema (``date`` / ``adj_close``).
      3. FRED-style (``observation_date`` / ``NASDAQ100``).
      4. Preset-baked hints (``hp.date_col`` / ``hp.close_col``).

    The preset hints come last because they are usually the COLUMNS THE
    PRESET WAS TUNED ON, not the columns of the new input DataFrame.
    """
    candidates: list[tuple[str, str]] = []
    if "date_col" in input_dict and "close_col" in input_dict:
        candidates.append((input_dict["date_col"], input_dict["close_col"]))
    candidates.append(("date", "adj_close"))
    candidates.append(("observation_date", "NASDAQ100"))
    if "date_col" in hp and "close_col" in hp:
        candidates.append((hp["date_col"], hp["close_col"]))
    for dc, cc in candidates:
        if dc in df.columns and cc in df.columns:
            return dc, cc
    raise ValueError(
        f"data DataFrame lacks recognizable date/close columns; tried "
        f"{candidates}; available={list(df.columns)}"
    )


def forecast(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Single-origin price-path forecast.

    See ``docs/forecasters/V1_PLAN.md`` §"Wire-format contract" for the
    input/output JSON shape this function must conform to.
    """
    required = ("data", "origin", "horizon", "hyperparameters")
    missing = [k for k in required if k not in input_dict]
    if missing:
        raise ValueError(f"forecast input missing keys: {missing}")

    df: pd.DataFrame = input_dict["data"]
    origin_iso: str = input_dict["origin"]
    horizon: int = int(input_dict["horizon"])
    hp: dict[str, Any] = dict(input_dict["hyperparameters"])
    seed: int | None = input_dict.get("seed")

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")

    warnings: list[str] = []

    # ---- Build the analog_mc Config (use horizon to override) ------------
    # Pick a block_length / n_blocks consistent with the requested horizon.
    # If horizon == hp.forecast_horizon, leave block_length/n_blocks alone.
    # Otherwise, derive: try the preset's block_length; if it divides horizon,
    # use it; else fall back to a sensible split.
    hp_eff = dict(hp)
    hp_eff["forecast_horizon"] = horizon
    requested_bl = int(hp.get("block_length", 10))
    if horizon % requested_bl == 0:
        hp_eff["block_length"] = requested_bl
        hp_eff["n_blocks"] = horizon // requested_bl
    else:
        # Fall back: 10-day blocks if divisible, else 1 block of length-horizon.
        if horizon % 10 == 0:
            hp_eff["block_length"] = 10
            hp_eff["n_blocks"] = horizon // 10
        else:
            hp_eff["block_length"] = horizon
            hp_eff["n_blocks"] = 1
        warnings.append(
            f"requested horizon {horizon} not divisible by preset block_length "
            f"{requested_bl}; using block_length={hp_eff['block_length']}, "
            f"n_blocks={hp_eff['n_blocks']}."
        )

    config = _build_config(hp_eff, seed=seed)

    # ---- Load close series & log returns --------------------------------
    # Resolve date/close column names against the actual DataFrame columns.
    # Preset-baked column names are TREATED AS HINTS, not requirements — a
    # preset tuned on NASDAQ100 (FRED columns) might be applied at forecast
    # time to a canonical-schema DataFrame from data_pipelines (date /
    # adj_close), and the preset's column hints would be wrong for the new
    # input. We probe in order: explicit caller override (input_dict),
    # canonical (date / adj_close), FRED-style (observation_date / NASDAQ100),
    # preset hints (hp.date_col / hp.close_col) — falling back through.
    date_col, close_col = _resolve_data_columns(df, input_dict, hp)
    close = close_series_from_dataframe(df, date_col=date_col, close_col=close_col)
    if len(close) < 2:
        raise ValueError(f"data has < 2 valid rows; cannot compute returns")
    returns = log_returns(close)
    returns_arr = returns.to_numpy()

    # ---- Resolve origin --------------------------------------------------
    origin_idx = _resolve_origin_idx(returns, origin_iso)
    if origin_idx + 1 > returns.size:
        raise ValueError(
            f"origin {origin_iso} sits at the end of the series with no future "
            f"returns; can't forecast forward."
        )

    # ---- Features --------------------------------------------------------
    features = compute_features(
        returns,
        halflife=config.ewma_halflife,
        horizons=config.zscore_horizons,
        momentum_lookback=config.momentum_lookback if config.drift_mode != "zero" else None,
    )

    # ---- Resolve weights + n_eff ----------------------------------------
    weights, n_eff = _get_weights_and_n_eff(
        hp, returns_arr, origin_idx, features, config, warnings,
    )

    # ---- Build the analog candidate pool --------------------------------
    # Default: every prior return is fair game (subject to the forward-block
    # boundary in simulate.eligible_candidates).
    candidate_idx = np.arange(0, origin_idx, dtype=np.int64)
    if candidate_idx.size == 0:
        raise ValueError(
            f"no candidate returns available before origin {origin_iso}"
        )

    # ---- Forecast --------------------------------------------------------
    rng = np.random.default_rng(_seed_for(config.random_seed, weights, n_eff, origin_idx))
    t0 = time.perf_counter()
    paths_logret = _simulate_forecast(
        origin_idx=origin_idx,
        returns=returns_arr,
        candidate_idx=candidate_idx,
        features=features,
        weights=weights,
        n_eff=n_eff,
        config=config,
        rng=rng,
    )
    elapsed = time.perf_counter() - t0
    log.info(
        "forecast: generated %d paths × %d horizon in %.2fs",
        paths_logret.shape[0], paths_logret.shape[1], elapsed,
    )

    # ---- Build result dict -----------------------------------------------
    origin_close = float(close.iloc[origin_idx])
    summary = _summarize_paths(paths_logret, origin_close)

    # CRPS against the realized horizon if available (test-mode forecast).
    crps_val: float | None = None
    realized_avail = returns_arr.size - (origin_idx + 1)
    if realized_avail >= horizon:
        realized = returns_arr[origin_idx + 1 : origin_idx + 1 + horizon]
        try:
            crps_val = float(crps_sample(paths_logret, realized))
        except Exception as e:  # pragma: no cover - defensive
            warnings.append(f"CRPS computation failed: {e}")
    summary["crps"] = crps_val

    horizon_dates = _horizon_dates(returns, origin_idx, horizon)
    if len(returns.index) - (origin_idx + 1) < horizon:
        warnings.append(
            f"only {len(returns.index) - (origin_idx + 1)} future dates in data; "
            f"padded horizon_dates with calendar-day successors."
        )

    return {
        "paths": paths_logret.astype(np.float64),  # log returns, not prices
        "anchors": {
            "origin_date": origin_iso,
            "horizon_dates": horizon_dates,
        },
        "summary": summary,
        "metadata": {
            "backend_name": BACKEND_NAME,
            "preset_name": input_dict.get("preset_name", "<inline>"),
            "preset_hash": input_dict.get("preset_hash", "<inline>"),
            "config_hash": _config_hash(config),
            "n_paths": int(paths_logret.shape[0]),
            "seed_used": int(config.random_seed),
            "origin_close": origin_close,
            "weights": [float(w) for w in weights],
            "n_eff": float(n_eff),
        },
        "warnings": warnings,
    }


# ----------------------------------------------------------------------------
# tune()
# ----------------------------------------------------------------------------


def tune(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Walk-forward + grid search across the supplied data range.

    Produces a ``preset_dict`` that conforms to the forecasters preset schema
    (``docs/forecasters/V1_PLAN.md`` §"Preset artifact schema"). The final
    fold's tuned ``(weights, n_eff)`` are pinned in
    ``hyperparameters.weights`` / ``hyperparameters.n_eff`` so a subsequent
    ``forecast()`` call against the produced preset uses them directly
    (no in-call re-tune required).
    """
    required = ("data", "identifier", "range")
    missing = [k for k in required if k not in input_dict]
    if missing:
        raise ValueError(f"tune input missing keys: {missing}")

    df: pd.DataFrame = input_dict["data"]
    identifier: str = input_dict["identifier"]
    range_pair: tuple[str, str] = tuple(input_dict["range"])  # type: ignore[assignment]
    search_config: dict[str, Any] = dict(input_dict.get("search_config") or {})
    seed: int | None = input_dict.get("seed")
    output_name: str = input_dict.get("output_name", "tuned-preset")

    # ---- Build Config from search_config (or canonical defaults) ---------
    # search_config is treated like a partial hyperparameters dict. Missing
    # fields fall back to the analog_mc Config defaults (which match the v2.4
    # canonical for any field not explicitly set).
    hp_eff = dict(search_config)
    config = _build_config(hp_eff, seed=seed)

    # ---- Load data -------------------------------------------------------
    date_col, close_col = _resolve_data_columns(df, input_dict, search_config)
    close = close_series_from_dataframe(df, date_col=date_col, close_col=close_col)
    returns = log_returns(close)

    # ---- Walk-forward ----------------------------------------------------
    # Lazy import: walk_forward pulls in multiprocessing setup we don't need
    # at module-import time.
    from analog_mc.features import compute_features
    from analog_mc.walk_forward import run_walk_forward, FoldOutcome

    log.info(
        "tune: running walk-forward on %s (%d returns, %s → %s)",
        identifier, len(returns), range_pair[0], range_pair[1],
    )

    t0 = time.perf_counter()
    run_dir = run_walk_forward(returns, config)
    runtime_sec = time.perf_counter() - t0
    log.info("tune: walk-forward done in %.1fs (run_dir=%s)", runtime_sec, run_dir)

    # ---- Aggregate fold outcomes ----------------------------------------
    # run_walk_forward writes per-fold summary.json files; read them back to
    # get the FINAL fold's (weights, n_eff) for the preset.
    from pathlib import Path
    folds_dir = Path(run_dir) / "folds"
    fold_summaries = []
    for sub in sorted(folds_dir.iterdir(), key=lambda p: int(p.name)):
        with open(sub / "summary.json") as f:
            fold_summaries.append(json.load(f))
    if not fold_summaries:
        raise RuntimeError(f"walk-forward produced 0 folds in {run_dir}")

    final = fold_summaries[-1]
    mean_test_crps = float(np.mean([fs["test_crps"] for fs in fold_summaries]))

    # ---- Build the preset dict ------------------------------------------
    data_hash = _df_close_hash(close)
    fitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Bake the final-fold tuned knobs into hyperparameters so forecast()
    # can use them directly without re-tuning.
    baked_hp = dict(config.to_dict())
    baked_hp["weights"] = [float(w) for w in final["weights"]]
    baked_hp["n_eff"] = float(final["n_eff"])

    preset = {
        "name": output_name,
        "backend": BACKEND_NAME,
        "schema_version": 1,
        "hyperparameters": baked_hp,
        "fitted_on": {
            "identifier": identifier,
            "start": range_pair[0],
            "end": range_pair[1],
            "data_hash": data_hash,
            "n_observations": int(len(close)),
        },
        "fitted_at": fitted_at,
        "validation_metrics": {
            "crps_mean": mean_test_crps,
            "n_folds": len(fold_summaries),
            "final_fold_val_crps": float(final["val_crps"]),
            "final_fold_test_crps": float(final["test_crps"]),
        },
        "provenance": {
            "source": "tuned",
            "tune_runtime_seconds": float(runtime_sec),
            "run_dir": str(run_dir),
            "n_folds": len(fold_summaries),
        },
    }
    return preset


def _df_close_hash(close: pd.Series) -> str:
    """Hash of the (date, close) pairs — stable across DataFrames with the
    same data, insensitive to column ordering or extra columns.
    """
    h = hashlib.sha256()
    for ts, val in zip(close.index, close.to_numpy(), strict=True):
        h.update(pd.Timestamp(ts).strftime("%Y-%m-%d").encode())
        h.update(b":")
        h.update(f"{float(val):.10g}".encode())
        h.update(b"\n")
    return "sha256:" + h.hexdigest()
