"""Tests for the analog_mc public forecast/tune API (forecasters wire format).

These tests assert the wire-format CONTRACT — what dispatcher.py later checks
at runtime. They use the NASDAQ100 CSV fixture and a slim hyperparameter
configuration (small n_paths, explicit weights/n_eff) to keep wall time low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analog_mc.data import close_series_from_dataframe, load_close_series
from analog_mc.forecaster import BACKEND_NAME, forecast


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nasdaq100_df() -> pd.DataFrame:
    """The NASDAQ100 CSV exposed as a canonical-schema DataFrame.

    The fixture renames columns into the data_pipelines canonical schema
    (``date`` / ``adj_close``) so it round-trips through the framework's
    DataFrame-input path. The original CSV (FRED-style columns) is the
    project default per [[project-data-source]].
    """
    raw = pd.read_csv("data/NASDAQ100.csv")
    return pd.DataFrame({
        "date": pd.to_datetime(raw["observation_date"]),
        "adj_close": raw["NASDAQ100"].astype(float),
    })


def _slim_hyperparameters() -> dict:
    """A v1-shaped hyperparameter dict with explicit weights/n_eff so the
    forecast call skips its in-fold mini-search and runs in seconds.
    """
    return {
        "forecast_horizon": 60,
        "block_length": 10,
        "n_blocks": 6,
        "n_paths": 200,  # slim — full canonical uses 1000
        "zscore_horizons": [20, 50, 200],
        "ewma_halflife": 20,
        "train_initial_size": 1000,
        "val_size": 60,
        "test_size": 60,
        "vol_clip_lower": 0.5,
        "vol_clip_upper": 3.0,
        "drift_mode": "zero",  # off for the contract test
        "conditional_block_sampling": False,
        "matcher_distance": "weighted_euclidean",
        "vol_model": "ewma",
        "weights": [0.33, 0.34, 0.33],
        "n_eff": 50,
        "random_seed": 42,
    }


# ----------------------------------------------------------------------------
# Wire-format contract on the result dict
# ----------------------------------------------------------------------------


def test_forecast_returns_contract_conformant_result(nasdaq100_df: pd.DataFrame) -> None:
    """The result must conform to the V1_PLAN wire-format JSON shape."""
    horizon = 60
    input_dict = {
        "data": nasdaq100_df,
        "origin": "2020-01-02",
        "horizon": horizon,
        "hyperparameters": _slim_hyperparameters(),
        "seed": 42,
    }
    result = forecast(input_dict)

    # Top-level keys.
    for k in ("paths", "anchors", "summary", "metadata", "warnings"):
        assert k in result, f"missing top-level key {k!r}"

    # paths
    paths = result["paths"]
    assert isinstance(paths, np.ndarray)
    assert paths.ndim == 2
    assert paths.shape[1] == horizon
    assert paths.dtype == np.float64
    assert np.isfinite(paths).all()
    assert paths.shape[0] == 200  # echoes hyperparameters.n_paths

    # anchors
    anchors = result["anchors"]
    assert anchors["origin_date"] == "2020-01-02"
    assert len(anchors["horizon_dates"]) == horizon
    assert all(isinstance(d, str) and len(d) == 10 for d in anchors["horizon_dates"])

    # summary
    summary = result["summary"]
    for k in ("median", "p05", "p25", "p75", "p95"):
        assert len(summary[k]) == horizon
        assert all(isinstance(v, float) for v in summary[k])
    # Monotone percentile ordering at every step.
    for h in range(horizon):
        assert summary["p05"][h] <= summary["p25"][h] <= summary["median"][h]
        assert summary["median"][h] <= summary["p75"][h] <= summary["p95"][h]
    assert summary["crps"] is None or isinstance(summary["crps"], float)

    # metadata
    md = result["metadata"]
    assert md["backend_name"] == BACKEND_NAME
    assert md["n_paths"] == 200
    assert md["seed_used"] == 42
    assert md["config_hash"].startswith("sha256:")
    assert len(md["weights"]) == 3
    assert md["n_eff"] == 50.0

    # warnings is a list, never None.
    assert isinstance(result["warnings"], list)


def test_forecast_is_deterministic_with_seed(nasdaq100_df: pd.DataFrame) -> None:
    """Same input → bit-identical paths."""
    input_dict = {
        "data": nasdaq100_df,
        "origin": "2020-01-02",
        "horizon": 60,
        "hyperparameters": _slim_hyperparameters(),
        "seed": 123,
    }
    r1 = forecast(input_dict)
    r2 = forecast(input_dict)
    assert np.array_equal(r1["paths"], r2["paths"])
    assert r1["metadata"]["config_hash"] == r2["metadata"]["config_hash"]


def test_forecast_crps_present_for_in_sample_origin(nasdaq100_df: pd.DataFrame) -> None:
    """CRPS must be computed when the realized horizon is available."""
    input_dict = {
        "data": nasdaq100_df,
        "origin": "2020-01-02",
        "horizon": 60,
        "hyperparameters": _slim_hyperparameters(),
        "seed": 42,
    }
    result = forecast(input_dict)
    crps = result["summary"]["crps"]
    assert crps is not None
    assert np.isfinite(crps)
    assert crps > 0.0


def test_forecast_horizon_must_be_positive(nasdaq100_df: pd.DataFrame) -> None:
    input_dict = {
        "data": nasdaq100_df,
        "origin": "2020-01-02",
        "horizon": 0,
        "hyperparameters": _slim_hyperparameters(),
    }
    with pytest.raises(ValueError, match="horizon"):
        forecast(input_dict)


def test_forecast_rejects_pre_history_origin(nasdaq100_df: pd.DataFrame) -> None:
    input_dict = {
        "data": nasdaq100_df,
        "origin": "1900-01-01",
        "horizon": 60,
        "hyperparameters": _slim_hyperparameters(),
    }
    with pytest.raises(ValueError, match="before"):
        forecast(input_dict)


def test_forecast_missing_required_key_raises(nasdaq100_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing keys"):
        forecast({
            "data": nasdaq100_df,
            "origin": "2020-01-02",
            "hyperparameters": _slim_hyperparameters(),
        })  # horizon missing


def test_forecast_accepts_fred_style_columns() -> None:
    """The NASDAQ100 CSV's native (observation_date, NASDAQ100) shape works
    without renaming — the loader has a documented fallback for that pair."""
    raw = pd.read_csv("data/NASDAQ100.csv")
    input_dict = {
        "data": raw,
        "origin": "2020-01-02",
        "horizon": 60,
        "hyperparameters": _slim_hyperparameters(),
        "seed": 42,
    }
    result = forecast(input_dict)
    assert result["paths"].shape == (200, 60)


def test_forecast_summary_in_price_space(nasdaq100_df: pd.DataFrame) -> None:
    """Summary percentiles are in PRICE space — median[0] should be close
    to (not identical to) the origin's close. We pin the loose bracket so
    the test is robust to MC noise but still catches a log-return-space
    accidental return shape."""
    input_dict = {
        "data": nasdaq100_df,
        "origin": "2020-01-02",
        "horizon": 60,
        "hyperparameters": _slim_hyperparameters(),
        "seed": 42,
    }
    result = forecast(input_dict)
    origin_close = result["metadata"]["origin_close"]
    median_step0 = result["summary"]["median"][0]
    # Day-1 median should be within a few percent of the origin close.
    assert 0.85 * origin_close < median_step0 < 1.15 * origin_close


def test_warnings_is_always_list(nasdaq100_df: pd.DataFrame) -> None:
    """A clean run still returns warnings as a list, never None."""
    input_dict = {
        "data": nasdaq100_df,
        "origin": "2020-01-02",
        "horizon": 60,
        "hyperparameters": _slim_hyperparameters(),
        "seed": 42,
    }
    result = forecast(input_dict)
    assert isinstance(result["warnings"], list)


def test_close_series_from_dataframe_canonical_schema(nasdaq100_df: pd.DataFrame) -> None:
    s = close_series_from_dataframe(nasdaq100_df, date_col="date", close_col="adj_close")
    # Should match what load_close_series produces on the original CSV.
    s_ref = load_close_series("data/NASDAQ100.csv", "observation_date", "NASDAQ100")
    assert len(s) == len(s_ref)
    assert (s.values == s_ref.values).all()
    assert (s.index == s_ref.index).all()
