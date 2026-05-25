"""End-to-end forecast tests using the real analog_mc backend.

These exercise the full ``/forecast`` flow (preset load → data prep →
dispatch → result validation → cache write) against the NASDAQ100 CSV.
Hyperparameters are slimmed (n_paths=100, explicit weights/n_eff) so the
test runs in seconds — the V5.A.2 baseline metrics come from a 1000-path
run, so we check only that the result conforms to the contract and that
the forecast is in a reasonable price-space range.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasters import dispatch_forecast, load_preset


@pytest.fixture
def overrides_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "overrides.yaml"
    p.write_text(textwrap.dedent(
        """
        n_paths: 100
        weights: [0.33, 0.34, 0.33]
        n_eff: 50
        """
    ).strip())
    return p


def test_forecast_e2e_nasdaq100_csv(tmp_path: Path, overrides_yaml: Path) -> None:
    """End-to-end forecast through the dispatcher on the canonical preset."""
    preset = load_preset("v24-default")
    raw = pd.read_csv("data/NASDAQ100.csv")
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["observation_date"]),
        "adj_close": raw["NASDAQ100"].astype(float),
    })
    hp = dict(preset["hyperparameters"])
    hp.update({"n_paths": 100, "weights": [0.33, 0.34, 0.33], "n_eff": 50})
    input_dict = {
        "data": df,
        "origin": "2020-01-02",
        "horizon": 60,
        "hyperparameters": hp,
        "seed": 42,
    }
    result = dispatch_forecast(preset, input_dict)

    # Contract.
    assert result["paths"].shape == (100, 60)
    assert len(result["summary"]["median"]) == 60
    assert result["metadata"]["backend_name"] == "analog_mc"
    assert result["metadata"]["preset_name"] == "v24-default"
    assert result["metadata"]["preset_hash"].startswith("sha256:")

    # No drift: forecasting on the same data the preset was fit on.
    drift_warnings = [w for w in result["warnings"]
                      if "uncalibrated" in w.lower()]
    assert drift_warnings == [], f"unexpected drift warning: {drift_warnings}"

    # CRPS present (in-sample horizon available).
    assert result["summary"]["crps"] is not None
    assert np.isfinite(result["summary"]["crps"])
    # CRPS for a slim 100-path forecast on a calm 2020-01-02 origin should
    # be reasonable — give a generous bracket so MC noise doesn't break the
    # test, but tight enough to catch a wholly wrong scale.
    assert 0.0 < result["summary"]["crps"] < 1.0


def test_forecast_e2e_determinism_through_dispatcher(overrides_yaml: Path) -> None:
    """Same preset + same data + same seed → bit-identical paths."""
    preset = load_preset("v24-default")
    raw = pd.read_csv("data/NASDAQ100.csv")
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["observation_date"]),
        "adj_close": raw["NASDAQ100"].astype(float),
    })
    hp = dict(preset["hyperparameters"])
    hp.update({"n_paths": 100, "weights": [0.33, 0.34, 0.33], "n_eff": 50})
    inp = {"data": df, "origin": "2020-01-02", "horizon": 60,
           "hyperparameters": hp, "seed": 42}
    r1 = dispatch_forecast(preset, inp)
    r2 = dispatch_forecast(preset, inp)
    np.testing.assert_array_equal(r1["paths"], r2["paths"])
