"""Cross-asset drift test: forecasting NYSE-style asset with v24-default
(fit on NASDAQ100) must surface the drift warning.

Uses the seeded data_pipelines cache (NASDAQ:AAPL) shipped with the repo.
Forecast is slim (n_paths=100, explicit weights/n_eff) for speed.
"""

from __future__ import annotations

import pytest

from forecasters import data_hash, dispatch_forecast, load_preset, prepare_data


@pytest.fixture(scope="module")
def aapl_df():
    """Pull NASDAQ:AAPL from the seeded processed.db. Skip if absent."""
    try:
        df = prepare_data(
            identifier="NASDAQ:AAPL",
            start="2018-01-01",
            end="2024-12-31",
        )
    except Exception as e:
        pytest.skip(f"NASDAQ:AAPL not available in processed.db: {e}")
    if df is None or len(df) == 0:
        pytest.skip("NASDAQ:AAPL slice is empty in cache")
    return df


def test_aapl_with_v24_default_surfaces_drift(aapl_df) -> None:
    preset = load_preset("v24-default")
    hp = dict(preset["hyperparameters"])
    hp.update({"n_paths": 100, "weights": [0.33, 0.34, 0.33], "n_eff": 50})
    input_dict = {
        "data": aapl_df,
        "origin": "2020-01-02",
        "horizon": 60,
        "hyperparameters": hp,
        "seed": 42,
    }
    result = dispatch_forecast(preset, input_dict)
    # Drift warning is present.
    drift_warnings = [w for w in result["warnings"]
                      if "Hyperparameters may be uncalibrated" in w]
    assert drift_warnings, (
        f"expected drift warning, got warnings={result['warnings']}"
    )
    # Warning quantifies BOTH the preset's data_hash and the current one.
    msg = drift_warnings[0]
    fitted_hash = preset["fitted_on"]["data_hash"]
    current_hash = data_hash(aapl_df)
    assert fitted_hash in msg
    assert current_hash in msg
    # Forecast still ran — warnings are not errors.
    assert result["paths"].shape == (100, 60)
