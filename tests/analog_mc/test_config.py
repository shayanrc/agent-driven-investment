"""Tests for the Config dataclass: invariants + YAML round-trip."""

from __future__ import annotations

import pytest

from analog_mc.config import Config


def test_default_config_is_valid() -> None:
    cfg = Config()
    assert cfg.forecast_horizon == cfg.n_blocks * cfg.block_length
    assert len(cfg.zscore_horizons) == 3
    assert max(cfg.zscore_horizons) < cfg.train_initial_size
    assert cfg.vol_clip_lower < 1.0 < cfg.vol_clip_upper


def test_invariant_forecast_horizon_consistency() -> None:
    with pytest.raises(ValueError, match="forecast_horizon"):
        Config(forecast_horizon=50, n_blocks=6, block_length=10)


def test_invariant_three_zscore_horizons() -> None:
    with pytest.raises(ValueError, match="zscore_horizons"):
        Config(zscore_horizons=(20, 50))
    with pytest.raises(ValueError, match="zscore_horizons"):
        Config(zscore_horizons=(20, 50, 100, 200))


def test_invariant_zscore_within_train_size() -> None:
    with pytest.raises(ValueError, match="train_initial_size"):
        Config(zscore_horizons=(20, 50, 5000), train_initial_size=1000)


def test_invariant_vol_clip_brackets_one() -> None:
    with pytest.raises(ValueError, match="vol_clip_lower"):
        Config(vol_clip_lower=1.1, vol_clip_upper=2.0)
    with pytest.raises(ValueError, match="vol_clip_lower"):
        Config(vol_clip_lower=0.5, vol_clip_upper=0.9)


def test_invariant_drift_mode() -> None:
    with pytest.raises(ValueError, match="drift_mode"):
        Config(drift_mode="bogus")


def test_yaml_round_trip(tmp_path) -> None:
    original = Config(ticker="TEST", n_paths=500, zscore_horizons=(10, 30, 90))
    path = tmp_path / "cfg.yaml"
    original.to_yaml(path)
    loaded = Config.from_yaml(path)
    assert loaded == original
    # Tuples must be preserved (yaml emits lists; __post_init__ coerces back).
    assert isinstance(loaded.zscore_horizons, tuple)
    assert isinstance(loaded.n_eff_values, tuple)
    assert isinstance(loaded.acf_lags, tuple)


def test_yaml_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("ticker: X\nnot_a_real_field: 42\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_yaml(path)
