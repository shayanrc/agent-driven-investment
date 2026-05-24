"""Stage 5 tests: USEquitiesConfig invariants."""

from __future__ import annotations

import pytest

from data_pipelines.domains.us_equities.config import USEquitiesConfig


def test_defaults_valid():
    c = USEquitiesConfig()
    assert c.big_gap_threshold_days == 90
    assert c.chain_seed == "stooq"
    assert c.chain_update == ("tiingo", "yfinance")
    assert c.tiingo_api_key_env == "TIINGO_API_KEY"
    assert "^SPX" in c.stooq_index_slugs


def test_zero_threshold_rejected():
    with pytest.raises(ValueError, match="big_gap_threshold_days"):
        USEquitiesConfig(big_gap_threshold_days=0)


def test_negative_max_retries_rejected():
    with pytest.raises(ValueError, match="tiingo_max_retries"):
        USEquitiesConfig(tiingo_max_retries=-1)


def test_empty_env_var_name_rejected():
    with pytest.raises(ValueError, match="tiingo_api_key_env"):
        USEquitiesConfig(tiingo_api_key_env="")


def test_empty_update_chain_rejected():
    with pytest.raises(ValueError, match="chain_update"):
        USEquitiesConfig(chain_update=())


def test_zero_timeout_rejected():
    with pytest.raises(ValueError, match="timeouts"):
        USEquitiesConfig(tiingo_timeout_sec=0)
