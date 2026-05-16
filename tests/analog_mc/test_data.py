"""Tests for analog_mc.data — CSV loading and log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analog_mc.config import Config
from analog_mc.data import load_close_series, load_returns, log_returns


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "px.csv"
    path.write_text(
        "observation_date,FOO\n"
        "2024-01-02,100.0\n"
        "2024-01-03,101.0\n"
        "2024-01-04,99.0\n"
        "2024-01-05,102.0\n"
    )
    return path


def test_load_close_series(csv_path) -> None:
    s = load_close_series(csv_path, date_col="observation_date", close_col="FOO")
    assert len(s) == 4
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.is_monotonic_increasing
    assert s.iloc[0] == 100.0


def test_load_close_drops_duplicates_keep_last(tmp_path) -> None:
    path = tmp_path / "dups.csv"
    path.write_text(
        "d,c\n2024-01-02,1.0\n2024-01-02,2.0\n2024-01-03,3.0\n"
    )
    s = load_close_series(path, date_col="d", close_col="c")
    assert len(s) == 2
    assert s.iloc[0] == 2.0


def test_load_close_drops_nans(tmp_path) -> None:
    path = tmp_path / "na.csv"
    path.write_text("d,c\n2024-01-02,1.0\n2024-01-03,\n2024-01-04,3.0\n")
    s = load_close_series(path, date_col="d", close_col="c")
    assert len(s) == 2


def test_log_returns_values() -> None:
    close = pd.Series(
        [100.0, 110.0, 99.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    r = log_returns(close)
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(np.log(110.0 / 100.0))
    assert r.iloc[1] == pytest.approx(np.log(99.0 / 110.0))


def test_log_returns_rejects_non_positive() -> None:
    close = pd.Series([100.0, 0.0, 99.0])
    with pytest.raises(ValueError, match="non-positive"):
        log_returns(close)


def test_load_returns_via_config(csv_path) -> None:
    cfg = Config(
        ticker="FOO",
        data_path=str(csv_path),
        date_col="observation_date",
        close_col="FOO",
    )
    r = load_returns(cfg)
    assert len(r) == 3  # n - 1 because the first row has no prior close
    assert isinstance(r.index, pd.DatetimeIndex)


def test_load_nasdaq100_real_file() -> None:
    """Smoke-test against the actual NASDAQ100.csv shipped in data/."""
    cfg = Config()  # defaults point at data/NASDAQ100.csv
    try:
        r = load_returns(cfg)
    except FileNotFoundError:
        pytest.skip("data/NASDAQ100.csv not present")
    assert len(r) > 10_000
    # Sanity: daily log returns should generally be small.
    assert r.abs().median() < 0.02
