"""Tests for forecasters.data.prepare_data + data_hash."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forecasters.data import data_hash, prepare_data


def test_prepare_data_requires_exactly_one_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        prepare_data(start="2020-01-01", end="2020-12-31")
    with pytest.raises(ValueError, match="exactly one"):
        prepare_data(identifier="X", data_path=tmp_path / "f.csv",
                     start="2020-01-01", end="2020-12-31")


def test_prepare_data_path_csv_canonical_columns(tmp_path: Path) -> None:
    p = tmp_path / "small.csv"
    p.write_text("date,adj_close\n2020-01-01,100\n2020-01-02,101\n2020-01-03,102\n")
    df = prepare_data(data_path=p)
    assert len(df) == 3
    assert "date" in df.columns and "adj_close" in df.columns


def test_prepare_data_path_csv_slices_by_date(tmp_path: Path) -> None:
    p = tmp_path / "small.csv"
    p.write_text("date,adj_close\n2020-01-01,100\n2020-02-01,101\n2020-03-01,102\n")
    df = prepare_data(data_path=p, start="2020-01-15", end="2020-02-15")
    assert len(df) == 1


def test_prepare_data_path_fred_style_columns_slice() -> None:
    """The project's NASDAQ100 CSV uses observation_date / NASDAQ100."""
    df = prepare_data(
        data_path="data/NASDAQ100.csv",
        start="2020-01-01",
        end="2020-12-31",
    )
    assert "observation_date" in df.columns
    assert "NASDAQ100" in df.columns
    # Sliced by date.
    assert df["observation_date"].min() >= pd.Timestamp("2020-01-01")
    assert df["observation_date"].max() <= pd.Timestamp("2020-12-31")


def test_prepare_data_path_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_data(data_path=tmp_path / "nope.csv")


def test_prepare_data_identifier_requires_dates() -> None:
    with pytest.raises(ValueError, match="start and end"):
        prepare_data(identifier="NASDAQ:AAPL")


def test_data_hash_canonical_schema() -> None:
    df1 = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "adj_close": [100.0, 101.0],
    })
    df2 = df1.copy()
    assert data_hash(df1) == data_hash(df2)
    # Adding an extra column doesn't change the hash.
    df3 = df1.copy()
    df3["volume"] = [1, 2]
    assert data_hash(df3) == data_hash(df1)
    # Changing a value DOES change the hash.
    df4 = df1.copy()
    df4.loc[1, "adj_close"] = 999.0
    assert data_hash(df4) != data_hash(df1)


def test_data_hash_falls_back_to_fred_style() -> None:
    df = pd.DataFrame({
        "observation_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "NASDAQ100": [100.0, 101.0],
    })
    h = data_hash(df)
    assert h.startswith("sha256:")


def test_data_hash_missing_columns_raises() -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    with pytest.raises(ValueError, match="cannot find"):
        data_hash(df)
