"""fred_macro schema: (date, value) with a nullable value column."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipelines.domains.fred_macro.schema import FRED_SCHEMA
from data_pipelines.errors import SchemaMismatch


def _df(dates, values):
    return pd.DataFrame({
        "date": pd.to_datetime(dates).astype("datetime64[ns]"),
        "value": pd.Series(values, dtype="float64"),
    })


def test_column_names():
    assert FRED_SCHEMA.column_names == ["date", "value"]


def test_accepts_nan_value():
    # value is nullable — NaN (FRED "." / suppressed observation) is legal.
    FRED_SCHEMA.validate(_df(["2020-01-01", "2020-01-02"], [1.0, np.nan]))


def test_rejects_nan_date():
    df = _df(["2020-01-01", "2020-01-02"], [1.0, 2.0])
    df.loc[1, "date"] = pd.NaT
    with pytest.raises(SchemaMismatch):
        FRED_SCHEMA.validate(df)


def test_normalize_casts_strings():
    raw = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "value": ["1.5", "2.5"]})
    out = FRED_SCHEMA.normalize(raw)
    assert str(out["date"].dtype) == "datetime64[ns]"
    assert str(out["value"].dtype) == "float64"
    assert out["value"].tolist() == [1.5, 2.5]
