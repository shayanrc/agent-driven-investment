"""Stage 1 tests: schema primitives (ColumnSpec, Schema, validate, normalize)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipelines.errors import SchemaMismatch
from data_pipelines.schema import ColumnSpec, Schema


@pytest.fixture
def toy_schema() -> Schema:
    return Schema(columns=(
        ColumnSpec("date", "datetime64[ns]"),
        ColumnSpec("value", "float64"),
        ColumnSpec("count", "int64"),
    ))


@pytest.fixture
def good_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]).astype("datetime64[ns]"),
        "value": [1.5, 2.5],
        "count": [10, 20],
    })


class TestSchemaConstruction:
    def test_duplicate_column_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            Schema(columns=(
                ColumnSpec("a", "float64"),
                ColumnSpec("a", "int64"),
            ))

    def test_column_names_property(self, toy_schema: Schema):
        assert toy_schema.column_names == ["date", "value", "count"]


class TestValidate:
    def test_good_df_passes(self, toy_schema: Schema, good_df: pd.DataFrame):
        toy_schema.validate(good_df)  # no raise

    def test_missing_column(self, toy_schema: Schema, good_df: pd.DataFrame):
        df = good_df.drop(columns=["count"])
        with pytest.raises(SchemaMismatch, match="column mismatch"):
            toy_schema.validate(df)

    def test_wrong_order(self, toy_schema: Schema, good_df: pd.DataFrame):
        df = good_df[["value", "date", "count"]]
        with pytest.raises(SchemaMismatch, match="column mismatch"):
            toy_schema.validate(df)

    def test_wrong_dtype(self, toy_schema: Schema, good_df: pd.DataFrame):
        df = good_df.copy()
        df["value"] = df["value"].astype("float32")
        with pytest.raises(SchemaMismatch, match="dtype mismatch"):
            toy_schema.validate(df)

    def test_nan_in_non_nullable(self, toy_schema: Schema, good_df: pd.DataFrame):
        df = good_df.copy()
        df.loc[0, "value"] = np.nan
        with pytest.raises(SchemaMismatch, match="non-nullable"):
            toy_schema.validate(df)

    def test_nan_allowed_in_nullable(self):
        schema = Schema(columns=(
            ColumnSpec("value", "float64", nullable=True),
        ))
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
        schema.validate(df)  # no raise

    def test_extra_columns_rejected_by_validate(self, toy_schema: Schema, good_df: pd.DataFrame):
        df = good_df.assign(extra=[0, 0])
        with pytest.raises(SchemaMismatch, match="column mismatch"):
            toy_schema.validate(df)

    def test_provider_and_identifier_in_error(self, toy_schema: Schema, good_df: pd.DataFrame):
        df = good_df.drop(columns=["count"])
        with pytest.raises(SchemaMismatch) as exc:
            toy_schema.validate(df, provider="testprov", identifier="TEST:X")
        assert exc.value.provider == "testprov"
        assert exc.value.identifier == "TEST:X"


class TestNormalize:
    def test_rename_cast_reorder(self, toy_schema: Schema):
        # Source has wrong column names, wrong order, wrong-but-coercible dtypes.
        src = pd.DataFrame({
            "Count": [10, 20],
            "Value": ["1.5", "2.5"],  # strings — cast to float64
            "Date": ["2026-01-01", "2026-01-02"],
            "extra_garbage": [None, None],
        })
        out = toy_schema.normalize(
            src,
            source_column_map={"Date": "date", "Value": "value", "Count": "count"},
        )
        assert list(out.columns) == ["date", "value", "count"]
        assert str(out["date"].dtype) == "datetime64[ns]"
        assert str(out["value"].dtype) == "float64"
        assert str(out["count"].dtype) == "int64"
        toy_schema.validate(out)  # round-trip

    def test_missing_canonical_column_raises(self, toy_schema: Schema):
        src = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "value": [1.0]})
        with pytest.raises(SchemaMismatch, match="missing columns"):
            toy_schema.normalize(src)

    def test_uncoercible_dtype_raises(self, toy_schema: Schema):
        src = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01"]),
            "value": ["not_a_number"],
            "count": [1],
        })
        with pytest.raises(SchemaMismatch, match="cannot cast"):
            toy_schema.normalize(src)

    def test_extra_source_columns_dropped(self, toy_schema: Schema, good_df: pd.DataFrame):
        src = good_df.assign(noise=[0.0, 0.0])
        out = toy_schema.normalize(src)
        assert list(out.columns) == ["date", "value", "count"]
