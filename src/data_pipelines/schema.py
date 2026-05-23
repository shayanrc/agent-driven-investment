"""Schema primitives for per-domain canonical schemas.

A Schema is a domain's contract for what `processed/<domain>/` parquet looks
like. Adapters return DataFrames; the framework calls Schema.normalize() to
rename/cast/reorder source-specific columns into canonical shape, then
Schema.validate() before any cache write. Schema mismatch is a SchemaMismatch
exception, never a silent coercion (D1).

Schemas are instantiated per-domain (us_equities OHLCV is one Schema; FRED's
(date, value) would be another). The framework knows nothing about OHLCV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipelines.errors import SchemaMismatch


@dataclass(frozen=True)
class ColumnSpec:
    """Declaration for one column of a canonical schema.

    dtype is a numpy / pandas dtype string ("float64", "int64",
    "datetime64[ns]", ...). nullable=False means validate() rejects any NaN
    in that column.
    """

    name: str
    dtype: str
    nullable: bool = False


@dataclass(frozen=True)
class Schema:
    """Ordered set of ColumnSpecs defining a domain's canonical DataFrame shape.

    Two operations:
      - normalize(df, source_column_map): rename source columns to canonical
        names, cast dtypes, reorder. Used at the adapter→cache boundary.
      - validate(df): raise SchemaMismatch if columns / dtypes / nullability
        don't match. Called after normalize and before every cache write.
    """

    columns: tuple[ColumnSpec, ...]

    def __post_init__(self):
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate column names in Schema: {names}")

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def normalize(
        self,
        df: pd.DataFrame,
        source_column_map: dict[str, str] | None = None,
        provider: str = "<unknown>",
        identifier: str = "<unknown>",
    ) -> pd.DataFrame:
        """Rename per `source_column_map` (source_name → canonical_name),
        cast each column to its declared dtype, reorder to canonical order.

        Extra columns in the source are dropped silently — adapters often
        return more columns than the canonical schema needs. Missing required
        columns raise SchemaMismatch.
        """
        out = df.rename(columns=source_column_map or {})

        missing = [c.name for c in self.columns if c.name not in out.columns]
        if missing:
            raise SchemaMismatch(
                provider, identifier,
                f"missing columns after rename: {missing} (have: {list(out.columns)})",
            )

        out = out[self.column_names].copy()

        for col in self.columns:
            try:
                out[col.name] = out[col.name].astype(col.dtype)
            except (ValueError, TypeError) as e:
                raise SchemaMismatch(
                    provider, identifier,
                    f"cannot cast {col.name!r} to {col.dtype}: {e}",
                ) from e

        return out.reset_index(drop=True)

    def validate(
        self,
        df: pd.DataFrame,
        provider: str = "<unknown>",
        identifier: str = "<unknown>",
    ) -> None:
        """Raise SchemaMismatch if df does not match this schema exactly.

        Checks: column names match in order; dtypes match declared; no NaN in
        non-nullable columns.
        """
        actual_cols = list(df.columns)
        if actual_cols != self.column_names:
            raise SchemaMismatch(
                provider, identifier,
                f"column mismatch — expected {self.column_names}, got {actual_cols}",
            )

        for col in self.columns:
            actual_dtype = str(df[col.name].dtype)
            if not _dtype_matches(actual_dtype, col.dtype):
                raise SchemaMismatch(
                    provider, identifier,
                    f"dtype mismatch on {col.name!r} — expected {col.dtype}, got {actual_dtype}",
                )

            if not col.nullable and df[col.name].isna().any():
                n_na = int(df[col.name].isna().sum())
                raise SchemaMismatch(
                    provider, identifier,
                    f"non-nullable column {col.name!r} has {n_na} NaN values",
                )


def _dtype_matches(actual: str, expected: str) -> bool:
    """Loose dtype equivalence: 'datetime64[ns]' matches the same; numeric
    families compare by canonical numpy name.
    """
    if actual == expected:
        return True
    try:
        return np.dtype(actual) == np.dtype(expected)
    except TypeError:
        return False
