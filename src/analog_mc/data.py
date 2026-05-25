"""Data loading and walk-forward fold generation for the analog_mc pipeline.

v1 reads a single CSV with configurable column names (FRED-style or
yfinance-style both work). Returns a pandas Series of log returns indexed by
trading date, sorted ascending, with NaN/duplicate rows dropped.

Walk-forward folds enforce constraint **C6**: expanding Train starts at the
earliest data and grows; fixed-size Val/Test windows march strictly forward,
and no date appears in more than one Test block across folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analog_mc.config import Config


def load_close_series(
    path: str | Path,
    date_col: str,
    close_col: str,
) -> pd.Series:
    """Load a price series from CSV.

    Returns a Series of close prices indexed by parsed dates, sorted ascending,
    with duplicate dates and NaNs removed.
    """
    df = pd.read_csv(path, usecols=[date_col, close_col])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col, close_col])
    df = df.drop_duplicates(subset=[date_col], keep="last")
    df = df.sort_values(date_col).reset_index(drop=True)
    series = pd.Series(df[close_col].to_numpy(), index=pd.DatetimeIndex(df[date_col]), name="close")
    return series


def close_series_from_dataframe(
    df: pd.DataFrame,
    date_col: str = "date",
    close_col: str = "adj_close",
) -> pd.Series:
    """Build a close-series from an in-memory DataFrame.

    Mirrors ``load_close_series`` but takes a pre-loaded DataFrame, which is
    how the ``forecasters`` framework hands data to the analog_mc backend (it
    fetches via ``data_pipelines.fetch()`` and passes the result through, per
    docs/forecasters/V1_PLAN.md §"Wire-format contract").

    Defaults track the canonical data_pipelines schema (``date``,
    ``adj_close``); pass explicit names for FRED-style CSVs.

    The CSV-first contract (``load_close_series``) is the project default per
    ``[[project-data-source]]``; this is the additive DataFrame path.
    """
    if date_col not in df.columns or close_col not in df.columns:
        raise ValueError(
            f"DataFrame missing required columns; need date_col={date_col!r} and "
            f"close_col={close_col!r}, have {list(df.columns)}"
        )
    sub = df[[date_col, close_col]].copy()
    sub[date_col] = pd.to_datetime(sub[date_col])
    sub = sub.dropna(subset=[date_col, close_col])
    sub = sub.drop_duplicates(subset=[date_col], keep="last")
    sub = sub.sort_values(date_col).reset_index(drop=True)
    series = pd.Series(
        sub[close_col].to_numpy(),
        index=pd.DatetimeIndex(sub[date_col]),
        name="close",
    )
    return series


def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns: log(close[t] / close[t-1]).

    The first row is dropped (no prior close to difference against).
    """
    if (close <= 0).any():
        raise ValueError("close series contains non-positive values; cannot take log")
    r = np.log(close / close.shift(1))
    r = r.dropna()
    r.name = "log_return"
    return r


def load_returns(config: Config) -> pd.Series:
    """Convenience wrapper: load CSV per config and return log returns."""
    close = load_close_series(config.data_path, config.date_col, config.close_col)
    return log_returns(close)


# ---------------------------------------------------------------------------
# Walk-forward fold generation (Stage 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold.

    Indices are positional (integer) indices into the returns Series, not date
    labels. Use ``returns.iloc[fold.train_idx]`` etc. to materialize.

    Layout per fold (with T = train_initial_size + k * (val_size + test_size)):
      * train_idx : [0, T)              -- expanding from earliest data
      * val_idx   : [T, T + V)          -- tuning window
      * test_idx  : [T + V, T + V + Ts) -- held-out evaluation window
    """

    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_val(self) -> int:
        return int(self.val_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)


def generate_folds(returns: pd.Series, config: Config) -> list[Fold]:
    """Generate walk-forward folds per **C6**.

    Each fold expands the train window to include all data strictly before the
    current val block. Val and Test march forward by (val_size + test_size)
    per fold so the next fold's train absorbs the previous val + test, and no
    Test block ever overlaps another.

    Stops when the next fold's test block would run past the end of the data.
    """
    n = len(returns)
    val_size = config.val_size
    test_size = config.test_size
    block = val_size + test_size

    folds: list[Fold] = []
    cursor = config.train_initial_size
    fold_idx = 0
    while cursor + block <= n:
        train_idx = np.arange(0, cursor, dtype=np.int64)
        val_idx = np.arange(cursor, cursor + val_size, dtype=np.int64)
        test_idx = np.arange(cursor + val_size, cursor + block, dtype=np.int64)
        folds.append(Fold(index=fold_idx, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
        cursor += block
        fold_idx += 1

    if not folds:
        raise ValueError(
            f"No folds could be generated: series length {n} is too short for "
            f"train_initial_size={config.train_initial_size} + val_size={val_size} "
            f"+ test_size={test_size}."
        )
    return folds
