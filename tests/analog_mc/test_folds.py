"""Tests for analog_mc.data.generate_folds (constraint C6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analog_mc.config import Config
from analog_mc.data import Fold, generate_folds


def _make_returns(n: int) -> pd.Series:
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    return pd.Series(np.zeros(n), index=idx, name="log_return")


@pytest.fixture
def cfg_small() -> Config:
    # Small enough that we get several folds without hitting other invariants.
    return Config(
        train_initial_size=500,
        val_size=60,
        test_size=60,
        zscore_horizons=(20, 50, 200),
    )


def test_generate_folds_layout(cfg_small: Config) -> None:
    returns = _make_returns(1000)
    folds = generate_folds(returns, cfg_small)
    # Expect floor((1000 - 500) / (60 + 60)) = 4 folds.
    assert len(folds) == 4

    # Fold 0 layout.
    f0 = folds[0]
    assert f0.index == 0
    assert f0.train_idx[0] == 0 and f0.train_idx[-1] == 499
    assert f0.val_idx[0] == 500 and f0.val_idx[-1] == 559
    assert f0.test_idx[0] == 560 and f0.test_idx[-1] == 619


def test_train_is_expanding(cfg_small: Config) -> None:
    returns = _make_returns(1000)
    folds = generate_folds(returns, cfg_small)
    for prev, curr in zip(folds, folds[1:]):
        assert curr.train_idx[0] == 0  # always starts from earliest data
        assert curr.n_train > prev.n_train
        # Train absorbs everything strictly before the new val block.
        assert curr.train_idx[-1] + 1 == curr.val_idx[0]


def test_test_blocks_never_overlap(cfg_small: Config) -> None:
    """C6: no date appears in more than one Test block across folds."""
    returns = _make_returns(1000)
    folds = generate_folds(returns, cfg_small)
    seen: set[int] = set()
    for f in folds:
        as_set = set(int(i) for i in f.test_idx)
        assert as_set.isdisjoint(seen), f"Test overlap detected in fold {f.index}"
        seen |= as_set


def test_val_and_test_are_contiguous_within_fold(cfg_small: Config) -> None:
    returns = _make_returns(1000)
    for f in generate_folds(returns, cfg_small):
        assert f.val_idx[-1] + 1 == f.test_idx[0]
        assert f.n_val == 60 and f.n_test == 60


def test_val_block_never_overlaps_train(cfg_small: Config) -> None:
    returns = _make_returns(1000)
    for f in generate_folds(returns, cfg_small):
        assert f.train_idx[-1] < f.val_idx[0]
        assert set(int(i) for i in f.train_idx).isdisjoint(set(int(i) for i in f.val_idx))


def test_stops_when_test_block_would_run_past_end(cfg_small: Config) -> None:
    # 500 + 4 * 120 = 980, so 980+120 = 1100 fits exactly; 1099 doesn't.
    folds_exact = generate_folds(_make_returns(980 + 120), cfg_small)
    folds_short = generate_folds(_make_returns(980 + 120 - 1), cfg_small)
    assert len(folds_exact) == len(folds_short) + 1
    # Last fold's test must end strictly within the series.
    last = folds_exact[-1]
    assert last.test_idx[-1] == 980 + 120 - 1


def test_raises_when_series_too_short(cfg_small: Config) -> None:
    with pytest.raises(ValueError, match="too short"):
        generate_folds(_make_returns(cfg_small.train_initial_size + 50), cfg_small)


def test_fold_is_frozen() -> None:
    f = Fold(index=0, train_idx=np.array([0]), val_idx=np.array([1]), test_idx=np.array([2]))
    with pytest.raises(Exception):
        f.index = 1  # type: ignore[misc]


def test_generate_folds_with_real_data() -> None:
    """Sanity test against NASDAQ100.csv: should produce many folds."""
    from analog_mc.data import load_returns

    cfg = Config()
    try:
        r = load_returns(cfg)
    except FileNotFoundError:
        pytest.skip("data/NASDAQ100.csv not present")
    folds = generate_folds(r, cfg)
    # ~10500 returns, train_initial=1000, block=120 -> ~79 folds.
    assert len(folds) > 50
    # All test blocks disjoint, sorted forward.
    last_end = -1
    for f in folds:
        assert f.test_idx[0] > last_end
        last_end = int(f.test_idx[-1])
