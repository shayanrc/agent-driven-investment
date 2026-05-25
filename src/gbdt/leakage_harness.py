"""Synthetic-data harness for detecting causal-feature violations.

The harness builds a tiny OHLCV panel, plants a known "leak signal" at a
chosen row ``t_leak``, and verifies that *no* feature value at ``t < t_leak``
is altered when the leak is injected. Any feature whose value at ``t`` shifts
because of data at ``t' > t`` is non-causal — exactly the silent failure
mode CLAUDE.md C1 forbids.

Usage:

    from gbdt.leakage_harness import LeakageHarness
    h = LeakageHarness()
    h.assert_causal(my_feature_fn)         # raises if non-causal

Or, for ad-hoc one-row spike tests:

    from gbdt.leakage_harness import synthetic_leak_test
    synthetic_leak_test(my_feature_fn)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import inf  # noqa: F401  (kept for explicit import in tests)


# Default panel shape used by every harness check. Two tickers so the harness
# also exercises cross-sectional feature pipelines that compute on the
# date-grouped cross-section. Length is small so unit tests are fast.
_DEFAULT_N_ROWS = 120
_DEFAULT_N_TICKERS = 2
_DEFAULT_LEAK_ROW = 80                   # last 40 rows are post-leak
_LEAK_MULTIPLIER = 5.0                   # 5x bump = "spectacular" leak


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------


def make_synthetic_panel(
    n_rows: int = _DEFAULT_N_ROWS,
    n_tickers: int = _DEFAULT_N_TICKERS,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a baseline (no-leak) panel.

    Long-format MultiIndex ``(date, ticker)`` with OHLCV columns. Prices are
    a deterministic random walk seeded by ``seed`` so multiple harness calls
    line up exactly.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    frames = []
    for i in range(n_tickers):
        # Per-ticker walk, deterministic offset so prices differ across tickers.
        rets = rng.normal(loc=0.0, scale=0.01, size=n_rows)
        close = 100.0 * np.exp(np.cumsum(rets))
        high = close * (1.0 + rng.uniform(0.0, 0.005, size=n_rows))
        low = close * (1.0 - rng.uniform(0.0, 0.005, size=n_rows))
        open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n_rows))
        volume = rng.integers(1_000_000, 5_000_000, size=n_rows).astype(np.int64)
        df = pd.DataFrame({
            "date": dates,
            "ticker": f"TKR{i}",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        })
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel.set_index(["date", "ticker"]).sort_index()


def plant_leak(
    panel: pd.DataFrame,
    *,
    leak_row: int = _DEFAULT_LEAK_ROW,
    multiplier: float = _LEAK_MULTIPLIER,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``panel`` with a price spike at one row.

    The leak is large (default 5x) so any non-causal feature would show
    a clearly different value vs. the baseline at every pre-leak row that
    references the leaked row.
    """
    out = panel.copy()
    dates = out.index.get_level_values("date").unique()
    leak_date = dates[leak_row]
    if ticker is None:
        tickers = out.index.get_level_values("ticker").unique()
        target_ticker = tickers[0]
    else:
        target_ticker = ticker
    for col in ("open", "high", "low", "close", "adj_close"):
        if (leak_date, target_ticker) in out.index:
            out.loc[(leak_date, target_ticker), col] *= multiplier
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class LeakageReport:
    """What :func:`synthetic_leak_test` returns."""

    causal: bool
    leak_row: int
    n_pre_leak: int
    max_abs_diff_pre_leak: float
    columns_with_diff: list[str]


class LeakageHarness:
    """Reusable harness that exercises a feature function on the same fixture
    twice (no-leak vs leak-planted) and verifies pre-leak values are stable.
    """

    def __init__(
        self,
        n_rows: int = _DEFAULT_N_ROWS,
        n_tickers: int = _DEFAULT_N_TICKERS,
        leak_row: int = _DEFAULT_LEAK_ROW,
        *,
        seed: int = 0,
        atol: float = 1e-9,
    ) -> None:
        self.n_rows = n_rows
        self.n_tickers = n_tickers
        self.leak_row = leak_row
        self.seed = seed
        self.atol = atol

    def check(self, feature_fn) -> LeakageReport:
        """Run ``feature_fn(panel) -> DataFrame`` on no-leak + leak panels.

        ``feature_fn`` must return a DataFrame indexed on the same
        ``(date, ticker)`` MultiIndex as the input panel. Pre-leak rows are
        compared cell-by-cell; any difference > ``atol`` flags the feature
        as non-causal.
        """
        base_panel = make_synthetic_panel(self.n_rows, self.n_tickers, seed=self.seed)
        leaky_panel = plant_leak(base_panel, leak_row=self.leak_row)

        base_feat = feature_fn(base_panel)
        leak_feat = feature_fn(leaky_panel)

        if not isinstance(base_feat, pd.DataFrame):
            base_feat = base_feat.to_frame()
        if not isinstance(leak_feat, pd.DataFrame):
            leak_feat = leak_feat.to_frame()

        # Align on the index of the no-leak features (a feature pipeline may
        # legitimately drop the leading rows where lookbacks aren't filled).
        common_idx = base_feat.index.intersection(leak_feat.index)
        base_feat = base_feat.loc[common_idx]
        leak_feat = leak_feat.loc[common_idx]

        dates = common_idx.get_level_values("date").unique()
        leak_date = pd.date_range("2020-01-01", periods=self.n_rows, freq="B")[self.leak_row]
        pre_leak_mask = common_idx.get_level_values("date") < leak_date

        base_pre = base_feat[pre_leak_mask]
        leak_pre = leak_feat[pre_leak_mask]

        # NaN-equal-NaN: rolling features have NaN heads which must compare
        # equal. Treat (NaN, NaN) as zero diff and (NaN, value) or (value, NaN)
        # as a real difference.
        both_nan = base_pre.isna() & leak_pre.isna()
        either_nan = base_pre.isna() ^ leak_pre.isna()
        raw_diff = (leak_pre.fillna(0.0) - base_pre.fillna(0.0)).abs()
        diff = raw_diff.mask(both_nan, 0.0).mask(either_nan, np.inf)
        max_diff = float(diff.values.max()) if diff.size else 0.0
        cols_changed = [
            c for c in diff.columns if diff[c].max(skipna=True) > self.atol
        ]
        causal = max_diff <= self.atol
        return LeakageReport(
            causal=causal,
            leak_row=self.leak_row,
            n_pre_leak=int(pre_leak_mask.sum()),
            max_abs_diff_pre_leak=max_diff,
            columns_with_diff=cols_changed,
        )

    def assert_causal(self, feature_fn) -> None:
        """Raise ``AssertionError`` if ``feature_fn`` is non-causal."""
        report = self.check(feature_fn)
        if not report.causal:
            raise AssertionError(
                f"non-causal feature: max abs diff on pre-leak rows = "
                f"{report.max_abs_diff_pre_leak:.6g}; "
                f"columns affected = {report.columns_with_diff}"
            )


def synthetic_leak_test(feature_fn) -> LeakageReport:
    """One-shot convenience: build the default harness and check ``feature_fn``."""
    return LeakageHarness().check(feature_fn)
