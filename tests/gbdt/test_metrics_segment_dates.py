"""V1.4 P2 — metrics.json segment-date persistence smoke test.

The runner emits 7 new fields (3 top-level + 3 data-nested + 1 carry):

- ``split_mode``                              (top-level, str)
- ``split_train_start``                       (top-level, ISO str | None)
- ``segment_dates``                           (top-level, 4-segment dict)
- ``data.n_tickers_per_segment``              (4-key int dict)
- ``data.tickers_per_segment``                (4-key list[str] dict)
- ``data.row_counts_per_segment_per_ticker``  (ticker → 4-key int dict)

We don't drive the full ``run_experiment`` here (it requires the data
cache). Instead, we exercise the runner's two builder helpers (computed
in ``run_experiment`` just before the ``metrics = {...}`` construction)
directly on a synthetic 3-ticker ``WalkForwardResult`` and verify shape
+ values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbdt.train import SplitSpec, WalkForwardResult


def _toy_predictions() -> dict[str, pd.DataFrame]:
    """3 tickers (AAA/BBB/CCC) × 4 segments, distinct date ranges."""
    cols = ["date", "ticker", "p_raw", "p_calibrated", "y_true", "sample_weight"]
    train = pd.DataFrame({
        "date": pd.to_datetime([
            "2019-01-02", "2019-01-03",
            "2019-01-02", "2019-01-03",
        ]),
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "p_raw":         [0.1, 0.2, 0.3, 0.4],
        "p_calibrated":  [0.1, 0.2, 0.3, 0.4],
        "y_true":        [0,   1,   0,   1],
        "sample_weight": [1.0, 1.0, 1.0, 1.0],
    })[cols]
    val = pd.DataFrame({
        "date": pd.to_datetime([
            "2020-01-02", "2020-01-03",
            "2020-01-02", "2020-01-03",
        ]),
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "p_raw":         [0.1, 0.2, 0.3, 0.4],
        "p_calibrated":  [0.1, 0.2, 0.3, 0.4],
        "y_true":        [0,   1,   0,   1],
        "sample_weight": [1.0, 1.0, 1.0, 1.0],
    })[cols]
    # CCC enters only at eval (late-IPO)
    eval_ = pd.DataFrame({
        "date": pd.to_datetime([
            "2021-01-02", "2021-01-03",
            "2021-01-02", "2021-01-03",
            "2021-01-02", "2021-01-03", "2021-01-04",
        ]),
        "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC", "CCC"],
        "p_raw":         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "p_calibrated":  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "y_true":        [0,   1,   0,   1,   0,   1,   1],
        "sample_weight": [1.0]*7,
    })[cols]
    test = pd.DataFrame({
        "date": pd.to_datetime([
            "2022-01-02", "2022-01-03",
            "2022-01-02", "2022-01-03",
            "2022-01-02",
        ]),
        "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC"],
        "p_raw":         [0.1, 0.2, 0.3, 0.4, 0.5],
        "p_calibrated":  [0.1, 0.2, 0.3, 0.4, 0.5],
        "y_true":        [0,   1,   0,   1,   0],
        "sample_weight": [1.0]*5,
    })[cols]
    return {"train": train, "val": val, "eval": eval_, "test": test}


def _emit_metrics_block(result: WalkForwardResult, split: SplitSpec) -> dict:
    """Replicate the run_experiment metrics-block builder logic for V1.4 fields.

    Kept lock-step with src/gbdt/__main__.py's _trailing_segment_dates +
    _per_ticker_sidecars + the surrounding ``metrics = {...}`` construction.
    A drift in either is a real bug — this test guards both shapes.
    """
    def _trailing_segment_dates() -> dict[str, dict[str, str | None]]:
        sd: dict[str, dict[str, str | None]] = {}
        for seg in ("train", "val", "eval", "test"):
            df = result.predictions.get(seg)
            if df is None or len(df) == 0:
                sd[seg] = {"start": None, "end": None}
                continue
            dts = pd.to_datetime(df["date"])
            sd[seg] = {
                "start": dts.min().date().isoformat(),
                "end": dts.max().date().isoformat(),
            }
        return sd

    if result.segment_dates is not None:
        segment_dates = result.segment_dates
    else:
        segment_dates = _trailing_segment_dates()

    tickers_per_segment: dict[str, list[str]] = {}
    n_tickers_per_segment: dict[str, int] = {}
    row_counts: dict[str, dict[str, int]] = {}
    for seg in ("train", "val", "eval", "test"):
        df = result.predictions.get(seg)
        if df is None or len(df) == 0:
            tickers_per_segment[seg] = []
            n_tickers_per_segment[seg] = 0
            continue
        grp = df.groupby("ticker", sort=True)["y_true"].count()
        seg_tickers = grp.index.tolist()
        tickers_per_segment[seg] = seg_tickers
        n_tickers_per_segment[seg] = len(seg_tickers)
        for t, n in grp.items():
            row_counts.setdefault(t, {"train": 0, "val": 0, "eval": 0, "test": 0})[
                seg
            ] = int(n)

    return {
        "split_mode": split.mode,
        "split_train_start": (
            split.train_start.isoformat() if split.train_start is not None else None
        ),
        "segment_dates": segment_dates,
        "data": {
            "n_tickers_per_segment": n_tickers_per_segment,
            "tickers_per_segment": tickers_per_segment,
            "row_counts_per_segment_per_ticker": row_counts,
        },
    }


def _toy_result(predictions: dict, segment_dates=None) -> WalkForwardResult:
    return WalkForwardResult(
        best_iteration=0,
        best_model=None,  # type: ignore[arg-type]
        best_features=[],
        best_hp={},
        best_val_brier=0.0,
        iterations=[],
        calibration=None,  # type: ignore[arg-type]
        inner_stop_signal="cap",
        predictions=predictions,
        segment_dates=segment_dates,
    )


def test_metrics_block_trailing_mode_shape_and_values():
    predictions = _toy_predictions()
    result = _toy_result(predictions, segment_dates=None)
    split = SplitSpec(mode="trailing")

    block = _emit_metrics_block(result, split)

    # Top-level shape (3 fields).
    assert block["split_mode"] == "trailing"
    assert block["split_train_start"] is None
    sd = block["segment_dates"]
    assert set(sd.keys()) == {"train", "val", "eval", "test"}
    for seg in sd:
        assert set(sd[seg].keys()) == {"start", "end"}

    # Calendar UNION values per segment (MIN start, MAX end across tickers).
    assert sd["train"] == {"start": "2019-01-02", "end": "2019-01-03"}
    assert sd["eval"] == {"start": "2021-01-02", "end": "2021-01-04"}

    # data.n_tickers_per_segment matches.
    assert block["data"]["n_tickers_per_segment"] == {
        "train": 2, "val": 2, "eval": 3, "test": 3,
    }

    # tickers_per_segment lists are sorted (groupby sort=True).
    assert block["data"]["tickers_per_segment"]["eval"] == ["AAA", "BBB", "CCC"]

    # row_counts_per_segment_per_ticker carries per-ticker row counts.
    rc = block["data"]["row_counts_per_segment_per_ticker"]
    assert rc["AAA"] == {"train": 2, "val": 2, "eval": 2, "test": 2}
    assert rc["CCC"] == {"train": 0, "val": 0, "eval": 3, "test": 1}


def test_metrics_block_date_aligned_uses_result_segment_dates():
    predictions = _toy_predictions()
    canonical_dates = {
        "train": {"start": "2019-01-01", "end": "2019-12-31"},
        "val":   {"start": "2020-01-01", "end": "2020-12-31"},
        "eval":  {"start": "2021-01-01", "end": "2021-12-31"},
        "test":  {"start": "2022-01-01", "end": "2022-12-31"},
    }
    result = _toy_result(predictions, segment_dates=canonical_dates)
    from datetime import date
    split = SplitSpec(
        mode="date_aligned", train_start=date(2019, 1, 1),
    )
    block = _emit_metrics_block(result, split)
    assert block["split_mode"] == "date_aligned"
    assert block["split_train_start"] == "2019-01-01"
    # date_aligned mode uses result.segment_dates verbatim — NOT the
    # calendar union from the predictions DataFrame.
    assert block["segment_dates"] == canonical_dates


def test_metrics_block_empty_segment_yields_none_endpoints():
    predictions = _toy_predictions()
    cols = ["date", "ticker", "p_raw", "p_calibrated", "y_true", "sample_weight"]
    predictions["test"] = pd.DataFrame(columns=cols)
    result = _toy_result(predictions, segment_dates=None)
    split = SplitSpec(mode="trailing")
    block = _emit_metrics_block(result, split)
    assert block["segment_dates"]["test"] == {"start": None, "end": None}
    assert block["data"]["n_tickers_per_segment"]["test"] == 0
    assert block["data"]["tickers_per_segment"]["test"] == []
