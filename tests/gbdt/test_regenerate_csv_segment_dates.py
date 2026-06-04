"""V1.4 P4 — regenerate_r_precision_at_k_csv writes 8 segment-date cols.

Idempotence vs P3's backfill: running regenerate on an artifact dir
whose ``metrics.json`` carries ``segment_dates`` should populate the 8
columns identically to backfill's read-from-metrics path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.gbdt.regenerate_r_precision_at_k_csv import (
    _DATE_COLS,
    _segment_dates_for_artifact,
)


def _make_artifact_dir(
    tmp_path: Path,
    *,
    with_metrics_segment_dates: bool,
    with_predictions: bool,
) -> Path:
    art = tmp_path / "results" / "gbdt" / "experiments" / "fake_cell"
    art.mkdir(parents=True, exist_ok=True)
    if with_metrics_segment_dates:
        (art / "metrics.json").write_text(json.dumps({
            "experiment_name": "fake_cell",
            "segment_dates": {
                "train": {"start": "2019-01-02", "end": "2022-03-15"},
                "val":   {"start": "2022-03-16", "end": "2023-10-31"},
                "eval":  {"start": "2023-11-01", "end": "2024-08-22"},
                "test":  {"start": "2024-08-23", "end": "2025-01-14"},
            },
        }))
    if with_predictions:
        pred_dir = art / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for seg, dates in (
            ("train", ["2019-02-01", "2022-03-10"]),
            ("val",   ["2022-04-01", "2023-10-01"]),
            ("eval",  ["2023-12-01", "2024-08-01"]),
            ("test",  ["2024-09-01", "2025-01-10"]),
        ):
            pd.DataFrame({
                "date": dates,
                "ticker": ["AAA", "BBB"],
                "p_raw": [0.1, 0.2],
                "p_calibrated": [0.1, 0.2],
                "y_true": [0, 1],
                "sample_weight": [1.0, 1.0],
            }).to_csv(pred_dir / f"{seg}.csv", index=False)
    return art


def test_regenerate_uses_metrics_segment_dates_when_present(tmp_path):
    art = _make_artifact_dir(
        tmp_path, with_metrics_segment_dates=True, with_predictions=True,
    )
    sd = _segment_dates_for_artifact(art)
    # metrics.json wins over predictions when both are present.
    assert sd["train"]["start"] == "2019-01-02"
    assert sd["train"]["end"]   == "2022-03-15"
    assert sd["test"]["end"]    == "2025-01-14"


def test_regenerate_falls_back_to_calendar_union(tmp_path):
    art = _make_artifact_dir(
        tmp_path, with_metrics_segment_dates=False, with_predictions=True,
    )
    sd = _segment_dates_for_artifact(art)
    # No metrics.json::segment_dates → calendar UNION across predictions.
    assert sd["train"] == {"start": "2019-02-01", "end": "2022-03-10"}
    assert sd["test"]  == {"start": "2024-09-01", "end": "2025-01-10"}


def test_regenerate_empty_artifact_yields_none(tmp_path):
    art = _make_artifact_dir(
        tmp_path, with_metrics_segment_dates=False, with_predictions=False,
    )
    sd = _segment_dates_for_artifact(art)
    for seg in ("train", "val", "eval", "test"):
        assert sd[seg] == {"start": None, "end": None}


def test_eight_date_columns_constant():
    """Schema lock — the 8 column names + order must not drift."""
    assert _DATE_COLS == (
        "train_start", "train_end",
        "val_start",   "val_end",
        "eval_start",  "eval_end",
        "test_start",  "test_end",
    )
