"""FredAdapter.parse — CSV & JSON transports, "." → NaN, daily densify,
reprocess determinism, schema invariance across both transports."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data_pipelines.domains.fred_macro.adapters.fred import FredAdapter
from data_pipelines.domains.fred_macro.schema import FRED_SCHEMA

FREQ = {"DGS10": "daily", "CPIAUCSL": "monthly", "GDPC1": "quarterly"}


def _adapter():
    return FredAdapter(frequency_map=FREQ)


def _write(tmp_path, series_id, name, text):
    """Lay the raw file out where parse() expects it:
    <root>/<SERIES_ID>/<file> — parse recovers the series id from parent.name.
    """
    d = tmp_path / series_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(text)
    return p


def _isodates(df):
    return list(df["date"].dt.strftime("%Y-%m-%d"))


def test_parse_csv_daily_densifies_missing_weekday(tmp_path):
    # FRED omits the Mon 2020-01-06 row entirely; densify must still produce it
    # (NaN) so the weekday grid converges.
    p = _write(
        tmp_path, "DGS10", "a.csv",
        "observation_date,DGS10\n2020-01-02,1.88\n2020-01-03,1.79\n2020-01-07,1.83\n",
    )
    df = _adapter().parse(p)
    assert _isodates(df) == ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    v = df.loc[df["date"] == pd.Timestamp("2020-01-06"), "value"].iloc[0]
    assert np.isnan(v)
    FRED_SCHEMA.validate(df)


def test_parse_csv_dot_becomes_nan(tmp_path):
    p = _write(
        tmp_path, "DGS10", "b.csv",
        "observation_date,DGS10\n2020-01-02,.\n2020-01-03,1.79\n",
    )
    df = _adapter().parse(p)
    v = df.loc[df["date"] == pd.Timestamp("2020-01-02"), "value"].iloc[0]
    assert np.isnan(v)


def test_parse_json_monthly_not_densified(tmp_path):
    obs = {"observations": [
        {"date": "2020-01-01", "value": "257.9"},
        {"date": "2020-03-01", "value": "258.1"},  # Feb intentionally absent
    ]}
    p = _write(tmp_path, "CPIAUCSL", "c.json", json.dumps(obs))
    df = _adapter().parse(p)
    # monthly series are NOT densified — no phantom 2020-02-01 weekday fill.
    assert _isodates(df) == ["2020-01-01", "2020-03-01"]
    FRED_SCHEMA.validate(df)


def test_csv_and_json_agree(tmp_path):
    # Same daily data via both transports → identical canonical frame (D1).
    pc = _write(
        tmp_path, "DGS10", "d.csv",
        "observation_date,DGS10\n2020-01-02,1.88\n2020-01-03,1.79\n",
    )
    pj = _write(tmp_path, "DGS10", "e.json", json.dumps({"observations": [
        {"date": "2020-01-02", "value": "1.88"},
        {"date": "2020-01-03", "value": "1.79"},
    ]}))
    pd.testing.assert_frame_equal(_adapter().parse(pc), _adapter().parse(pj))


def test_reprocess_determinism(tmp_path):
    # Same raw bytes → same DataFrame, every time (D8).
    p = _write(
        tmp_path, "DGS10", "f.csv",
        "observation_date,DGS10\n2020-01-02,1.88\n2020-01-03,1.79\n",
    )
    a = _adapter()
    pd.testing.assert_frame_equal(a.parse(p), a.parse(p))


def test_unknown_series_not_densified(tmp_path):
    # Out-of-frequency-map series: best-effort, no densify (left as returned).
    p = _write(
        tmp_path, "MYSTERY", "g.csv",
        "observation_date,MYSTERY\n2020-01-02,1.0\n2020-01-07,2.0\n",
    )
    df = _adapter().parse(p)
    assert _isodates(df) == ["2020-01-02", "2020-01-07"]  # no weekday fill


def test_source_column_map_is_none():
    assert FredAdapter.source_column_map is None
