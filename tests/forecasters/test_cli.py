"""Smoke tests for the scripts/forecasters/run.py CLI.

We invoke the CLI in-process via main(argv) for speed; subprocess invocation
is also valid but adds dependency on shell environment which the in-process
path side-steps.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def main():
    # Lazy import — the CLI module pulls matplotlib at import time.
    from scripts.forecasters.run import main as _main
    return _main


@pytest.fixture
def overrides_yaml(tmp_path: Path) -> Path:
    """Slim overrides so the CLI runs in seconds."""
    p = tmp_path / "overrides.yaml"
    p.write_text(textwrap.dedent(
        """
        n_paths: 100
        weights: [0.33, 0.34, 0.33]
        n_eff: 50
        """
    ).strip())
    return p


def test_forecast_subcommand_smoke(main, tmp_path: Path, overrides_yaml: Path, capsys) -> None:
    rc = main([
        "forecast",
        "--preset", "v24-default",
        "--data-path", "data/NASDAQ100.csv",
        "--start", "2018-01-01",
        "--end", "2024-12-31",
        "--origin", "2020-01-02",
        "--horizon", "60",
        "--seed", "42",
        "--config-overrides", str(overrides_yaml),
        "--cache-path", str(tmp_path / "fcache"),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    out_dir = Path(captured.out.strip())
    assert out_dir.is_dir()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "paths.npz").is_file()
    assert (out_dir / "warnings.json").is_file()
    assert (out_dir / "fan_chart.png").is_file()
    # paths array shape
    paths = np.load(out_dir / "paths.npz")["paths"]
    assert paths.shape == (100, 60)


def test_forecast_cache_hit_on_repeat(main, tmp_path: Path, overrides_yaml: Path, capsys) -> None:
    common = [
        "forecast",
        "--preset", "v24-default",
        "--data-path", "data/NASDAQ100.csv",
        "--start", "2018-01-01",
        "--end", "2024-12-31",
        "--origin", "2020-01-02",
        "--horizon", "60",
        "--seed", "42",
        "--config-overrides", str(overrides_yaml),
        "--cache-path", str(tmp_path / "fcache"),
    ]
    main(common)
    out1 = capsys.readouterr().out.strip()
    main(common)
    out2 = capsys.readouterr().out.strip()
    assert out1 == out2  # same cache key → same output dir
    # Confirm the contents weren't recomputed (mtime unchanged on summary.json).
    p = Path(out1) / "summary.json"
    mtime_first = p.stat().st_mtime
    main(common)
    mtime_third = p.stat().st_mtime
    assert mtime_first == mtime_third


def test_forecast_data_correction_misses_cache(main, tmp_path: Path, overrides_yaml: Path, capsys) -> None:
    """V2_TBD #18 regression, end-to-end through the CLI: populate the forecast
    cache, then revise ONE bar's VALUE in the input CSV (same dates, same
    --start/--end/--origin — the pre-fix key is identical) and re-run. The
    data_hash key component must force a cache MISS (a different output dir,
    i.e. a recompute), never serve the stale cached forecast."""
    import pandas as pd

    csv = tmp_path / "data.csv"
    csv.write_text(Path("data/NASDAQ100.csv").read_text())
    common = [
        "forecast",
        "--preset", "v24-default",
        "--data-path", str(csv),
        "--start", "2018-01-01",
        "--end", "2024-12-31",
        "--origin", "2020-01-02",
        "--horizon", "60",
        "--seed", "42",
        "--config-overrides", str(overrides_yaml),
        "--cache-path", str(tmp_path / "fcache"),
    ]
    main(common)
    out1 = capsys.readouterr().out.strip()
    main(common)
    assert capsys.readouterr().out.strip() == out1  # sanity: unchanged data hits

    # Values-only correction: one bar inside the sliced range, dates untouched.
    df = pd.read_csv(csv)
    row = df.index[df["observation_date"] == "2019-06-03"][0]
    df.loc[row, "NASDAQ100"] = float(df.loc[row, "NASDAQ100"]) * 1.01
    df.to_csv(csv, index=False)

    main(common)
    out3 = capsys.readouterr().out.strip()
    assert out3 != out1, (
        "corrected input data served the stale cached forecast — the cache key "
        "must include a content hash of the fetched-and-sliced data"
    )
    assert (Path(out3) / "summary.json").is_file()  # recomputed + cached anew


def test_forecast_no_cache_bypasses(main, tmp_path: Path, overrides_yaml: Path, capsys) -> None:
    args = [
        "forecast",
        "--preset", "v24-default",
        "--data-path", "data/NASDAQ100.csv",
        "--start", "2018-01-01",
        "--end", "2024-12-31",
        "--origin", "2020-01-02",
        "--horizon", "60",
        "--seed", "42",
        "--config-overrides", str(overrides_yaml),
        "--cache-path", str(tmp_path / "fcache"),
        "--no-cache",
    ]
    main(args)
    out_dir = Path(capsys.readouterr().out.strip())
    # --no-cache routes the output to a tempdir, not the cache root.
    assert not out_dir.is_relative_to(tmp_path / "fcache")
    assert (out_dir / "summary.json").is_file()


def test_forecast_output_dir_explicit(main, tmp_path: Path, overrides_yaml: Path, capsys) -> None:
    out_root = tmp_path / "explicit"
    args = [
        "forecast",
        "--preset", "v24-default",
        "--data-path", "data/NASDAQ100.csv",
        "--start", "2018-01-01",
        "--end", "2024-12-31",
        "--origin", "2020-01-02",
        "--horizon", "60",
        "--seed", "42",
        "--config-overrides", str(overrides_yaml),
        "--output-dir", str(out_root),
    ]
    main(args)
    out_dir = Path(capsys.readouterr().out.strip())
    assert out_dir.is_relative_to(out_root)
    assert (out_dir / "summary.json").is_file()


def test_forecast_unknown_preset_exits_nonzero(main, tmp_path: Path, capsys) -> None:
    rc = main([
        "forecast",
        "--preset", "does-not-exist",
        "--data-path", "data/NASDAQ100.csv",
        "--start", "2018-01-01",
        "--end", "2024-12-31",
        "--origin", "2020-01-02",
        "--horizon", "60",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "UnknownPresetError" in err


def test_list_presets_table(main, capsys) -> None:
    rc = main(["list-presets"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "v24-default" in out
    assert "canonical" in out
    assert "analog_mc" in out


def test_list_presets_json(main, capsys) -> None:
    rc = main(["list-presets", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    names = {r["name"] for r in parsed}
    assert "v24-default" in names


def test_list_presets_filter_backend_unknown_returns_empty(main, capsys) -> None:
    rc = main(["list-presets", "--backend", "imaginary"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no presets found)" in out
