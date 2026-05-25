"""Smoke test for scripts/forecasters/run_acceptance_demo.py.

The full live demo runs the analog_mc tune on multi-year NIFTY 500 history
(~hours of compute) and is a manual / pre-merge step. This CI smoke test
exercises the report-writing path (phase verify) against pre-computed
state.json so the orchestrator's verify branch is exercised end-to-end
without paying the tune cost.

The phase-fetch / phase-tune / phase-forecast branches are smoke-tested via
the underlying components (test_skill_runner, test_cli, test_forecast_e2e);
this file's job is the verify+report glue.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_fake_state(work_dir: Path, preset_path: Path,
                     origin_iso: str, horizon: int = 60) -> dict:
    """Write a synthetic state.json + forecast_paths.npz so verify can run."""
    # Use a deterministic synthetic forecast: paths drawn from a small-vol
    # Gaussian, centered on zero. The realized path is also a small random
    # walk so coverage lands in [0.5, 1.0].
    rng = np.random.default_rng(7)
    n_paths = 200
    daily_sigma = 0.005
    paths = rng.normal(0.0, daily_sigma, size=(n_paths, horizon))
    np.savez_compressed(work_dir / "forecast_paths.npz", paths=paths)
    state = {
        "fetch": {
            "phase": "fetch",
            "identifier": "NIFTY:NIFTY500",
            "range": {"start": "2024-01-01", "end": "2024-12-31"},
            "rows": 250,
            "first_date": "2024-01-01",
            "last_date": "2024-12-31",
            "cache_was_cold": True,
            "gaps_filled": [],
            "providers_failed": [],
        },
        "tune": {
            "phase": "tune",
            "tune_end": origin_iso,
            "preset_path": str(preset_path.resolve()),
            "tune_runtime_seconds": 1.0,
            "cmd": ["python", "-m", "scripts.forecasters.run", "tune", "..."],
        },
        "forecast": {
            "phase": "forecast",
            "origin": origin_iso,
            "horizon": horizon,
            "preset_name": preset_path.stem,
            "preset_path": str(preset_path.resolve()),
            "result_summary": {"crps": None, "median": [], "p05": [], "p25": [], "p75": [], "p95": []},
            "warnings": [],
            "metadata": {},
            "anchors": {},
        },
    }
    (work_dir / "state.json").write_text(json.dumps(state, indent=2))
    return state


def _write_fixture_data(tmp_path: Path) -> Path:
    """Write a small synthetic NIFTY 500 CSV the orchestrator can read."""
    # Approximate ~250 trading days in 2024 + 60 trading days held-out into
    # the test horizon.
    dates = pd.bdate_range("2024-01-01", "2024-12-31")
    rng = np.random.default_rng(42)
    closes = 20000 + np.cumsum(rng.normal(0, 50, size=len(dates)))
    df = pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "adj_close": closes,
        "volume": (rng.integers(1_000_000_000, 3_000_000_000, size=len(dates))).astype("int64"),
    })
    p = tmp_path / "nifty500_fixture.csv"
    df.to_csv(p, index=False)
    return p


def _write_minimal_preset(path: Path, data_hash_str: str) -> None:
    """A minimal valid preset YAML for the verify step's schema check."""
    import yaml
    body = {
        "name": path.stem,
        "backend": "analog_mc",
        "schema_version": 1,
        "hyperparameters": {"n_eff": 50, "block_length": 10},
        "fitted_on": {
            "identifier": "NIFTY:NIFTY500",
            "start": "2024-01-01",
            "end": "2024-09-30",
            "data_hash": data_hash_str,
            "n_observations": 200,
        },
        "fitted_at": "2026-05-24T12:00:00Z",
        "validation_metrics": {"crps_mean": 0.05},
    }
    path.write_text(yaml.safe_dump(body, sort_keys=False))


def test_verify_phase_writes_pass_report(tmp_path: Path, monkeypatch) -> None:
    """End-to-end exercise of the verify+report glue using a fixture."""
    from scripts.forecasters import run_acceptance_demo as mod

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    preset_path = tmp_path / "fixture-preset.yaml"
    _write_minimal_preset(preset_path, "sha256:fixture")

    # Monkeypatch the data fetch to read from our local fixture instead of
    # going through data_pipelines. The verify phase calls fetch_with_meta
    # twice (forecast.compute_realized uses it for the realized horizon).
    fixture_csv = _write_fixture_data(tmp_path)
    fixture_df = pd.read_csv(fixture_csv)
    fixture_df["date"] = pd.to_datetime(fixture_df["date"])

    class FakeMeta:
        identifier = "NIFTY:NIFTY500"
        domain = "nse_equities"
        range = {"start": "2024-01-01", "end": "2024-12-31"}
        row_count = len(fixture_df)
        cache_was_cold = False
        gaps_filled: list = []
        providers_failed: list = []

    monkeypatch.setattr(mod, "fetch_with_meta", lambda *args, **kwargs: (fixture_df, FakeMeta()))

    # Origin: 60 trading days before the last fixture date.
    origin_iso = "2024-09-25"  # picked so >= 60 future trading days remain
    state = _make_fake_state(work_dir, preset_path, origin_iso)

    # Run verify directly.
    from types import SimpleNamespace
    args = SimpleNamespace(
        start="2024-01-01",
        end="2024-12-31",
        work_dir=str(work_dir),
    )

    monkeypatch.chdir(tmp_path)  # so docs/forecasters/_acceptance_demo.md writes here
    report = mod.phase_verify(args, state["forecast"], state["fetch"], state["tune"], preset_path)

    # Assertions on the report payload.
    assert report["phase"] == "verify"
    assert report["identifier"] == "NIFTY:NIFTY500"
    assert "crps_mean" in report
    assert "coverage_90" in report
    assert isinstance(report["assertions"], dict)
    for k in ("preset_validates", "forecast_warnings_empty",
              "coverage_90_in_range", "crps_finite", "crps_beats_baseline"):
        assert k in report["assertions"]

    # Report file written.
    md_path = tmp_path / "docs/forecasters/_acceptance_demo.md"
    assert md_path.is_file()
    md = md_path.read_text()
    assert "Verdict:" in md
    assert "NIFTY:NIFTY500" in md
    assert "Acceptance criteria" in md
