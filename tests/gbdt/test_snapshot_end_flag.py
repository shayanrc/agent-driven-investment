"""Bug #226 Phase 3 — ``--snapshot-end`` CLI flag.

Three small tests cover the contract laid out in the bug-226 P3 brief:

1. **Override applied**: when ``--snapshot-end YYYY-MM-DD`` is passed and the
   spec carries a different ``date_range.end``, the runner pins the override
   for the lifetime of the run and persists the ISO string into
   ``metrics.json::preflight.snapshot_end_override``.

2. **No override**: when the flag is NOT passed, the spec's
   ``date_range.end`` is used unchanged and the persisted
   ``snapshot_end_override`` field is ``None``.

3. **Invalid date**: a malformed ISO string (``2025-13-99``) causes
   ``main(...)`` to exit with a non-zero status and emit a clear stderr
   message — without ever calling into ``run_experiment``.

Tests 1+2 reuse the synthetic-panel + ``callback_mode: default`` harness
(the same load_panel monkeypatch trick as ``tests/gbdt/test_phase4_smoke``)
so the runner finalizes in one process and writes ``metrics.json``. Test 3
is a pure CLI parser test — it can be a subsecond unit test.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

import gbdt.data as gbdt_data
from gbdt.__main__ import main, run_experiment
from gbdt.data import TickerStatus, UniversePanel


# ---------------------------------------------------------------------------
# Synthetic panel + spec (lifted from test_phase4_smoke; minor trim — default
# callback_mode here so the run finalizes in one shot)
# ---------------------------------------------------------------------------


def _synthetic_panel(n_per_ticker: int = 360, n_tickers: int = 4,
                      seed: int = 3) -> UniversePanel:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_per_ticker, freq="B")
    frames = []
    statuses = []
    for i in range(n_tickers):
        rets = rng.normal(0.0003, 0.012, n_per_ticker)
        c = 100.0 * np.exp(np.cumsum(rets))
        ticker = f"NSE:T{i}"
        frames.append(pd.DataFrame({
            "date": dates, "ticker": ticker,
            "open": c,
            "high": c * (1 + np.abs(rng.normal(0, 0.004, n_per_ticker))),
            "low": c * (1 - np.abs(rng.normal(0, 0.004, n_per_ticker))),
            "close": c, "adj_close": c,
            "volume": rng.integers(100_000, 500_000, n_per_ticker),
        }))
        statuses.append(TickerStatus(
            ticker=ticker, rows=n_per_ticker, kept=True, reason="",
            cache_last_date="2016-05-01", cache_age_days=1, is_stale=False,
        ))
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()

    ic = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n_per_ticker)))
    index_series = pd.DataFrame({
        "date": dates, "open": ic, "high": ic * 1.003, "low": ic * 0.997,
        "close": ic, "adj_close": ic,
        "volume": rng.integers(1_000_000, 5_000_000, n_per_ticker),
    }).set_index("date")

    return UniversePanel(
        universe="smoke_synth",
        panel=panel,
        index_series=index_series,
        annualization_factor=250,
        statuses=statuses,
        stale_tickers=[],
        staleness_days_threshold=gbdt_data.DEFAULT_STALENESS_DAYS,
    )


def _write_spec(out_dir, artifact_dir, *, spec_end: str | None = None) -> "object":
    """Tiny synthetic spec; ``callback_mode: default`` so the run finalizes
    in one shot. ``spec_end`` (when supplied) lands in ``date_range.end`` so
    the override-vs-spec contract can be asserted from the persisted
    ``preflight.snapshot_end_override``.
    """
    spec = {
        "target": {
            "universe": "smoke_synth",
            "direction": "up",
            "threshold_pct": 5,
            "horizon_days": 10,
            "max_drawdown": None,
        },
        "split": {
            "train_rows": 180, "val_rows": 90, "eval_rows": 50, "test_rows": 30,
            "min_rows_per_ticker": 350,
        },
        "features": {
            "candidates": ["F2", "F4"],
            "lookback_windows": [5, 10, 20],
            "exclude": [],
        },
        "backend": {
            "library": "catboost",
            "calibration_method": "conditional_isotonic",
            "fs_hp_loop": {
                "max_iterations": 1,
                "callback_mode": "default",
                "plateau_threshold": 0.0,
                "degradation_gate": 1.0,
            },
            "hp_starting": {
                "iterations": 15, "depth": 2, "learning_rate": 0.1,
                "l2_leaf_reg": 3.0, "boosting_type": "Plain",
                "early_stopping_rounds": 10,
            },
        },
        "artifacts": {"experiment_dir": str(artifact_dir)},
        "random_seed": 42,
    }
    if spec_end is not None:
        spec["date_range"] = {"end": spec_end}
    spec_path = out_dir / "snapshot_end_synth.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


@pytest.fixture()
def smoke_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "0")

    def _fake_load_panel(universe, *args, **kwargs):
        assert universe == "smoke_synth"
        return _synthetic_panel()

    monkeypatch.setattr(gbdt_data, "load_panel", _fake_load_panel)

    art_dir = tmp_path / "artifacts"
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    return tmp_path, spec_dir, art_dir


# ---------------------------------------------------------------------------
# Test 1 — override applied: metrics.json carries the override ISO string
# ---------------------------------------------------------------------------


def test_snapshot_end_override_persists_into_metrics(smoke_env):
    _tmp, spec_dir, art_dir = smoke_env
    spec_path = _write_spec(spec_dir, art_dir, spec_end="2025-01-01")

    # Override the spec's date_range.end with --snapshot-end 2025-12-31.
    final_dir = run_experiment(spec_path, snapshot_end=date(2025, 12, 31))

    metrics = json.loads((final_dir / "metrics.json").read_text())
    assert metrics["preflight"]["snapshot_end_override"] == "2025-12-31"

    # The snapshot of the per-experiment spec also reflects the effective
    # end-date (so the artifact's ``spec.yaml`` doesn't drift from the run).
    snapshot = yaml.safe_load((final_dir / "spec.yaml").read_text())
    snap_end = snapshot.get("date_range", {}).get("end")
    # YAML round-trips a date to a ``datetime.date`` — accept either repr.
    assert snap_end in (date(2025, 12, 31), "2025-12-31")


# ---------------------------------------------------------------------------
# Test 2 — no override: spec's date_range.end used unchanged, override = None
# ---------------------------------------------------------------------------


def test_snapshot_end_absent_leaves_spec_end_intact(smoke_env):
    _tmp, spec_dir, art_dir = smoke_env
    spec_path = _write_spec(spec_dir, art_dir, spec_end="2025-06-30")

    final_dir = run_experiment(spec_path)  # no snapshot_end kwarg

    metrics = json.loads((final_dir / "metrics.json").read_text())
    # Override field is present (always) and explicitly None when not passed.
    assert "snapshot_end_override" in metrics["preflight"]
    assert metrics["preflight"]["snapshot_end_override"] is None

    # The persisted spec snapshot still carries the spec's value (no mutation).
    snapshot = yaml.safe_load((final_dir / "spec.yaml").read_text())
    snap_end = snapshot.get("date_range", {}).get("end")
    assert snap_end in (date(2025, 6, 30), "2025-06-30")


# ---------------------------------------------------------------------------
# Test 3 — invalid date string exits non-zero with a clear stderr message
# ---------------------------------------------------------------------------


def test_invalid_snapshot_end_exits_with_clear_message(tmp_path, capsys):
    # The spec file path doesn't need to exist — argparse validates the
    # flag's value BEFORE main() dispatches to run_experiment.
    bogus_spec = tmp_path / "nonexistent.yaml"
    bogus_spec.write_text("target: {}\n")  # never read; runner exits first

    rc = main([
        "experiment", str(bogus_spec),
        "--snapshot-end", "2025-13-99",
    ])
    assert rc == 2, f"expected exit code 2, got {rc}"

    captured = capsys.readouterr()
    err = captured.err
    assert "--snapshot-end" in err, f"missing flag name in stderr: {err!r}"
    assert "2025-13-99" in err, f"missing offending value in stderr: {err!r}"
