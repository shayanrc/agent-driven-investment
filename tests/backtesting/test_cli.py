"""Stage 8 tests for the `python -m backtesting <config.yaml>` CLI.

Coverage:
- End-to-end smoke run against the shipped example YAML
- Subprocess invocation (exercises the `__main__.py` entry point)
- Result-file shape (JSON keys, exit code 0)
- Config-error path (exit code 1, error on stderr)
- Strategy variants: hold, fixed_weight, scripted

All tests are fully self-contained — no fetch against /tmp/exp_data,
no network, no external state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backtesting.cli import main, run_from_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = (
    REPO_ROOT / "configs" / "backtesting" / "examples" / "smoketest_30day.yaml"
)

EXPECTED_RESULT_KEYS = {
    "n_steps",
    "terminal_done",
    "initial_equity",
    "final_equity",
    "total_return",
    "n_fills",
    "n_rejected_overdraw",
    "n_rejected_untradeable",
    "n_rejected_invalid",
    "final_positions",
    "final_cash",
}


# ---------------------------------------------------------------------------
# In-process: run_from_config
# ---------------------------------------------------------------------------
def test_example_config_runs_in_process():
    """The shipped example YAML must execute end-to-end."""
    import yaml

    config = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    result = run_from_config(config)
    assert set(result.keys()) == EXPECTED_RESULT_KEYS
    assert result["terminal_done"] is True
    assert result["initial_equity"] == pytest.approx(100_000.0)
    # Two fills (one per asset) from the fixed-weight strategy.
    assert result["n_fills"] == 2


def test_hold_strategy_config():
    config = {
        "backtest": {"initial_cash": 50_000.0, "lookback": 3},
        "synthetic_feed": {
            "n_days": 10,
            "assets": [{"name": "X", "start_price": 100.0}],
        },
        "strategy": {"type": "hold"},
    }
    result = run_from_config(config)
    assert result["n_fills"] == 0
    assert result["initial_equity"] == 50_000.0
    assert result["final_equity"] == 50_000.0
    assert result["total_return"] == 0.0


def test_scripted_strategy_config():
    config = {
        "backtest": {
            "initial_cash": 100_000.0,
            "lookback": 3,
            "fill_mode": "current_close",
        },
        "synthetic_feed": {
            "n_days": 10,
            "assets": [{"name": "X", "start_price": 100.0}],
        },
        "strategy": {
            "type": "scripted",
            "actions": [
                {"type": "order", "orders": [{"asset": "X", "qty": 10}]},
                None,
                {"type": "order", "orders": [{"asset": "X", "qty": -5}]},
            ],
        },
    }
    result = run_from_config(config)
    assert result["n_fills"] == 2
    assert result["final_positions"] == {"X": 5}


def test_unknown_strategy_raises():
    config = {
        "backtest": {"lookback": 3},
        "synthetic_feed": {
            "n_days": 10,
            "assets": [{"name": "X", "start_price": 100.0}],
        },
        "strategy": {"type": "ml_blackbox"},
    }
    with pytest.raises(ValueError, match="unknown strategy"):
        run_from_config(config)


def test_max_steps_caps_terminal_done_false():
    config = {
        "backtest": {"lookback": 3},
        "synthetic_feed": {
            "n_days": 100,
            "assets": [{"name": "X", "start_price": 100.0}],
        },
        "strategy": {"type": "hold"},
        "run": {"max_steps": 5},
    }
    result = run_from_config(config)
    assert result["terminal_done"] is False
    assert result["n_steps"] == 5


# ---------------------------------------------------------------------------
# In-process: main(argv)
# ---------------------------------------------------------------------------
def test_main_writes_to_output_file(tmp_path: Path):
    output = tmp_path / "result.json"
    rc = main([str(EXAMPLE_CONFIG), "--output", str(output)])
    assert rc == 0
    assert output.exists()
    payload = json.loads(output.read_text())
    assert set(payload.keys()) == EXPECTED_RESULT_KEYS


def test_main_missing_config_returns_nonzero(tmp_path: Path):
    bogus = tmp_path / "does_not_exist.yaml"
    rc = main([str(bogus)])
    assert rc == 1


def test_main_malformed_yaml_returns_nonzero(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_mapping_just_a_string\n", encoding="utf-8")
    rc = main([str(bad)])
    assert rc == 1


# ---------------------------------------------------------------------------
# Subprocess: actually exercises python -m backtesting
# ---------------------------------------------------------------------------
def test_python_m_backtesting_smoketest_subprocess(tmp_path: Path):
    """End-to-end: launch `python -m backtesting <config>` as a subprocess
    and verify exit code, stdout JSON shape."""
    proc = subprocess.run(
        [sys.executable, "-m", "backtesting", str(EXAMPLE_CONFIG)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    payload = json.loads(proc.stdout)
    assert set(payload.keys()) == EXPECTED_RESULT_KEYS
    assert payload["terminal_done"] is True


def test_python_m_backtesting_missing_config_subprocess(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, "-m", "backtesting", str(tmp_path / "nope.yaml")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 1
    assert "error" in proc.stderr.lower()
