"""CLI entry point for the backtesting engine.

Usage:
    python -m backtesting <config.yaml> [--output result.json]

The YAML config drives a self-contained smoketest of the Stages 1-7
pipeline against **synthetic, in-memory data** — no I/O against
``data_pipelines`` cache, no external fetch. Per ``spec.md`` § 6, a
real YAML-driven config layer is deferred to v1.1; this CLI is a thin
convenience for end-to-end smoke-testing the engine via a single
command, not the public configuration contract.

The schema is deliberately small:

```yaml
backtest:
  initial_cash: 100000.0
  lookback: 5
  fill_mode: next_open          # or current_close
  gap_policy: ffill_zero_volume # or raise
  lot_sizes: {AAPL: 1, MSFT: 1}
  default_lot_size: 1

synthetic_feed:
  feed_name: equities
  start_date: 2024-01-01
  n_days: 30
  freq: B
  assets:
    - {name: AAPL, start_price: 100.0}
    - {name: MSFT, start_price: 200.0}

strategy:
  type: hold                    # one of hold | fixed_weight | scripted
  # type=fixed_weight extra fields:
  # weights: {AAPL: 0.3, MSFT: 0.3}
  # type=scripted extra fields:
  # actions: [...]              # list of action dicts (or null)

run:
  max_steps: null               # optional cap
```

Exit codes:
- 0 on success.
- 1 on config-validation or runtime error (printed to stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from backtesting.backtest import Backtest
from backtesting.results import summarize_run
from backtesting.strategy import (
    FixedWeightStrategy,
    HoldStrategy,
    ScriptedActionStrategy,
    Strategy,
    run_strategy,
)


# ---------------------------------------------------------------------------
# Synthetic-feed construction
# ---------------------------------------------------------------------------
def _build_synthetic_feed(
    spec: dict[str, Any],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build a synthetic single-feed dict matching DataHandler's contract.

    Each asset's OHLCV is a strictly monotone ramp from ``start_price``
    over ``n_days`` business days; deterministic by construction so
    smoke-test outputs are reproducible.
    """
    feed_name = spec.get("feed_name", "equities")
    start_date = spec.get("start_date", "2024-01-01")
    n_days = int(spec.get("n_days", 30))
    freq = spec.get("freq", "B")
    assets_spec = spec.get("assets") or []
    if not assets_spec:
        raise ValueError("synthetic_feed.assets must contain at least one entry")
    dates = pd.date_range(start_date, periods=n_days, freq=freq)
    feed: dict[str, pd.DataFrame] = {}
    for asset in assets_spec:
        name = asset["name"]
        start_price = float(asset.get("start_price", 100.0))
        base = np.arange(n_days, dtype=float) + start_price
        feed[name] = pd.DataFrame(
            {
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.5,
                "volume": np.full(n_days, 1_000.0),
            },
            index=dates,
        )
    return {feed_name: feed}


def _build_strategy(spec: dict[str, Any]) -> Strategy:
    s_type = spec.get("type", "hold")
    if s_type == "hold":
        return HoldStrategy()
    if s_type == "fixed_weight":
        weights = spec.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("strategy.weights must be a dict for fixed_weight")
        return FixedWeightStrategy({k: float(v) for k, v in weights.items()})
    if s_type == "scripted":
        actions = spec.get("actions")
        if not isinstance(actions, list):
            raise ValueError("strategy.actions must be a list for scripted")
        return ScriptedActionStrategy(actions)
    raise ValueError(
        f"unknown strategy.type {s_type!r}; "
        "expected one of: hold, fixed_weight, scripted"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Construct + run + summarize a backtest from a parsed config dict.

    Returns a JSON-serializable result dict with the RunSummary fields
    plus the resolved config echo for traceability.
    """
    bt_cfg = config.get("backtest") or {}
    feed_cfg = config.get("synthetic_feed") or {}
    strat_cfg = config.get("strategy") or {"type": "hold"}
    run_cfg = config.get("run") or {}

    data_feeds = _build_synthetic_feed(feed_cfg)
    bt = Backtest(
        data_feeds=data_feeds,
        initial_cash=float(bt_cfg.get("initial_cash", 100_000.0)),
        lookback=int(bt_cfg.get("lookback", 20)),
        lot_sizes={k: int(v) for k, v in (bt_cfg.get("lot_sizes") or {}).items()},
        default_lot_size=int(bt_cfg.get("default_lot_size", 1)),
        fill_mode=bt_cfg.get("fill_mode", "next_open"),
        gap_policy=bt_cfg.get("gap_policy", "ffill_zero_volume"),
    )
    strategy = _build_strategy(strat_cfg)
    max_steps = run_cfg.get("max_steps")
    history = run_strategy(
        bt, strategy, max_steps=int(max_steps) if max_steps is not None else None
    )
    summary = summarize_run(history)
    return {
        "n_steps": summary.n_steps,
        "terminal_done": summary.terminal_done,
        "initial_equity": summary.initial_equity,
        "final_equity": summary.final_equity,
        "total_return": summary.total_return,
        "n_fills": len(summary.fills),
        "n_rejected_overdraw": len(summary.rejected_overdraw),
        "n_rejected_untradeable": len(summary.rejected_untradeable),
        "n_rejected_invalid": len(summary.rejected_invalid),
        "final_positions": (
            summary.final_state["portfolio"]["positions"]
            if summary.final_state is not None
            else {}
        ),
        "final_cash": (
            summary.final_state["portfolio"]["cash"]
            if summary.final_state is not None
            else None
        ),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m backtesting",
        description="Run a synthetic-data backtest from a YAML config.",
    )
    p.add_argument(
        "config",
        type=Path,
        help="Path to YAML config file (see configs/backtesting/examples/).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON result. Default: stdout.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        with args.config.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        if not isinstance(config, dict):
            raise ValueError(
                f"config root must be a mapping; got {type(config).__name__}"
            )
        result = run_from_config(config)
    except Exception as err:  # noqa: BLE001 — surface any error to stderr
        print(f"backtesting: error: {err}", file=sys.stderr)
        return 1

    output = json.dumps(result, indent=2, default=str)
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
