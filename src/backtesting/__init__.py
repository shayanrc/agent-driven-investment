"""backtesting — a correct, configurable, multi-asset backtesting engine.

See ``docs/backtesting/goal.md`` and ``docs/backtesting/spec.md`` for the
design contract. The public entry point is :class:`backtesting.Backtest`.
"""

from backtesting.backtest import Backtest
from backtesting.results import (
    RunSummary,
    summarize_run,
    validate_info_schema,
)
from backtesting.strategy import (
    FixedWeightStrategy,
    HoldStrategy,
    ScriptedActionStrategy,
    Strategy,
    run_strategy,
)

__all__ = [
    "Backtest",
    "Strategy",
    "HoldStrategy",
    "FixedWeightStrategy",
    "ScriptedActionStrategy",
    "run_strategy",
    "RunSummary",
    "summarize_run",
    "validate_info_schema",
]
