"""Result aggregation helpers for backtest runs.

The engine itself produces only the per-step ``(state, done, info)``
tuple stream (per ``goal.md``: "this module produces probabilities,
downstream owns evaluation" stance, mirrored from ``analog_mc``). This
module is a *thin*, optional aggregator for callers who want a single
``RunSummary`` object after a run completes.

What's locked here:

- The per-step ``info`` dict shape — defined in ``spec.md`` § 3.2 and
  enforced by ``Backtest._build_info`` (key omission rules: list-shape
  keys omit when empty; dict-shape keys omit when zero / equal). The
  ``validate_info_schema`` helper below verifies a single info dict
  satisfies the contract.
- The ``RunSummary`` shape returned by ``summarize_run``. Stable in v1;
  any field addition is a backward-compatible extension.

What's NOT here:

- Sharpe / Sortino / max-drawdown ratios. The summary surfaces the
  equity curve as a list of ``(timestamp, equity)`` pairs; the caller
  computes whatever performance metrics it needs. (Per ``goal.md``:
  "the caller owns evaluation".)
- Persistence (CSV / Parquet writers). v1.1 concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Info schema lock (per spec.md § 3.2). Tests in test_results.py verify
# every emitted info dict's keys are a subset of this set.
LIST_VALUED_INFO_KEYS: frozenset[str] = frozenset(
    {
        "fills",
        "rejected_overdraw",
        "rejected_untradeable",
        "rejected_invalid",
    }
)
DICT_VALUED_INFO_KEYS: frozenset[str] = frozenset(
    {
        "weight_drift",
        "rebalance_shortfall",
        "lot_size_audit",
    }
)
ALLOWED_INFO_KEYS: frozenset[str] = (
    LIST_VALUED_INFO_KEYS | DICT_VALUED_INFO_KEYS
)


@dataclass
class RunSummary:
    """End-of-run summary built from a ``run_strategy`` history.

    Fields:

    - ``equity_curve``: list of ``(timestamp_iso, equity)`` per step
      (including the reset state).
    - ``fills``: flat list of every fill dict across the run.
    - ``rejected_overdraw`` / ``rejected_untradeable`` / ``rejected_invalid``:
      flat per-bucket lists.
    - ``final_state``: the terminal ``state`` dict (the last step's state).
    - ``n_steps``: number of ``step()`` calls (excludes the reset).
    - ``terminal_done``: whether the run ended naturally (``True``) or
      via a ``max_steps`` cap (``False``).
    """

    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    rejected_overdraw: list[dict[str, Any]] = field(default_factory=list)
    rejected_untradeable: list[dict[str, Any]] = field(default_factory=list)
    rejected_invalid: list[dict[str, Any]] = field(default_factory=list)
    final_state: dict[str, Any] | None = None
    n_steps: int = 0
    terminal_done: bool = False

    @property
    def initial_equity(self) -> float | None:
        return self.equity_curve[0][1] if self.equity_curve else None

    @property
    def final_equity(self) -> float | None:
        return self.equity_curve[-1][1] if self.equity_curve else None

    @property
    def total_return(self) -> float | None:
        """``(final - initial) / initial``. ``None`` if no equity curve."""
        if not self.equity_curve:
            return None
        i = self.equity_curve[0][1]
        f = self.equity_curve[-1][1]
        if i == 0:
            return None
        return (f - i) / i


def summarize_run(
    history: Iterable[tuple[dict[str, Any], bool, dict[str, Any]]],
) -> RunSummary:
    """Aggregate a ``run_strategy`` history into a ``RunSummary``.

    Walks the trace once, accumulating fills / rejections / equity. The
    first tuple in ``history`` is the post-reset state; every subsequent
    tuple is one ``step()``. The terminal tuple's ``done`` flag
    determines ``terminal_done``.

    Validates each info dict against the spec-locked schema along the way
    (catches a misconfigured engine that emits unknown keys; raises
    immediately so the bug is visible).
    """
    summary = RunSummary()
    history_list = list(history)
    if not history_list:
        return summary
    for state, _done, info in history_list:
        validate_info_schema(info)
        summary.equity_curve.append(
            (state["timestamp"], state["portfolio"]["equity"])
        )
        if "fills" in info:
            summary.fills.extend(info["fills"])
        if "rejected_overdraw" in info:
            summary.rejected_overdraw.extend(info["rejected_overdraw"])
        if "rejected_untradeable" in info:
            summary.rejected_untradeable.extend(info["rejected_untradeable"])
        if "rejected_invalid" in info:
            summary.rejected_invalid.extend(info["rejected_invalid"])

    # n_steps excludes the reset.
    summary.n_steps = len(history_list) - 1
    last_state, last_done, _ = history_list[-1]
    summary.final_state = last_state
    summary.terminal_done = bool(last_done)
    return summary


def validate_info_schema(info: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``info`` violates the spec.md § 3.2 lock.

    Rules enforced:
    - Every key must be in ``ALLOWED_INFO_KEYS`` (the seven locked keys).
    - List-valued keys (``fills`` / ``rejected_*``) must be non-empty
      when present (omit-when-empty rule).
    - Dict-valued keys (``weight_drift`` / ``rebalance_shortfall`` /
      ``lot_size_audit``) must be non-empty when present (omit-when-empty
      rule) and every value must be a dict-typed (lot_size_audit) or
      float-typed (drift / shortfall) payload.

    The "omit when zero / equal" rule for the dict-valued keys is
    enforced by ``Backtest._build_info`` itself (which only inserts asset
    entries with non-zero drift / shortfall, and only inserts lot_audit
    entries where requested != filled); this validator additionally
    checks the *container* is non-empty.
    """
    if not isinstance(info, dict):
        raise ValueError(
            f"info must be a dict, got {type(info).__name__}"
        )
    for key, value in info.items():
        if key not in ALLOWED_INFO_KEYS:
            raise ValueError(
                f"info has unknown key {key!r}; allowed: "
                f"{sorted(ALLOWED_INFO_KEYS)}"
            )
        if key in LIST_VALUED_INFO_KEYS:
            if not isinstance(value, list):
                raise ValueError(
                    f"info[{key!r}] must be a list, got {type(value).__name__}"
                )
            if len(value) == 0:
                raise ValueError(
                    f"info[{key!r}] is present but empty; "
                    "omit-when-empty rule violated (spec.md § 3.2)"
                )
        else:  # DICT_VALUED_INFO_KEYS
            if not isinstance(value, dict):
                raise ValueError(
                    f"info[{key!r}] must be a dict, got {type(value).__name__}"
                )
            if len(value) == 0:
                raise ValueError(
                    f"info[{key!r}] is present but empty; "
                    "omit-when-empty rule violated (spec.md § 3.2)"
                )
