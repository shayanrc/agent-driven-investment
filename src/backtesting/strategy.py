"""Strategy callback protocol + run-loop helper.

Strategies in v1 are deliberately not a class hierarchy — per ``goal.md``
the engine is strategy-agnostic. A strategy is any callable that maps a
``(state, info)`` observation to an ``action`` dict (or ``None``); this
module pins down the type contract and provides one helper that drives
the loop from ``reset()`` to ``done=True``.

The Strategy protocol is intentionally minimal:

- one ``__call__`` method, ``(state, info) -> action`` (or ``None``);
- ``action`` follows ``spec.md`` § 3.1 (``{"type": "order", ...}``,
  ``{"type": "weight", ...}``, or ``None``);
- the strategy can hold private state across calls (a plain function with
  ``functools.partial``-bound state, a closure, a class instance — all
  valid, none required).

The engine NEVER calls a strategy. The caller calls the strategy. The
caller passes the strategy's output into ``Backtest.step(action)``. This
preserves D1 (step-loop, not event-driven callbacks) — the strategy
protocol is a *convention* for callers, not an engine hook.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from backtesting.backtest import Backtest


@runtime_checkable
class Strategy(Protocol):
    """The single-method strategy callback contract.

    Implementations may be plain functions, callable classes, lambdas, or
    closures. The engine does not import this protocol — it's a
    convenience for callers who want a typed signature for hand-coded or
    composed strategies.

    Parameters passed by the caller (NOT by the engine):
    - ``state``: the dict returned by ``Backtest.step()`` (or ``reset()``).
      Shape per ``spec.md`` § 3.2.
    - ``info``: the per-step diagnostic dict. Shape per ``spec.md`` § 3.2
      (only non-empty Q10 keys appear). On the first call (post-reset),
      ``info`` is ``{}``.

    Returns:
        An ``action`` dict (``{"type": "order", "orders": [...]}`` or
        ``{"type": "weight", "target_weights": {...}}``) or ``None`` to
        hold for the bar.
    """

    def __call__(
        self, state: dict[str, Any], info: dict[str, Any]
    ) -> dict[str, Any] | None: ...


# ---------------------------------------------------------------------------
# Example strategies (also reusable in tests as fixtures).
# ---------------------------------------------------------------------------
class HoldStrategy:
    """The simplest possible strategy: do nothing, ever.

    Useful as a baseline (equity curve = initial cash) and as a smoke
    test for the run-loop machinery.
    """

    def __call__(
        self, state: dict[str, Any], info: dict[str, Any]
    ) -> None:
        return None


class FixedWeightStrategy:
    """Submit a fixed target_weights vector on the first step, hold after.

    The caller passes the weight dict at construction time. The strategy
    issues the rebalance exactly once (the post-reset step) and returns
    ``None`` on every subsequent step.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        if sum(weights.values()) > 1.0 + 1e-12:
            raise ValueError(
                f"FixedWeightStrategy weights sum to {sum(weights.values())} "
                "(> 1.0); the engine would reject this at parse time"
            )
        self._weights = dict(weights)
        self._submitted = False

    def __call__(
        self, state: dict[str, Any], info: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self._submitted:
            return None
        self._submitted = True
        return {"type": "weight", "target_weights": dict(self._weights)}

    def reset(self) -> None:
        """Reset the internal submission latch (mirrors Backtest.reset)."""
        self._submitted = False


class ScriptedActionStrategy:
    """Replay a pre-built list of actions, one per step.

    The strategy returns ``actions[i]`` on call ``i``; once the script is
    exhausted, it returns ``None`` for every subsequent step. Useful for
    deterministic regression tests.
    """

    def __init__(self, actions: list[dict[str, Any] | None]) -> None:
        self._actions = list(actions)
        self._i = 0

    def __call__(
        self, state: dict[str, Any], info: dict[str, Any]
    ) -> dict[str, Any] | None:
        if self._i >= len(self._actions):
            return None
        a = self._actions[self._i]
        self._i += 1
        return a

    def reset(self) -> None:
        self._i = 0


# ---------------------------------------------------------------------------
# Run-loop helper
# ---------------------------------------------------------------------------
def run_strategy(
    backtest: Backtest,
    strategy: Strategy,
    max_steps: int | None = None,
) -> list[tuple[dict[str, Any], bool, dict[str, Any]]]:
    """Drive a backtest from reset() to done with a strategy callback.

    Returns a list of ``(state, done, info)`` tuples — one per step, in
    chronological order, including the post-reset initial state and the
    terminal state. The terminal tuple has ``done=True``; everything
    before has ``done=False``.

    Parameters
    ----------
    backtest:
        A constructed ``Backtest`` instance. The function calls
        ``backtest.reset()`` first.
    strategy:
        Any callable matching the ``Strategy`` protocol.
    max_steps:
        Optional hard cap on the number of ``step()`` calls. ``None`` (the
        default) runs to natural termination (``done=True``). Useful for
        bounded smoke tests.

    The engine still owns all state. The strategy never sees the broker,
    portfolio, or data handler directly — only the per-step state /
    info dicts. This keeps the strategy-agnostic property of the engine
    intact (per ``goal.md``).
    """
    if not callable(strategy):
        raise TypeError(
            f"strategy must be callable; got {type(strategy).__name__}"
        )
    history: list[tuple[dict[str, Any], bool, dict[str, Any]]] = []
    state, done, info = backtest.reset()
    history.append((state, done, info))
    steps_taken = 0
    while not done:
        if max_steps is not None and steps_taken >= max_steps:
            break
        action = strategy(state, info)
        state, done, info = backtest.step(action)
        history.append((state, done, info))
        steps_taken += 1
    return history
