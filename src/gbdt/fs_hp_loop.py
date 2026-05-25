"""Inner-stop logic + best-checkpoint selection for the FS+HP loop.

Per V1_PLAN.md Stage 6:
- plateau: val Brier improvement < ``plateau_threshold`` (default 0.005
  absolute) across the last 2 iterations → stop.
- degradation: val Brier > ``(1 + degradation_gate) × best_val_brier``
  (default 1% degrade from best) → stop.
- cap: iteration count ≥ ``max_iterations`` (default 8) → stop.

The artifact at the end of the loop is the **best-Brier checkpoint**, not
the final iteration, so exploration into slightly worse iterations doesn't
penalize the final emit.
"""

from __future__ import annotations

from typing import Sequence


def inner_stop_check(
    val_briers: Sequence[float],
    *,
    plateau_threshold: float = 0.005,
    degradation_gate: float = 0.01,
    max_iterations: int = 8,
) -> tuple[bool, str | None]:
    """Return ``(should_stop, signal_name | None)``.

    Evaluates against the full history of val Brier scores so far (most
    recent last). The agent calls this after each iteration's bundle is in
    hand.

    Signal precedence: ``degradation`` first (more diagnostic), then
    ``plateau``, then ``cap``.
    """
    n = len(val_briers)
    if n == 0:
        return False, None

    best = min(val_briers)

    # Degradation: latest > (1 + gate) * best
    if val_briers[-1] > (1.0 + degradation_gate) * best:
        return True, "degradation"

    # Plateau: last two iterations both improved by < threshold
    if n >= 3:
        d_last = val_briers[-2] - val_briers[-1]
        d_prev = val_briers[-3] - val_briers[-2]
        if d_last < plateau_threshold and d_prev < plateau_threshold:
            return True, "plateau"

    # Cap
    if n >= max_iterations:
        return True, "cap"

    return False, None


def best_checkpoint(val_briers: Sequence[float]) -> int:
    """Return the index of the lowest val Brier (best checkpoint to ship)."""
    if not len(val_briers):
        raise ValueError("empty val_briers")
    return int(min(range(len(val_briers)), key=lambda i: val_briers[i]))


__all__ = ["inner_stop_check", "best_checkpoint"]
