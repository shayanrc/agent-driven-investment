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

Tie-breaking (L1 from ``_187``)
-------------------------------
On HP-ceiling cells (``_187``, ``_147``) the entire HP grid lands val Brier
in a sub-1% band while the train-val gap varies up to ~9× and Spiegelhalter
Z varies ~2× across configs. A pure val-Brier ``argmin`` cannot distinguish
a clean (low-gap, well-calibrated) config from an overfit one at equal val
Brier. To break ties:

1. Define a tie band ``[min_val_brier, min_val_brier + tie_band]``. All
   iterations whose ``val_brier`` lands inside the band are *tied with best*.
2. Among tied configs, prefer (a) lower ``train_val_gap``, then (b)
   Spiegelhalter ``|z|`` closer to 0 (lexicographic).
3. When the tie set is a singleton (no other config within the band), the
   strict val-Brier winner is returned unchanged — backwards-compatible.

The default ``tie_band`` is ``0.5 × plateau_threshold`` (= 0.0025 absolute
val Brier at the default ``plateau_threshold=0.005``). Rationale: this is
**half the per-iteration improvement floor** the plateau gate already
considers "no further movement worth chasing" — so any cluster of configs
inside it is, by the loop's own definition, val-Brier-equivalent. The
``_187`` motivating cell spans 0.00108 absolute val Brier across 9 configs,
well within the default band; the d4/λ6/mcw10 hygiene config (gap 0.0017,
``|z|=9.4``) wins over the d8/λ3 overfit config (gap 0.0146, ``|z|=13.0``)
at ~equal val Brier — which is the L1 fix.
"""

from __future__ import annotations

from typing import Sequence


def inner_stop_check(
    val_briers: Sequence[float],
    *,
    plateau_threshold: float = 0.005,
    degradation_gate: float = 0.01,
    max_iterations: int = 8,
    disable_plateau: bool = False,
) -> tuple[bool, str | None]:
    """Return ``(should_stop, signal_name | None)``.

    Evaluates against the full history of val Brier scores so far (most
    recent last). The agent calls this after each iteration's bundle is in
    hand.

    Signal precedence: ``degradation`` first (more diagnostic), then
    ``plateau``, then ``cap``.

    ``disable_plateau`` (task #204): when True, the plateau signal is
    suppressed. ``degradation`` + ``cap`` still fire normally. Set by the
    runner when ``callback_mode == "agent_file_protocol"`` — in agent mode
    the runner defers loop-continuation calls to the agent, which should be
    free to pivot to a structurally-different knob (e.g. ``colsample``
    after ``min_child_weight`` plateaued) instead of being auto-stopped on a
    single-knob val_brier flatline. ``default`` (sweep) mode keeps the
    plateau gate active — there's no agent to defer to.
    """
    n = len(val_briers)
    if n == 0:
        return False, None

    best = min(val_briers)

    # Degradation: latest > (1 + gate) * best
    if val_briers[-1] > (1.0 + degradation_gate) * best:
        return True, "degradation"

    # Plateau: last two iterations both improved by < threshold.
    # Gated off in agent mode (task #204) — the agent decides when to stop.
    if not disable_plateau and n >= 3:
        d_last = val_briers[-2] - val_briers[-1]
        d_prev = val_briers[-3] - val_briers[-2]
        if d_last < plateau_threshold and d_prev < plateau_threshold:
            return True, "plateau"

    # Cap
    if n >= max_iterations:
        return True, "cap"

    return False, None


# Default tie band as a fraction of plateau_threshold. See module docstring
# for rationale.
DEFAULT_TIE_BAND_FRACTION = 0.5


def _resolve_tie_band(
    tie_band: float | None,
    plateau_threshold: float | None,
) -> float:
    """Resolve the effective tie band.

    - Explicit ``tie_band`` (including 0.0 to disable tie-breaking) wins.
    - Otherwise fall back to ``DEFAULT_TIE_BAND_FRACTION * plateau_threshold``.
    - If neither is supplied, return 0.0 (no tie-breaking; strict argmin).
    """
    if tie_band is not None:
        return float(tie_band)
    if plateau_threshold is not None:
        return float(plateau_threshold) * DEFAULT_TIE_BAND_FRACTION
    return 0.0


def best_checkpoint(
    val_briers: Sequence[float],
    *,
    train_val_gaps: Sequence[float | None] | None = None,
    spiegelhalter_zs: Sequence[float | None] | None = None,
    tie_band: float | None = None,
    plateau_threshold: float | None = None,
) -> int:
    """Return the index of the best checkpoint to ship.

    Default behaviour (no tie-break inputs): the strict val-Brier ``argmin``,
    bit-for-bit identical to the v1 contract.

    With tie-break inputs (``train_val_gaps`` and/or ``spiegelhalter_zs``):
    among configs whose val Brier lands within
    ``[min_val_brier, min_val_brier + effective_tie_band]``, return the
    iteration with (a) lowest ``train_val_gap``, then (b) smallest
    ``|spiegelhalter_z|``, then (c) earliest iteration index as a final
    deterministic fallback. ``None`` gap / z entries are treated as the
    worst possible value on their key — i.e. iterations with missing
    metrics never out-prefer iterations with present metrics.

    ``effective_tie_band`` is resolved via :func:`_resolve_tie_band` from the
    explicit ``tie_band`` argument (overrides everything; pass ``0.0`` to
    disable tie-breaking) or, failing that, from ``plateau_threshold``
    scaled by :data:`DEFAULT_TIE_BAND_FRACTION`.

    When the tie set is a singleton (no other iteration's val Brier falls
    inside the band, ignoring rounding noise), the strict val-Brier winner
    is returned unchanged — backwards-compatible.
    """
    n = len(val_briers)
    if not n:
        raise ValueError("empty val_briers")

    strict_best = int(min(range(n), key=lambda i: val_briers[i]))

    # Backwards-compatible fast path: no tie-break inputs at all → strict argmin.
    if train_val_gaps is None and spiegelhalter_zs is None:
        return strict_best

    effective_band = _resolve_tie_band(tie_band, plateau_threshold)
    if effective_band <= 0.0:
        return strict_best

    min_vb = float(val_briers[strict_best])
    threshold = min_vb + effective_band

    # Find the tie set (all configs within the band).
    tied = [i for i in range(n) if float(val_briers[i]) <= threshold]
    if len(tied) <= 1:
        return strict_best

    # Worst-case sentinels for missing-metric configs: +inf gap, +inf |z|.
    # Earlier iter index breaks the final remaining tie deterministically
    # (favours the simpler/earlier config — typically the unperturbed base).
    def sort_key(i: int) -> tuple[float, float, int]:
        gap = (
            float(train_val_gaps[i])
            if (train_val_gaps is not None and train_val_gaps[i] is not None)
            else float("inf")
        )
        z = (
            abs(float(spiegelhalter_zs[i]))
            if (spiegelhalter_zs is not None and spiegelhalter_zs[i] is not None)
            else float("inf")
        )
        return (gap, z, i)

    return min(tied, key=sort_key)


__all__ = [
    "inner_stop_check",
    "best_checkpoint",
    "DEFAULT_TIE_BAND_FRACTION",
]
