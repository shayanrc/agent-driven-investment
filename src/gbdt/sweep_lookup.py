"""V1.3 Option A — canonical sweep-CSV lookup for the anti-AUC flag.

The canonical CSV lives at ``results/gbdt/data/r_precision_at_k.csv`` and
carries one row per completed experiment with the columns ``experiment``,
``rows``, ``Q_days``, ``base_rate``, ``AUC``, ``R_precision_at_{1,3,5,10,20}``
(see ``.claude/memories/project-r-precision-methodology.md`` for the
formula + regeneration recipe).

The lookup converts an experiment's cell tuple ``(universe, direction,
threshold_pct, horizon_days, max_drawdown)`` into the canonical experiment
name suffix used in the CSV and returns the matching row (preferring the
sweep-baseline row over any ``_agentloop`` / ``_manual_xgb`` / etc.
follow-up variants for the same cell). The flag's job is to capture the
cell-SHAPE property (per D3); follow-up variants are particular
*responses* to that shape, not the shape itself.

Returns None if no matching row is found — the caller surfaces that as
``anti_auc_flag == "unknown"`` so the loop's auto-disables safely default
to NOT firing on new cells without a sweep row yet.
"""

from __future__ import annotations

import csv
from pathlib import Path


# Suffix priority: the bare sweep row wins over any per-cell follow-up
# variants. Each cell can have at most one bare row (e.g.
# ``nasdaq100_up_10pct_50d_dd5pct``); the variants are
# ``<bare>_agentloop``, ``<bare>_manual_xgb``, etc. The first match in
# this list (with the bare match listed implicitly first) wins.
_VARIANT_SUFFIXES = (
    "",            # bare sweep row (preferred)
    "_agentloop",
    "_agentloop_mix",
    "_agentloop_mix_mcw3",
    "_agentloop_gamma",
    "_agentloop_colsample",
    "_manual_xgb",
    "_xgb_phase8",
    "_xgb_acceptance",
    "_catboost_phase8",
    "_pilot",
)


def cell_key_to_experiment_name(
    universe: str,
    direction: str,
    threshold_pct: float | int,
    horizon_days: int,
    max_drawdown: float | None,
) -> str:
    """Build the canonical experiment name for one cell.

    Format matches the existing CSV rows:
    ``<universe>_<direction>_<threshold_pct>pct_<horizon_days>d_dd<max_drawdown_pct>pct``
    (or without the ``_dd<…>pct`` suffix when ``max_drawdown`` is None).

    Examples:
        ``("nasdaq100", "up", 10, 50, 0.05)`` ->
            ``"nasdaq100_up_10pct_50d_dd5pct"``
        ``("sp500", "up", 20, 25, 0.10)`` ->
            ``"sp500_up_20pct_25d_dd10pct"``
        ``("nifty50", "up", 10, 20, None)`` -> ``"nifty50_up_10pct_20d"``
    """
    # threshold_pct may arrive as int (10) or float (10.0); the CSV uses int.
    thr = int(round(float(threshold_pct)))
    base = f"{universe}_{direction}_{thr}pct_{int(horizon_days)}d"
    if max_drawdown is None:
        return base
    # max_drawdown is in (0, 1) per the spec validator (e.g. 0.05 -> 5pct).
    dd_pct = int(round(float(max_drawdown) * 100))
    return f"{base}_dd{dd_pct}pct"


def lookup_sweep_row(
    cell_key: str,
    csv_path: Path,
) -> dict | None:
    """Return the canonical sweep row for ``cell_key``, or None if absent.

    Reads ``csv_path`` (typically
    ``results/gbdt/data/r_precision_at_k.csv``) and finds the row matching
    ``cell_key`` exactly, OR (preferred) the bare-cell row when
    ``cell_key`` itself is a variant. The numeric columns (``AUC``,
    ``R_precision_at_*``, ``base_rate``) are coerced to floats; the rest
    are returned as strings.

    Suffix-stripping (per plan § 3.3 / D3): if ``cell_key`` carries one of
    the known variant suffixes, the bare version is searched first.
    Otherwise, ``cell_key`` is searched as-is and a suffixed variant of
    the same bare name is the second-choice fallback.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    # Build the bare-cell name (suffix-strip cell_key if it carries one).
    bare = cell_key
    for suffix in _VARIANT_SUFFIXES:
        if suffix and cell_key.endswith(suffix):
            bare = cell_key[: -len(suffix)]
            break

    rows_by_name: dict[str, dict] = {}
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("experiment", "").strip()
                if not name:
                    continue
                # Coerce numeric columns; leave strings on failure.
                for k in ("base_rate", "AUC", "R_precision_at_1",
                          "R_precision_at_3", "R_precision_at_5",
                          "R_precision_at_10", "R_precision_at_20"):
                    if k in row and row[k] not in (None, ""):
                        try:
                            row[k] = float(row[k])
                        except ValueError:
                            pass
                rows_by_name[name] = row
    except OSError:
        return None

    # Priority order: cell_key exact, then bare, then bare+each variant
    # suffix in registry order.
    candidates: list[str] = []
    candidates.append(cell_key)
    if bare != cell_key:
        candidates.append(bare)
    for suffix in _VARIANT_SUFFIXES:
        candidate = bare + suffix
        if candidate not in candidates:
            candidates.append(candidate)

    for name in candidates:
        if name in rows_by_name:
            return rows_by_name[name]
    return None


__all__ = [
    "cell_key_to_experiment_name",
    "lookup_sweep_row",
]
