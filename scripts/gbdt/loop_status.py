"""Monitoring surface for the agent-driven FS+HP loop (task #177).

Reads the persistent on-disk observability artifacts a running (or stalled)
``agent_file_protocol`` experiment writes — ``loop/status.json`` (single-shot
machine-readable position + liveness) and ``loop/progress.log`` (append-only
milestone + heartbeat trail) — and renders a compact human-readable status.

Two modes:

- ``python -m scripts.gbdt.loop_status <artifact_dir>`` — full status for one
  cell: the parsed ``status.json`` (run/iter/phase/awaiting_decision/
  best_val_brier/stop_reason), the last ~15 ``progress.log`` lines, and a
  derived ``ALIVE (heartbeat Ns ago)`` / ``STALE (Ns)`` / ``DONE (reason=…)``
  verdict (``now - last_heartbeat_utc``).

- ``python -m scripts.gbdt.loop_status`` (no arg) — scan
  ``results/gbdt/experiments/*/loop/`` and one-line-summarize every cell that
  has a ``status.json``.

Read-only — it never writes loop state. The liveness verdict + tail logic live
in :mod:`gbdt.loop_observability` so the runner and the reader share one
implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gbdt import loop_observability as obs


def _load_status(artifact_dir: Path) -> dict | None:
    """Parse ``<artifact_dir>/loop/status.json``; ``None`` if absent/unparseable."""
    p = obs.status_path(artifact_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def render_one(artifact_dir: Path, *, tail_n: int = 15) -> str:
    """Render the full single-cell status block as a string."""
    artifact_dir = Path(artifact_dir)
    status = _load_status(artifact_dir)
    lines: list[str] = []
    lines.append(f"=== {artifact_dir} ===")
    if status is None:
        lines.append(
            f"  no status.json at {obs.status_path(artifact_dir)} "
            f"(not an agent_file_protocol run, or never started)"
        )
        return "\n".join(lines)

    verdict = obs.liveness_verdict(status)
    lines.append(f"  verdict          : {verdict}")
    lines.append(f"  run_id           : {status.get('run_id')}")
    lines.append(f"  iter_idx         : {status.get('iter_idx')}")
    lines.append(f"  phase            : {status.get('phase')}")
    lines.append(f"  awaiting_decision: {status.get('awaiting_decision')}")
    lines.append(f"  best_val_brier   : {status.get('best_val_brier')}")
    lines.append(f"  stop_reason      : {status.get('stop_reason')}")
    lines.append(f"  last_update_utc  : {status.get('last_update_utc')}")
    lines.append(f"  last_heartbeat   : {status.get('last_heartbeat_utc')}")

    tail = obs.tail_lines(obs.progress_log_path(artifact_dir), n=tail_n)
    lines.append(f"  --- progress.log (last {len(tail)}) ---")
    if tail:
        for ln in tail:
            lines.append(f"    {ln}")
    else:
        lines.append("    (no progress.log)")
    return "\n".join(lines)


def render_summary_line(artifact_dir: Path) -> str | None:
    """One-line summary for the scan mode. ``None`` if no parseable status.json."""
    status = _load_status(Path(artifact_dir))
    if status is None:
        return None
    verdict = obs.liveness_verdict(status)
    name = status.get("run_id") or Path(artifact_dir).name
    awaiting = "AWAIT-DECISION" if status.get("awaiting_decision") else ""
    return (
        f"{name:<40} iter={status.get('iter_idx')!s:<4} "
        f"phase={status.get('phase')!s:<10} "
        f"{verdict} {awaiting}".rstrip()
    )


def _discover_cells(experiments_root: Path) -> list[Path]:
    """Every ``<experiments_root>/*/`` whose ``loop/status.json`` exists, sorted."""
    if not experiments_root.exists():
        return []
    cells = []
    for child in sorted(experiments_root.iterdir()):
        if child.is_dir() and obs.status_path(child).exists():
            cells.append(child)
    return cells


def render_scan(experiments_root: Path) -> str:
    """Render the no-arg scan over all cells with a status.json."""
    cells = _discover_cells(experiments_root)
    if not cells:
        return (
            f"no agent-loop cells with a status.json under {experiments_root} "
            f"(nothing running / never started an agent_file_protocol run)"
        )
    out = [f"scanning {experiments_root} — {len(cells)} cell(s) with status.json:"]
    for cell in cells:
        line = render_summary_line(cell)
        if line is not None:
            out.append(f"  {line}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.gbdt.loop_status",
        description="Report position + liveness for agent-driven FS+HP loop cells.",
    )
    parser.add_argument(
        "artifact_dir",
        nargs="?",
        default=None,
        type=Path,
        help="One experiment artifact dir (results/gbdt/experiments/<cell>/). "
             "Omit to scan all cells under --experiments-root.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("results/gbdt/experiments"),
        help="Root scanned in no-arg mode (default: results/gbdt/experiments).",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=15,
        help="Number of trailing progress.log lines to show (single-cell mode).",
    )
    args = parser.parse_args(argv)

    if args.artifact_dir is not None:
        print(render_one(args.artifact_dir, tail_n=args.tail))
    else:
        print(render_scan(args.experiments_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
