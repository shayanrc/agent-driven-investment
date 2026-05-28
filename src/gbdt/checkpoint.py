"""Checkpoint/resume scaffolding for the V1.1 agent-driven FS+HP loop.

The exit-and-resume architecture (see
``docs/gbdt/V1.1_agent_driven_fs_hp_loop_plan.md`` § 0) hands control back to
the agent each iteration: the runner trains, emits a diagnostic bundle, writes
a resume checkpoint, and **exits**. On the next launch ``--resume <run_id>``
loads the checkpoint and continues from the last good iteration.

V1.1 Phase 1 ships only the JSON round-trip primitives + the co-located path
helper (plan § 0.6). The actual exit-and-resume control flow lands in Phase 2.
These helpers are intentionally small and pure (no runner state, no imports
from ``gbdt.__main__``) so they can be unit-tested in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

# Bumped on any breaking change to the checkpoint dict shape (plan § 12 R6).
SCHEMA_VERSION = "v1"

# Co-located under the artifact dir per plan § 0.6:
#   results/gbdt/experiments/<cell>/loop/checkpoint.json
_LOOP_SUBDIR = "loop"
_CHECKPOINT_FILENAME = "checkpoint.json"


def checkpoint_path(run_id_or_dir: str | Path) -> Path:
    """Resolve the checkpoint path for a run.

    ``run_id_or_dir`` may be either the artifact directory (a ``Path`` or a
    path-like string that already points at ``results/.../<cell>/``) — in which
    case the checkpoint lands at ``<dir>/loop/checkpoint.json`` — or a bare
    run-id string, in which case it is treated as a relative directory name
    under the current working directory. Callers in the runner pass the
    existing ``out_dir`` so the checkpoint co-locates with the artifacts.
    """
    base = Path(run_id_or_dir)
    return base / _LOOP_SUBDIR / _CHECKPOINT_FILENAME


def write_checkpoint(run_id_or_dir: str | Path, state: dict) -> Path:
    """Write ``state`` as the run's checkpoint JSON; return the path written.

    A ``schema_version`` field is stamped onto the persisted dict (plan § 12
    R6). The input ``state`` is not mutated. Parent directories are created as
    needed.
    """
    path = checkpoint_path(run_id_or_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, **dict(state)}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_checkpoint(run_id_or_dir: str | Path) -> dict | None:
    """Read the run's checkpoint JSON; return ``None`` if it does not exist.

    The returned dict includes the stamped ``schema_version`` field.
    """
    path = checkpoint_path(run_id_or_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())
