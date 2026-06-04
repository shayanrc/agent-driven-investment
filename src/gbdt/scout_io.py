"""V1.3 Option B — agent_file_protocol scout/ subdir I/O contract.

The exit-resume cycles between the runner and the agent communicate via
three JSON file pairs under ``<artifact_dir>/scout/``:

- ``scout_results.jsonl`` (runner-written, cycle 1) — per-config metrics
  from Phase 1.5; one row per ``ScoutResult``.
- ``scout_bundle.json`` (runner-written, cycle 1) — summary metadata +
  the speed-biased agent prompt.
- ``combine_request.json`` (runner-written, cycle 1) — pre-filled with
  the lex auto-compose as zeroth candidate so the agent always has a
  "vs auto-compose" comparator visible.
- ``combine_decision.json`` (agent-written) — N ≤ 50 mix configs.
- ``combine_results.json`` (runner-written, cycle 2) — per-config fit
  metrics for every config in combine_decision.
- ``iter_0_decision.json`` (agent-written) — the winning HP overlay for
  iter_0.

Hard rule: every ``--resume`` invocation MUST re-pass ``--snapshot-end``
when the spec uses ``callback_mode == "agent_file_protocol"``. Enforced
in ``__main__.py::run_experiment`` against the contract written here.

D3b.A — ``combine_decision.json`` is hard-capped at 50 mix configs; the
runner rejects payloads exceeding the cap at parse time.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# D3b.A — combine cap (hard).
COMBINE_DECISION_MAX_CONFIGS = 50


def scout_dir(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / "scout"


def scout_results_path(artifact_dir: str | Path) -> Path:
    return scout_dir(artifact_dir) / "scout_results.jsonl"


def scout_bundle_path(artifact_dir: str | Path) -> Path:
    return scout_dir(artifact_dir) / "scout_bundle.json"


def combine_request_path(artifact_dir: str | Path) -> Path:
    return scout_dir(artifact_dir) / "combine_request.json"


def combine_decision_path(artifact_dir: str | Path) -> Path:
    return scout_dir(artifact_dir) / "combine_decision.json"


def combine_results_path(artifact_dir: str | Path) -> Path:
    return scout_dir(artifact_dir) / "combine_results.json"


def iter_0_decision_path(artifact_dir: str | Path) -> Path:
    return scout_dir(artifact_dir) / "iter_0_decision.json"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_scout_results(
    artifact_dir: str | Path, results: list[dict],
) -> Path:
    """Write ``scout_results.jsonl`` (one row per ScoutResult.to_dict())."""
    path = scout_results_path(artifact_dir)
    payload = "\n".join(json.dumps(r) for r in results)
    if payload:
        payload += "\n"
    _atomic_write_text(path, payload)
    return path


def write_scout_bundle(
    artifact_dir: str | Path, summary: dict,
) -> Path:
    """Write ``scout_bundle.json``."""
    path = scout_bundle_path(artifact_dir)
    _atomic_write_text(path, json.dumps(summary, indent=2, default=str))
    return path


def write_combine_request(
    artifact_dir: str | Path, payload: dict,
) -> Path:
    """Write ``combine_request.json`` — the runner-prefilled request."""
    path = combine_request_path(artifact_dir)
    _atomic_write_text(path, json.dumps(payload, indent=2, default=str))
    return path


def write_combine_results(
    artifact_dir: str | Path, results: list[dict],
) -> Path:
    """Write ``combine_results.json``."""
    path = combine_results_path(artifact_dir)
    _atomic_write_text(path, json.dumps({"configs": results}, indent=2,
                                          default=str))
    return path


# ---------------------------------------------------------------------------
# Decision reading + validation
# ---------------------------------------------------------------------------


class CombineDecisionError(ValueError):
    """Raised when ``combine_decision.json`` fails validation (missing,
    malformed, or exceeds D3b.A cap)."""


def read_combine_decision(artifact_dir: str | Path) -> dict:
    """Read + validate ``combine_decision.json``.

    Validation:
    - File exists + parses as JSON.
    - ``configs`` key holds a list of dicts.
    - List length ≤ ``COMBINE_DECISION_MAX_CONFIGS`` (D3b.A).
    - Each config has an ``hp`` dict (the overlay).

    Raises :class:`CombineDecisionError` on any failure.
    """
    path = combine_decision_path(artifact_dir)
    if not path.exists():
        raise CombineDecisionError(
            f"combine_decision.json not found at {path}. "
            f"The agent must write it before --resume."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CombineDecisionError(
            f"combine_decision.json at {path} is not valid JSON: {exc}"
        ) from exc
    configs = payload.get("configs")
    if not isinstance(configs, list):
        raise CombineDecisionError(
            f"combine_decision.json::configs must be a list (got "
            f"{type(configs).__name__})"
        )
    if len(configs) > COMBINE_DECISION_MAX_CONFIGS:
        raise CombineDecisionError(
            f"combine_decision.json::configs has {len(configs)} entries, "
            f"which exceeds the D3b.A cap of "
            f"{COMBINE_DECISION_MAX_CONFIGS}."
        )
    for i, cfg in enumerate(configs):
        if not isinstance(cfg, dict):
            raise CombineDecisionError(
                f"combine_decision.json::configs[{i}] must be a dict, "
                f"got {type(cfg).__name__}"
            )
        if "hp" not in cfg:
            raise CombineDecisionError(
                f"combine_decision.json::configs[{i}] missing 'hp' key"
            )
        if not isinstance(cfg["hp"], dict):
            raise CombineDecisionError(
                f"combine_decision.json::configs[{i}]['hp'] must be a "
                f"dict (got {type(cfg['hp']).__name__})"
            )
    return payload


class Iter0DecisionError(ValueError):
    """Raised when ``iter_0_decision.json`` fails validation."""


def read_iter_0_decision(artifact_dir: str | Path) -> dict:
    """Read + validate ``iter_0_decision.json`` (cycle 3).

    Expected shape: ``{"hp": {...}}``. The HP dict overlays on top of
    iter_0's defaults to seed Phase 2.
    """
    path = iter_0_decision_path(artifact_dir)
    if not path.exists():
        raise Iter0DecisionError(
            f"iter_0_decision.json not found at {path}. "
            f"The agent must write it before --resume."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise Iter0DecisionError(
            f"iter_0_decision.json at {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload.get("hp"), dict):
        raise Iter0DecisionError(
            f"iter_0_decision.json::hp must be a dict (got "
            f"{type(payload.get('hp')).__name__})"
        )
    return payload


# ---------------------------------------------------------------------------
# Cycle detection — what phase are we resuming into?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleState:
    has_combine_request: bool
    has_combine_decision: bool
    has_combine_results: bool
    has_iter_0_decision: bool


def detect_cycle_state(artifact_dir: str | Path) -> CycleState:
    return CycleState(
        has_combine_request=combine_request_path(artifact_dir).exists(),
        has_combine_decision=combine_decision_path(artifact_dir).exists(),
        has_combine_results=combine_results_path(artifact_dir).exists(),
        has_iter_0_decision=iter_0_decision_path(artifact_dir).exists(),
    )


__all__ = [
    "COMBINE_DECISION_MAX_CONFIGS",
    "CombineDecisionError",
    "Iter0DecisionError",
    "CycleState",
    "scout_dir",
    "scout_results_path",
    "scout_bundle_path",
    "combine_request_path",
    "combine_decision_path",
    "combine_results_path",
    "iter_0_decision_path",
    "write_scout_results",
    "write_scout_bundle",
    "write_combine_request",
    "write_combine_results",
    "read_combine_decision",
    "read_iter_0_decision",
    "detect_cycle_state",
]
