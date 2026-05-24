"""Preset artifact loader, validator, and enumeration.

A preset is a fitted-artifact YAML carrying the backend name, hyperparameters,
fit metadata, and validation metrics — see ``docs/forecasters/V1_PLAN.md``
§"Preset artifact schema" for the full layout.

Two preset directories are searched, in order:

  1. ``configs/forecasters/presets/``  — canonical, checked-in presets.
  2. ``results/forecasters/presets/``  — user-tuned presets (gitignored).

If the same name appears in both, the canonical one wins (per the
"canonical presets ship in the repo" rule in ``goal.md``).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from forecasters.errors import PresetSchemaError, UnknownPresetError


# Project-root relative defaults. Tests / scripts override via ``preset_roots``.
DEFAULT_CANONICAL_ROOT = Path("configs/forecasters/presets")
DEFAULT_USER_ROOT = Path("results/forecasters/presets")


REQUIRED_TOP_LEVEL_KEYS = (
    "name",
    "backend",
    "schema_version",
    "hyperparameters",
    "fitted_on",
    "fitted_at",
    "validation_metrics",
)

REQUIRED_FITTED_ON_KEYS = (
    "identifier",
    "start",
    "end",
    "data_hash",
    "n_observations",
)

CURRENT_SCHEMA_VERSION = 1


def _preset_roots(
    canonical_root: Path | None = None,
    user_root: Path | None = None,
) -> list[Path]:
    """Resolve preset search roots. Order: canonical first, then user-tuned."""
    return [
        Path(canonical_root) if canonical_root is not None else DEFAULT_CANONICAL_ROOT,
        Path(user_root) if user_root is not None else DEFAULT_USER_ROOT,
    ]


def resolve_preset_path(
    name: str,
    canonical_root: Path | None = None,
    user_root: Path | None = None,
) -> Path:
    """Find the YAML file backing a preset name.

    Raises:
        UnknownPresetError: if no matching file is found.
    """
    roots = _preset_roots(canonical_root, user_root)
    searched: list[str] = []
    for root in roots:
        candidate = root / f"{name}.yaml"
        searched.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise UnknownPresetError(name, searched)


def preset_content_hash(preset_path: Path) -> str:
    """Stable content hash of a preset YAML — used for cache keys."""
    h = hashlib.sha256(Path(preset_path).read_bytes()).hexdigest()
    return f"sha256:{h}"


def load_preset(
    name: str,
    canonical_root: Path | None = None,
    user_root: Path | None = None,
) -> dict[str, Any]:
    """Load + validate a preset YAML by name; return the parsed dict.

    The returned dict carries two synthetic top-level keys not present in the
    YAML itself, populated by the loader:
      * ``__source_path__`` (str): absolute path of the loaded YAML.
      * ``__content_hash__`` (str): ``sha256:...`` of the YAML bytes.

    These are not part of the on-disk schema — they're loader-supplied for
    drift detection and cache keying downstream. They are stripped before any
    re-serialization (``__`` prefix marks them as private).
    """
    path = resolve_preset_path(name, canonical_root, user_root)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise PresetSchemaError(str(path), "top-level YAML is not a mapping")
    validate_preset(raw, source_path=path)
    raw["__source_path__"] = str(path.resolve())
    raw["__content_hash__"] = preset_content_hash(path)
    return raw


def validate_preset(preset: dict[str, Any], source_path: Path | str | None = None) -> None:
    """Assert that ``preset`` conforms to the v1 preset schema.

    Raises:
        PresetSchemaError: on any violation, with the offending field named.

    The ``source_path`` arg (if provided) is used to (a) include the file path
    in error messages and (b) check that ``name`` matches the file stem.
    """
    src = str(source_path) if source_path is not None else "<in-memory>"
    if not isinstance(preset, dict):
        raise PresetSchemaError(src, "preset is not a dict")

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in preset]
    if missing:
        raise PresetSchemaError(src, f"missing required top-level keys: {missing}")

    if preset["schema_version"] != CURRENT_SCHEMA_VERSION:
        raise PresetSchemaError(
            src,
            f"unsupported schema_version {preset['schema_version']!r} "
            f"(loader supports {CURRENT_SCHEMA_VERSION})",
        )

    if not isinstance(preset["name"], str) or not preset["name"]:
        raise PresetSchemaError(src, f"name must be a non-empty string; got {preset['name']!r}")

    if not isinstance(preset["backend"], str) or not preset["backend"]:
        raise PresetSchemaError(src, f"backend must be a non-empty string; got {preset['backend']!r}")

    if source_path is not None:
        expected = Path(source_path).stem
        if preset["name"] != expected:
            raise PresetSchemaError(
                src,
                f"preset name {preset['name']!r} does not match filename stem {expected!r}; "
                "rename the file or rewrite the `name` field so they agree.",
            )

    hp = preset["hyperparameters"]
    if not isinstance(hp, dict) or not hp:
        raise PresetSchemaError(src, "hyperparameters must be a non-empty dict")

    fitted_on = preset["fitted_on"]
    if not isinstance(fitted_on, dict):
        raise PresetSchemaError(src, "fitted_on must be a dict")
    missing_fo = [k for k in REQUIRED_FITTED_ON_KEYS if k not in fitted_on]
    if missing_fo:
        raise PresetSchemaError(src, f"fitted_on missing keys: {missing_fo}")
    if not isinstance(fitted_on["data_hash"], str) or not fitted_on["data_hash"].startswith("sha256:"):
        raise PresetSchemaError(
            src,
            "fitted_on.data_hash must be a 'sha256:<hex>' string",
        )

    fitted_at = preset["fitted_at"]
    if not isinstance(fitted_at, str):
        raise PresetSchemaError(src, "fitted_at must be a UTC ISO-8601 string")
    try:
        _parse_utc(fitted_at)
    except ValueError as e:
        raise PresetSchemaError(src, f"fitted_at is not a UTC ISO-8601 timestamp: {e}") from e

    vm = preset["validation_metrics"]
    if not isinstance(vm, dict):
        raise PresetSchemaError(src, "validation_metrics must be a dict")


def _parse_utc(s: str) -> datetime:
    """Parse a UTC ISO-8601 timestamp; reject naive / non-UTC strings."""
    # Accept both 'Z' and '+00:00' suffixes.
    normalized = s.replace("Z", "+00:00") if s.endswith("Z") else s
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        raise ValueError("missing tzinfo (must end in 'Z' or '+00:00')")
    offset = dt.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("must be UTC ('Z' or '+00:00')")
    return dt


def list_presets(
    canonical_root: Path | None = None,
    user_root: Path | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """Enumerate every preset across both roots; return summary dicts.

    Summary fields (one row per preset):
        - ``name``: preset name
        - ``source``: ``"canonical"`` (configs/) or ``"user-tuned"`` (results/)
        - ``backend``: ``preset.backend``
        - ``fitted_on_identifier``: ``preset.fitted_on.identifier``
        - ``fitted_on_start`` / ``fitted_on_end``: range strings
        - ``fitted_at``: UTC ISO string
        - ``crps_mean``: ``preset.validation_metrics.crps_mean`` if present, else None
        - ``path``: source YAML path
        - ``error``: validation error message if the file fails schema validation;
                     otherwise None. (Failed presets still appear in the list so
                     ``/list-presets`` surfaces them rather than silently hiding.)

    Names that appear in both roots are reported with ``source="canonical"``
    only — the canonical version wins.

    Args:
        backend: if set, only presets with this ``backend`` field are returned.
    """
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    roots = _preset_roots(canonical_root, user_root)
    labels = ("canonical", "user-tuned")
    for root, source in zip(roots, labels, strict=True):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.yaml")):
            name = path.stem
            if name in seen:
                continue
            seen.add(name)
            row: dict[str, Any] = {
                "name": name,
                "source": source,
                "path": str(path.resolve()),
                "backend": None,
                "fitted_on_identifier": None,
                "fitted_on_start": None,
                "fitted_on_end": None,
                "fitted_at": None,
                "crps_mean": None,
                "error": None,
            }
            try:
                raw = yaml.safe_load(path.read_text()) or {}
                validate_preset(raw, source_path=path)
                row["backend"] = raw["backend"]
                fo = raw["fitted_on"]
                row["fitted_on_identifier"] = fo.get("identifier")
                row["fitted_on_start"] = fo.get("start")
                row["fitted_on_end"] = fo.get("end")
                row["fitted_at"] = raw["fitted_at"]
                vm = raw["validation_metrics"]
                row["crps_mean"] = vm.get("crps_mean") if isinstance(vm, dict) else None
            except PresetSchemaError as e:
                row["error"] = e.violated
            except Exception as e:  # pragma: no cover - defensive
                row["error"] = f"unexpected: {e}"
            rows.append(row)
    if backend is not None:
        rows = [r for r in rows if r.get("backend") == backend]
    return rows
