"""forecasters — backend-agnostic forecasting surface.

See ``docs/forecasters/goal.md`` for what success looks like and
``docs/forecasters/V1_PLAN.md`` for the implementation spec.

v1 ships a single backend (analog_mc) wired through a thin dispatcher; the
public surface is three skills (``/forecast``, ``/tune-preset``,
``/list-presets``) plus two bundled ``data_pipelines``-owned skills.
"""

from forecasters.errors import (
    PresetSchemaError,
    ResultContractError,
    UnknownBackendError,
    UnknownPresetError,
)
from forecasters.presets import (
    list_presets,
    load_preset,
    preset_content_hash,
    resolve_preset_path,
    validate_preset,
)

__all__ = [
    "PresetSchemaError",
    "ResultContractError",
    "UnknownBackendError",
    "UnknownPresetError",
    "list_presets",
    "load_preset",
    "preset_content_hash",
    "resolve_preset_path",
    "validate_preset",
]
