"""forecasters — backend-agnostic forecasting surface.

See ``docs/forecasters/goal.md`` for what success looks like and
``docs/forecasters/V1_PLAN.md`` for the implementation spec.

v1 ships a single backend (analog_mc) wired through a thin dispatcher; the
public surface is three skills (``/forecast``, ``/tune-preset``,
``/list-presets``) plus two bundled ``data_pipelines``-owned skills.
"""

from forecasters.cache import (
    cache_key,
    read_cached,
    write_cached,
)
from forecasters.data import data_hash, prepare_data
from forecasters.dispatch import (
    dispatch_forecast,
    dispatch_tune,
    known_backends,
)
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
    "cache_key",
    "data_hash",
    "dispatch_forecast",
    "dispatch_tune",
    "known_backends",
    "list_presets",
    "load_preset",
    "prepare_data",
    "preset_content_hash",
    "read_cached",
    "resolve_preset_path",
    "validate_preset",
    "write_cached",
]
