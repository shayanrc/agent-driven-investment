"""Typed errors for the forecasters framework.

Each error carries enough context (backend / preset / violated constraint) for
a caller — human or agent — to act on the failure without having to read the
traceback. Bare ``str``-based ``raise`` is discouraged in this module.
"""

from __future__ import annotations


class PresetSchemaError(ValueError):
    """A preset YAML failed schema validation on load.

    Raised by ``forecasters.presets.validate_preset`` when a required key is
    missing, a value has the wrong type, or the ``name`` field doesn't match
    the file's stem.
    """

    def __init__(self, preset_path: str, violated: str):
        self.preset_path = preset_path
        self.violated = violated
        super().__init__(f"preset at {preset_path}: {violated}")


class UnknownPresetError(LookupError):
    """No preset by that name found under either preset directory."""

    def __init__(self, preset_name: str, searched: list[str]):
        self.preset_name = preset_name
        self.searched = searched
        joined = ", ".join(searched)
        super().__init__(
            f"no preset named {preset_name!r} found (searched: {joined})"
        )


class UnknownBackendError(LookupError):
    """``preset.backend`` doesn't appear in the dispatcher's backend table."""

    def __init__(self, backend_name: str, known: list[str]):
        self.backend_name = backend_name
        self.known = known
        joined = ", ".join(known) if known else "<none>"
        super().__init__(
            f"unknown backend {backend_name!r}; dispatcher knows: {joined}"
        )


class ResultContractError(ValueError):
    """A backend returned a result that violates the wire-format contract.

    This is a backend bug, not a user bug — the dispatcher refuses to write a
    malformed result to the cache.
    """

    def __init__(self, backend_name: str, violated: str):
        self.backend_name = backend_name
        self.violated = violated
        super().__init__(
            f"backend {backend_name!r} returned a contract-violating result: "
            f"{violated}"
        )
