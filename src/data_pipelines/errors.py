"""Typed exceptions for the data_pipelines module.

D5 (provider failure semantics) and D6 (API key safety) require that failures
be explicit and typed, never silent. Adapters raise these; the dispatch layer
catches the chain-fallthrough subset and re-raises AllProvidersFailed when
every tier is exhausted.

Error messages must NEVER contain API keys or other secrets (D6).
"""

from __future__ import annotations


class DataPipelinesError(Exception):
    """Root of the module's exception hierarchy."""


class ProviderError(DataPipelinesError):
    """Adapter call failed: HTTP error, network timeout, malformed payload."""

    def __init__(self, provider: str, identifier: str, reason: str):
        self.provider = provider
        self.identifier = identifier
        self.reason = reason
        super().__init__(f"[{provider}] {identifier}: {reason}")


class EmptyPayload(ProviderError):
    """Provider returned 200 OK with zero rows (e.g., delisted ticker)."""

    def __init__(self, provider: str, identifier: str):
        super().__init__(provider, identifier, "empty payload")


class SchemaMismatch(DataPipelinesError):
    """Adapter-returned DataFrame fails the domain's schema validator (D1)."""

    def __init__(self, provider: str, identifier: str, details: str):
        self.provider = provider
        self.identifier = identifier
        self.details = details
        super().__init__(f"[{provider}] {identifier}: schema mismatch — {details}")


class MissingAPIKey(DataPipelinesError):
    """Required API key env var is unset. Raised BEFORE any network call (D6)."""

    def __init__(self, provider: str, env_var: str):
        self.provider = provider
        self.env_var = env_var
        super().__init__(
            f"[{provider}] missing API key: env var {env_var} is unset"
        )


class AllProvidersFailed(DataPipelinesError):
    """Every tier in the domain's adapter chain failed. Cache is untouched."""

    def __init__(self, identifier: str, failures: list[ProviderError]):
        self.identifier = identifier
        self.failures = failures
        summary = "; ".join(
            f"{f.provider}: {f.reason}" if isinstance(f, ProviderError) else str(f)
            for f in failures
        )
        super().__init__(f"{identifier}: all providers failed — {summary}")


class UnknownDomain(DataPipelinesError):
    """No domain is registered for the given identifier prefix."""

    def __init__(self, identifier: str, known_prefixes: list[str]):
        self.identifier = identifier
        self.known_prefixes = known_prefixes
        super().__init__(
            f"no domain registered for {identifier!r}; known prefixes: {known_prefixes}"
        )
