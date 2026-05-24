"""Generic time-series ingestion module.

See docs/data_pipelines/goal.md and docs/data_pipelines/V1_IMPLEMENTATION_PLAN.md.

v1 stage: framework primitives only (schema, errors, adapter, domain).
Public `fetch()` is wired in Stage 4.
"""

from data_pipelines.env import load_env

# Eager: load .env into os.environ before any adapter reads keys. Existing
# env vars take precedence over .env (D6 — CI / shell exports always win).
load_env()

from data_pipelines.dispatch import FetchMeta, fetch, fetch_with_meta
from data_pipelines.errors import (
    AllProvidersFailed,
    EmptyPayload,
    MissingAPIKey,
    ProviderError,
    SchemaMismatch,
    UnknownDomain,
)

__all__ = [
    "AllProvidersFailed",
    "EmptyPayload",
    "FetchMeta",
    "MissingAPIKey",
    "ProviderError",
    "SchemaMismatch",
    "UnknownDomain",
    "fetch",
    "fetch_with_meta",
]
