"""Shared retry primitive (exponential backoff + jitter).

Introduced in v1.7 for the NSE adapter chain (jugaad → nselib → yfinance), but
deliberately lifted to the framework root so the Tiingo adapter can adopt it
later. Lives here — not under any domain/ — because adapters across domains
should share one well-tested retry implementation.

Jitter rationale: bulk seeding N tickers without jitter produces synchronized
retry storms when many tickers hit a transient 429 in the same window. Adding
uniform[0, base_delay] jitter spreads the second-attempt fan-out.

Non-retryable exceptions (EmptyPayload, MissingAPIKey, SchemaMismatch) propagate
immediately — retrying them is wasted I/O and pollutes logs. The do_not_retry
tuple takes precedence over retry_on: if an exception matches both, it's NOT
retried.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from data_pipelines.errors import (
    EmptyPayload,
    MissingAPIKey,
    ProviderError,
    SchemaMismatch,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for call_with_retry.

    max_retries is retries AFTER the initial attempt — so 3 means up to 4 total
    calls. Setting max_retries=0 disables retry entirely (one attempt).
    """

    max_retries: int = 3
    base_delay_sec: float = 1.0
    max_delay_sec: float = 30.0
    jitter: bool = True
    retry_on: tuple[type[BaseException], ...] = (ProviderError,)
    do_not_retry: tuple[type[BaseException], ...] = field(
        default_factory=lambda: (EmptyPayload, MissingAPIKey, SchemaMismatch)
    )

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay_sec < 0:
            raise ValueError("base_delay_sec must be >= 0")
        if self.max_delay_sec < self.base_delay_sec:
            raise ValueError("max_delay_sec must be >= base_delay_sec")
        if not self.retry_on:
            raise ValueError("retry_on must contain at least one exception type")


def _compute_delay(attempt: int, policy: RetryPolicy) -> float:
    """Backoff delay before retry `attempt` (1-indexed: 1 = first retry).

    Formula per V1_IMPLEMENTATION_PLAN.md §retry-policy:
        min(base * 2^(attempt-1), max) + (uniform[0, base] if jitter else 0)

    Using attempt-1 in the exponent so the first retry sleeps `base`, second
    `2*base`, etc. — matches the principle-of-least-surprise for "exponential
    backoff" without needing a +0 magic offset.
    """
    exp = policy.base_delay_sec * (2 ** (attempt - 1))
    capped = min(exp, policy.max_delay_sec)
    if policy.jitter:
        return capped + random.uniform(0, policy.base_delay_sec)
    return capped


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    provider: str,
    identifier: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`, retrying on `policy.retry_on` up to `policy.max_retries` times.

    `sleep` is injectable for tests (pass a no-op or recorder).
    `provider` and `identifier` are used purely for log breadcrumbs; never
    embed secret material in either (D6).
    """
    last_exc: BaseException | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return fn()
        except policy.do_not_retry:
            # Explicit non-retryable wins over retry_on.
            raise
        except policy.retry_on as exc:
            last_exc = exc
            if attempt >= policy.max_retries:
                logger.warning(
                    "%s/%s: exhausted retries (%d/%d): %s",
                    provider, identifier, attempt, policy.max_retries, exc,
                )
                raise
            delay = _compute_delay(attempt + 1, policy)
            logger.info(
                "%s/%s: attempt %d/%d failed (%s); sleeping %.2fs",
                provider, identifier, attempt + 1, policy.max_retries + 1,
                exc, delay,
            )
            sleep(delay)
    # Unreachable; the loop either returns or raises.
    assert last_exc is not None
    raise last_exc
