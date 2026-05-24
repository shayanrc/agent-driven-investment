"""Retry primitive tests — backoff timing, jitter spread, non-retryable propagation."""

from __future__ import annotations

import random

import pytest

from data_pipelines.errors import (
    EmptyPayload,
    MissingAPIKey,
    ProviderError,
    SchemaMismatch,
)
from data_pipelines.retry import RetryPolicy, _compute_delay, call_with_retry


# ---- RetryPolicy validation ----


def test_retry_policy_rejects_negative_max_retries():
    with pytest.raises(ValueError, match="max_retries"):
        RetryPolicy(max_retries=-1)


def test_retry_policy_rejects_negative_base_delay():
    with pytest.raises(ValueError, match="base_delay"):
        RetryPolicy(base_delay_sec=-0.1)


def test_retry_policy_rejects_max_lt_base():
    with pytest.raises(ValueError, match="max_delay"):
        RetryPolicy(base_delay_sec=10.0, max_delay_sec=1.0)


def test_retry_policy_rejects_empty_retry_on():
    with pytest.raises(ValueError, match="retry_on"):
        RetryPolicy(retry_on=())


# ---- _compute_delay backoff shape ----


def test_compute_delay_no_jitter_is_exponential():
    p = RetryPolicy(base_delay_sec=1.0, max_delay_sec=60.0, jitter=False)
    assert _compute_delay(1, p) == 1.0   # base * 2^0
    assert _compute_delay(2, p) == 2.0   # base * 2^1
    assert _compute_delay(3, p) == 4.0
    assert _compute_delay(4, p) == 8.0


def test_compute_delay_caps_at_max():
    p = RetryPolicy(base_delay_sec=1.0, max_delay_sec=5.0, jitter=False)
    assert _compute_delay(10, p) == 5.0   # would be 512, capped


def test_compute_delay_with_jitter_within_band(monkeypatch):
    p = RetryPolicy(base_delay_sec=2.0, max_delay_sec=60.0, jitter=True)
    # uniform(0, base) → record draws, check shape
    draws = []

    def fake_uniform(a, b):
        draws.append((a, b))
        return (a + b) / 2

    monkeypatch.setattr(random, "uniform", fake_uniform)
    d = _compute_delay(1, p)
    # base=2 * 2^0 = 2 + uniform(0, 2)=1 → 3
    assert d == 3.0
    assert draws == [(0, 2.0)]


def test_compute_delay_jitter_spread_is_bounded():
    """Jitter must always stay in [0, base_delay_sec]."""
    p = RetryPolicy(base_delay_sec=0.5, max_delay_sec=60.0, jitter=True)
    samples = [_compute_delay(2, p) for _ in range(200)]
    base_exp = 0.5 * 2  # = 1.0
    assert all(base_exp <= s <= base_exp + 0.5 for s in samples)


# ---- call_with_retry success/failure paths ----


def test_call_returns_immediately_on_success():
    calls = []
    sleeps = []
    result = call_with_retry(
        lambda: (calls.append(1), "ok")[1],
        RetryPolicy(max_retries=3, base_delay_sec=0.001),
        provider="p", identifier="id",
        sleep=sleeps.append,
    )
    assert result == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_call_retries_until_success():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ProviderError("p", "id", "transient")
        return "done"

    sleeps = []
    result = call_with_retry(
        flaky,
        RetryPolicy(max_retries=3, base_delay_sec=0.001, jitter=False),
        provider="p", identifier="id",
        sleep=sleeps.append,
    )
    assert result == "done"
    assert len(attempts) == 3
    assert len(sleeps) == 2   # slept between attempts 1→2 and 2→3


def test_call_raises_after_max_retries():
    attempts = []

    def always_fail():
        attempts.append(1)
        raise ProviderError("p", "id", "always")

    sleeps = []
    with pytest.raises(ProviderError, match="always"):
        call_with_retry(
            always_fail,
            RetryPolicy(max_retries=2, base_delay_sec=0.001, jitter=False),
            provider="p", identifier="id",
            sleep=sleeps.append,
        )
    assert len(attempts) == 3   # initial + 2 retries
    assert len(sleeps) == 2


def test_max_retries_zero_means_one_attempt():
    attempts = []

    def always_fail():
        attempts.append(1)
        raise ProviderError("p", "id", "fail")

    sleeps = []
    with pytest.raises(ProviderError):
        call_with_retry(
            always_fail,
            RetryPolicy(max_retries=0, base_delay_sec=0.001),
            provider="p", identifier="id",
            sleep=sleeps.append,
        )
    assert len(attempts) == 1
    assert sleeps == []


# ---- non-retryable propagation ----


def test_empty_payload_propagates_immediately():
    attempts = []

    def empty():
        attempts.append(1)
        raise EmptyPayload("p", "id")

    sleeps = []
    with pytest.raises(EmptyPayload):
        call_with_retry(
            empty,
            RetryPolicy(max_retries=3, base_delay_sec=0.001),
            provider="p", identifier="id",
            sleep=sleeps.append,
        )
    assert len(attempts) == 1   # not retried even though ProviderError is in retry_on
    assert sleeps == []


def test_missing_api_key_propagates_immediately():
    attempts = []

    def no_key():
        attempts.append(1)
        raise MissingAPIKey("p", "FAKE_KEY_ENV")

    with pytest.raises(MissingAPIKey):
        call_with_retry(
            no_key,
            RetryPolicy(max_retries=3, base_delay_sec=0.001),
            provider="p", identifier="id",
        )
    assert len(attempts) == 1


def test_schema_mismatch_propagates_immediately():
    attempts = []

    def bad_schema():
        attempts.append(1)
        raise SchemaMismatch("p", "id", "missing column")

    with pytest.raises(SchemaMismatch):
        call_with_retry(
            bad_schema,
            RetryPolicy(max_retries=3, base_delay_sec=0.001),
            provider="p", identifier="id",
        )
    assert len(attempts) == 1


def test_unlisted_exception_propagates_without_retry():
    """A ValueError isn't in retry_on, so it should bubble up immediately."""
    attempts = []

    def boom():
        attempts.append(1)
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        call_with_retry(
            boom,
            RetryPolicy(max_retries=3, base_delay_sec=0.001),
            provider="p", identifier="id",
        )
    assert len(attempts) == 1


# ---- custom retry_on tuple ----


def test_custom_retry_on_accepts_only_listed():
    class TransientNetwork(Exception):
        pass

    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 2:
            raise TransientNetwork("blip")
        return "ok"

    sleeps = []
    result = call_with_retry(
        flaky,
        RetryPolicy(
            max_retries=3,
            base_delay_sec=0.001,
            retry_on=(TransientNetwork,),
            do_not_retry=(),
        ),
        provider="p", identifier="id",
        sleep=sleeps.append,
    )
    assert result == "ok"
    assert len(attempts) == 2
    assert len(sleeps) == 1


def test_sleep_delays_use_backoff_sequence():
    """Verify the actual sleep durations follow the documented backoff curve."""
    attempts = []
    sleeps: list[float] = []

    def always_fail():
        attempts.append(1)
        raise ProviderError("p", "id", "fail")

    with pytest.raises(ProviderError):
        call_with_retry(
            always_fail,
            RetryPolicy(
                max_retries=3,
                base_delay_sec=1.0,
                max_delay_sec=60.0,
                jitter=False,
            ),
            provider="p", identifier="id",
            sleep=sleeps.append,
        )
    # 4 attempts → 3 sleeps, delays 1, 2, 4
    assert sleeps == [1.0, 2.0, 4.0]
