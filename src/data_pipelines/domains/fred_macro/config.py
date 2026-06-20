"""fred_macro per-domain config.

Transport is *auto* (per the approved plan): the keyless ``fredgraph.csv``
download endpoint by default, switching to the official keyed
``api.stlouisfed.org`` JSON API when ``FRED_API_KEY`` is set in the
environment. Both endpoints are configured here; the adapter chooses at fetch
time based on key presence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FredMacroConfig:
    default_universe: str = "macro"

    # Transport (auto: keyed JSON API when the key env var is set, else
    # keyless CSV). The key is read from env at fetch time only and is never
    # logged, embedded in a raw filename, or echoed in an error (D6).
    fred_api_key_env: str = "FRED_API_KEY"
    fred_api_base_url: str = "https://api.stlouisfed.org/fred"
    fredgraph_base_url: str = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    timeout_sec: float = 30.0

    # Retry policy (shared call_with_retry primitive — same as the NSE chain).
    retry_max_retries: int = 3
    retry_base_delay_sec: float = 1.0
    retry_max_delay_sec: float = 30.0
    retry_jitter: bool = True

    def __post_init__(self):
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        if self.retry_max_retries < 0:
            raise ValueError("retry_max_retries must be >= 0")
        if self.retry_base_delay_sec < 0:
            raise ValueError("retry_base_delay_sec must be >= 0")
        if self.retry_max_delay_sec < self.retry_base_delay_sec:
            raise ValueError("retry_max_delay_sec must be >= retry_base_delay_sec")
        if not self.fred_api_key_env:
            raise ValueError("fred_api_key_env must be a non-empty env-var name")
