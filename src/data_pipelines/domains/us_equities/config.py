"""us_equities per-domain config (V1_IMPLEMENTATION_PLAN.md §Configuration)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class USEquitiesConfig:
    # Dispatch
    big_gap_threshold_days: int = 90       # trading days; gap > this → use Stooq seed
    default_universe: str = "sp500"

    # Adapter chain (names; resolved to instances at domain wire-up)
    chain_seed: str = "stooq"
    chain_update: tuple[str, ...] = ("tiingo", "yfinance")

    # Provider settings
    tiingo_api_key_env: str = "TIINGO_API_KEY"
    tiingo_base_url: str = "https://api.tiingo.com"
    tiingo_timeout_sec: float = 10.0
    tiingo_max_retries: int = 3

    stooq_base_url: str = "https://stooq.com/q/d/l/"
    stooq_timeout_sec: float = 30.0
    # Stooq's CSV endpoint is gated for some IPs (probably high-volume / IP
    # reputation based, not universal — see investigation notes). When gated,
    # the response is a help page redirecting to the apikey registration:
    #   https://stooq.com/q/d/?s=<sym>.us&get_apikey
    # The documented URL param name (per Stooq's own help text) is 'apikey':
    #   https://stooq.com/q/d/l/?s=aapl.us&i=d&apikey=XXX...
    stooq_api_key_env: str = "STOOQ_API_KEY"
    stooq_api_key_param: str = "apikey"

    yfinance_enabled: bool = True

    # Shared retry/backoff (data_pipelines.retry.call_with_retry) for the
    # yfinance fallback tier — same knobs as the nse_equities and
    # us_fundamentals domains.
    retry_max_retries: int = 3
    retry_base_delay_sec: float = 1.0
    retry_max_delay_sec: float = 30.0
    retry_jitter: bool = True

    # Index symbol → Stooq URL slug. Stooq uses lowercase '^spx' (no '.us').
    stooq_index_slugs: dict[str, str] = field(default_factory=lambda: {
        "^SPX": "^spx",
        "^NDX": "^ndx",
        "^DJI": "^dji",
        "^RUT": "^rut",
    })

    def __post_init__(self):
        if self.big_gap_threshold_days < 1:
            raise ValueError("big_gap_threshold_days must be >= 1")
        if self.tiingo_max_retries < 0:
            raise ValueError("tiingo_max_retries must be >= 0")
        if self.retry_max_retries < 0:
            raise ValueError("retry_max_retries must be >= 0")
        if self.retry_base_delay_sec < 0:
            raise ValueError("retry_base_delay_sec must be >= 0")
        if self.retry_max_delay_sec < self.retry_base_delay_sec:
            raise ValueError("retry_max_delay_sec must be >= retry_base_delay_sec")
        if not self.tiingo_api_key_env:
            raise ValueError("tiingo_api_key_env must be a non-empty env-var name")
        if not self.stooq_api_key_env:
            raise ValueError("stooq_api_key_env must be a non-empty env-var name")
        if not self.stooq_api_key_param:
            raise ValueError("stooq_api_key_param must be a non-empty URL param name")
        if not self.chain_update:
            raise ValueError("chain_update must contain at least one adapter name")
        if self.tiingo_timeout_sec <= 0 or self.stooq_timeout_sec <= 0:
            raise ValueError("timeouts must be > 0")
