"""nse_equities per-domain config (V1_IMPLEMENTATION_PLAN.md §Configuration)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NSEEquitiesConfig:
    # Dispatch
    default_universe: str = "nifty50"
    # All three NSE adapters are range-aware (accept from_date/to_date), so
    # unlike us_equities there's no seed-vs-update split. The chain stays the
    # same regardless of gap size or cache state.
    chain_order: tuple[str, ...] = ("jugaad", "nselib", "yfinance")

    # Retry policy (shared across all three NSE adapters via call_with_retry)
    retry_max_retries: int = 3
    retry_base_delay_sec: float = 1.0
    retry_max_delay_sec: float = 30.0
    retry_jitter: bool = True

    # Library timeouts (best-effort — neither jugaad nor nselib exposes a
    # timeout knob through their public API in current versions, so these are
    # honored on yfinance and ignored elsewhere unless we monkey-patch).
    jugaad_timeout_sec: float = 30.0
    nselib_timeout_sec: float = 30.0
    yfinance_enabled: bool = True

    # NIFTY: alias → yahoo finance index symbol. Keep small; expand only when
    # a downstream consumer asks. nselib + jugaad use upstream NSE names
    # resolved via registry.NIFTY_INDEX_SLUGS — kept separate so each provider
    # can have its own naming scheme without one having to match the other.
    yfinance_index_slugs: dict[str, str] = field(default_factory=lambda: {
        "50": "^NSEI",
        "BANK": "^NSEBANK",
    })

    def __post_init__(self):
        if self.retry_max_retries < 0:
            raise ValueError("retry_max_retries must be >= 0")
        if self.retry_base_delay_sec < 0:
            raise ValueError("retry_base_delay_sec must be >= 0")
        if self.retry_max_delay_sec < self.retry_base_delay_sec:
            raise ValueError("retry_max_delay_sec must be >= retry_base_delay_sec")
        if self.jugaad_timeout_sec <= 0 or self.nselib_timeout_sec <= 0:
            raise ValueError("timeouts must be > 0")
        if not self.chain_order:
            raise ValueError("chain_order must contain at least one adapter name")
