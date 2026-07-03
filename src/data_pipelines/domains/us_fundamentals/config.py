"""us_fundamentals per-domain config.

Three providers, three politeness postures:

- **macrotrends** (primary) — scraped HTML pages; throttled with a minimum
  interval + jitter between requests and a browser User-Agent. Bulk seeds are
  sequential by design (the throttle is the rate limit, not thread count).
- **SEC EDGAR** (secondary) — official API; SEC fair-access policy requires a
  User-Agent that identifies the caller with contact info (env-overridable)
  and ≤10 req/s, which the shared retry/backoff respects trivially.
- **yfinance** (tertiary) — library-managed transport; no knobs here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class USFundamentalsConfig:
    default_universe: str = "all"

    # --- macrotrends (primary) ---
    macrotrends_base_url: str = "https://www.macrotrends.net"
    # Minimum seconds between any two macrotrends requests, plus uniform
    # jitter in [0, jitter_sec]. Live-measured (2026-07-03 sp500 seed):
    # macrotrends 429s at a sustained rate faster than ~1 req/5-6 s; at
    # 1.5 s the retry/backoff machinery absorbed the 429s (self-healing but
    # ~3 attempts/request). 4.5 s + jitter clears the limit with clean 200s
    # at the same effective throughput.
    macrotrends_min_interval_sec: float = 4.5
    macrotrends_jitter_sec: float = 1.0
    # A browser UA: macrotrends serves the same HTML to browsers; a bare
    # python UA invites blocking.
    browser_user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) "
        "Gecko/20100101 Firefox/127.0"
    )

    # --- SEC EDGAR (secondary) ---
    edgar_base_url: str = "https://data.sec.gov"
    sec_www_base_url: str = "https://www.sec.gov"
    # SEC fair-access policy wants a contact in the UA. Overridable via env;
    # the default is the repo owner's public git-commit email (no new
    # exposure — it is already in every commit header).
    edgar_contact_env: str = "SEC_EDGAR_CONTACT"
    edgar_contact_default: str = "shayan.roychoudhury@gmail.com"

    # --- shared transport ---
    timeout_sec: float = 30.0
    retry_max_retries: int = 3
    retry_base_delay_sec: float = 1.0
    retry_max_delay_sec: float = 30.0
    retry_jitter: bool = True

    # --- calendar ---
    # Grid dates are demanded only once this many days old (10-Q deadline for
    # large/accelerated filers is 40-45 days; see calendar.py).
    reporting_lag_days: int = 45

    def __post_init__(self):
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        if self.macrotrends_min_interval_sec < 0:
            raise ValueError("macrotrends_min_interval_sec must be >= 0")
        if self.macrotrends_jitter_sec < 0:
            raise ValueError("macrotrends_jitter_sec must be >= 0")
        if self.retry_max_retries < 0:
            raise ValueError("retry_max_retries must be >= 0")
        if self.retry_base_delay_sec < 0:
            raise ValueError("retry_base_delay_sec must be >= 0")
        if self.retry_max_delay_sec < self.retry_base_delay_sec:
            raise ValueError("retry_max_delay_sec must be >= retry_base_delay_sec")
        if self.reporting_lag_days < 0:
            raise ValueError("reporting_lag_days must be >= 0")
