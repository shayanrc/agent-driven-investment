"""in_fundamentals per-domain config.

One provider, two politeness postures:

- The **main-site API** (``www.nseindia.com/api/...``) is bot-guarded: it
  needs a browser UA and a session cookie from a warmup request (which may
  itself 403 — its Set-Cookie headers still count). Throttled conservatively.
- The **archives host** (``nsearchives.nseindia.com``) serves static XBRL
  files and tolerates a faster cadence; it is the known-reliable NSE host
  from this machine (`[[project-nse-data-quirks]]`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InFundamentalsConfig:
    default_universe: str = "nifty500"

    # --- NSE endpoints ---
    nse_base_url: str = "https://www.nseindia.com"
    results_api_path: str = "/api/corporates-financial-results"
    # Referer the API expects (part of the bot-guard heuristics).
    results_referer: str = (
        "https://www.nseindia.com/companies-listing/"
        "corporate-filings-financial-results"
    )

    # Minimum seconds between main-site API requests / archive downloads,
    # plus uniform jitter in [0, jitter].
    api_min_interval_sec: float = 2.5
    api_jitter_sec: float = 0.5
    archive_min_interval_sec: float = 0.8
    archive_jitter_sec: float = 0.4

    browser_user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) "
        "Gecko/20100101 Firefox/127.0"
    )

    # --- shared transport ---
    timeout_sec: float = 30.0
    retry_max_retries: int = 3
    retry_base_delay_sec: float = 2.0
    retry_max_delay_sec: float = 60.0
    retry_jitter: bool = True

    # --- calendar ---
    # SEBI LODR results deadlines: 45 d after quarter end (Q1-Q3), 60 d for
    # the audited Q4/annual — one knob covers both, so grid dates are demanded
    # only once 60 days old.
    reporting_lag_days: int = 60

    # --- history floor ---
    # XBRL filings are uniform from the Ind-AS era; older filings drift
    # through taxonomy generations. Records with a quarter end before this
    # year are skipped at parse (raw metadata still lands for the audit
    # trail).
    min_year: int = 2016

    def __post_init__(self):
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")
        for f in ("api_min_interval_sec", "api_jitter_sec",
                  "archive_min_interval_sec", "archive_jitter_sec",
                  "retry_base_delay_sec"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be >= 0")
        if self.retry_max_retries < 0:
            raise ValueError("retry_max_retries must be >= 0")
        if self.retry_max_delay_sec < self.retry_base_delay_sec:
            raise ValueError("retry_max_delay_sec must be >= retry_base_delay_sec")
        if self.reporting_lag_days < 0:
            raise ValueError("reporting_lag_days must be >= 0")
