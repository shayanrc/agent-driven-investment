# data_pipelines — Goal

**This is the generic time-series ingestion module for the project.** It handles fetch, process, cache, on-demand load, and source-sync for any time-series data — US equities, non-US equities, interest rates, macro series, commodities, FX, alt data — independent of source or type. Each kind of data is a **domain** plugged into the same framework: defining its own schema, identifier convention, universe, adapter set, and calendar. The framework provides everything else (raw/processed cache layout, atomic writes, gap detection, provider tiering and fallback, reprocess workflow, public `fetch()` dispatch).

**v1 ships exactly one domain: US equities** (NYSE + NASDAQ, S&P 500 universe, Stooq + Tiingo + yfinance adapters, daily OHLCV schema). Future domains — NSE equities, FRED macro, commodities, etc. — plug in alongside without changing the framework. The right framework abstraction crystallizes when domain #2 is built; v1's job is to ship US equities cleanly with the right seams that make adding domain #2 a plug-in rather than a refactor.

This document states what `data_pipelines` is optimizing for and what trade-offs are unacceptable. Read it before editing any file under `src/data_pipelines/`, `tests/data_pipelines/`, `configs/data_pipelines/`, or `docs/data_pipelines/`.

For *how* it works (architecture, stages, constraints), see `V1_IMPLEMENTATION_PLAN.md`. This file is the *why* and *what success looks like*.

---

## What this module is optimizing for

Provide **reliable, fast, source-agnostic access to any historical time-series data** the rest of this repo needs, with one defining rule:

> **The consumer should never have to know which provider supplied the data, and should not have to learn a new API for each domain.** Schema, units, adjustment semantics, date handling — all canonical *per domain*. Provider selection, caching, fallbacks, rate-limit handling, and source-sync — all hidden behind a single `fetch()` call that takes a domain-qualified identifier.

The module exists so that `analog_mc` (US-equity OHLCV today; potentially macro or rates inputs later) and every future data-consuming module can call one function and get back a DataFrame in a known-per-domain shape, without taking on a coupling to any specific data vendor or to the details of a particular data type.

## What success looks like

A consumer module's data-access code should look like this and nothing more:

```python
from data_pipelines import fetch
df = fetch("NYSE:AAPL", start="2010-01-01", end="2026-05-15", frequency="daily")
# Later, when other domains land:
# df = fetch("FRED:DGS10", start="2010-01-01", end="2026-05-15")     # macro
# df = fetch("NSE:RELIANCE", start="2010-01-01", end="2026-05-15")   # India equities
```

The identifier prefix (`NYSE:`, `FRED:`, `NSE:`, ...) is what routes the call to the right domain. Within a domain, the call must satisfy all of the following:

- **Canonical per-domain schema** — identical columns and dtypes regardless of which adapter served the request *within that domain*. For US equities: `date` (datetime64), `open`, `high`, `low`, `close`, `adj_close` (float64), `volume` (int64). No conditional column presence within a domain. No per-source flags the consumer has to handle. Different domains have different schemas — `FRED:DGS10` returns `(date, value)`, not OHLCV — and consumers know which shape to expect because they know which domain they're calling.
- **Cache-first, two-layer** — repeated calls for overlapping ranges hit local storage, not external APIs. The cache is split into `data/raw/` (immutable per-provider downloads, source-native format) and `data/processed/` (canonical-schema parquet served to consumers). `fetch()` reads `processed/` first, identifies gaps, calls providers to fill them, lands the raw bytes under `raw/`, normalizes to canonical, and merges into `processed/`. Raw is the audit trail and the reprocess source — if normalization logic changes, we re-derive `processed/` from `raw/` without re-paying for API calls.
- **Deterministic** — same call + same cache state → bit-identical result. No silent reordering, no nondeterministic provider dispatch.
- **Honest about failure** — provider outages, missing tickers, schema mismatches surface as explicit errors with provider context. Never silently return partial or stale data without a flag.
- **Adjustment semantics documented and enforced** — `adj_close` always means split-and-dividend adjusted, regardless of source. Sources that don't natively provide this are either reconciled or rejected for that column.
- **API keys never leak** — keys load from env vars or a gitignored secrets file. Never logged, never committed, never embedded in cache filenames or error messages.

## Provider strategy (per domain)

The framework supports tiered seed → update → fallback chains per domain. Each domain declares its own chain in its config. The chain *shape* is generic (cold-start vs incremental vs last-resort); the *providers* are domain-specific.

**US equities chain (v1):**

| Tier | Provider | Role | When invoked |
|---|---|---|---|
| Seed | **Stooq** | Bulk cold-start, deep backfill | Empty cache, OR missing range > `BIG_GAP_THRESHOLD` (default 90 trading days) |
| Update | **Tiingo** | Routine incremental updates, small gaps | Default for all non-seed fetches; requires API key |
| Fallback | **yfinance** | Per-ticker last resort | Only when Tiingo fails on a specific ticker (HTTP error, empty payload, schema mismatch) |

**Why this US-equities tiering:**

- Stooq is free and has multi-decade history — perfect for bulk seed. As of v1 smoke testing (2026-05-23) Stooq now requires a free API key (captcha-only registration at https://stooq.com/q/d/?get_apikey) for its CSV endpoint; read from env var `STOOQ_API_KEY`. Its `close` is split-adjusted only (not dividend-adjusted), so it's *wrong* as a daily update source for total-return calculations.
- Tiingo is reliable, has clean dividend-and-split-adjusted closes, well-documented REST, and a generous free tier (500 req/day). Right for routine updates.
- yfinance is known-unreliable (frequent endpoint breakage, rate-limit instability, occasionally returns adjusted-as-close), so it earns its slot only as a third-tier emergency fallback.

The `BIG_GAP_THRESHOLD` exists because under it, Tiingo can fill a gap in one cheap per-ticker call; above it, you're burning quota for hundreds of tickers when one Stooq call would do the job.

Future domains will declare their own chains — e.g., FRED macro will likely be single-source (FRED API only, no fallback) with a different threshold semantic since FRED series update on per-series schedules, not daily.

## What this module is *not*

- **Not domain-restricted in principle, just in current implementation.** The architecture supports any time-series domain. v1 ships only the US-equities domain; that's a scoping decision, not a structural limit. New domains (NSE, FRED, commodities, ...) plug in here, not in sibling modules.
- **Not a trading system** — no order routing, execution, broker APIs.
- **Not real-time / intraday in v1** — daily-frequency historical data only. Intraday is a deferred concern; the schema-primitive design should not preclude it.
- **Not a backtester** — produces data, doesn't run strategies on it.
- **Not a fundamentals normalizer** — earnings, filings, balance sheets are a separate concern (would likely be a different domain in this module rather than a separate module, but deferred).
- **Not a forecaster** — doesn't predict anything. `analog_mc` and future forecasting modules consume from here.
- **Not a pre-extracted framework.** v1 builds US equities with the right module boundaries (schema, adapters, universe, registry isolated per-domain), but does not crystallize a `core/` framework abstraction until domain #2 actually needs it. Three similar lines beat one premature abstraction; the right time to factor out shared code is when domain #2 is being added and the shape of the abstraction is informed by two real use cases, not one.

## Scope for v1

v1 = framework + exactly one domain.

**Framework (generic, domain-agnostic):**
- Schema definition primitives — declare a domain's columns + dtypes + validators in one place.
- Two-layer cache: `data/raw/<provider>/...` (immutable downloads, source-native format) + `data/processed/<domain>/...` (canonical-schema parquet served to consumers).
- Atomic writes (D2), raw immutability + reprocess determinism (D8), gap detection, atomic merge.
- Adapter ABC + tiered fallback chain (seed → update → fallback), with semantics driven by per-domain config.
- Public `fetch(identifier, start, end, ...) → DataFrame` that dispatches to the right domain by identifier prefix.
- CLI scaffolding: `fetch`, `seed`, `reprocess`, `health`.

**Domain #1: US equities.**
- Identifier scheme: `NYSE:AAPL`, `NASDAQ:MSFT`, `INDEX:^SPX` (`INDEX:^SPX`, `INDEX:^NDX`, `INDEX:^DJI`, `INDEX:^RUT` for the v1 supported index set).
- Universe: **S&P 500 constituents** + the supported indices, pinned in `configs/data_pipelines/domains/us_equities/universe_sp500.yaml`. `fetch()` doesn't hard-restrict to this universe — out-of-universe tickers fetch with a warning — but bulk-seed, integration tests, and the agent-tool surface treat S&P 500 as the supported set.
- Schema: daily OHLCV + dividend-and-split-adjusted close.
- Adapter chain: **Stooq → Tiingo → yfinance** wired end-to-end.
- Calendar: NYSE trading calendar.
- CLI: `python -m data_pipelines fetch NYSE:AAPL --start 2010-01-01`, plus `seed --domain us_equities --universe sp500`.
- Tests covering the schema invariant per adapter, raw→processed reproducibility, cache hit/miss logic, gap-detection threshold, and provider fallback chain.

**Explicitly deferred to later domains / versions:**
- NSE (India) — shipped in v1.7; LSE (UK), TSE (Japan), and other non-US equity domains still deferred.
- FRED macroeconomic series, US Treasury rates, BEA series.
- Commodity prices, FX rates, crypto.
- Intraday bars (any domain).
- Corporate actions as separate tables (splits, dividends, mergers — adj_close captures what consumers need for v1).
- Streaming / WebSocket connections.
- Multi-currency / cross-listings.
- A Streamlit dashboard (probably not needed; the CLI + agent-tool surface should suffice).

## Eventual deployment shape

Currently the planned surface is: a Python `fetch()` function, a CLI, and a parquet cache. These reflect the **build-and-validate phase**.

Intended end state: an **agent-callable tool/skill** that takes a ticker (or symbol set), date range, and frequency, and returns the data — JSON-serializable summary metadata + a path to the cached DataFrame, or the DataFrame inline for small requests. The agent (likely an `analog_mc` forecasting agent) calls this without knowing or caring which provider served the bytes.

When designing new APIs in this module, prefer shapes that wrap cleanly as a tool call later:
- Clean function signature, all args explicit, no positional ambiguity.
- JSON-serializable return metadata (ticker, range covered, source(s) used, rows, cache hit/miss).
- Errors raise typed exceptions with provider context, not bare strings.
- No hidden CLI state or environment-dependent behavior beyond the API key.

## How to apply this when working on the module

- **Schema invariance is non-negotiable.** Every adapter PR ships with a test that asserts the canonical schema on the processed-layer output. A schema regression in any adapter is a release blocker.
- **Raw layer is immutable.** Once an adapter writes a raw download, that file is never modified. Reprocessing means reading raw and writing fresh `processed/`. Tests must cover the round-trip: same raw bytes → same processed parquet.
- **Cache writes must be atomic.** Partial writes during a provider failure are the canonical silent-failure mode for this kind of module. Both raw and processed writes use temp-file + fsync + rename.
- **When tempted to add a provider**, ask: does the tiering strategy already cover this need? If yes, don't add. If no, where does it slot — seed, update, or fallback? Don't introduce a fourth tier.
- **When tempted to add new data types** (rates, macro, etc.), check the deferred list above. v1 is equities only — adding a second asset class to v1 means adding an asset-class dimension to the canonical schema before any provider has been validated, which is premature abstraction.
- **When asked "is the cache stale?"** — the answer comes from the gap-detection logic in `cache.py`, not from a heuristic in calling code. Consumers don't reason about freshness; they just call `fetch()`.
- **API keys** load only from env (`TIINGO_API_KEY`) or a gitignored secrets file. Never log a key, never embed it in a filename, never echo it in an error. Tests run without a key; integration tests gate on key presence and skip otherwise.

## Implementation discipline

The how-it-works specification is `docs/data_pipelines/V1_IMPLEMENTATION_PLAN.md`. The stages and the correctness constraints (D1–D8) defined there are non-negotiable. Don't silently change architectural decisions — surface the deviation and ask first.
