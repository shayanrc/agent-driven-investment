# data_pipelines — V1 Implementation Plan

## Build status

- **v1 (us_equities domain):** shipped (228 tests, 520-ticker S&P 500 + indices seeded).
- **v1.7 (nse_equities domain):** shipped. Specification, implementation findings, and known limitations are in the "v1.7 — NSE equities (India) domain" section at the end of this document.

For the *why* (what success looks like, anti-goals, deployment intent), see `goal.md`. This document is the *how* — architecture, schema, constraints, stages.

The v1.x revision history below is one continuous series: revisions of the plan, scope changes, hardening fixes, and feature increments all live in the same sequence. v1.7 (NSE) is the next entry — its full specification is appended as a dedicated section at the end of this document.

## Revision history

- **v1.0** *(2026-05-22, drafted)* — Initial plan: Stooq seed + Tiingo update + yfinance fallback, parquet cache at `data/equities/<exchange>/<ticker>/daily.parquet`, canonical OHLCV schema, NYSE/NASDAQ daily equities only for v1.
- **v1.1** *(2026-05-23)* — Two-layer cache: `data/raw/` (immutable per-provider downloads) and `data/processed/` (canonical OHLCV). Adapters write raw and return parsed DataFrames; cache.py runs normalization → validation → atomic merge into processed. Added D8 (raw immutability + reprocess determinism). Universe pinned to S&P 500 constituents via `configs/data_pipelines/universe_sp500.yaml`; bulk-seed CLI defaults to it.
- **v1.2** *(2026-05-23)* — Scope explicitly tightened to **US equities only**. Non-US markets (NSE/India, LSE/UK, TSE/Japan, etc.) will be sibling modules following the same architectural pattern (canonical schema, raw/processed split, tiered providers, atomic writes) — replicated, not abstracted. Non-US-equity data types (interest rates, macro, commodities) also become sibling modules pending decision in open question 6. Added open questions about module rename (`data_pipelines` → `us_market_data`) and US-non-equity data-type policy.
- **v1.3** *(2026-05-23)* — **Architectural reframe: `data_pipelines` is the generic time-series ingestion module, NOT a US-equities-only module.** All data types and markets (US equities, NSE, FRED, commodities, FX, etc.) plug in here as **domains** rather than as sibling modules. The framework provides: schema primitives, raw/processed cache layout, atomic writes, gap detection, adapter ABC, tiered fallback chain logic, reprocess workflow, public `fetch()` dispatch. Each domain provides: schema definition, identifier conventions, universe, adapter set, calendar. v1 still ships only the US-equities domain — building it with clean per-domain seams so domain #2 is a plug-in, not a refactor. Open questions 6 (non-equity policy) and 7 (rename) resolved by this reframe and removed.
- **v1.4** *(2026-05-23)* — **Stooq added an API-key gate** (discovered during v1 online smoke test). The CSV endpoint at `https://stooq.com/q/d/l/?s=<sym>.us&i=d` now returns a "Get your apikey" help page unless `&k=<key>` is appended. Registration is free and captcha-only (`https://stooq.com/q/d/?get_apikey`). Adapter and config updated: `STOOQ_API_KEY` env var, pre-flight `MissingAPIKey` raise if unset, dedicated `_is_apikey_required` payload check raises `ProviderError` if Stooq returns the help page (e.g., invalid key). Stooq's tier role (seed, big-gap fills) is unchanged.
- **v1.5** *(2026-05-24)* — **Processed-layer storage swapped from parquet to SQLite.** Single global DB at `data/processed.db`. Per-domain tables are auto-created from `domain.schema`: `<domain>_data(ticker, <schema cols>...)` with composite PK `(ticker, <time_column>)` and `<domain>_meta(ticker PK, schema_version, row_count, range_start, range_end, last_fetch_utc, sources_json TEXT)`. Atomicity (D2) is delivered by SQLite's `BEGIN ... COMMIT`; deterministic reads (D3) by `ORDER BY <time_column>` plus dtype-coerced canonical-schema return; the meta row is still the commit marker (D2 read-side). Raw layer is unchanged. New `cache.py` API: `processed_db_path`, `read_processed`, `write_processed_atomic`, `list_cached_identifiers`, `purge_identifier`. One-shot migration script (`scripts/data_pipelines/migrate_parquet_to_sqlite.py`) ports any existing parquet+meta entries into the SQLite cache without re-paying API costs. Motivation: cross-ticker SQL queries (e.g., "all S&P 500 closes on 2025-01-15"), single-file backup/portability, simpler agent-tool surface.
- **v1.6** *(2026-05-24)* — **Document renamed `V1_IMPLEMENTATION_PLAN.md`.** Single doc for all v1.x revisions and feature increments of the data_pipelines module; future v2 work would get its own `V2_IMPLEMENTATION_PLAN.md`.
- **v1.7** *(2026-05-24, shipped)* — **NSE equities (India) added as domain #2.** Adapter chain `jugaad-data → nselib → yfinance(.NS)` with a new shared retry primitive (exponential backoff + jitter, 3 retries) at `src/data_pipelines/retry.py`. NIFTY 50 universe + index seeded end-to-end (51/51 identifiers cached). Framework code unchanged beyond the planned retry lift — the plug-in surface held up cleanly. Triggered open question 6's framework-extraction decision: hold the line on further extraction until domain #3 produces concrete evidence (see `adding_a_domain.md` v1.7 follow-up section). Findings inlined in the v1.7 section below.

---

## Purpose

This is an implementation specification for a **generic time-series ingestion module**. It handles fetch, process, cache, on-demand load, and source-sync for any time-series data — US equities, non-US equities, interest rates, macro series, commodities, FX, alt data. Each kind of data is a **domain** plugged into the same framework.

**v1 scope:** ship the framework and exactly one domain (US equities — NYSE + NASDAQ daily bars, S&P 500 universe). The framework should not over-crystallize: build it as much as US equities actually needs, with clean per-domain seams that make adding domain #2 a plug-in rather than a refactor.

The end-state public surface is a single function:

```python
fetch(identifier: str, start: str | date, end: str | date, frequency: str = "daily") -> pd.DataFrame
```

The identifier prefix routes the call to the right domain — `NYSE:AAPL` and `NASDAQ:MSFT` to the US-equities domain in v1; later, `NSE:RELIANCE` to NSE, `FRED:DGS10` to FRED macro, etc. Within a domain, the returned DataFrame conforms to the domain's canonical schema, sourced from the domain's tiered adapter chain, cached in the shared raw/processed two-layer store.

The plan is the output of a design conversation. Every decision documented here was made for a reason. **Do not silently change architectural decisions.** If implementation reveals a problem with a decision, surface it explicitly and ask before deviating.

---

## High-level architecture

Two layers in the design:

- **Framework** (domain-agnostic): identifier parsing + dispatch, two-layer cache (raw + processed), atomic writes, gap detection, adapter ABC, fallback chain logic, schema validation primitives, reprocess workflow, public `fetch()` and CLI.
- **Domains** (plug-ins): each domain provides its schema (using framework primitives), identifier conventions, universe loader, adapter set, calendar, and a per-domain config (chain ordering, thresholds, paths).

Two-layer cache: `raw/` is the immutable per-provider download history (any source format); `processed/<domain>/` is the canonical-schema consumer-facing parquet, schema-per-domain. `fetch()` always reads `processed/` first; only gaps cause adapter calls; adapter calls land in `raw/` and are then normalized + merged into `processed/`.

```
                           ┌─────────────────────────────────┐
              consumer ──▶ │  fetch(identifier, ...)         │  ← public API (also CLI)
                           └────────────────┬────────────────┘
                                            │
                                ┌───────────▼────────────┐
                                │   dispatch.py          │  ← parse identifier prefix,
                                │   (framework)          │     route to domain
                                └───────────┬────────────┘
                                            │
                                ┌───────────▼────────────┐
                                │  domain registry       │  ← look up domain by prefix
                                │  NYSE/NASDAQ/INDEX     │     (v1: us_equities only)
                                │  → us_equities domain  │
                                └───────────┬────────────┘
                                            │
                                ┌───────────▼────────────┐
                                │   cache.py             │  ← (1) read processed/<domain>/,
                                │   (framework)          │     (2) detect gaps (domain calendar),
                                │                        │     (3) orchestrate raw→processed,
                                │                        │     (4) atomic merge & return
                                └───────────┬────────────┘
                                            │ (only when gaps exist)
                                            ▼
                                ┌────────────────────────┐
                                │ domain chain dispatch  │  ← per-domain config: which adapter
                                │ seed / update / fb     │     for which gap size
                                └───────────┬────────────┘
                                            │
                           ┌────────────────┼────────────────┐
                           ▼                ▼                ▼
                     ┌──────────┐    ┌──────────┐    ┌──────────────┐
                     │ stooq.py │    │tiingo.py │    │ yfinance.py  │   ← us_equities adapters
                     │  (seed)  │    │ (update) │    │  (fallback)  │     (future domains:
                     └────┬─────┘    └────┬─────┘    └──────┬───────┘      different adapters)
                          │               │                  │
                          ▼               ▼                  ▼
                       writes raw via raw_store.py (framework, generic)
                       data/raw/<provider>/<domain>/<exchange>/<ticker>/<ts>_<range>.<ext>
                          │               │                  │
                          └───────────────┼──────────────────┘
                                          ▼
                                ┌────────────────────────┐
                                │ schema.normalize(df,   │  ← per-domain normalizer
                                │   domain=us_equities)  │
                                │ schema.validate(df,    │  ← per-domain validator
                                │   domain.schema)       │     (D1 + D8 enforcement)
                                └───────────┬────────────┘
                                            ▼
                                ┌────────────────────────┐
                                │  merge into            │
                                │  processed/<domain>/   │
                                │  (atomic write)        │
                                └────────────────────────┘
```

Dispatch logic (framework, in `dispatch.py` + `cache.py`):

```
domain = registry.resolve(identifier)             # "NYSE:AAPL" → us_equities domain
cached = read_processed(domain, identifier)
missing = domain.calendar.gap(requested_range, cached.range)
if not cached or len(missing) > domain.config.big_gap_threshold:
    adapter = domain.chain.seed                    # us_equities: stooq
    raw_path = adapter.fetch(identifier)
    df = adapter.parse(raw_path)
elif missing is non-empty:
    for tier in domain.chain.update_then_fallback: # us_equities: [tiingo, yfinance]
        try:
            raw_path = tier.fetch(identifier, missing.start, missing.end)
            df = tier.parse(raw_path)
            break
        except (ProviderError, SchemaMismatch, EmptyPayload):
            continue
    else:
        raise AllProvidersFailed(...)
normalized = domain.schema.normalize(df, source=adapter.name)
domain.schema.validate(normalized)                # raises on D1 violation
write_processed_atomic(merge_cache(cached, normalized, domain), meta_update)
return slice(read_processed(domain, identifier), requested_range)
```

Note that raw writes happen *before* parsing — if normalization fails on the bytes, the raw is still saved for inspection / reprocess, which is the whole point of the split. The framework knows nothing about OHLCV; that's the us_equities domain's schema.

---

## Module structure

```
src/data_pipelines/
│
├── __init__.py                  # exports: fetch, exceptions
├── __main__.py                  # CLI dispatcher (`python -m data_pipelines <sub>`)
│
│  # ---- framework (domain-agnostic) -----------------------------------
├── errors.py                    # generic exceptions: ProviderError, SchemaMismatch, etc.
├── schema.py                    # primitives: ColumnSpec, Schema, validate(df, schema), normalize()
├── raw_store.py                 # write_raw_atomic(domain, provider, identifier, payload, range) → Path
├── cache.py                     # processed read/write, gap detection (uses domain calendar),
│                                # atomic merge (uses domain adjustment-quality semantics)
├── dispatch.py                  # public fetch(); parses identifier, resolves domain, orchestrates
├── domain.py                    # Domain ABC + DomainRegistry; concrete domains register here
├── adapter.py                   # Adapter ABC: fetch / parse / health_check
│
│  # ---- domains (plug-ins) --------------------------------------------
└── domains/
    └── us_equities/             # v1's only domain
        ├── __init__.py          # registers the domain with DomainRegistry on import
        ├── schema.py            # OHLCV Schema instance using framework primitives
        ├── registry.py          # NYSE:AAPL / NASDAQ:MSFT / INDEX:^SPX parser
        ├── universe.py          # load_universe("sp500") → list[str]
        ├── calendar.py          # NYSE trading calendar (holidays, gap arithmetic)
        ├── config.py            # us_equities-specific config (chain ordering, threshold, paths)
        └── adapters/
            ├── __init__.py
            ├── stooq.py         # HTTP CSV download → raw/; parse → DataFrame
            ├── tiingo.py        # REST API client → raw/; parse → DataFrame
            └── yfinance.py      # yfinance wrapper → raw/; parse → DataFrame

tests/data_pipelines/
│
│  # ---- framework tests ----------------------------------------------
├── test_schema_primitives.py     # ColumnSpec / Schema / validate / normalize on a synthetic domain
├── test_cache.py                 # processed read/write/gap-detection/atomicity/merge (with mocked domain)
├── test_raw_store.py             # raw landing atomicity, immutability, naming convention
├── test_dispatch.py              # identifier parsing → domain resolution; framework-level fallback chain
├── test_domain_registry.py       # domain registration + lookup
│
│  # ---- domain tests (us_equities) ------------------------------------
├── domains/
│   └── us_equities/
│       ├── test_schema.py        # OHLCV schema invariants
│       ├── test_registry.py      # NYSE:AAPL / INDEX:^SPX parsing
│       ├── test_universe.py      # S&P 500 YAML load + membership queries
│       ├── test_calendar.py      # NYSE holidays, gap arithmetic
│       ├── test_reprocess.py     # raw → processed round-trip determinism (D8)
│       └── adapters/
│           ├── test_stooq.py     # recorded raw fixtures; one online smoke test
│           ├── test_tiingo.py    # recorded raw fixtures; one online smoke test (gated on key)
│           └── test_yfinance.py  # recorded raw fixtures; online smoke test
│
├── fixtures/                     # committed raw payload samples per (domain, provider)
│   └── us_equities/
│       ├── stooq/
│       ├── tiingo/
│       └── yfinance/
└── conftest.py                   # fixtures: synthetic OHLCV df, tmp data root, fake domain

configs/data_pipelines/
├── default.yaml                  # framework: data_root, raw_subdir, processed_subdir
└── domains/
    └── us_equities/
        ├── default.yaml          # us_equities config: chain ordering, threshold, provider settings
        └── universe_sp500.yaml   # S&P 500 ticker list (current membership; manual updates)

docs/data_pipelines/
├── goal.md                  # source of truth for what success means
└── V1_IMPLEMENTATION_PLAN.md   # this document
```

Top-level `data/` is the canonical storage root, shared across all domains. Layout is keyed on `<domain>` under both `raw/` and `processed/`:

```
data/
├── NASDAQ100.csv                                                    # legacy, kept for analog_mc compatibility
├── raw/                                                             # immutable per-provider downloads
│   └── <provider>/<domain>/<exchange>/<ticker>/
│       └── <UTC_timestamp>_<range>.<ext>                            # one file per fetch call
│
│   # v1 (us_equities domain only):
│   #   raw/stooq/us_equities/NYSE/AAPL/20260523T143022Z_1986-01-02_2026-05-22.csv
│   #   raw/tiingo/us_equities/NYSE/AAPL/20260523T143022Z_2026-05-20_2026-05-22.json
│   #   raw/yfinance/us_equities/NYSE/AAPL/20260523T143022Z_2026-05-20_2026-05-22.parquet
│   # future (e.g., FRED macro):
│   #   raw/fred/fred_macro/-/DGS10/20260523T143022Z_2010-01-01_2026-05-22.json
│
└── processed/                                                       # consumer-facing canonical layer
    └── <domain>/<exchange>/<ticker>/
        ├── daily.parquet                                            # canonical-schema (per domain)
        └── _meta.json                                               # see below
    
    # v1: processed/us_equities/NYSE/AAPL/daily.parquet
    # future: processed/fred_macro/-/DGS10/daily.parquet, processed/nse_equities/NSE/RELIANCE/daily.parquet
```

The `<exchange>` segment is whatever the domain uses to namespace identifiers within itself — `NYSE`/`NASDAQ`/`INDEX` for us_equities, `-` (placeholder) for domains without an exchange concept like FRED.

**Raw layer:**
- One file per fetch call, named `<UTC_timestamp>_<start>_<end>.<ext>` (e.g., `20260523T143022Z_2020-01-01_2026-05-22.json`). Timestamp + range in the filename lets you eyeball the directory and see what's there.
- Format is whatever the provider returns natively. Stooq → CSV, Tiingo → JSON, yfinance → parquet of the raw `Ticker.history()` DataFrame (yfinance's "raw" is already a DataFrame, so parquet preserves dtypes and metadata).
- **Never modified after write.** Reprocessing means reading raw and writing fresh `processed/`. This is D8.
- Gitignored. Audit trail and reprocess source only.

**Processed layer `_meta.json`:**

```json
{
  "schema_version": 1,
  "ticker": "AAPL",
  "exchange": "NYSE",
  "frequency": "daily",
  "row_count": 4123,
  "range": {"start": "2010-01-04", "end": "2026-05-22"},
  "last_fetch_utc": "2026-05-23T14:30:22Z",
  "sources": [
    {"provider": "stooq", "raw_file": "20260520T100012Z_1986-01-02_2026-05-19.csv",
     "covers": {"start": "1986-01-02", "end": "2026-05-19"}, "adjustment_quality": "split_only"},
    {"provider": "tiingo", "raw_file": "20260523T143022Z_2026-05-20_2026-05-22.json",
     "covers": {"start": "2026-05-20", "end": "2026-05-22"}, "adjustment_quality": "full"}
  ]
}
```

The `sources` array gives a per-fetch trail: which raw file contributed which date range to the current processed parquet. That trail is the audit surface for D4 (adjustment semantics) and the input for reprocessing.

---

## Canonical schema (per domain)

Schemas are defined per-domain using framework primitives (`schema.ColumnSpec`, `schema.Schema`). Each domain owns its schema; the framework validates against whatever the domain declares.

This section documents the **us_equities-domain** schema — daily OHLCV + adjusted close. Other domains will have other schemas (FRED macro will be `(date, value)`, not OHLCV; commodities may add contract-month metadata; etc.).

This is the **processed-layer** us_equities schema. Raw files keep their source-native format; normalization happens at the cache.py boundary using `domain.schema.normalize()`. Every adapter's `parse(raw_path)` returns a DataFrame in this exact shape:

| Column | dtype | Notes |
|---|---|---|
| `date` | `datetime64[ns]` | Trading-day timestamp at midnight UTC. Index OR column — pick one and document. |
| `open` | `float64` | Raw, unadjusted |
| `high` | `float64` | Raw, unadjusted |
| `low` | `float64` | Raw, unadjusted |
| `close` | `float64` | Raw, unadjusted |
| `adj_close` | `float64` | Split-AND-dividend adjusted |
| `volume` | `int64` | Shares traded |

**Recommendation:** `date` as a column, sorted ascending, NOT as the DataFrame index. Reason: parquet round-trips cleaner with date-as-column; consumers can `.set_index("date")` if they want.

**`adj_close` for Stooq:** Stooq's `close` is split-adjusted but NOT dividend-adjusted. The adapter must surface this honestly — either:
- (a) Set `adj_close = close` and tag `_meta.json` with `adjustment_quality: "split_only"`, OR
- (b) Refuse to populate `adj_close` and require Tiingo to fill it on a follow-up call.

**Pick (a)** for v1 — keeps the schema honest (no NaN columns) and the metadata flag warns consumers that doing total-return calculations on Stooq-only data is incorrect. When Tiingo subsequently updates the cache, it overwrites the `adj_close` column for any overlapping rows.

---

## Configuration

Two layers of config: a generic framework config (cache paths, behavior toggles) and per-domain configs (chain ordering, provider credentials, thresholds).

### Framework config

```python
@dataclass
class FrameworkConfig:
    # ---- Cache layout (shared across all domains) ----------------------
    data_root: str = "data"                   # parent of raw/ and processed/
    raw_subdir: str = "raw"                   # data_root/raw/<provider>/<domain>/...
    processed_subdir: str = "processed"       # data_root/processed/<domain>/...
    storage_format: str = "parquet"           # processed-layer format; only "parquet" in v1

    # ---- Generic behavior ----------------------------------------------
    default_frequency: str = "daily"
    fail_on_schema_mismatch: bool = True      # if False, log and skip; default = raise
    allow_partial_cache_returns: bool = False # if True, return whatever is cached if all providers fail
```

### Per-domain config (us_equities)

```python
@dataclass
class USEquitiesConfig:
    # ---- Dispatch ------------------------------------------------------
    big_gap_threshold_days: int = 90          # trading days; gap > this → use Stooq
    default_universe: str = "sp500"           # bulk-seed default; reads
                                              # configs/data_pipelines/domains/us_equities/universe_sp500.yaml

    # ---- Adapter chain (ordered: seed, then update tiers in order) ----
    chain_seed: str = "stooq"
    chain_update: list[str] = ("tiingo", "yfinance")  # tried in order on missing slices

    # ---- Providers -----------------------------------------------------
    tiingo_api_key_env: str = "TIINGO_API_KEY"   # name of env var, NOT the key itself
    tiingo_base_url: str = "https://api.tiingo.com"
    tiingo_timeout_sec: float = 10.0
    tiingo_max_retries: int = 3

    stooq_base_url: str = "https://stooq.com/q/d/l/"
    stooq_timeout_sec: float = 30.0           # bulk downloads are slower

    yfinance_enabled: bool = True             # third-tier fallback toggle
```

**Config invariants** (validate at construction):

- `big_gap_threshold_days >= 1`
- `storage_format == "parquet"` in v1 (CSV reserved for future)
- `tiingo_max_retries >= 0`
- `tiingo_api_key_env` is a non-empty string; the actual key is read from `os.environ[that_name]` at fetch time, not at config load.
- `chain_update` contains at least one adapter; every name resolves to a registered adapter for this domain.

**Future domains** will define their own per-domain config classes — FRED macro won't have `big_gap_threshold_days` (irrelevant since FRED is single-source) but will need `series_metadata_cache_ttl`, etc. The framework doesn't presume any specific knob set.

---

## Critical correctness constraints

These are non-negotiable. They are the silent-failure modes most likely to make this module quietly return wrong data — which downstream forecasting modules would then build on without knowing.

### D1. Schema invariance across providers (per domain)

Every adapter's returned DataFrame MUST match its domain's canonical schema exactly: same columns, same dtypes, same ordering, same adjustment semantics. A unit test per adapter must assert this on a real (or recorded-fixture) fetch.

**Why this is the most important constraint:** if Stooq returns `Volume` capitalized and Tiingo returns `volume`, the cache reads will silently break when the source flips between calls. If `adj_close` means "split-only" from Stooq and "split+dividend" from Tiingo within the same domain, total-return calculations downstream will produce subtly wrong numbers that pass plausibility checks but fail audits.

The framework's `schema.validate(df, domain.schema)` runs after every adapter call and BEFORE every cache write. Failure raises `SchemaMismatch` with the offending columns/dtypes.

Schema invariance applies *within a domain*. Different domains have different schemas — `fetch("NYSE:AAPL", ...)` returns OHLCV, `fetch("FRED:DGS10", ...)` would return `(date, value)` — and that's intentional, not a violation. The framework only enforces "the DataFrame matches the schema the resolved domain declared."

### D2. Atomic cache writes (raw and processed)

Both raw and processed writes must be all-or-nothing. A provider failure mid-write, a Ctrl-C, a disk-full error — none of these may leave a partially-written file on disk that subsequent reads would treat as authoritative.

Pattern (applies to all three layers — raw payload, processed parquet, `_meta.json`): write to `<target>.tmp`, fsync, then `os.replace()` to the final name. Order matters: write raw atomically *first* (so even if normalization crashes, the bytes are preserved for inspection), then the processed parquet, then update `_meta.json` last.

**Why:** partial writes are the canonical silent-failure mode for this class of module. A truncated parquet that pandas reads as "valid but short" is indistinguishable from a real "ticker has limited history" case downstream. The raw layer's atomicity matters too — a half-written raw file would block reprocessing later.

### D3. Deterministic cache reads

Two `fetch()` calls with identical arguments and identical cache state must return bit-identical DataFrames. No nondeterministic provider selection, no random tie-breaking on overlap days, no relying on dict ordering for column placement.

**Why:** consumer modules (especially `analog_mc`) treat data as the deterministic ground truth their causality discipline depends on. A nondeterministic data layer poisons every downstream test.

### D4. Adjustment semantics enforcement

`adj_close` always means split-and-dividend adjusted. When Stooq populates `adj_close` (with split-only adjustment), `_meta.json` must record `adjustment_quality: "split_only"`. When Tiingo or yfinance overwrite those rows, the quality flag flips to `"full"`.

**Why:** an `adj_close` column whose meaning silently varies row-by-row is a guaranteed source of wrong total-return numbers. The flag makes the difference auditable.

A Stooq-only ticker that's never been updated by Tiingo is allowed to exist in the cache — but `fetch()` must propagate the quality flag in its return metadata so callers know.

### D5. Provider failure semantics

Failures must be explicit and typed:

- `ProviderError(provider, ticker, reason)` — HTTP error, network timeout, malformed payload.
- `EmptyPayload(provider, ticker)` — provider returned 200 OK with no rows (e.g., delisted ticker on Stooq).
- `SchemaMismatch(provider, ticker, details)` — adapter returned data that fails `validate_df()`.
- `AllProvidersFailed(ticker, failures: list)` — every tier exhausted; cache untouched.

Falling through the chain is silent (Tiingo failure → try yfinance is normal). But final exhaustion raises loudly; it does NOT return partial cached data unless `allow_partial_cache_returns=True` is explicitly set.

**Why:** "I'll just return what I have" is how downstream consumers end up computing on stale or wrong-shape data without noticing.

### D6. API key safety

- Tiingo key loads only from `os.environ[Config.tiingo_api_key_env]` at fetch time.
- Never log the key. Never embed it in error messages, cache file names, or `_meta.json`.
- If the key is missing, raise `MissingAPIKey(provider, env_var_name)` BEFORE attempting any network call.
- Unit tests run without a key; integration tests check for presence and skip otherwise.

### D7. Date / timezone discipline

- All dates stored as `datetime64[ns]` at midnight UTC. No timezone-aware columns in parquet (round-trip issues).
- Trading days only — weekends and US-market holidays should not appear in cached series.
- "End of day" means the close print as reported by the provider — typically next-business-day availability for most providers. The adapter must NOT fabricate today's row if the provider hasn't reported it yet.
- `start` / `end` arguments are interpreted as inclusive trading-day boundaries in US/Eastern time, then normalized to UTC midnight for storage.

**Why:** timezone bugs and "is today's bar real or projected?" bugs are silent and devastating. Pin both behaviors at the adapter boundary.

### D8. Raw immutability and reprocess determinism

Once a raw file is written, it is never modified or deleted by code in this module. The on-disk filename includes the UTC fetch timestamp, so multiple fetches for overlapping ranges produce distinct files; none overwrites another.

Reprocessing — re-deriving `processed/` from `raw/` — must be **bit-deterministic** given the same set of raw files. Specifically: feed the same raw files into `schema.normalize_df()` + `cache.merge_cache()` in the same order, get the same processed parquet. No timestamps, no random tie-breaking, no environment-dependent paths in the output.

A unit test (`test_reprocess.py`) freezes a set of raw fixtures and asserts byte-identical processed output across runs.

**Why:** the whole point of separating raw and processed is to be able to fix a normalization bug, re-derive, and audit the difference against historical processed output — without re-paying for API calls. That's only useful if the transform is deterministic. Nondeterminism here turns the audit trail into noise.

---

## Step-by-step implementation order

Implement and unit-test in this order. **Do not skip ahead.** Framework primitives (Stages 1–4) come first, then the us_equities domain plugs in on top (Stages 5–9), then the surface layers (Stages 10–11). Building an adapter before the framework primitives exist means refactoring it later.

### Stage 1: Framework primitives — schema, errors, abstractions

1. `schema.py`:
   - `ColumnSpec(name, dtype, nullable=False)` — single-column declaration.
   - `Schema(columns: list[ColumnSpec])` — domain schema; `validate(df) → None | raises`; `normalize(df, source_column_map) → df` (rename + cast + reorder).
   - Designed to be instantiated per-domain, not as a singleton.
2. `errors.py`: `ProviderError`, `EmptyPayload`, `SchemaMismatch`, `MissingAPIKey`, `AllProvidersFailed`, `UnknownDomain`.
3. `adapter.py`: `Adapter` ABC with `name`, `fetch(identifier, start=None, end=None) → Path` (writes raw, returns raw path), `parse(raw_path) → DataFrame`, `health_check() → bool`. Every concrete adapter inherits.
4. `domain.py`: `Domain` ABC declaring required attributes (`name`, `schema`, `identifier_parser`, `calendar`, `chain`, `config`); `DomainRegistry` (singleton: `register(domain)`, `resolve(identifier) → Domain`).
5. **Unit tests:** `Schema` validates against a synthetic 3-column schema (rejects missing, wrong dtype, wrong order); `normalize` handles common source variations; `DomainRegistry` resolves and rejects unknown prefixes.

### Stage 2: Framework primitives — raw store

6. `raw_store.py`:
   - `write_raw_atomic(domain, provider, identifier, payload: bytes, range) → Path` — temp file + fsync + rename; filename encodes UTC timestamp + range; path layout is `data/raw/<provider>/<domain>/<exchange>/<ticker>/<ts>_<range>.<ext>`.
   - `list_raw(domain, provider, identifier) → list[Path]` — for reprocess workflows.
   - **Never writes if a file with the same name already exists** (timestamp resolution is seconds; collisions are programmer error).
7. **Unit tests:** atomic write survives mock crash; immutability check (can't overwrite); filename parser round-trips; correct path layout for known inputs.

### Stage 3: Framework primitives — processed cache

8. `cache.py`:
   - `read_processed(domain, identifier) → (DataFrame, meta) | (None, None)`.
   - `write_processed_atomic(domain, identifier, df, meta)` — parquet temp + fsync + rename; meta updated last.
   - `detect_gaps(cached_df, requested_start, requested_end, calendar) → list[(gap_start, gap_end)]` — uses the domain's calendar object for trading-day arithmetic, so each domain controls its own gap semantics.
   - `merge_cache(existing_df, new_df, existing_meta, new_source, domain) → (DataFrame, updated_meta)` — handles overlap; precedence rules driven by domain-supplied policy (us_equities prefers full-quality adj_close over split-only); appends to the meta `sources` list.
9. **Unit tests:** atomic processed writes survive mock crash; gap detection on edge cases against a fake calendar (cache covers exactly, prefix only, internal gap); merge precedence with a fake domain policy; meta `sources` array grows correctly.

### Stage 4: Framework primitives — dispatch

10. `dispatch.py`:
    - Public `fetch(identifier, start, end, frequency="daily") → DataFrame`.
    - Logic: `DomainRegistry.resolve(identifier)` → read_processed → detect_gaps via domain.calendar → choose adapter per domain.config.chain + threshold → adapter.fetch (writes raw) → adapter.parse → domain.schema.normalize → domain.schema.validate → merge → atomic write → return requested slice.
    - Warn if identifier is outside the domain's universe (don't reject).
    - Return metadata accessible via `fetch_with_meta(...) → (DataFrame, FetchMeta)` for the agent-tool path.
11. **Unit tests:** with a fake domain + mocked adapters: cold cache → seed; small gap → first update adapter; big gap → seed; update adapter failure → falls through to fallback; all-fail → raises `AllProvidersFailed`; unknown prefix → `UnknownDomain`.

### Stage 5: us_equities domain — schema, registry, universe, calendar

12. `domains/us_equities/schema.py`: instantiate a `Schema` with the OHLCV columns. Adjustment-quality precedence policy ("full" beats "split_only").
13. `domains/us_equities/registry.py`: `parse_identifier("NYSE:AAPL") → (exchange="NYSE", symbol="AAPL")`. Accept `NYSE:`, `NASDAQ:`, `INDEX:`. Handle bare `"AAPL"` (warn; default to NYSE).
14. `domains/us_equities/universe.py`: `load_universe(name="sp500") → list[str]` reading `configs/data_pipelines/domains/us_equities/universe_<name>.yaml`; `is_in_universe(identifier, universe_name) → bool`.
15. `domains/us_equities/calendar.py`: NYSE trading calendar. `gap(requested_range, cached_range) → list[(start, end)]` using business-day arithmetic + NYSE holidays.
16. **Unit tests:** identifier parsing edge cases (lowercase, INDEX:^SPX, bare symbol); universe load + membership; NYSE calendar excludes known holidays; schema rejects malformed OHLCV.

### Stage 6: us_equities domain — Stooq adapter

17. `domains/us_equities/adapters/stooq.py`:
    - `fetch(identifier, start=None, end=None) → Path` — full-history pull (Stooq doesn't take range params in the simple CSV endpoint), HTTPS GET `https://stooq.com/q/d/l/?s=<symbol>.us&i=d`, write response bytes via `raw_store.write_raw_atomic`, return path.
    - `parse(raw_path) → DataFrame` — reads CSV, normalizes via the us_equities Schema. Sets `adj_close = close`, tags `adjustment_quality: "split_only"`.
    - Handle "200 + empty body" → raise `EmptyPayload` before writing raw.
18. **Unit tests:** recorded raw fixtures for a known ticker, empty-body case, malformed CSV; parse round-trips a fixture; one online smoke test gated on `PYTEST_ONLINE=1`.

### Stage 7: us_equities domain — Tiingo adapter

19. `domains/us_equities/adapters/tiingo.py`:
    - `fetch(identifier, start, end) → Path` — GET `https://api.tiingo.com/tiingo/daily/<symbol>/prices?startDate=...&endDate=...&token=...`, write raw JSON, return path.
    - `parse(raw_path) → DataFrame` — reads JSON, normalizes via the us_equities Schema. Tags `adjustment_quality: "full"`.
    - Key from `os.environ[config.tiingo_api_key_env]`; raise `MissingAPIKey` if absent.
    - Retry on 429 with exponential backoff up to `tiingo_max_retries`.
20. **Unit tests:** recorded raw fixtures for normal, 401, 429-with-retry, empty response; parse round-trips; online smoke test gated on `TIINGO_API_KEY`.

### Stage 8: us_equities domain — yfinance fallback adapter

21. `domains/us_equities/adapters/yfinance.py`:
    - `fetch(identifier, start, end) → Path` — `yfinance.Ticker(symbol).history(start=start, end=end, auto_adjust=False)`, serialize the returned DataFrame to parquet under raw, return path.
    - `parse(raw_path) → DataFrame` — reads the parquet, normalizes multi-index columns and `Adj Close` to canonical, tags `adjustment_quality: "full"`.
22. **Unit tests:** recorded raw fixtures, schema-mismatch case (extra columns), empty result.

### Stage 9: us_equities domain — wire-up + reprocess test (D8 enforcement)

23. `domains/us_equities/__init__.py`: instantiate the `USEquitiesDomain(...)` (schema + registry parser + universe loader + calendar + chain config + adapter instances) and `DomainRegistry.register(...)` it on import.
24. `domains/us_equities/test_reprocess.py`:
    - Freeze a fixture set of raw files (one per provider × one ticker).
    - Run normalize + merge in a defined order; assert byte-identical processed parquet across runs.
    - Run again in CI; assert same hash.
25. This stage is the canary for D8 — if a future change introduces nondeterminism in normalization or merge, this test fails before the change ships.

### Stage 10: CLI

26. `__main__.py`:
    - Subcommands (domain is inferred from identifier prefix where applicable):
      - `fetch NYSE:AAPL --start 2010-01-01 --end 2026-05-22` — single-identifier fetch.
      - `seed --domain us_equities --universe sp500` — bulk-seed every identifier in the named universe.
      - `reprocess [--identifier NYSE:AAPL | --domain us_equities | --all]` — re-derive `processed/` from `raw/` without hitting any API.
      - `list-cached [--domain us_equities]` / `purge [--domain us_equities --identifier ...]` / `health [--domain us_equities]` (provider reachability + key presence).
      - `list-domains` — show all registered domains and their chain config.
27. **Unit tests:** argparse routing; integration smoke test against the cache; `seed` runs against a mocked-adapter 5-ticker universe and verifies all 5 land in `processed/us_equities/`.

### Stage 11: Documentation, examples, agent-tool surface prep

28. `README.md` for the module (brief — most content already in `goal.md` and this plan).
29. Worked-example notebook: cold-start fetch via Stooq, incremental update via Tiingo, gap-fill, simulated provider failure, reprocess from raw after a normalization change.
30. Design note (separate doc) sketching the agent-tool wrapper shape — what `fetch_with_meta` returns in JSON, error envelope, partial-success semantics, multi-domain identifier discovery. Not implemented in v1; documented so v2 can land it without re-litigating the API.
31. Internal `docs/data_pipelines/adding_a_domain.md` — a "how to add a new domain" guide written *after* us_equities ships, with the seams it found and the patterns it crystallized. This doc is the input to the framework-extraction decision at the point domain #2 is added.

---

## Reproducibility requirements

- Single `Config` object defines the entire run; pass through explicitly, no globals.
- Cache file format and `_meta.json` schema versioned (a `schema_version: 1` field) so future format migrations are detectable.
- Adapter responses recorded as fixtures (committed under `tests/data_pipelines/fixtures/`) so offline tests are bit-deterministic.
- No network calls in default test runs. Online smoke tests gated on env vars (`PYTEST_ONLINE=1`, `TIINGO_API_KEY`).

---

## What not to do

- **Don't crystallize the framework abstraction prematurely.** v1 builds the framework primitives (`schema.py`, `cache.py`, `raw_store.py`, `dispatch.py`, `domain.py`, `adapter.py`) only as much as the us_equities domain actually needs. Don't add framework hooks for hypothetical future domains (intraday, streaming, vintage-revised data, etc.). The right time to enrich the framework is when domain #2 is being added and the gap is informed by two real use cases.
- **Don't** put OHLCV-specific logic in the framework layer. The framework knows about "a domain has a schema and a calendar"; it does not know what columns are in any specific schema. If something is hardcoded as OHLCV outside `domains/us_equities/`, it's a bug.
- **Don't** put US-equity-specific logic outside `domains/us_equities/`. The S&P 500 universe file, NYSE calendar, Stooq URL format, Tiingo endpoint shape, etc. all live in the domain. The framework dispatches; it doesn't know.
- **Don't** add a second domain in v1. The right way to know what the framework should look like is to ship one domain cleanly, see what hurts, then add the second informed by that. Two domains shipped together is the canonical over-abstraction trap.
- **Don't** retry silently across providers without logging which provider failed and why. The fallback chain is observable.
- **Don't** introduce a "smart" cache that proactively pre-fetches. The cache only fills on demand.
- **Don't** add corporate-actions tables (splits, dividends as separate series) in v1. The `adj_close` column captures what consumers need; explicit corporate-action data is a v2 concern if a consumer actually needs it.
- **Don't** wrap `pandas-datareader` instead of writing direct adapters. Indirect through a maintained-by-others wrapper introduces breakage we can't control and obscures the schema-normalization boundary.
- **Don't** cache by REST query string. Cache by `(domain, exchange, identifier, frequency)` tuple → canonical on-disk path. The cache is *what data exists for this identifier*, not *what was requested last*.
- **Don't** add intraday support in v1. The schema, cache layout, and provider list would all need different shapes; ship daily cleanly first.
- **Don't** add a Streamlit dashboard yet. The CLI + (eventual) agent tool covers the needed surface.
- **Don't** auto-detect domain from a bare symbol. Always require a `<DOMAIN_PREFIX>:<SYMBOL>` identifier (or warn loudly on bare symbol with an explicit default).

---

## Open questions

These are flagged for the user to decide before / during implementation. None are blockers for the plan; they shape specific stages.

1. **S&P 500 constituent list — source and update cadence.** v1 pins constituents in `configs/data_pipelines/universe_sp500.yaml`, hand-maintained. Options for keeping it fresh: (a) leave as fully manual (acceptable since S&P 500 churns ~25 tickers/year), (b) one-shot script that scrapes Wikipedia's S&P 500 page on demand and regenerates the YAML, (c) commit to a paid constituents-history feed later. **Recommendation:** start with (a); add (b) as a `python -m data_pipelines refresh-universe sp500` command when the manual maintenance friction is actually felt. Historical (point-in-time) constituent membership is a separate, harder problem — defer.
2. **Index symbols** (`^SPX`, `^NDX`, `^DJI`). Stooq's URL scheme for these is `^spx` not `^spx.us`. Adapter needs a small lookup. Worth pinning the v1 supported index set up front (recommend: `^SPX`, `^NDX`, `^DJI`, `^RUT`).
3. **Overlap reconciliation policy** on a day where both Stooq (split-only `adj_close`) and Tiingo (full `adj_close`) provide values. Recommendation: Tiingo wins on `adj_close` (its semantics match the column meaning); both should agree on raw OHLC (sanity-check warn if they diverge >0.5%, which would suggest a data error somewhere).
4. **Cache-format version migration plan.** The raw/processed split makes this much cheaper: if the processed schema changes, run `reprocess --all` against existing raw, no API calls needed. The remaining concern is the raw-file naming convention itself (timestamp + range encoding) — once tools rely on parsing those filenames, changing the format is breaking. Worth pinning the exact pattern in `raw_store.py` early and adding a docstring contract test.
5. **Whether `analog_mc.data.load_close_series` migrates to consume `data_pipelines.fetch()`** or keeps its standalone CSV reader. Recommend: leave `analog_mc` untouched in v1 (its CSV loader is fine), but expose a parallel `analog_mc.data.load_from_pipelines(identifier, ...)` thin wrapper as a v2 task.
6. **Framework extraction trigger.** v1 ships framework primitives only as much as us_equities needs. When does domain #2 (whatever it is — NSE, FRED, commodities) trigger refactoring the framework toward a more crystallized shape? Likely candidates: discovering that `merge_cache`'s adjustment-quality precedence policy needs to be a generic interface, that the calendar abstraction needs a richer API, that `Adapter.fetch()` signature needs `frequency` to be a first-class parameter, that bulk-fetch vs incremental-fetch needs to be more cleanly split. **Recommendation:** when domain #2 lands, write `docs/data_pipelines/adding_a_domain.md` (Stage 11) capturing what *had* to be refactored vs what *should* be refactored, and decide framework extraction from that evidence — not from theoretical generality.
7. **Identifier scheme convention.** `<DOMAIN_PREFIX>:<SYMBOL>` with the domain prefix matching the exchange for equities (`NYSE:AAPL`) and the data source / type for non-equity (`FRED:DGS10`). Edge case: same identifier could be valid in multiple domains in principle (rare). Recommendation: register one domain per prefix; rejection of duplicate registration is a programmer error. Index symbols under `INDEX:` keeps NYSE/NASDAQ for actual exchange listings.

These belong in their own follow-up notes once v1 lands; mentioning here so they aren't forgotten.

---

# v1.7 — NSE equities (India) domain

Status: planned. Not implemented. Branch: TBD (per CLAUDE.md, this v1.7 work gets its own git branch when implementation starts; the plan lives here in the V1 doc).

This section describes what's *added* on top of v1's framework to bring up the NSE (National Stock Exchange of India) domain. Architectural decisions about the framework itself stay in the main body above; this section assumes those decisions.

## Purpose

Add NSE daily equities as the second domain plugged into the framework. This is the first real exercise of the "domain #2 plug-in" abstraction promised by v1.3, and the trigger for the framework-extraction decision in [open question 6](#open-questions).

## What v1.7 ships

- A new `domains/nse_equities/` plug-in: schema, calendar, identifier parser, universe loader, three adapters wired through the existing `Domain` ABC.
- One new framework primitive: a shared **retry policy** (exponential backoff with jitter, 3 retries) used by all three new adapters. Tiingo in us_equities retrofits onto the same primitive in a follow-up commit (keeps existing semantics; just removes the bespoke retry loop).
- `configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml` (start with the smallest Nifty universe; broader universes added as separate YAML files when needed).
- Tests + fixtures per adapter; D8 reprocess-determinism test for nse_equities; one online smoke test gated on `PYTEST_ONLINE=1`.

## Architectural fit

NSE plugs into the same seams us_equities uses — the framework code (`dispatch.py`, `cache.py`, `raw_store.py`, `schema.py`, `adapter.py`, `domain.py`, `env.py`) is **untouched** apart from the retry-utility lift.

```
data_pipelines/
├── (framework — unchanged)
└── domains/
    ├── us_equities/           # v1 — already shipped
    └── nse_equities/          # v1.7 — new
        ├── __init__.py        # registers NSEDomain
        ├── schema.py          # OHLCV (same shape as us_equities)
        ├── registry.py        # NSE: / BSE: / NIFTY: parser
        ├── universe.py        # load Nifty 50/100/200/500
        ├── calendar.py        # NSE trading calendar
        ├── config.py          # NSEEquitiesConfig
        └── adapters/
            ├── jugaad.py      # primary  — jugaad-data
            ├── nselib.py      # secondary — nselib
            └── yfinance.py    # fallback — yf.Ticker("RELIANCE.NS")
```

The processed-layer SQLite cache (v1.5) sees per-domain tables already; the new `nse_equities_data` and `nse_equities_meta` tables are created on first write — no schema migration needed.

## Schema (nse_equities)

Daily OHLCV — same column shape as us_equities for v1.7:

| Column | dtype | Notes |
|---|---|---|
| `date` | `datetime64[ns]` | UTC midnight |
| `open` | `float64` | INR, unadjusted |
| `high` | `float64` | INR, unadjusted |
| `low` | `float64` | INR, unadjusted |
| `close` | `float64` | INR, unadjusted |
| `adj_close` | `float64` | Split-and-dividend adjusted (INR) |
| `volume` | `int64` | Shares traded |

**Currency note.** Values are in INR throughout. There is no FX normalization in v1.7 — consumers that mix INR + USD time series are responsible for the conversion. A future FX-conversion concern is deferred.

**Adjustment semantics (D4 carry-over):**
- jugaad-data: returns both raw + adjusted series for equities; verify which column is split-vs-full-adjusted against a known split (RELIANCE 2017, INFY 2018) before pinning `adjustment_quality`.
- nselib: same verification step.
- yfinance .NS: same caveat as us_equities — `Close` is silently split-adjusted regardless of `auto_adjust`; `adj_close` is the trustworthy one.

The per-source `adjustment_quality` tag (`"full"`, `"split_only"`, `"unknown"`) goes into `_meta.sources[]` as in us_equities.

## Identifier scheme

```
NSE:RELIANCE      NSE:TCS      NSE:HDFCBANK
BSE:RELIANCE      BSE:TCS                       (BSE listings, where available)
NIFTY:NIFTY50     NIFTY:NIFTY100    NIFTY:NIFTY500    NIFTY:BANKNIFTY
```

- `NSE:`, `BSE:`, `NIFTY:` are the registered prefixes.
- `BSE:` is supported by the parser but adapters fall through on `BSE:` symbols if the provider doesn't have BSE coverage (jugaad and nselib are NSE-first; yfinance covers both via `.NS` / `.BO` suffixes).
- Use `NIFTY:` (NOT `INDEX:`) for NSE indices — `INDEX:` is already registered by us_equities and DomainRegistry's invariant is one-prefix-one-domain.

## Universe

Start with **Nifty 50**, pinned in `configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml`. Source (scrape on demand): the NSE constituents page or the index-history page. Hand-maintained YAML, same pattern as `universe_sp500.yaml`.

Expand to Nifty 100, 200, 500 in later versions as needed — each as its own `universe_*.yaml`. Bulk-seed CLI takes `--universe nifty50` exactly as for `sp500`.

## Calendar (NSECalendar)

NSE trading calendar:
- Trading days: Monday – Friday.
- Closed: weekends + ~17 published holidays/year. Many Indian holidays (Holi, Diwali, Eid) follow lunar calendars and can't be computed from simple rules — hand-pin them per-year, similar to how unscheduled NYSE closures are pinned in v1.

Implementation note: the `holidays` Python lib has an `IN` package with NSE support. Worth using if it stays maintained; otherwise hand-rolled per year. Treat library choice as an implementation detail (the framework only sees the `Calendar` protocol).

## Adapter chain (per the v1.7 brief)

**jugaad-data → nselib → yfinance**, all three using the new retry primitive (exponential backoff with jitter, 3 retries).

| Tier | Adapter | Library | Role | When invoked |
|---|---|---|---|---|
| Primary | `jugaad.py` | [jugaad-data](https://pypi.org/project/jugaad-data/) | seed + update | Default for every gap |
| Secondary | `nselib.py` | [nselib](https://pypi.org/project/nselib/) | fallback on jugaad fail | jugaad raises ProviderError / EmptyPayload |
| Fallback | `yfinance.py` | [yfinance](https://pypi.org/project/yfinance/) | last resort | both jugaad + nselib failed |

Both jugaad-data and nselib are range-aware (accept `from_date`/`to_date`), so unlike us_equities there's no "seed=full-history vs update=incremental" distinction — `chain_for_gap` returns the same ordered list `[jugaad, nselib, yfinance]` regardless of gap size or cache state.

The yfinance adapter is largely a shape-shift of the existing us_equities yfinance adapter — same `Ticker(...).history(...)` flow, just with `.NS` symbol-suffix mapping (`RELIANCE` → `RELIANCE.NS`). Keep parallel implementations across the two domains until a third domain needs the same library (avoids premature abstraction).

## Retry policy (new framework primitive)

The v1.7 brief specifies **exponential backoff with jitter, 3 retries** for all NSE adapters. To avoid copy-paste and to set the stage for retrofitting Tiingo in us_equities, this lands as a shared utility:

```
src/data_pipelines/retry.py

@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_sec: float = 1.0
    max_delay_sec: float = 30.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = (...)   # e.g., ProviderError

def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    provider: str,
    identifier: str,
) -> T: ...
```

Behavior:
- Attempt `fn()`. If it raises one of `retry_on`, sleep
  `min(base_delay_sec * 2**attempt, max_delay_sec) + random.uniform(0, base_delay_sec)`
  if `jitter` else just the exponential value.
- After `max_retries`, re-raise the last exception.
- Non-retryable exceptions (e.g., `MissingAPIKey`) propagate immediately.

Jitter rationale: bulk seed across N tickers without jitter creates synchronized retry storms when many tickers hit a transient 429 in the same window. Jitter (uniform in `[0, base_delay_sec]`) spreads the retries.

**Retrofit follow-up** (separate small commit after v1.7 lands): replace Tiingo's bespoke `_request_with_retry` in `domains/us_equities/adapters/tiingo.py` with `call_with_retry(...)` so both domains share the same code path. The Tiingo circuit breaker stays in place (different concern — across-call rate-limit state, not per-call retry).

## Configuration (NSEEquitiesConfig)

```python
@dataclass(frozen=True)
class NSEEquitiesConfig:
    # Dispatch
    default_universe: str = "nifty50"
    chain_order: tuple[str, ...] = ("jugaad", "nselib", "yfinance")

    # Retry policy (shared across all three adapters)
    retry_max_retries: int = 3
    retry_base_delay_sec: float = 1.0
    retry_max_delay_sec: float = 30.0
    retry_jitter: bool = True

    # Library-specific timeouts (provider native HTTP layers respect these)
    jugaad_timeout_sec: float = 30.0
    nselib_timeout_sec: float = 30.0
    yfinance_enabled: bool = True
```

No API keys for any of the three providers in v1.7 — none of jugaad-data, nselib, or yfinance require one for NSE data. (Some advanced nselib endpoints may need a session/cookie warm-up; treat as adapter internal.)

## Stages (implementation order)

Mirror the original 11-stage build order. Skip stages that are pure framework primitives (already shipped) and add a new stage for the retry utility.

12. **Retry primitive** — `src/data_pipelines/retry.py` + tests. Pure utility, no domain coupling. Land first so adapter stages can rely on it.
13. **nse_equities schema** — OHLCV schema instance (same shape as us_equities, separate `Schema` object to avoid coupling).
14. **nse_equities identifier parser + universe** — `parse_identifier("NSE:RELIANCE")`, `load_universe("nifty50")`.
15. **nse_equities calendar** — `NSECalendar` with hand-pinned holidays per year (or `holidays` library wrapper, whichever proves easier).
16. **jugaad adapter** — `jugaad-data` wrapper, normalize columns to canonical schema, validate adjustment-quality against a known split event in tests.
17. **nselib adapter** — `nselib` wrapper, same column-normalization + adjustment-validation.
18. **yfinance adapter (.NS)** — fork of us_equities yfinance adapter; uses `.NS` symbol suffix; carries same "split-adjusted-disguised-as-raw OHLC" D4 caveat.
19. **NSEDomain wire-up** — `__init__.py` instantiates and registers.
20. **D8 reprocess determinism test** — same shape as `test_reprocess.py` for us_equities.
21. **CLI integration** — `python -m data_pipelines seed --universe nifty50` just works once the domain is registered. Smoke-test against the cache.
22. **Cross-source parity audit** — `scripts/data_pipelines/parity_check_nse.py`. Findings inlined under "Implementation findings" below.
23. **Documentation update** — README, `adding_a_domain.md` follow-up capturing what *had* to be refactored vs what just plugged in (this is the evidence base for open question 6's framework-extraction decision).

## Tests

Mirror the us_equities tests under `tests/data_pipelines/domains/nse_equities/`. Fixtures: committed raw payload samples per (adapter, ticker) under `tests/data_pipelines/fixtures/nse_equities/{jugaad,nselib,yfinance}/`. Pick RELIANCE (long history, 2017 split), TCS (no recent splits), INFY (long history, NYSE+NSE dual listing — exercises the cross-listing edge case lightly) as canonical fixtures.

New shared tests:
- `tests/data_pipelines/test_retry.py` — retry primitive: backoff timing, jitter spread, max-retries cap, non-retryable propagation.

## Open questions (resolved during v1.7 build)

8. **Universe ordering — Nifty 50 first or skip to Nifty 100/500?** Shipped Nifty 50. Other Nifty universes deferred to v1.7.1 when a consumer asks.
9. **BSE coverage**: deferred. Parser accepts `BSE:` but jugaad/nselib raise `EmptyPayload` for it (both NSE-only); yfinance `.BO` is parser-routable but not validated.
10. **jugaad-data + nselib library liveness on Python 3.13**: both work. Pinned `jugaad-data==0.33.1` (Mar 2026), `nselib==2.5.1` (May 2026).
11. **Currency tag in schema vs. meta?** Omitted from schema; recorded per-source as `currency: "INR"` in `_meta.sources[]`.
12. **Time zone**: `date` stored as naive midnight UTC. Adapters explicitly convert from IST (jugaad's `2025-04-29 18:30:00` = April 30 IST trade date) before storing. See `domains/nse_equities/calendar.py` for the policy.
13. **Holiday data source**: using `holidays.financial_holidays('NSE')` — 12–16 holidays/year, properly named. Hand-pinned unscheduled-closure set kept empty pending the first surfaced omission.

## Implementation findings

Substantive things discovered during the v1.7 build that future maintainers and downstream consumers should know. Adapter-source comments cross-reference here.

### Seed validation

End-to-end seed of NIFTY 50 (2020-01-01 → 2025-12-31) succeeded for all 51 identifiers (50 equities + 1 index), 72,273 rows total. Provider mix: jugaad served 50, nselib served the index. Raw audit footprint: ~50 MB.

### Cross-source parity (RELIANCE/TCS/INFY/HDFCBANK, Jan–Apr 2025)

Run `scripts/data_pipelines/parity_check_nse.py` for the table; max-relative-diff headline:

- **Raw OHLC agrees within ~1% jugaad ↔ nselib ↔ yfinance on no-corporate-action tickers.** Both jugaad and nselib pull from NSE bhav; yfinance independently agrees.
- **HDFCBANK shows 2× divergence yfinance ↔ others** on Jan–Apr 2025 — jugaad/nselib close = 1793.75 on 2025-01-02, yfinance = 896.88. HDFCBANK had a 1:1 bonus issue effective **August 2025**, *after* the query window, and yfinance has forward-applied it. Same D4 trap as us_equities: yfinance's `Close` is silently split-/bonus-adjusted regardless of `auto_adjust=False`. **Use jugaad/nselib for raw traded prices/volumes pre-corporate-action; use yfinance's `adj_close` for total-return.**
- **adj_close diverges 0.4–5% jugaad ↔ yfinance** as expected. jugaad/nselib have no adjustment (`QUALITY_NONE`, `adj_close = close`); yfinance back-applies dividends (`QUALITY_FULL`). `merge_overlap_nse_equities` preserves `QUALITY_FULL` adj_close over `QUALITY_NONE`.
- **Single-day volume divergences up to ~50%** between providers exist even when most days agree to the share (RELIANCE 2025-01-02 matches to the last share across all three). Likely cause: one provider missing or post-correcting a single-day adjustment. Not blocking; flag in user-facing dashboards.

### Known upstream limitations

1. **jugaad-data `index_df` was broken upstream; patched in our vendored copy.** The niftyindices.com `/Backpage.aspx/getHistoricaldatatabletoString` endpoint changed contract around mid-2026: it now requires a single `cinfo` parameter holding a JSON-encoded string with `name`/`startDate`/`endDate`/`indexName`. Unmodified upstream sent the three flat fields and got HTTP 500 (`"missing value for parameter: 'cinfo'"`), surfacing as `KeyError: 'd'` in `_index`. Diagnosed and fixed in our git-subtree vendor at `vendor/jugaad-data/jugaad_data/nse/history.py` (search for `LOCAL PATCH`); `jugaad_data.nse.index_df` works again. **`JugaadAdapter` still short-circuits `NIFTY:` identifiers with `EmptyPayload`** because jugaad's index payload provides only OHLC — no VOLUME — and the canonical schema requires non-null int64 volume; nselib's `TRADED_QTY` fills that role. Re-enable jugaad for `NIFTY:` only if nselib breaks or volume is made nullable. Vendor sync instructions: `vendor/README.md`.

2. **nselib index history caps at ~3 fiscal years; resolved via per-identifier chain + partial-fill continuation.** `index_data("NIFTY 50", 2020-01-01, 2025-12-31)` returns ~422 of expected ~1,492 rows — the upstream endpoint silently truncates older chunks. Originally the dispatcher locked the partial coverage in (cache-clip + first-success-wins). **Both framework lifts now done:**
   - `Domain.chain_for_gap(identifier, gap_size, has_cache)` — per-prefix chain composition.
   - Dispatcher continues the chain after each successful write, re-detecting remaining sub-gaps and asking the next provider in the chain to fill them.

   NIFTY: now routes through `[nselib, yfinance]`: nselib fills what it can with true `TRADED_QTY`; yfinance backfills the rest. Verified end-to-end: `NIFTY:50` for 2020-01-01 → 2025-12-31 now caches **1,488 rows** (99.7% coverage) — 422 with nselib's authoritative TRADED_QTY, 1,063 with yfinance's smaller ^NSEI volume (recorded as separate sources in meta so consumers can tell which is which).

3. **jugaad returns multi-series rows** (EQ + BL + T0 + …) even when caller passes `series="EQ"` — the library translates `"EQ"` to `"ALL"` upstream. Non-EQ rows would collide on the cache's `(ticker, date)` PK. Fixed in `JugaadAdapter.parse()` by filtering `CH_SERIES == "EQ"`. Regression test added.

4. **yfinance silently bonus/split-adjusts the "raw" OHLC columns regardless of `auto_adjust=False`** *(both domains, all yfinance-sourced rows for tickers with any historical split or bonus issue).* The `auto_adjust=False` flag only controls dividend adjustment; splits and bonus issues are always back-applied internally — there is no way to extract true-raw historical OHLC from yfinance. Concrete instances surfaced during parity audits:
   - **HDFCBANK** (NSE) — 1:1 bonus effective Aug 2025; yfinance reports half-prices for Jan–Apr 2025 (close 896.88 vs jugaad/nselib 1793.75) and **2x the volume**.
   - **AAPL** — 2014 7-for-1 + 2020 4-for-1; yfinance pre-split OHLC differs from Tiingo by 7× and 4× respectively.
   - **RELIANCE** — 2017 1:1 bonus; analogous pattern.

   `adj_close` (`QUALITY_FULL` from yfinance) is *consistent* across sources — divergence is OHLC-only. `merge_overlap` preserves Tiingo/nselib/jugaad OHLC when they overlap with yfinance, so the trap only bites on dates yfinance is the only source for (notably: all 4 US indices and the ~1063 yfinance-backfilled NIFTY:50 dates). **Use `adj_close` for any return / volatility / log-return calculation;** treat yfinance-sourced OHLC for split tickers as cosmetic. Pinned in `domains/us_equities/schema.py` and `domains/nse_equities/schema.py` docstrings.

5. **yfinance index volume is meaningful only loosely.** Two flavors:
   - US (`^SPX`, `^NDX`, `^DJI`, `^RUT`): yfinance reports a **constituent-volume aggregate** (~2.6B/day for ^SPX) as `Volume`. Not the same metric as a true index-traded volume; useful as a coarse activity proxy, not as a tradeable quantity. All 4 US indices in cache (~9k rows each) are 100% yfinance-sourced and carry these numbers.
   - NSE (`^NSEI`): yfinance reports a number ~3 orders of magnitude smaller than nselib's `TRADED_QTY` on the same day (~3.1e5 vs ~3.7e8). Origin unclear; possibly index-ETF retail only. For `NIFTY:50`, the cache has 422 rows from nselib (true `TRADED_QTY`) and 1,063 rows backfilled from yfinance (smaller, less-meaningful number). Source is recorded per range in `_meta.sources[]`.

   Mitigation: for NIFTY:50, prefer nselib's volume on the dates it covers (the cache already does this via merge precedence). For US indices, no better source is currently wired — Stooq's index volume is unknown to us (gated on this IP), Tiingo doesn't cover indices. If a downstream consumer needs true-volume metrics, wire a new provider (e.g., a futures-volume feed) and re-route via `chain_for_gap("INDEX:...", ...)`.

### Concurrency contract

**Treat seeds as single-process operations per `data_root`.** Two simultaneous `seed` processes against the same domain collide on D8 immutability (same UTC-second collision in `write_raw_atomic` → `FileExistsError`) and on the cache's `(ticker, date)` PK (overlapping inserts → `UNIQUE constraint failed`). Intra-process concurrency works (SQLite WAL); cross-process does not. If parallelism is needed, partition the universe across workers each with a distinct `--data-root`, or wrap in an external lock.

### Retrofit deferred

Tiingo's bespoke `_request_with_retry` in `domains/us_equities/adapters/tiingo.py` should be replaced with `call_with_retry(...)` so both domains share one code path. The Tiingo circuit breaker stays in place (different concern). Pending separate commit.

## What not to do (v1.7)

- **Don't** crystallize a "shared OHLCV schema" abstraction yet. NSE happens to have the same column set as us_equities; that's coincidental. Other domains (FRED, FX, commodities) will not. Two domains using the same `Schema(...)` instance can be revisited in v2 when there's a real pattern.
- **Don't** unify the yfinance adapters across us_equities and nse_equities until a third domain needs the same library. Two copies is cheaper than one premature abstraction.
- **Don't** add intraday or tick data — same scope discipline as v1.
- **Don't** add a Streamlit dashboard, fundamentals normalizer, or trading features. Same scope discipline as v1.
- **Don't** ship without the retry primitive's tests. The retry semantics are the kind of thing that silently regresses if untested.
