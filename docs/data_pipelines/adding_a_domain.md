# Adding a new domain to `data_pipelines`

A "domain" is one kind of time-series data — US equities (v1), NSE equities,
FRED macro, commodities, FX, etc. Each domain plugs into the same framework
with its own schema, identifier convention, universe, calendar, and adapter
set.

This guide is written after shipping the us_equities domain. It captures the
seams that mattered, the abstractions that crystallized, and the parts of the
framework that will likely need to flex once a *second* domain lands. Treat it
as starting evidence, not a contract — the right place to revise the framework
is when domain #2 forces a refactor, not before.

## Checklist

1. **Pick an identifier prefix scheme.** `<DOMAIN_PREFIX>:<SYMBOL>` (e.g.,
   `FRED:DGS10`, `NSE:RELIANCE`). Pick something that won't collide with
   existing prefixes (`NYSE`, `NASDAQ`, `INDEX`). Register one domain per
   prefix; duplicate registration is a programmer error.

2. **Define the canonical schema** under `src/data_pipelines/domains/<name>/schema.py`:
   ```python
   from data_pipelines.schema import ColumnSpec, Schema

   FRED_SCHEMA = Schema(columns=(
       ColumnSpec("date", "datetime64[ns]"),
       ColumnSpec("value", "float64", nullable=True),  # FRED has gaps
   ))
   ```
   Different domain → different shape. Don't shoehorn into OHLCV.

3. **Write a calendar** under `domains/<name>/calendar.py`. The framework asks
   one method of it:
   ```python
   class FREDCalendar:
       def trading_days(self, start: date, end: date) -> list[date]: ...
   ```
   For FRED-style series the "valid days" might be all weekdays, or every
   month-end depending on the series. If your calendar varies per series,
   inject it via the parser or expose a `calendar_for(symbol)` method on the
   domain rather than over-generalizing the Calendar protocol.

4. **Write an identifier parser** under `domains/<name>/registry.py`:
   ```python
   def parse_identifier(identifier: str) -> tuple[str, str]:
       """Split 'FRED:DGS10' → ('-', 'DGS10') for cache pathing.

       The first element is the path segment under data/{raw,processed}/<domain>/<segment>/<symbol>/.
       Use '-' (or similar placeholder) when there's no exchange/namespace concept.
       """
   ```

5. **Add a universe loader** (optional but recommended) under
   `domains/<name>/universe.py` and YAML files under
   `configs/data_pipelines/domains/<name>/universe_*.yaml`. Universe is a soft
   guardrail — dispatch warns but does not reject out-of-universe identifiers.

6. **Write the adapters** under `domains/<name>/adapters/`. Each adapter is a
   subclass of `data_pipelines.adapter.Adapter` with three methods:
   - `fetch(identifier, start, end, *, data_root) → Path` — download, write
     raw via `raw_store.write_raw_atomic`, return raw path.
   - `parse(raw_path) → pd.DataFrame` — read raw, return source-native shape.
     Pure function; deterministic.
   - `health_check() → bool` — best-effort liveness.

   Two class attributes the framework reads:
   - `source_column_map: dict[str, str] | None` — passed to `Schema.normalize`
     for rename. Use None if `parse` already returns canonical names.
   - `extra_meta: dict` — merged into the `sources[]` entry in `_meta.json`.
     Use for per-source provenance flags like `adjustment_quality` for
     us_equities.

   Errors must be typed: `ProviderError`, `EmptyPayload`, `MissingAPIKey`. The
   framework's chain dispatcher catches these as fall-through-the-chain
   signals (D5).

7. **Write a per-domain config dataclass** under `domains/<name>/config.py`.
   Domains have wildly different knob sets — don't try to share a base class.
   us_equities has `big_gap_threshold_days`; FRED probably won't.

8. **Wire it up** in `domains/<name>/__init__.py`:
   ```python
   class FREDDomain(Domain):
       @property
       def name(self): return "fred_macro"
       @property
       def identifier_prefixes(self): return ("FRED",)
       @property
       def schema(self): return FRED_SCHEMA
       @property
       def calendar(self): return FREDCalendar()
       def parse_identifier(self, identifier):
           return ("-", identifier.split(":", 1)[1])
       def chain_for_gap(self, gap_size_trading_days, has_cache):
           return [self._adapters["fred"]]   # single-source, no fallback
       # Optionally override merge_overlap for per-column precedence

   _DOMAIN = FREDDomain()
   DomainRegistry.register(_DOMAIN)
   ```
   Side-effect-on-import is the convention; users `import` the package to make
   the domain available to `fetch`.

9. **Tests**: a fixture-driven schema invariance test per adapter, a reprocess
   determinism test (D8), and a chain-routing test through the public
   `fetch()` API with mocked adapters. Online smoke tests gated on env vars.

## Seams the framework deliberately leaves to domains

- **Adjustment / precedence policy.** us_equities preserves "full" adj_close
  over "split_only" via `merge_overlap` override. Generalizing this is
  premature; each domain knows its own column semantics.
- **Threshold semantics.** us_equities has `big_gap_threshold_days`. FRED
  series update on per-series schedules — a different `chain_for_gap`
  implementation makes more sense there.
- **Universe maintenance.** v1 ships a hand-maintained YAML. Domains that
  need point-in-time / drifting universes (NSE constituents over time,
  S&P 500 historical membership) will likely want their own scraper or
  vendor feed — that's a per-domain concern, not framework infrastructure.

## Likely framework refactors when domain #2 lands

These are predictions, not promises — wait for the actual pain before doing them:

- **Calendar API may need to grow.** `trading_days()` is enough for daily
  bars. Intraday data needs `trading_minutes()` or similar. Monthly series
  may want `valid_periods()` with period objects, not dates.
- **`Adapter.fetch()` signature may need `frequency`.** v1 hardcodes daily.
  Mixing daily + intraday in one domain (or supporting both via per-call
  override) means `frequency` becomes a real parameter.
- **`merge_overlap` precedence may need column-level granularity.**
  us_equities preserves only `adj_close` from existing rows; other columns
  flip to new. If multiple domains need this pattern, lift to a declarative
  `MergeStrategy` interface instead of duplicating the logic.
- **Identifier parsing.** v1 uses `<PREFIX>:<SYMBOL>`. If a domain needs
  multi-part identifiers (currency pairs `FX:EUR/USD`, futures contracts
  with month codes `FUT:CL.M26`), the parser interface may need to return
  richer structured info than `(exchange, symbol)`.

When you trip on one of these, write a short note in this file documenting
*what hurt* and *what minimal change unblocks it* — that's what we'll use to
inform the framework crystallization decision.

---

## v1.7 follow-up — evidence from the NSE domain build (2026-05-24)

Domain #2 (`nse_equities`) is now landed. This section records what actually plugged in vs the predictions above, and the framework-extraction decisions justified by the new evidence (open question 6 in `V1_IMPLEMENTATION_PLAN.md`).

### What plugged in unchanged

- `Schema` / `SchemaMismatch` / `Schema.normalize` — separate instance, same shape (intentional, per "don't crystallize a shared OHLCV abstraction yet").
- `DomainRegistry` — added `NSE` / `BSE` / `NIFTY` prefixes alongside `NYSE` / `NASDAQ` / `INDEX`. No collision; no API change.
- `raw_store.write_raw_atomic` + D8 immutability — three new file extensions (`.json` for jugaad, `.csv` for nselib, `.parquet` for yfinance) flowed through with no code change.
- `cache.py` SQLite layer — auto-derived `nse_equities_data` / `nse_equities_meta` tables from `domain.schema` on first write. Zero code change.
- `dispatch.py` cache-clip + soft-fail-on-EmptyPayload — both reused unchanged.
- `Adapter` ABC + `source_column_map` + `extra_meta` — exactly the right shape; no signature change.
- CLI — 4-line patch (import the new domain module + add the universe-loader entry in `_load_universe_for_domain`). All subcommands worked.

### What was lifted (already in the v1.7 plan)

- **`src/data_pipelines/retry.py`** (`RetryPolicy` + `call_with_retry`). All three NSE adapters use it. Tiingo retrofit deferred to a follow-up commit.

### Where the predictions held vs missed

| Prediction (v1)                                | Actual (v1.7)                                                |
|------------------------------------------------|--------------------------------------------------------------|
| Calendar API may need to grow                  | **No.** `trading_days()` covered NSE fine — different holiday set, same protocol. |
| `Adapter.fetch()` may need `frequency`         | **No.** All v1.7 data is daily; no pressure to add it.       |
| `merge_overlap` may need column-level granularity | **Partly.** NSE adopted the same "preserve high-quality adj_close" pattern as us_equities — two domains using identical logic with one provider-name swap. If domain #3 wants the same thing, lift to a small `MergeStrategy` helper. |
| Identifier parsing may need richer structure   | **No.** `<PREFIX>:<SYMBOL>` worked. The NIFTY: short-alias slug map lives inside `registry.py`, not as a framework concern. |

### New friction surfaced (not in v1's predictions)

1. **~~Chain composition for partial-coverage providers.~~ Lifted.** `Domain.chain_for_gap` now takes an `identifier` arg; domains can return different chains per prefix. us_equities routes `INDEX:*` to `[yfinance, stooq]` (skip dead Tiingo leg); nse_equities routes `NIFTY:*` to `[nselib, yfinance]`, `BSE:*` to `[yfinance]`, and `NSE:*` to `[jugaad, nselib, yfinance]`. Adapter-internal `EmptyPayload` short-circuits stay as defense in depth.

2. **~~Dispatcher stops after partial-fill.~~ Lifted.** After each provider's successful write, dispatch re-detects remaining sub-gaps inside the original top-level gap and asks the next provider in chain to fill them. Bounded: each provider runs at most once per top-level gap. Soft-fail kicks in when the chain is exhausted but at least one provider returned data (or returned authoritative EmptyPayload with existing cache); the residual uncovered range is recorded in `providers_failed[]`. Concrete win measured during the lift: `NIFTY:50` 2020-01-01 → 2025-12-31 went from **422 → 1,488 rows** (99.7% coverage) — nselib supplies true `TRADED_QTY` where available, yfinance backfills the older OHLC.

3. **Concurrency across processes.** Two `seed` processes racing on the same identifier collide on D8 immutability (same-second raw filename) and on the cache's `(ticker, date)` PK. Current contract: seeds are single-process operations per `data_root`. Documented in `V1_IMPLEMENTATION_PLAN.md` §"Concurrency contract".

4. **Provider library quirks live in adapters, not framework.** jugaad returns multi-series rows even when caller passes `series="EQ"`; nselib returns Indian-comma-formatted strings with unicode-rupee suffixes; jugaad's index endpoint is broken upstream. All three correctly handled in the adapter — none warranted framework abstraction. The `parse()` boundary is the right place for this.

5. **Currency / units have no first-class home.** v1.7 stores `currency: "INR"` in `_meta.sources[]`. No consumer in v1.7 needed it. When the first cross-currency consumer appears, lift to `Schema.column_units` or similar.

### Open-question 6 verdict

Two of the five candidate refactors from v1 lit up with concrete evidence during the v1.7 NIFTY 50 work and were lifted (per-identifier chain ordering + dispatcher partial-fill continuation). The other three (currency/units, row-level provenance, shared OHLCV schema) still wait. Hold the line on those until a real consumer produces pressure.

---

## v2 follow-up — evidence from the FRED build (2026-06-20)

Domain #3 (`fred_macro`) is now landed — the first **non-equity, non-OHLCV**
domain: single `(date, value)` macro series, three publication cadences (daily /
monthly / quarterly), one logical source with auto keyless/keyed transport.
Plan: `docs/data_pipelines/V2_FRED_MACRO_PLAN.md`.

### What plugged in unchanged

- `Adapter` ABC + `source_column_map` + `extra_meta` — no signature change. One
  `FredAdapter` with two transports (CSV / JSON) dispatched in `parse()` on raw
  file extension; `.csv` and `.json` flowed through `raw_store` / D8 with zero
  framework change (as predicted by the v1.7 multi-extension evidence).
- `cache.py` table auto-derivation — a `(date, value)` schema auto-created
  `fred_macro_data` / `fred_macro_meta` on first write. Zero DDL code change.
- `Schema` / `Schema.normalize` / `Schema.validate` — the nullable `value`
  column (`ColumnSpec(..., nullable=True)`) worked as designed; validate accepts
  NaN there and rejects it in `date`.
- `dispatch.py` cache-clip + partial-fill + soft-fail — reused unchanged.
- `DomainRegistry` — added the `FRED` prefix; no collision, no API change.
- CLI — the same 4-line patch shape as NSE (import the module + a
  `_load_universe_for_domain` branch).
- `retry.py` `call_with_retry` — adopted directly for the FRED GET.

### Prediction #1 (Calendar API) — **lit up**, and was lifted

The v1 doc predicted "Calendar API may need to grow" and suggested
`calendar_for(symbol)` over generalizing the `Calendar` protocol. FRED's three
cadences forced it: a monthly series checked against a daily grid shows ~250
phantom gaps/year. Minimal lift, exactly as predicted:

- **`Domain.calendar_for(identifier) -> Calendar`** added to the ABC with a
  default returning `self.calendar`. Equity domains inherit it → **zero
  behavioural change** (the single-calendar case is the default).
- `dispatch.py` now resolves `cal = domain.calendar_for(identifier)` once per
  fetch and uses it for every `detect_gaps` / `trading_days` call (4 sites).
- `FREDMacroDomain.calendar_for` returns a business-day / monthly / quarterly
  calendar keyed on the series' declared frequency.

The `Calendar` **protocol itself did not change** — `trading_days(start, end)`
still suffices; we just needed per-identifier *selection*, not a richer method.
That validates the v1 instinct to extend selection on the Domain rather than
fatten the Calendar protocol.

### New friction surfaced (not in v1's predictions)

1. **Universe-YAML shape is OHLCV-specific.** `test_universe_yaml_lint.py`
   enforces `{universe, listed_at, indices, tickers}` with domain-prefixed
   tickers — no slot for per-series `frequency`, and `indices`/`listed_at` are
   meaningless for macro series. Rather than fork the lint + its spec doc, the
   FRED config uses a distinct filename (`series_<name>.yaml`, outside the
   lint's `universe_*.yaml` glob) with a fit-for-purpose shape carrying
   `frequency`/`category`/`description`. **Open question for domain #4:** if a
   third config shape appears, consider generalizing the lint to dispatch on a
   declared `kind:` field instead of accreting filename conventions.

2. **`cache.py` `iterrows()` could not store NaN.** `_replace_data` built insert
   rows via `df.iterrows()`, which coerces a row's cells to one dtype — a NaN
   float in a row that also has a datetime column became `NaT` and crashed the
   SQLite bind. Latent since v1.5 (no equity column is nullable); FRED's
   nullable `value` is the first to exercise it. Fixed by extracting
   column-wise and mapping missing → SQL `NULL`. **Lesson:** the first nullable
   column in any new domain should ship with an explicit cache NaN round-trip
   test (`test_cache.py::TestNullableRoundTrip`).

3. **Per-call provenance vs static `extra_meta`.** The transport (keyless CSV vs
   keyed JSON) varies per fetch, but `extra_meta` is a static class attribute.
   We left transport provenance to the raw file extension (`.csv`/`.json`)
   rather than fight the static-meta shape. If a future domain needs per-call
   provenance flags, that's the seam to revisit.

### Single-source, no-fallback chain

`chain_for_gap` returning one adapter for every identifier worked cleanly — the
dispatcher's chain machinery handles a length-1 chain with no special-casing.
Confirms the chain abstraction degrades gracefully to the single-provider case.
