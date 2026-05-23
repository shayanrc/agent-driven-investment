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
