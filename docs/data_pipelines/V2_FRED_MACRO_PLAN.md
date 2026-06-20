# data_pipelines V2 — FRED macro domain

Domain #3. The first **non-equity, non-OHLCV** domain: single `(date, value)`
macro series (rates, curve, credit spreads, breakevens, inflation, labor,
growth) sourced from the St. Louis Fed (FRED). Lands the macro-regime context
the project previously approximated only with a crude SMA200 gate.

`goal.md` and `adding_a_domain.md` both named `FRED:DGS10` as the canonical
"domain #3" example; this is that domain, built to the `adding_a_domain.md`
checklist.

## What shipped

```
src/data_pipelines/domains/fred_macro/
  __init__.py        FREDMacroDomain — registers on import; single-source chain
  schema.py          FRED_SCHEMA = (date datetime64, value float64 NULLABLE)
  calendar.py        BusinessDayCalendar / MonthlyCalendar / QuarterlyCalendar
  registry.py        VALID_PREFIXES=("FRED",); parse_identifier → ('-', SERIES_ID)
  config.py          FredMacroConfig (endpoints, FRED_API_KEY env, retry knobs)
  universe.py        load_universe + load_frequency_map (reads series_macro.yaml)
  adapters/fred.py   FredAdapter — auto keyless/keyed transport
configs/data_pipelines/domains/fred_macro/series_macro.yaml   ~25 curated series
```

Plus a 4-line CLI wire-up in `src/data_pipelines/__main__.py`, a framework
extension in `domain.py` + `dispatch.py`, and a framework bug-fix in `cache.py`
(see below).

Usage:

```python
from data_pipelines import fetch
import data_pipelines.domains.fred_macro     # registers the FRED prefix
df = fetch("FRED:DGS10", "2010-01-01", "2026-06-20")   # (date, value)
```

```
uv run python -m data_pipelines fetch FRED:CPIAUCSL --start 2010-01-01 --end 2026-06-20
uv run python -m data_pipelines seed --domain fred_macro --universe macro --start 2000-01-01 --end 2026-06-20
```

## Design decisions

### Schema — `(date, value)`, value nullable
FRED encodes "no data" for a day as `"."` (federal holidays on a daily series,
suppressed/unreleased observations). The adapter maps `"."` → `NaN`; the cache
stores `NaN` as SQL `NULL`. Deliberately not shoehorned into the equities OHLCV
shape — the framework reads `domain.schema` and knows nothing about either.

### Transport — auto (keyless default, keyed when `FRED_API_KEY` set)
One logical source, two transports in `FredAdapter`:
- **keyless** (default, zero setup): `GET fredgraph.csv?id=…&cosd=…&coed=…` → CSV.
- **keyed** (when `FRED_API_KEY` is in env): `GET <api>/series/observations?…&api_key=…&file_type=json` → JSON.

Chosen at fetch time on key presence. Both yield identical canonical
`(date, value)`. `parse()` dispatches on the raw file extension (`.json` vs
`.csv`) so reprocess-from-raw (D8) works regardless of which transport produced
the file. Single-source domain — `chain_for_gap` returns just the FRED adapter
(no seed/update/fallback tiering). D6: the key is read from env at fetch time
only and never reaches a raw filename, log line, or error message (HTTP errors
are sanitized to a status code with `from None`, since `HTTPError` echoes the
request URL).

### Per-series-cadence calendars — the framework extension
FRED dates observations at **period-start**: daily = business day, monthly =
1st of month, quarterly = 1st of Jan/Apr/Jul/Oct. A single domain calendar
can't model that — a monthly series checked against a daily grid shows ~250
phantom gaps/year. So this domain motivated the extension `adding_a_domain.md`
predicted first ("Calendar API may need to grow"):

- **`Domain.calendar_for(identifier) -> Calendar`** (new, in `domain.py`), with a
  default that returns `self.calendar`. Equity domains inherit the default →
  **zero behavioural change**.
- `dispatch.py` resolves `cal = domain.calendar_for(identifier)` once per call
  and uses it for all gap detection (4 call sites swapped from `domain.calendar`).
- `FREDMacroDomain.calendar_for` looks up the series' frequency (from
  `series_macro.yaml`) and returns the daily / monthly / quarterly calendar.
  Out-of-universe series → business-day calendar + a logged warning (best-effort).

### Daily densification
A daily series is reindexed in `parse()` to every weekday in its covered span
(`NaN` on no-data days) so federal-holiday gaps don't read as perpetual 1-day
gaps that re-fetch on every refresh. Monthly/quarterly series are left as FRED
dates them (period-start), matching their calendars. This makes the exact
holiday set irrelevant to convergence — no SIFMA holiday table to maintain.

### Series config — `series_macro.yaml` (NOT `universe_*.yaml`)
The equity universe shape (`universe`/`listed_at`/`indices`/`tickers`,
validated by `tests/data_pipelines/test_universe_yaml_lint.py`) is OHLCV-specific
and has no slot for per-series `frequency`, which this domain needs for calendar
selection. The FRED config uses a distinct filename (`series_<name>.yaml`,
outside the lint's glob) and a fit-for-purpose shape:

```yaml
universe: macro
series:
  - {id: DGS10,    frequency: daily,     category: rates,     description: ...}
  - {id: CPIAUCSL, frequency: monthly,   category: inflation, description: ...}
  - {id: GDPC1,    frequency: quarterly, category: growth,    description: ...}
```

Keeping the equity lint + its spec doc untouched was deliberate. The ~25 seed
series are a soft set — `fetch("FRED:<anything>")` works for unlisted series.

## Framework bug fixed in passing — `cache.py` NaN→NaT

`_replace_data` built insert rows via `df.iterrows()`, which coerces every cell
in a row to one common dtype: a `NaN` float in a row that also carries a
`datetime64` column comes back as `NaT` and crashes the SQLite bind
(`type 'NaTType' is not supported`). Equity domains never store `NaN`, so this
latent bug never fired; `fred_macro`'s nullable `value` is the first to hit it.
Fix: extract column-wise (preserving each column's dtype) and map any missing
value → SQL `NULL`. Pinned by `tests/data_pipelines/test_cache.py::TestNullableRoundTrip`.

## Out of scope (deliberate)

Wiring FRED series as **gbdt model features** is not in this PR. The gbdt panel
is asset-agnostic and cross-sectionally pooled; macro series are
constant-across-tickers per date — a modelling decision, not an ingestion one.
That lands as a separate plan under `docs/gbdt/`. This PR's deliverable is the
ingestion surface: `fetch("FRED:…")` returns canonical `(date, value)`,
cache-first, with a seedable universe.

Also deferred: **weekly-cadence** series (FRED's "week-ending" dating varies per
series and would need a 4th calendar) — add when a chosen series needs it.

## Tests

`tests/data_pipelines/domains/fred_macro/` (22 tests):
- `test_schema` — nullable value accepted; NaN-in-date rejected; normalize casts.
- `test_calendar` — weekday/monthly/quarterly enumerations vs FRED dating.
- `test_adapter` — CSV & JSON parse agree (D1); `"."`→NaN; daily densify;
  monthly not densified; reprocess determinism (D8).
- `test_dispatch` — fetch() with the network stubbed: daily convergence,
  **NaN survives the SQLite round-trip**, **monthly uses the monthly calendar**
  (3 rows, not a daily grid), single-source chain.

Plus `test_cache.py::TestNullableRoundTrip` (framework-level NaN→NULL guard).

Full suite green: `uv run python -m pytest tests/data_pipelines -q` → 531 passed,
2 skipped (online/key-gated smokes).

## Verification status

Offline verification complete (the suite above, plus a registration +
`list-domains` smoke). **Live network verification is pending** — this build
host's egress is firewalled to `stlouisfed.org` (`http=000`), so the real
`fredgraph`/API fetch could not be exercised here. The failure path *was*
confirmed correct (timeout → retry ×3 → `AllProvidersFailed`, cache untouched).
Run on a networked host to confirm the happy path:

```
uv run python -m data_pipelines fetch FRED:DGS10   --start 2015-01-01 --end <today> --head 5
uv run python -m data_pipelines fetch FRED:CPIAUCSL --start 2015-01-01 --end <today> --head 5
```
Expect: dense weekday rows with NaN holidays (DGS10); month-first rows (CPI); a
re-run fills no new daily gaps (calendar converges).
