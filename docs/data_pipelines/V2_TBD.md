# data_pipelines — V2 TBD

Parking lot for work surfaced during v1.7 that is *not* shipping in this
branch. Each entry should be small enough that "convert to a real plan
doc" is one short paragraph; if an entry grows past that, lift it to its
own `V<N>_PLAN.md` and a branch.

Consolidation rule: when a coherent slice of these (or a new piece of
work) crosses the "this is a project, not a chore" threshold, promote it
into a numbered plan + branch. This file is the staging area, not a
spec.

---

## Issue #3 — NSE volume divergence: three sub-fixes

Source: parity audit, Jan–Apr 2025, `NSE:RELIANCE / TCS / INFY` across
jugaad / nselib / yfinance.

Reality after digging in: 79–80/82–84 days match exactly across all
three sources. The headline "50% divergence" was the worst single day
(`2025-03-18`) normalized by the window's max-volume day.

### #3a — nselib `Series == "EQ"` filter *(latent bug; should fix)*

nselib returns the same multi-series rows (`EQ` + `BL` + `T0` + …) as
jugaad. We filter jugaad in `JugaadAdapter.parse()`; the analogous
filter is **missing** in `NSElibAdapter.parse()`. It hasn't fired in
production because jugaad serves all NSE equities first (nselib only
serves indices in our chain). Any future chain reorder, jugaad outage,
or new equity prefix routed through nselib would hit
`UNIQUE constraint failed` on the cache's `(ticker, date)` PK on every
block-deal day.

Concrete evidence — RELIANCE 2025-03-18 raw nselib response:
```
Symbol    Series  Date         OpenPrice  ClosePrice  TotalTradedQuantity
RELIANCE  EQ      18-Mar-2025  1,244.70   1,238.80    1,57,45,877
RELIANCE  BL      18-Mar-2025  1,238.85   1,238.85    1,70,000
```

Fix: mirror `JugaadAdapter.parse()`'s filter inside `NSElibAdapter.parse()`.
Re-capture `tests/data_pipelines/fixtures/nse_equities/nselib/RELIANCE_2025-04.csv`
to include at least one BL row, add a regression test analogous to
`test_jugaad_parse_drops_non_eq_series`.

### #3b — parity script metric is misleading

`scripts/data_pipelines/parity_check_nse.py` currently reports
`max(|a-b|) / max(a.max(), b.max())` per column. A single-day outlier
(e.g. `volume = 0` from yfinance on 2025-03-18) saturates the metric
and reads as "~50% disagreement" when the truth is "1 day in 82 differs."

Replace with: `(% of common days where values agree within tolerance,
top-N divergent days listed with values)`. Tolerance per column:
- date / OHLC: rel-diff > 0.5%
- volume: rel-diff > 5% **and** abs-diff > 1000 shares
- adj_close: skip (legitimate cross-source variance)

### #3c — yfinance occasional `volume = 0` on trading days

yfinance reported `volume = 0` for RELIANCE / TCS / INFY on 2025-03-18
while jugaad + nselib both had real values. NSE was open and trading.
This is a yfinance data hole, not a real zero-volume day.

Fix: in `YFinanceNSEAdapter.parse()` (and `YFinanceAdapter` for
us_equities), drop rows where `Volume == 0` for an asset whose other
rows have non-zero volume. Equivalent on the upstream side: log a
warning. Document the dropout in the schema docstring.

---

## Consensus mechanism — design parked, not shipping

User proposal: when merging into the cache, if a new row disagrees with
the cached value, fetch from another source, then keep only the values
that match across sources. Log warnings + write to
`data/consensus_log.jsonl`.

Failure modes identified (see chat transcript for full discussion):

1. **Third source often doesn't exist.** NIFTY: only nselib + yfinance.
   BSE: yfinance only. US INDEX: yfinance only. No tiebreaker available.
2. **Three-way split with no majority** collapses to existing precedence
   policy (which `merge_overlap` already implements).
3. **Disagreement is sometimes legitimate** — yfinance silent bonus/split
   adjustment on raw OHLC (HDFCBANK 2025), nselib `TRADED_QTY` vs
   yfinance ^NSEI Volume (different metrics, not disagreement). Majority
   vote would systematically discard the `QUALITY_FULL` adj_close.
4. **Disagreement is sometimes structural noise** (multi-series rows) —
   fixed at parse time, doesn't need consensus.
5. **Per-column vs whole-row semantics.** Per-column consensus implies
   per-column provenance, which the schema explicitly does not have yet
   (v2 row-level provenance is on the open-questions list).
6. **Cost amplification** — every overlap-with-disagreement triggers an
   extra fetch; non-idempotent under re-seed.
7. **`merge_cache` is currently pure / deterministic** — adding network
   calls inside it breaks the contract; consensus has to live a level up
   in dispatch.

Proposed shape (when we revisit):

- **Layer 1 (cheap, always on):** at `merge_cache` boundary, when a new
  write overlaps cached rows, compare per-column with tolerance (same
  thresholds as #3b). On trip, append structured entry to
  `data/consensus_log.jsonl` with timestamp, identifier, date, column,
  old + new values + sources, and the precedence decision applied.
  Default `merge_overlap` precedence unchanged.
- **Layer 2 (opt-in, on-demand):** `validate-consensus --identifier ...`
  CLI that explicitly fetches all configured providers for the requested
  range, compares row-by-row, prints divergence report + appends to
  `consensus_log.jsonl`. Does not change normal fetch flow.
- **Per-row sanity rules** in adapter `parse()` (yfinance volume=0
  trap, etc.) catch known structural noise before it enters the cache.

Open design question (need a decision before building): rows where only
one source is available (all US INDEX rows, ~1k NIFTY:50 backfill
rows, ~all pre-IPO ranges). Three positions —

- **Strict:** drop them. Lossy; regresses NIFTY:50 from 1488 → 422.
- **Strict-with-marker:** keep but mark `consensus: false` in meta.
  Preserves data; consumers decide whether to filter. Needs per-row
  meta which we don't have today.
- **Pragmatic:** accept as-is (today's behavior). Consensus checking
  only fires on overlap; per-row sanity rules are the safety net for
  single-source rows.

Wait until #3a–c land + we measure how often *real* (non-structural)
divergence happens. If <0.1% of rows after parse-time cleanup, the
mechanism is probably not worth building.

---

## Other known issues (not yet triaged)

Carried over from the v1.7 known-issues audit. Each needs its own
mini-decision (fix here vs defer vs document-only). The user is walking
through these one at a time.

### High priority

- **#1 yfinance silent bonus/split adjustment on raw OHLC** (both
  domains). Documented in
  `V1_IMPLEMENTATION_PLAN.md` §"Known upstream limitations" entry 4.
  `adj_close` is consistent across sources; raw OHLC is not for split
  tickers. No code fix planned — use `adj_close` for returns/volatility.
- **#2 yfinance index volume meaningless.** Documented in
  `V1_IMPLEMENTATION_PLAN.md` §"Known upstream limitations" entry 5.
  US `^SPX/^NDX/^DJI/^RUT` carry constituent-aggregate volume; NSE
  `^NSEI` carries a number ~1000× smaller than nselib `TRADED_QTY`.
  Mitigation requires wiring a futures-volume feed; not in v1.7 scope.

### Medium priority

- **#4 Stooq IP-gated on dev machine.** Works from other networks per
  user research; no code change needed. Document in adapter docstring
  if not already.
- **#5 Concurrent seeds collide** on D8 immutability (same-second raw
  filename) and `(ticker, date)` PK. Current contract: single-process
  per `data_root`. Documented; no fix planned for v1.
- **#6 Tiingo retry retrofit deferred.** Replace bespoke
  `_request_with_retry` in `TiingoAdapter` with the new shared
  `data_pipelines.retry.call_with_retry`. Pure refactor.

### Low priority

- **#7 Row-level provenance is approximate.** `_meta.sources[]` lists
  per-range provenance; row-level "which provider supplied this row"
  is heuristic. Lift to per-row column when a consumer needs it.
- **#8 Cache-format version migration informal.** Raw filename format
  is pinned (`{adapter}_{start}_{end}.{ext}`), but no `schema_version`
  migration story. Decide on a versioning scheme before the first
  breaking schema change.
- **#9 `BSE:` equity coverage not end-to-end validated.** Chain is
  `[yfinance]` only. Smoke-test a known BSE ticker (e.g. `BSE:500325`
  for Reliance) and capture in a regression test.
- **#10 Currency / units no first-class schema home.** Stored per-source
  in `_meta.sources[].currency`. Lift to `Schema.column_units` or
  similar when the first cross-currency consumer appears.

---

## PR #1 review follow-ups (deferred warnings)

From the PR #1 code review. Criticals (#1, #2) and trivial warnings
(#7, #8, #9 + goal.md nit) were fixed in the PR. The items below were
flagged as worth fixing but are non-trivial — parked here.

### `_replace_data` uses `iterrows()` — `cache.py:389`

Row-by-row iteration in the bulk write path. `df.itertuples()` or
`df.to_sql(if_exists="append")` would be measurably faster on bulk
seeds (500+ tickers × multi-year ranges). Not correctness; not currently
a bottleneck per perf observations, but worth a benchmark + swap if the
benchmark confirms the win. Touch points: `_replace_data` plus the
SQLite type-coercion (pd.Timestamp → ISO string, numpy scalars →
Python scalars) currently done inside the loop body — that coercion
has to land somewhere outside the loop too.

### `merge_overlap` NaN injection — `us_equities/schema.py` + `nse_equities/schema.py`

When `new_quality` is inferior to existing, the merge preserves the
existing `adj_close` via `resolved["date"].map(existing_indexed)`. For
any date in `new` not present in `existing`, the `.map()` produces
`NaN`, which silently survives into the processed layer.

Needs a design decision before fixing:
- **Raise** if any preserved column would be NaN after merge (strictest,
  surfaces the inconsistency to the caller).
- **Forward-fill** from `new`'s lower-quality value (silent fallback).
- **Forward-fill from the latest known good value** (carry forward, can
  drift significantly if the gap is large).

Not a 1-line fix — pick the policy first, then implement + test.

### Retry-everywhere pass (combines review #5, #6, and v1.7 known-issue #6)

Three related items, worth handling together as one consistent retry
hardening pass:

1. **US `YFinanceAdapter` does a bare `yf.Ticker().history()` call** —
   no retry, no circuit breaker. Asymmetric with `YFinanceNSEAdapter`
   (NSE) which correctly wraps the call in `call_with_retry`. Lift to
   `call_with_retry` like the NSE adapter does.
2. **`TiingoAdapter._consecutive_429s` / `_circuit_open_until` are
   class-level mutable state** mutated via read-modify-write (`+= 1`).
   Single-threaded today, latent concurrency bug. Either move to
   `threading.Lock`-guarded instance state, or replace the whole
   bespoke retry+circuit-breaker with `call_with_retry` + a small
   per-instance circuit-breaker helper.
3. **Tiingo retry retrofit** (already on the v1.7 known-issues list as
   #6) — replace `TiingoAdapter._request_with_retry` with
   `data_pipelines.retry.call_with_retry`.

Doing them together avoids touching the same adapters twice and lets
us pick a consistent retry surface across all 6 adapters
(stooq / tiingo / yfinance × 2 / jugaad / nselib).
