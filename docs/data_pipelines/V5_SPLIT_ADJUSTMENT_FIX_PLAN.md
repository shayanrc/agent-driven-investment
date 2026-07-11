# V5 — Split-adjustment correctness fix (gbdt prices + NSE adj_close)

**Status:** PLAN — discovered 2026-07-11 while investigating the NSE F18 coverage
cliff. Scope confirmed by the user: **full fix, both universes.**

## The bug

**gbdt trains on split-UNadjusted prices.** Every feature in `src/gbdt/features.py`
reads `panel["close"]` (18+ sites: returns, volatility, drawdown, momentum, GK/YZ
vol, trend, bollinger, …) and the target in `src/gbdt/targets.py` reads raw
`close`/`high`/`low`. None of them use `adj_close`. There is **no split adjustment
anywhere** in the gbdt loader (`src/gbdt/data.py`) or the equities domains.

Consequence: at every stock split/bonus the raw close jumps by the split ratio —
AAPL 500→129 (2020 4:1), RELIANCE 2655→1334 (2024 2:1) — injecting a fake
~−50–75% one-day return into **features** (and polluting every lookback window that
spans it — a 200-day vol feature is corrupted for 200 days) AND a fake
drawdown-breach / return into the **label** (the target's `max_drawdown` gate and
forward-return both key off raw close/high/low).

### Blast radius (measured)
- **nifty500:** 345 split-like spikes (|1d move| > 35%) across **118 of ~500
  tickers (~24%)**. 48/374 cached tickers have `adj_close == close` everywhere
  (fully unadjusted); more (incl. RELIANCE) have splits missing from `adj_close`
  even when partially dividend-adjusted.
- **sp500:** `us_equities` has a mostly-correct `adj_close` (AAPL smooth
  121.3→125.2), but **gbdt ignores it and uses raw `close`** → US splits inject the
  same spikes. **The deployed sp500 champions were trained on this.** US splits are
  rarer, so the impact is smaller but non-zero and was unmeasured.

## Root cause

1. **gbdt consumes the wrong column.** The `data_pipelines` contract
   (`docs/data_pipelines/goal.md`) is explicit: *"`adj_close` always means
   split-and-dividend adjusted, regardless of source."* The `valuation` module
   correctly uses `adj_close`. gbdt uses raw `close` — a latent bug from day one.
2. **NSE `adj_close` violates its own contract.** `nse_equities` merges
   provider-mixed data: yfinance supplies split-adjusted OHLC + `adj_close="full"`;
   jugaad-data / nselib supply **raw** NSE bhav OHLC + `adj_close="none"` (==close).
   The D4 merge preserves a "full" adj_close when one exists, but a ticker only ever
   fetched via jugaad/nselib (RELIANCE) keeps `adj_close == raw close` — splits
   never applied. So `adj_close` cannot be blindly trusted for splits on NSE.
3. **Provider-mixed raw OHLC.** Because yfinance back-applies splits to OHLC while
   jugaad/nselib do not, the raw `open/high/low/close` themselves are inconsistently
   adjusted across tickers/dates — and row-level provenance is not stored ("v2").

## Fix design

**Target state:** the OHLCV that gbdt features + target consume is **fully
split-adjusted** (open/high/low/close ÷ future-split-factor, volume × factor), in
BOTH universes, and `nse_equities.adj_close` honors its documented contract.

### The design fork (needs a call)
The clean adjustment factor is the **authoritative corporate-action series**
(`fetch_splits`, already used by `valuation`) — NOT the unreliable cached
`adj_close`. But applying a splits factor to OHLC that is *already* split-adjusted
(yfinance rows) double-adjusts. Two ways to get a consistent base:

- **(A) Detect-and-normalize (offline, deterministic, recommended).** For each
  ticker, test the raw `close` for a split-ratio discontinuity at each known split
  ex-date. If present → the series is raw → apply the cumulative split factor. If
  absent → already adjusted → leave. Produces a canonical split-adjusted OHLCV
  purely from cached data + the splits series. No network, reproducible.
- **(B) Re-seed from one consistent provider.** Re-fetch all NSE (and optionally US)
  tickers from a single split-adjusting provider (yfinance / Tiingo) so `close` is
  uniformly adjusted. Simpler logic but network-dependent (yfinance-.NS is flaky per
  `[[project-nse-data-quirks]]`) and re-pays API cost; changes raw cache bytes.

**Recommendation: (A)** — deterministic, offline, auditable, and it fixes the
`adj_close` contract at the same time (write the normalized series into `adj_close`).

### Phases
1. **Canonical split-adjuster** (`data_pipelines` or a gbdt data-layer helper):
   given cached OHLCV + `fetch_splits`, emit fully split-adjusted OHLCV via approach
   (A). Unit-test on RELIANCE (2024), AAPL (2020), a bonus ticker, and an
   already-adjusted yfinance ticker (must be a no-op).
2. **Repair `nse_equities.adj_close`** to the contract (split-and-dividend adjusted)
   for all rows — so `valuation` (market_cap) becomes correct too, retiring the
   pre-2019 backfill's split-basis caveat.
3. **gbdt loader** returns split-adjusted OHLCV (`src/gbdt/data.py`), so features +
   target are correct with NO change to feature code. Surface this in `gbdt/goal.md`
   decisions (it's a correctness fix, not an architecture change).
4. **Rebuild** the NSE valuation panel (+ re-apply the pre-2019 screener backfill on
   the corrected prices) and the gbdt feature-matrix caches.
5. **Re-run** all affected experiments: nifty500 lattice + the F18 sweep/finetune
   (#34) + stratified (#31); **re-validate the sp500 champions** on adjusted prices
   (`/daily-predictions` retrain + backtest deltas) — a champion may move.
6. **Verify:** no |1d move|>35% artifact remains except genuine gaps; champion
   backtest deltas quantified; F18 reruns now on clean features+labels.

## Invalidation
Every prior gbdt experiment on either universe used raw close → all carry a
split-artifact confound (severity ∝ split density). The nifty500 F18 memos already
flagged for the coverage cliff (`_285`/`_286`) compound with this. A blanket
caveat + a re-run manifest will be produced once Phase 3 lands.

## Out of scope / follow-ups
- Row-level provenance in `nse_equities` (the "v2" that would make detection
  unnecessary).
- Dividend-adjustment correctness (this plan targets SPLITS; total-return dividend
  adjustment is a separate axis — features are price-move based, so splits dominate).
