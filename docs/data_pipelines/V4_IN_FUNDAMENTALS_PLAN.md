# data_pipelines V4 — `in_fundamentals`: Indian quarterly fundamentals domain

**Status: in progress (2026-07-09).**

## Why

The nifty500 gbdt cells are technical-only: the F18 valuation features that robustly
helped the sp500 +50%/50d champion (`_272`/`_273`) have no Indian counterpart because
there is no Indian fundamentals data in the cache. The nifty500 200d aligned cells sit
at R-p@3 0.38–0.44 on base rates 0.29–0.49 — weak lift with an obvious missing feature
axis. **This plan is ingestion + storage only** (the V3 posture): fetch Indian quarterly
results, land them in the cache with the module's atomicity/immutability/determinism
guarantees. The INR valuation panel and the gbdt F18-IN join are separate follow-up
plans.

Domain #5, closing the `V3_TBD.md` "NSE fundamentals" deferral.

## Recon facts (verified live from this host, 2026-07-09)

- **NSE corporates-financial-results API works from this host**:
  `https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol=RELIANCE&period=Quarterly`
  returns 200 + full JSON even when the homepage warmup 403s (the cookie jar from the
  403 response suffices). RELIANCE: **130 records spanning 2005 → 2025**.
- Each record carries **exact point-in-time filing timestamps** —
  `filingDate: "16-Jan-2025 20:20"`, `broadCastDate` with seconds — plus `audited`,
  `consolidated`/`Non-Consolidated`, `bank` (Y/N), `period`, `relatingTo`
  (First/Second/Third quarter), `fromDate`/`toDate`, and an `xbrl` attachment URL.
  This is *better* than the US: filed dates are native, no enrichment pass needed.
- **XBRL files live on `nsearchives.nseindia.com`** — the host already known reliable
  from this machine (`[[project-nse-data-quirks]]`). Sample INDAS filing parses
  cleanly with the shared NSE/BSE **`in-bse-fin` Ind-AS results taxonomy**:
  `RevenueFromOperations`, `ProfitLossForPeriod`,
  `DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations`,
  `PaidUpValueOfEquityShareCapital`, segment blocks; `unitRef="INR"` absolute values
  (`decimals="-7"`); contexts distinguish the quarter (`OneD`) from YTD (`FourD`).
- **Quarterly cash flow does not exist in India**: SEBI LODR mandates cash-flow
  statements only half-yearly. Quarterly records carry P&L + EPS + capital only.
- Indian fiscal year is Apr–Mar, but **quarters end on calendar quarter-ends**
  (Q1 FY = Apr–Jun ends Jun 30) — the calendar-quarter grid works unchanged, and
  there is no 52/53-week wobble.
- BSE (`api.bseindia.com`) reachable (fallback candidate, same taxonomy);
  screener.in reachable (no filed dates); yfinance `.NS` ~5 quarters deep.

## Design decisions

1. **Domain** `in_fundamentals`, prefix **`INFUND:`** (`INFUND:RELIANCE`). Universe
   **derived at load time from the nse_equities universe YAMLs** (default
   `nifty500`; any nse_equities universe name accepted), `NSE:X` → `INFUND:X`,
   `INDEX:*`/`NIFTY:*` excluded — the V3 no-duplicate-ticker-list rule.
2. **Schema**: the 12 us_fundamentals columns **+ `consolidated`** (float64 nullable,
   1.0/0.0) appended last. Money in **INR millions**, shares in millions, EPS in
   INR/share. `ocf`/`capex`/`fcf` stay in the schema but are **expected-NaN**
   (recon: no quarterly CF in India) — schema symmetry keeps the valuation-panel
   code path reusable; fcf-based ratios will honestly be NaN. Half-yearly CF fills
   are parked in `V4_TBD.md`.
3. **Grid**: same calendar quarter-end grid + snap util (imported from
   us_fundamentals — domain #2 for fundamentals informs the abstraction; a shared
   import over a copy). Indian `toDate` values are exact quarter-ends, so snap is a
   no-op safety.
4. **Calendar**: reuse `QuarterEndCalendar` with **`lag_days=60`** (SEBI: 45 d for
   Q1–Q3 results, 60 d for Q4/annual audited — one knob covers both; late filers
   soft-fail and retry next seed).
5. **Adapter chain v1: single provider — `nse_xbrl`** (metadata JSON + XBRL
   downloads + native filed dates in one adapter). The fred_macro single-source
   precedent applies; BSE and yfinance-IN fallbacks are parked in `V4_TBD.md`
   (BSE endpoint params need their own recon; yfinance depth is marginal).
6. **Raw payloads** (immutable, reprocess-from-raw): one JSON envelope per fetch —
   `{"symbol", "fetched_utc", "metadata": [records...], "xbrl": {seqNumber: xml}}`
   under `data/raw/nse_xbrl/in_fundamentals/-/<TICKER>/...json`. The metadata
   records are kept verbatim (audit trail for filing dates + flags).
7. **Politeness**: session warmup (cookie jar) + browser UA; the main-site API
   throttled ~2.5 s; `nsearchives` XBRL downloads throttled ~0.8 s + jitter
   (archive host, static files); exponential backoff on 401/403/429 via the shared
   retry helper. Gap-bounded fetches: only XBRLs whose quarter-end falls in the
   requested `[start, end]` are downloaded (per-filing files make incremental
   fetches natural — unlike the US full-history pages).
8. **Point-in-time policy**: per (grid date, basis) the **earliest-filed record
   wins** (as-first-published; revised/re-filed results lose — `oldNewFlag`/`reInd`
   never used for identity). Per grid date, **consolidated is preferred over
   standalone** when both exist; the `consolidated` column records which basis
   served the row. Cache merge = per-cell first-written-wins + fcf recompute
   (identical `merge_overlap` policy to us_fundamentals).
9. **Banks** (`bank == "Y"`, ~10% of nifty500): the banking results taxonomy has no
   `RevenueFromOperations`. v1 ships a small bank-tag fallback list (interest
   earned + total income) and accepts honest-NaN revenue where tags miss;
   net_income/EPS tags are shared. Full bank mapping quality pass → `V4_TBD.md`.

## XBRL normalization design

- **Context selection**: parse `xbrli:context` elements → `(startDate, endDate)`
  per `contextRef`; the quarter context is the duration **70–115 d ending on the
  record's `toDate`** (rejects YTD/`FourD` contexts structurally rather than by
  name). Instant facts (paid-up capital) use the instant context at `toDate`.
- **Tag priority per metric** (namespace-agnostic local names; the taxonomy prefix
  drifts `in-bse-fin`/`in-capmkt` across eras):
  - revenue: `RevenueFromOperations` → `TotalRevenueFromOperations` →
    `RevenueFromInterestAndDividendOperations` (banks) → `InterestEarned` (banks) →
    `TotalIncome` (last resort, includes other income — documented approximation)
  - net_income: `ProfitLossForPeriod` → `ProfitLossForPeriodFromContinuingOperations`
  - eps_basic / eps_diluted:
    `{Basic,Diluted}EarningsLossPerShareFromContinuingAndDiscontinuedOperations` →
    `{Basic,Diluted}EarningsLossPerShare` → per-continuing-operations variants
  - shares: **derived** `shares_basic = net_income / eps_basic`,
    `shares_diluted = net_income / eps_diluted` (weighted-average-consistent by
    construction; INR-M / (INR/share) = M shares); fallback
    `PaidUpValueOfEquityShareCapital / FaceValueOfEquityShareCapital` (actual,
    not weighted — fallback only, same caveat class as the US dei skip).
  - ocf/capex/fcf: NaN (see D2).
- **Units**: `unitRef INR` absolute → ÷1e6 (INR M); `INRPerShare`-class units as-is;
  exact-unit-key discipline like EDGAR.
- **filed_date**: record `filingDate` (fallback `broadCastDate`), normalized to
  day resolution — same semantics as the US column (evening filings step the
  valuation panel on filing day; the F18 join's existing lag posture applies).
- **Determinism**: `parse()` is a pure function of the envelope (D8); re-parse of
  the same raw bytes is byte-identical.

## Phases

- **Phase 0 — worktree branch + plan doc.** [this commit]
- **Phase 1 — domain skeleton**: `schema.py` (+`consolidated`, shared snap import),
  `registry.py` (`INFUND:`, namespace `'-'`), `universe.py` (nse_equities-derived),
  `config.py`, `__init__.py` (Domain + registration + merge_overlap), `__main__.py`
  import patch. Tests: identifier parsing, universe derivation, schema/NaN
  round-trip, merge policy, calendar lag.
- **Phase 2 — nse_xbrl adapter**: session/cookie handling, metadata fetch,
  gap-bounded XBRL downloads, envelope raw, pure `parse()` per the design above.
  Fixture tests from the live RELIANCE recon payloads (metadata slice + one full
  XBRL): context selection, tag priority, unit scaling, consolidated preference,
  earliest-filed dedup, filed-date parsing, derived shares.
- **Phase 3 — pilot seed + validate**: RELIANCE (conglomerate), INFY (IT,
  ADR-listed), HDFCBANK (bank — exercises the bank-tag fallback), TCS. Spot-check
  against published quarterly numbers; verify cache round-trip + idempotent
  re-fetch.
- **Phase 4 — full nifty500 seed** (~500 tickers; XBRL era 2016+ target). Disk
  pre-flight; sequential with throttle; expect hours — run overnight/background.
  Record coverage/NULL-rate/provider stats here (V3 Outcome-section pattern).
- **Phase 5 — docs + ship**: `adding_a_domain.md` retrospective (domain #5),
  `goal.md` deferred-list update, CLAUDE.md data bullet, `V4_TBD.md` (BSE fallback,
  yfinance-IN, half-yearly CF fills, bank taxonomy pass, INR valuation panel plan
  pointer, screener cross-validation). PR + review/merge.

**Explicitly out of scope** (follow-up plans): INR valuation panel build
(`scripts/valuation` is USD/us_equities-coupled — needs its own small plan), gbdt
F18-IN feature wiring, balance-sheet items, annual statements, point-in-time index
membership.

## Verification

- `uv run python -m pytest tests/data_pipelines -q` green (existing domains
  untouched; framework unmodified).
- Live: `fetch INFUND:RELIANCE` returns rows on the calendar quarter grid with
  non-null `filed_date` everywhere, revenue(2024-12-31, consolidated) matching the
  published ₹ figure; second call is a pure cache hit.
- Reprocess-from-raw determinism: `parse()` twice on the same envelope →
  byte-identical frames.
