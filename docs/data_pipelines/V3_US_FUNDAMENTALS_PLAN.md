# data_pipelines V3 — `us_fundamentals`: quarterly company fundamentals domain

**Status: in progress (2026-07-03).**

## Why

The gbdt models consume only price/volume features (F1–F16). The next candidate edge is
**fundamentals** — revenue, net profit, free cash flow — from which per-share metrics
(EPS, revenue/share, FCF/share) and valuation ratios (PE, PS, P/FCF) can later be derived
by joining daily prices. **This plan is ingestion + storage only**: fetch the data, land
it in the cache with the module's atomicity/immutability/determinism guarantees. Feature
transforms and model wiring are a later, separate plan.

Per `goal.md` ("Not a fundamentals normalizer — … would likely be a different domain in
this module rather than a separate module, but deferred"): the deferral ends here. Like
the price domains, fundamentals get a **tiered adapter chain** — primary, secondary,
tertiary — not a single scraper.

## Recon facts (verified live from this host, 2026-07-03)

- `macrotrends.net` reachable (200, ~0.7 s/page). One request
  (`/assets/php/ticker_search_list.php`, 387 KB) returns the full **ticker→slug map**
  (6,601 entries). A wrong slug 301-redirects to the canonical URL but **drops the
  `?freq=Q` query** → resolve slugs from the map, never rely on redirects.
- Two pages per ticker carry everything: `income-statement?freq=Q` (22 line items:
  Revenue, Net Income, Shares basic/diluted, EPS, …) + `cash-flow-statement?freq=Q`
  (29 items: OCF, PP&E change, …), embedded as `var originalData = [...]` JSON,
  quarterly **2011→now** (59 quarters for AAPL), units **$M**.
- **Macrotrends does NOT snap off-cycle fiscal calendars**: WMT shows true fiscal
  month-ends (2025-01-31, 2025-04-30, …); AAPL's fiscal end 2026-03-28 shows as
  month-end 2026-03-31. Dates must be normalized to a uniform grid in `parse()`.
- **SEC EDGAR `companyfacts`** (official, free, ~10 req/s, requires a UA with contact
  email): full XBRL history ~2008→now **with `filed` dates** (point-in-time truth —
  AAPL's quarter ending 2026-03-28 was filed 2026-05-01).
- **yfinance quarterly statements**: work, but only the **last ~5 quarters** → tertiary
  "newest quarter last-resort", mirroring yfinance's role in the us_equities chain.

## Design decisions

1. **Domain** `us_fundamentals`, prefix **`FUND:`** (`FUND:AAPL`). Universe **derived at
   load-time from the us_equities universe YAMLs** (union sp500 ∪ russell1000 ∪
   nasdaq100, minus `INDEX:*`, exchange prefix stripped → ~1,010 identifiers) — no
   duplicate ticker list to drift.
2. **Schema** (wide, one row per ticker per grid date; all metric cols `nullable=True`):
   `date` (datetime64, **grid** = calendar quarter-end), `fiscal_period_end` (datetime64,
   true period end), `filed_date` (datetime64 nullable — EDGAR fills it; macrotrends
   NULL), `revenue, net_income, ocf, capex, fcf` (float64, **$M**; `fcf = ocf − capex`,
   NULL-propagating; capex stored positive-outflow), `shares_basic, shares_diluted`
   (M shares), `eps_basic, eps_diluted` (USD). A superset of the three headline metrics,
   but all from the same two requests — shares/EPS are required for the later per-share
   ratios.
3. **Grid snap (shared util, all adapters):** snap `fiscal_period_end` **forward** to the
   next calendar quarter-end, with a **7-day backward tolerance** (a fiscal end ≤ 7 days
   after a grid date snaps back to it — fixes the 52/53-week wobble: AAPL fiscal Q3
   FY2017 ended 2017-07-01 → grid 2017-06-30, not a Sep-30 collision). WMT Jan-31 →
   Mar-31 (conservative: the quarter ended before its grid date ⇒ no look-ahead).
   Post-snap collisions (fiscal-year-change stubs): keep the row whose
   `fiscal_period_end` is closest to the grid date, warn.
4. **Calendar** (gap detection): `QuarterEndCalendar(lag_days=45)` — emits a calendar
   quarter-end only once `today ≥ grid_date + 45 d`, so the chain isn't re-hit daily
   while companies haven't reported yet. Late filers → dispatch's existing soft-fail
   (cache preserved, retry next seed). Quarterly cadence ⇒ re-fetch bursts ~4×/year.
5. **Adapter chain** (`chain_for_gap`, any gap size): **macrotrends → edgar → yfinance**.
   Macrotrends pages always return full history, so seed and update are the same request.
6. **Raw payloads** (immutable, reprocess-from-raw): macrotrends = one JSON envelope
   `{"slug": …, "pages": {"income-statement": <html>, "cash-flow-statement": <html>},
   "urls": …}` per fetch (`ext="json"`; the Adapter contract is one raw Path per fetch);
   EDGAR = companyfacts JSON verbatim; yfinance = JSON of the frames. The slug map and
   the SEC CIK map (`company_tickers.json`) also land as raw (pseudo-ticker `_meta`) +
   in-process memo.
7. **Politeness:** macrotrends throttled ~1.5 s + jitter between requests (adapter-level
   config knob), exponential backoff on 403/429, browser UA; seed sequential
   (`--jobs 1`). EDGAR UA carries a contact email per SEC policy. Full-union seed ≈
   2,020 macrotrends requests ≈ 60–75 min, one-time; per-ticker failures fall through
   to EDGAR.
8. **Point-in-time policy:** stored values are earliest-filed (EDGAR) / as-published
   (macrotrends, which shows restated history — a documented divergence). `filed_date`
   is the causal-lag hook for the modeling phase; deriving features stays OUT of this
   plan.

## EDGAR normalization design (validated against the AAPL companyfacts fixture)

- **Tag priority lists** per metric — revenue:
  `RevenueFromContractWithCustomerExcludingAssessedTax` → `…IncludingAssessedTax` →
  `SalesRevenueNet` → `SalesRevenueGoodsNet` → `Revenues`; net_income: `NetIncomeLoss` →
  `ProfitLoss` → `NetIncomeLossAvailableToCommonStockholdersBasic`; ocf:
  `NetCashProvidedByUsedInOperatingActivities` → `…ContinuingOperations`; capex:
  `PaymentsToAcquirePropertyPlantAndEquipment` → `PaymentsToAcquireProductiveAssets`;
  shares/EPS: `WeightedAverageNumberOf{,Diluted}SharesOutstanding{Basic,}`,
  `EarningsPerShare{Basic,Diluted}` (+ `…BasicAndDiluted` fallbacks).
- **Dedup**: pool all tags' duration points per metric, key by `(start, end)`, pick by
  **min `(filed, tag_rank, accn)`** — earliest-filed-wins keeps point-in-time values;
  comparatives, 10-K/A restatements, and re-release 8-Ks lose automatically. (Fixture
  evidence: AAPL's Jan-2010 retrospective subscription-accounting restatement; the ASC
  606 transition where a naive prefer-first-tag would pick a 2019 comparative over the
  original 2018 10-Q.) `fy`/`fp` fields are **filing-relative** and never used for
  period identity.
- **Quarter recovery, one unified rule:** direct quarters = durations 70–115 d;
  **consecutive-YTD differencing** within same-`start` groups (all YTD durations in a
  fiscal year share the fiscal-year start) covers both the YTD-only cash-flow pattern
  (53/71 AAPL OCF quarters come from differencing) and Q4 = FY − 3Q-YTD with no special
  case. `filed_date` of a differenced quarter = max of the two components' filed dates.
- **Shares Q4** are never directly reported → `4×FY − ΣQ1..3` (validated: implies AAPL's
  reported Q4 EPS exactly); missing EPS filled by `net_income / shares` behind a
  `DERIVE_EPS = True` module constant.
- **Skip `dei:EntityCommonStockSharesOutstanding`** (instant fact, cover-page date,
  actual-not-weighted shares — wrong semantics for `shares_basic`).
- Units: `USD` → ÷1e6 ($M); `shares` → ÷1e6 (M); `USD/shares` as-is; exact unit key only.
- Fixture recoverability: revenue/NI 72 quarters, OCF/capex 71, shares 71 (2008→2026);
  every fiscal year 2009–2025 recovers 4/4 — no per-filing fetches needed.
- Known punts: banks/insurers with no revenue-analog tag → NULL revenue; non-USD filers
  → NULL; sub-70-day fiscal-transition stubs dropped; per-class EPS (multi-class
  entities) first-match; cross-tag YTD differencing at accounting transitions accepted
  as fallback-adapter risk.

## Phases

- **Phase 0 — branch + plan doc.** [this commit]
- **Phase 1 — domain skeleton** (`src/data_pipelines/domains/us_fundamentals/`):
  `schema.py` (Schema + shared snap util), `registry.py` (`FUND:` parser, namespace
  `'-'` like FRED), `calendar.py` (`QuarterEndCalendar`), `universe.py` (derived-union
  loader), `config.py`, `__init__.py` (Domain + registration), CLI patch in
  `__main__.py` (the 4-line fred_macro precedent). Tests: snap unit tests (WMT,
  AAPL-2017-07-01), cache NaN round-trip (the fred_macro lesson,
  `adding_a_domain.md`), calendar lag/grid, universe union, seed routing.
- **Phase 2 — macrotrends adapter** (primary): slug-map fetch/memo/raw (+ dash/dot
  normalization, e.g. BRK-B), 2-page fetch → JSON envelope raw, pure `parse()`,
  throttle+backoff. Fixture tests (trimmed AAPL + WMT HTML from recon).
- **Phase 3 — EDGAR adapter** (secondary): CIK map, companyfacts fetch, `parse()` per
  the design above. Trimmed-fixture tests: direct-vs-differenced equality, Q4
  derivation, earliest-filed dedup, snap collisions.
- **Phase 4 — yfinance adapter** (tertiary): thin; quarterly frames → canonical rows.
- **Phase 5 — seed + validate.** Disk pre-flight; seed **sp500 first**, validate
  (coverage %, NULL rates, rows/ticker, spot-checks AAPL/WMT/GOOGL/BRK-B, EDGAR
  fall-through table), then the remaining union. Record timings + failures below.
- **Phase 6 — docs + ship.** `adding_a_domain.md` retrospective (domain #4), `goal.md`
  deferred-list update, CLAUDE.md data bullet, `V3_TBD.md` follow-ups (annual
  statements, balance-sheet metrics, NSE fundamentals, macrotrends-vs-EDGAR
  cross-validation, `/fetch-fundamentals` skill, filed-date enrichment for
  macrotrends-served rows). PR + review/merge.

## Verification

- `uv run python -m pytest tests/data_pipelines -q` green.
- Live: `fetch FUND:AAPL` returns ~59 rows, canonical schema, revenue(2026-03-31) =
  111,184 $M; second call is a pure cache hit (deterministic, byte-identical).
- Chain: forced macrotrends failure → EDGAR serves; EDGAR AAPL 2026-03-31 row has
  `filed_date = 2026-05-01`.
- Existing domains untouched (no framework edits): full suite green.

## Outcome

**sp500 seed (2026-07-03, 15:37→17:23 IST, ~106 min): 503/503 succeeded, 0 failed.**

- **Rate limit discovered live:** macrotrends 429s at sustained rates faster than
  ~1 req/5-6 s. At the initial 1.5 s throttle every request burned ~3 retry
  attempts (self-healing via backoff — pace settled at ~13 s/ticker); the default
  was raised to 4.5 s + 1 s jitter for clean 200s at the same throughput. Exactly
  ONE ticker exhausted retries mid-run (ACGL) and was served by EDGAR instead —
  the chain fallthrough working as designed.
- **Provider breakdown:** 486 pure-macrotrends · 15 mixed (macrotrends + edgar +
  yfinance partial fills) · 2 edgar-only.
- **Coverage:** 28,672 rows; median 59 quarters/ticker (macrotrends 2011→now),
  max 68 (EDGAR-served, 2008→now), min 8 (recent IPOs). Earliest grid date
  2009-06-30, latest 2026-06-30 (off-cycle fiscal quarters land ahead of the
  demanded grid). The demanded grid starts at the gbdt canonical anchor
  (2019-01-01); stored history is deeper because providers return full history
  and the dispatcher's merge is unclipped.
- **NULL rates:** revenue 0.5% · net_income 0.6% · ocf 0.03% · capex/fcf 0.8% ·
  shares 0.6% · eps_diluted 0.01%. `filed_date` 98.6% NULL as designed (only
  EDGAR-served rows carry it — uniform enrichment pass parked in `V3_TBD.md` §4;
  do it before building causally-lagged model features).
- **Spot checks:** AAPL 2026-03-31 revenue 111,184 / OCF 28,702 / capex 1,971 /
  FCF 26,731 $M (all three providers agree independently); WMT fiscal Apr-30 →
  Jun-30 grid with a plausible negative-FCF quarter; GOOGL and BRK-B internally
  consistent (EPS ≈ NI/shares).
- **Footprint:** raw audit trail ~147 M (macrotrends HTML envelopes 115 M +
  EDGAR companyfacts 32 M) for sp500.
- Validation is re-runnable:
  `uv run python -m scripts.data_pipelines.validate_us_fundamentals`.

**Full-union seed (sp500 ∪ russell1000 ∪ nasdaq100 = 1,015 tickers): 1,014
cached, 1 genuine gap.** Run 17:35→19:12 IST at the 4.5 s throttle — **zero
rate-limit exhaustions** (the bump held).

- **53,659 rows.** Provider mix: 892 pure-macrotrends · 90 all-three (partial
  fills) · 11 macrotrends+yfinance · 8 edgar+yfinance · 7 edgar+macrotrends · 6
  edgar-only. So ~99% of tickers get macrotrends' deep history; EDGAR/yfinance
  fill the primary's misses.
- **Three raw seed "failures" triaged — two were adapter bugs, fixed:**
  - `CWEN-A` (Clearway Class A): macrotrends spells it `CWENA` (dash removed,
    not the `BRK.B` dot form). Added the dash-removed resolution variant →
    served by macrotrends, 29 quarters.
  - `CAI` (Caris): a corrupt slug-map entry (U+FFFD mojibake) crashed urllib's
    ASCII request-line encode as a raw `UnicodeEncodeError`, bypassing the
    chain. Non-URL-safe slugs are now rejected at resolution → clean
    `EmptyPayload` → EDGAR served it (8 quarters — 2025 IPO).
  - `ANSS` (Ansys): **genuine survivorship gap** — acquired by Synopsys and
    delisted in 2025, so macrotrends dropped the page, EDGAR has no CIK, and
    yfinance has no history. A delisted name still in the index YAML; nothing
    to fetch. Left uncached (per-ticker isolation logged it and moved on).
- **NULL rates** (union): revenue 1.4% · net_income 1.1% · ocf 0.2% ·
  capex/fcf 1.2% · shares ~2% · eps_diluted 0.3% — slightly higher than sp500
  (small/mid-caps miss more line items), all well within usable range.
  `filed_date` 95.5% NULL (EDGAR-served rows only; enrichment pass = `V3_TBD` §4).
- **History:** median 59 quarters/ticker, max 68 (EDGAR 2008+), min 4 (2025-26
  IPOs); grid 2009-06-30 → 2026-06-30.
- Spot checks unchanged from the sp500 pass (all three providers agree on AAPL;
  WMT off-cycle fiscal; GOOGL/BRK-B internally consistent).

**Bottom line:** 1,014/1,015 (99.9%) of the equity universe now has quarterly
fundamentals cached with full audit trail; the single miss is a legitimately
delisted ticker. Re-runnable via
`uv run python -m scripts.data_pipelines.validate_us_fundamentals`.
