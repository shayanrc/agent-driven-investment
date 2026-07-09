# data_pipelines V4 follow-ups (parking lot)

Out-of-scope items discovered during the `in_fundamentals` build
(`V4_IN_FUNDAMENTALS_PLAN.md`). Promote to a real plan when a coherent slice
is big enough.

## 1. Insurer history backfill (pre-Mar-2025) — recon complete, parked 2026-07-09

The 9 nifty500 insurers (GICRE, HDFCLIFE, ICICIGI, ICICIPRULI, LICI, NIACL,
NIVABUPA, SBILIFE, STARHEALTH) have **no structured exchange filings before
the Integrated Filing regime (Mar-2025)** — they were governed by IRDAI
disclosure norms and sat outside the exchanges' results-XBRL system (banks
were in; insurers got PDFs). Verified live 2026-07-09:

- **NSE**: classic results API returns zero records for insurers under ANY
  period label. The corporate-announcements archive HAS their results back to
  IPO but as board-meeting-outcome **PDFs** (e.g. SBILIFE Q3 FY23
  `BM21012023_*.pdf`, stamped 21-Jan-2023).
- **BSE**: API host reachable but endpoint params undiscovered (needs recon
  of their page JS). Since the XBRL exclusion was regulatory, BSE most
  likely holds the same PDFs (~15% odds of insurer XBRL pre-2025).
- **IRDAI public disclosures** (L-forms / NL-forms): quarterly, deep history,
  but per-company PDF/Excel on each insurer's site. No uniform API.
- **screener.in**: insurer quarterlies digitized ~10y back. Numbers yes,
  filed dates no.

**The pragmatic hybrid, if ever needed**: numbers from screener.in (raw layer
flags provenance) + **filed dates from the NSE announcements archive**
(results-PDF announcements carry exact timestamps; quarter → announcement
matching recovers true point-in-time dates). Medium effort. PDF parsing of
the exchange/IRDAI documents is NOT recommended (fragile, high effort for
1.8% of the universe).

Why parked: insurers already have 5 filed-dated quarters forward (Mar-2025 →
Mar-2026), clearing the 4-quarter TTM bar for revenue/NI today; the real
blocker for their valuation ratios is the missing share count (§2), which
history backfill does not fix.

## 2. Insurer shares/EPS (NaN by design in v1)

LI/GI instances carry NO EPS facts and no trustworthy share count: GI's
`NumberOfShares` fails the sanity check vs actual outstanding (ICICIGI:
242.9M filed vs ~492M actual); LI files paid-up capital but no face-value
fact (assuming ₹10 face silently breaks on ₹5-face names, e.g. GICRE).
Resolution: source insurer share counts from the equities side (corporate-
info API / shareholding-pattern filings), or an assumed-face-value table
with per-ticker verification. Until then insurer per-share/market-cap ratios
stay honestly NaN.

## 3. BSE fallback adapter

Dual-listed coverage + redundancy for NSE outages; same `in-bse-fin`/
`in-capmkt` taxonomy family, different session/blocking profile. Endpoint
recon needed (their AnnGetData/results APIs are param-fussy — read the page
JS). Would also settle §1's BSE question.

**Concrete motivating misses (full nifty500 seed, 2026-07-09):** three
tickers have NO fetchable quarterly XBRL on either NSE stream —
**ABBOTINDIA, BAYERCROP, MCX** (a single stale 2010 placeholder record on
the classic stream, zero integrated-filing records). All are BSE-primary
large-caps that clearly file quarterly somewhere; the NSE chain simply has
nothing for them. These are the first customers of a BSE adapter (and the
test that it closes the 497/500 → 500/500 gap).

## 4. Half-yearly cash-flow fills

SEBI mandates half-yearly CF statements; H1/FY filings could populate
`ocf`/`capex`/`fcf` on two grid rows per year (semi-annual TTM). Today the
columns are all-NaN for India (kept for schema symmetry).

## 5. Minority-interest share bias in derived shares

`shares = net_income / eps` uses `ProfitLossForPeriod` (includes non-
controlling interests) while EPS is owners-attributable — consolidated
conglomerates overstate shares by the minority share (RIL ~8%). Fix: prefer
an owners-attributable NI tag for the derivation when present, or accept the
bias (per-share ratios built from EPS directly are unaffected).

## 6. yfinance-IN tertiary adapter

`.NS` quarterly statements, last ~5 quarters, no filed dates (NaT). Marginal;
only worth it as a freshness stopgap if NSE blocks this host.

## 7. INR valuation panel + gbdt F18-IN wiring

The consumer side (separate plans): `valuation` panel for NSE tickers (INR,
split-basis via nse_equities `adj_close`, point-in-time on `filed_date`) and
the gbdt `all_fundamentals`-style opt-in token for nifty500 cells. Blocked
on: full seed validation (V4 Phase 4) + a shares source for insurers (§2) if
insurer coverage is wanted in the panel.

## 8. Old-GAAP era depth (pre-2016)

`min_year: 2016` floors the parse. Classic-stream records exist back to
~2005 but pre-XBRL rows carry `"-"` placeholder links, and the early-XBRL
era drifts through taxonomy generations (`NONINDAS` parsed fine in the two
files seen, but unvalidated at scale). Raising depth = validate NONINDAS
broadly + accept placeholder-era absences.
