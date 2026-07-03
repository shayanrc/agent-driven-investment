# data_pipelines V3 — TBD (deferred follow-ups)

Parking lot for work discovered during V3 (`V3_US_FUNDAMENTALS_PLAN.md`) but out of
scope for it. Promote to a real plan when a coherent slice is big enough to be a
project, not a chore.

## Data breadth

1. **Annual statements.** Only quarterly (`?freq=Q`) is ingested. Macrotrends annual
   pages go deeper (~2009) and would extend history for long-horizon features; annual
   values are also derivable by summing four quarters for flow metrics, so this is
   low-value until a consumer wants pre-2011 depth.

2. **Balance-sheet metrics.** The schema covers income + cash-flow items. Total assets,
   equity, debt, cash — needed for P/B, ROE, leverage features — are one more
   macrotrends page per ticker (`balance-sheet?freq=Q`) and the same EDGAR tag
   machinery (instant facts this time, simpler than durations: no YTD differencing).
   Add as schema v2 columns when the modeling phase wants them.

3. **NSE fundamentals.** Indian-market fundamentals for the nse_equities universes
   would be a sibling domain (`nse_fundamentals`); screener.in is the macrotrends
   analog. Entirely separate provider recon needed.

## Correctness / enrichment

4. **[DONE 2026-07-03] filed_date enrichment.** `scripts/data_pipelines/
   enrich_fundamentals_filed_date.py` fills `filed_date` for every cached ticker
   from the SEC **submissions** API — `EdgarAdapter.filing_dates()` maps each
   `reportDate → earliest 10-K/10-Q filingDate`, the authoritative "became public"
   date. Chosen over the companyfacts-derived date (the initial approach): exact
   for derived-Q4 quarters (a 10-K's own filing date, no differencing-max that
   read up to a *year* late — see the GOOGL before/after), ~25× smaller fetch
   (~150 KB vs 3.7 MB), and no XBRL derivation. **Authoritative semantics:**
   filed_date = the confirmed submissions date, else NaT (any stale/unconfirmed
   value is cleared) with two guards — reportDate↔fiscal-end proximity AND the
   causal invariant (filing after period end), which rejects the off-calendar
   mis-snaps (CAVA's 13-week restaurant quarters, FERG's July fiscal year).
   Outcome: 993/1014 tickers dated, 92.8% of rows, **0 rows with
   filed_date ≤ fiscal_period_end**, filing lag median 36 d / p95 58 d. The 21
   undated tickers are ADRs/foreign filers/delisted (no 10-K/Q). Pure matching
   core `resolve_filed_dates()` + regression tests. Re-runnable + idempotent.

5. **Macrotrends-vs-EDGAR cross-validation report.** Both providers now cover most
   tickers; a scripted diff (per metric, per quarter, relative error distribution)
   would quantify divergence (restatements, unit quirks, sign conventions) and flag
   tickers where the primary is silently wrong. The AAPL fixture agreement (exact on
   revenue/NI/OCF/capex) suggests divergence is rare; verify at universe scale.

6. **Multi-class share entities.** EDGAR per-class EPS/shares (GOOGL, BRK) currently
   first-match-wins on the tag priority list; macrotrends publishes consolidated
   values. Acceptable for screening features; revisit if per-share precision matters.

## Consumer surface

7. **`/fetch-fundamentals` skill.** A thin agent verb over
   `fetch("FUND:<TICKER>")` once a consumer exists (mirrors `/fetch-data`).

8. **Derived ratios (PE, PS, P/FCF).** Deliberately NOT stored — they need a price
   join and a point-in-time lag policy (filed_date where present, else a conservative
   deadline-based lag), which is modeling-phase logic (a future gbdt feature family
   or a `fundamentals_features` module), not ingestion. The cache stores only what
   filings state.

9. **Universe drift on re-seed.** The fundamentals universe derives from the equity
   universe YAMLs at load time; when those YAMLs are refreshed (constituent updates),
   newly added tickers seed on the next run but DELISTED tickers' cached fundamentals
   simply stop refreshing (rows retained — survivorship-friendly). Document/verify
   this behaviour when the first universe refresh lands.
