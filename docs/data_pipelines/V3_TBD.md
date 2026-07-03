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

4. **filed_date enrichment for macrotrends-served rows.** Only EDGAR fills
   `filed_date`; macrotrends-served rows carry NaT. The `merge_overlap` fill-holes
   policy already enriches a row when EDGAR happens to serve its ticker, but a
   deliberate one-shot enrichment pass (fetch EDGAR for ALL cached tickers, merge
   filed dates only) would make the point-in-time lag policy uniform before modeling.
   ~1,000 companyfacts requests ≈ 1-2 h polite. Do this BEFORE building
   fundamentals-derived model features that need causal lags.

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
