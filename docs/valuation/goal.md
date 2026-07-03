# valuation — Goal

**Point-in-time valuation ratios over the quarterly fundamentals + daily prices.**
Turns the `us_fundamentals` cache (revenue / net income / FCF / shares, with SEC
`filed_date`) and the `us_equities` daily prices into a **daily, causally-correct**
panel of PE / PS / P-FCF (plus the per-share intermediates and their inverse
yields) — the reusable feature layer the gbdt models consume.

Read this before editing anything under `src/valuation/`, `tests/valuation/`,
`docs/valuation/`, `scripts/valuation/`.

## What it optimizes for

**Causal correctness first.** Every value on calendar day `t` must be computable
from information available on day `t` — no look-ahead. Concretely:

- A quarter's fundamentals enter a ratio **only on/after its `filed_date`** (the
  date the 10-Q/10-K hit EDGAR — the whole reason the enrichment pass exists).
  Fundamentals **step only on filings**; price moves daily.
- **TTM** (trailing twelve months) = the last **4 quarters** by fiscal period,
  summed for flow metrics (revenue, net income, FCF). A TTM snapshot is effective
  on the **newest** of its 4 quarters' `filed_date`, forward-filled over trading
  days until the next filing.
- A quarter with **NaT `filed_date`** cannot be placed in time → any TTM window
  containing it is not emitted (the ratio starts once 4 consecutive dated
  quarters exist). Undated tickers (ADRs / foreign 20-F filers) produce no ratios.

**Split-basis consistency.** A ratio series must have no artificial jumps at stock
splits. Prices use `adj_close` (fully split-adjusted, and the one column with
consistent cross-provider semantics per `data_pipelines/goal.md`); as-reported
shares are adjusted to the same latest split basis via split factors, so
market-cap = `adj_close × shares_adj` is split-consistent within a ticker's
history. (adj_close is also dividend-adjusted → a small, slowly-varying,
documented offset on the absolute ratio *level*; immaterial for the relative /
cross-sectional signal the models use. Validated in the panel's validation gate.)

## What success looks like

`build_panel()` emits one row per `(date, ticker)` trading day with `pe`, `ps`,
`p_fcf`, `eps_ttm`, `rev_ps_ttm`, `fcf_ps_ttm`, `earnings_yield`, `sales_yield`,
`fcf_yield`, plus provenance (`asof_fiscal_period_end`, `asof_filed_date`) so each
row self-describes which filing it reflects. Re-running is deterministic. A
spot-check of computed PE/PS against an external reference is within tolerance
after split alignment, and a look-ahead probe (perturb a future filing → row `t`
unchanged) passes.

## What this is NOT

- **Not an ingestion layer** — it consumes the caches, never fetches fundamentals
  (splits are the one small external pull, cached). New raw financial data is a
  `data_pipelines` concern.
- **Not a model / strategy** — it produces features; gbdt consumes them (a
  separate opt-in feature family, `V3_TBD` / task #77). No PnL, no positions here.
- **Not a definitive-accounting source** — point-in-time as-first-filed values;
  restatements are intentionally not back-applied (matches the `filed_date`
  discipline). Absolute ratio levels carry the documented dividend-adjustment
  offset; the signal is the relative/temporal structure.

## Conventions

- Ratios reported as raw values; negative/zero denominators handled explicitly
  (see below). The **inverse yields** (`earnings_yield` etc.) are the
  modeling-preferred form — finite and continuous across the zero-earnings
  crossing where PE diverges.
- Money in $M, shares in M, per-share in $, matching the `us_fundamentals` schema.
