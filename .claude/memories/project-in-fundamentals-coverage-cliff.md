---
name: project-in-fundamentals-coverage-cliff
description: NSE in_fundamentals (F18-IN) only begins ~2019 — the NSE valuation panel starts 2019-01-16, raw XBRL data ~2017-2018, broad coverage 2019-2020. F18 is ~48-58% NaN across a 2015-start training window (100% NaN pre-2019). This CONFOUNDS any nifty500 F18 experiment with train_start < 2019 (the fund arm trains on half-missing F18). Backfill via screener.in annual (→ FY2015) in progress (#32). Contrast: US us_fundamentals is dense from ~2011, so sp500 F18 work is NOT confounded.
metadata:
  type: project
---

**NSE fundamentals (`in_fundamentals` / F18-IN) have a hard time-coverage cliff at
~2019.** Discovered 2026-07-11 while setting up the #30 stratified run.

Coverage facts (nifty500, verified against the `in_fundamentals_data` cache table
+ the `nifty500_up_30pct_100d_dd15pct_ffund` feature-matrix cache):
- **NSE valuation panel** (`valuation_panel_nse.parquet`, the F18 source) starts
  **2019-01-16**.
- Raw `in_fundamentals` fiscal_period_end starts 2017-03-31 (52 rows), ramps 2018
  (1,172) → broad by **2019–2020**. It's NSE XBRL machine-readable financial
  results, which SEBI/NSE only standardized ~2018–2019.
- In the feature matrix, `fund_earnings_yield` is **100% NaN for 2013–2018**, 44%
  NaN in 2019, 16% in 2020, ~2–4% from 2021. `fund_rev_ttm_yoy` is worse (100%
  NaN through 2019 — needs a year-ago TTM). **Across a 2015→2024 training window,
  F18 is ~48–58% NaN.**
- It is a **time cliff, NOT a ticker gap** — 312/315 tickers have fundamentals.

**Why it matters:** any nifty500 F18 experiment with `train_start` before ~2019
(the `_285` sweep, the `_286`/#29 finetune, the #30 stratified harness) trains the
fund arm with F18 **absent for ~40% of the window** — a confound that buries the
F18 signal. **Those results are invalidated** (memos carry an ⚠️ banner) pending a
backfill + rerun (#32 → #34).

**Contrast — sp500 is clean:** `us_fundamentals` is dense from ~2011 (2011 = 1,561
filings, 2012 = 3,149…). The sp500 F18 sweep (`_287`, train_start 2015) had F18
present throughout, so its "does not replicate" verdict is unconfounded.

**How to apply:**
- For a *fair* nifty500 F18 test before the backfill lands, use `train_start`
  **2019–2020** (F18 present throughout; 2022 bear still in-window) — NOT 2015.
- The backfill (#32) uses **screener.in annual P&L** (reachable HTTP 200, no login,
  annual data → FY2015; yfinance is too shallow ~5y, NSE archives unreachable).
  Hybrid: screener annual for 2015–2018 + NSE XBRL quarterly for 2019+. After the
  merge, rebuild `valuation_panel_nse.parquet` (`build_valuation_panel --domain nse`).
- When reading any nifty500 fundamentals result, check the F18 NaN fraction in the
  training window before trusting a fund-vs-base delta.
