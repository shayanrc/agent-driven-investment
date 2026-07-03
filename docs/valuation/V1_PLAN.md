# valuation V1 — point-in-time PE / PS / P-FCF panel

**Status: in progress (2026-07-03).** Branch `valuation-pit-ratios`.

## Context

The gbdt models use only price/volume features. The user's hypothesis is that
fundamentals are the next edge. V3 of `data_pipelines` landed the
`us_fundamentals` cache (quarterly revenue / net income / OCF / capex / FCF /
shares / EPS) and a `filed_date` enrichment (SEC submissions API, authoritative).
This module turns those + daily prices into the **point-in-time valuation ratios**
the user asked for: PE, PS, Price/FCF — "only updated from the filing date, using
the TTM values."

## Design (decisions locked; see `goal.md` for the why)

- **Home:** a new installed module `src/valuation/` (cross-cutting: reusable by
  gbdt features, dashboards, analysis — not gbdt-internal).
- **Ratio formulation (market-cap form):**
  `market_cap(t) = adj_close(t) × shares_adj(t)`;
  `PE = market_cap / net_income_ttm`, `PS = market_cap / revenue_ttm`,
  `P_FCF = market_cap / fcf_ttm`. Per-share: `eps_ttm = net_income_ttm / shares`,
  `rev_ps_ttm`, `fcf_ps_ttm`. Inverse **yields** (`earnings_yield`,
  `sales_yield`, `fcf_yield`) — finite across the zero crossing, modeling-preferred.
- **TTM:** rolling 4-quarter sum (revenue/net_income/fcf) by `fiscal_period_end`;
  shares = most-recent-filed diluted shares. Snapshot effective on the newest
  quarter's `filed_date`, forward-filled over trading days.
- **Causality:** a quarter enters TTM only when `filed_date ≤ t`; a window with
  any NaT `filed_date` is not emitted. `filed_date` never precedes fiscal end
  (guaranteed by the V3 enrichment invariant).
- **Split alignment:** `adj_close` (split+div adjusted, consistent) for price;
  as-reported shares adjusted to the latest split basis via cumulative split
  factors (yfinance `.splits`, cached). Split-consistent within-ticker; dividend
  offset on absolute level documented + validated. Negative/zero denominators →
  ratio NaN (yields stay finite and signed).
- **Output:** `results/valuation/data/valuation_panel.parquet` (regenerable;
  gitignored if large, a compact per-ticker-latest summary checked in) — one row
  per `(date, ticker)` with the 9 metrics + `asof_fiscal_period_end` /
  `asof_filed_date` provenance.

## Files

- `src/valuation/ttm.py` — pure point-in-time TTM engine (`build_ttm_timeline`).
- `src/valuation/ratios.py` — pure per-share + ratio + yield math.
- `src/valuation/prices.py` — FUND: → us_equities id map, daily `adj_close` read,
  split-factor fetch/cache + `adjust_shares_to_latest_basis`.
- `src/valuation/panel.py` — orchestration: `build_panel(tickers, start, end)`.
- `scripts/valuation/build_valuation_panel.py` — runner (disk pre-flight, seed
  splits, emit panel + a checked-in latest-snapshot summary).
- `tests/valuation/` — TTM causality, ratio math, split adjustment, panel
  no-look-ahead + external spot-check.

## Phases (tasks #72–#76)

- **#72 prices** — id map + `adj_close` read + split fetch/cache + share adjust.
- **#73 TTM engine** — `build_ttm_timeline` (pure) + tests.
- **#74 ratios** — per-share + PE/PS/P-FCF + yields (pure) + tests.
- **#75 panel** — join everything → daily point-in-time panel + runner.
- **#76 validation** — external spot-check (AAPL/GOOGL PE vs a reference),
  no-look-ahead probe, TTM-steps-only-on-filings, PS>0 sanity; memo + tests.

## Verification

- `uv run python -m pytest tests/valuation -q` green.
- `build_panel(["FUND:AAPL","FUND:GOOGL"], ...)` → sane PE/PS/P-FCF; row on a
  pre-earnings day reflects the prior filing, flips on the filing date.
- Split spot-check: AAPL PE across its 2020 4:1 split has no factor-of-4 jump.
- Downstream (#77, separate): opt-in gbdt fundamentals feature family + A/B.
