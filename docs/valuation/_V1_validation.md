# valuation V1 — validation

Panel built 2026-07-03 over the full cached universe: **1,902,642 rows · 971
tickers · 2018-01-02 → 2026-07-02** (`results/valuation/data/valuation_panel.parquet`,
gitignored; `valuation_latest.csv` checked in). Metric: point-in-time PE / PS /
P-FCF + per-share + yields.

## Causal correctness (unit tests, `tests/valuation/test_validation.py`)

- **No look-ahead.** Perturbing a *future* filing (relative to day `t`) leaves
  the row at `t` unchanged; perturbing an *already-filed* quarter does move it.
- **TTM steps only on filings.** Between two consecutive filing dates every TTM
  column is constant while PE tracks price exactly (`pe/price = const`).
- **No premature rows.** No ratio before the first full 4-quarter window is
  effective; undated quarters block any window containing them.

These are guaranteed structurally: a quarter enters a TTM window only when
`filed_date ≤ t` (V3 enrichment made `filed_date` authoritative), and
`asof_daily` is a backward `merge_asof` on `effective_date`.

## Split-basis alignment

AAPL PE is **smooth across its 2020-08-31 4:1 split** (34.2 → 36.2 → 38.0 → 32.8
over 2020-08-20…09-10) — no factor-of-4 discontinuity — because as-reported
shares are lifted to the latest split basis (4,355M × 4 = 17,419M) to match the
split-adjusted `adj_close`. Verified live + in `test_prices` / `test_panel`.

## Internal accuracy — computed `eps_ttm` vs Σ last-4 reported diluted EPS

`eps_ttm = net_income_ttm / shares_adj` (latest snapshot) vs the sum of the four
trailing reported `eps_diluted` from the `us_fundamentals` cache — two
independent paths through the data:

| ticker | as-of | eps_ttm | Σ4 reported | rel err |
|---|---|--:|--:|--:|
| AAPL | 2026-03-31 | 8.32 | 8.26 | 0.8% |
| MSFT | 2026-03-31 | 16.82 | 16.80 | 0.1% |
| NVDA | 2026-04-30 | 6.54 | 6.53 | 0.2% |
| WMT | 2026-04-30 | 2.84 | 2.84 | 0.1% |
| JPM | 2026-03-31 | 21.14 | 20.89 | 1.2% |
| XOM | 2026-03-31 | 6.02 | 5.94 | 1.4% |
| GOOGL | 2026-03-31 | 12.25 | 13.11 | 6.5% |

Most within ~1% (NI/weighted-shares vs as-reported EPS differ only by rounding).
GOOGL is the outlier (multi-class shares + a large one-off gain quarter, where
NI/diluted-shares diverges from per-class reported EPS) — a documented known
limitation, not a pipeline error.

## Distribution sanity (971-ticker latest snapshot)

| ratio | n | p25 | median | p75 |
|---|--:|--:|--:|--:|
| PE | 850 | 16.4 | 24.8 | 38.6 |
| PS | 961 | 1.4 | 2.8 | 5.5 |
| P/FCF | 827 | 13.1 | 20.5 | 33.8 |

Textbook US large-cap levels (market PE ~20-25×). 117 loss-makers correctly
carry NaN PE + a signed negative `earnings_yield`; `PS > 0` wherever defined.
Coverage: **86%** of daily rows have a finite PE, **99%** a finite
`earnings_yield`.

**Post-ship artifact sweep (2026-07-04 review).** A panel-wide invariant sweep
(all 1.9M rows, not just the latest snapshot) found **116 rows (0.006%) with
non-positive PS** on two tickers: ASTS 2021 (a SPAC-transition quarter with
`shares_diluted = 0` → PS = 0) and CACC 2023 (the EDGAR Q4-shares derivation
`4×FY − ΣQ1..3` went negative on a buyback year → negative market cap,
sign-flipped EPS). Root cause: upstream `shares ≤ 0` artifacts (10 quarters /
6 tickers cache-wide) were not masked. Fixed by a `shares > 0` guard in
`compute_ratios` (all ratio columns NaN on such rows) + regression tests; panel
rebuilt — the three artifact gates (no-look-ahead, filing-order, PS>0) all pass
at 0. Neither ticker is in the sp500, so the `_272` A/B was unaffected; the
other four affected tickers (incl. CEG pre-spinoff) never entered a valid TTM
window.

## Known limitations (carried into modeling)

- **Absolute level vs relative signal.** `adj_close` is split *and dividend*
  adjusted, so market-cap-based ratio *levels* carry a small, slowly-varying
  dividend offset (immaterial for low-yield names; larger for high-yield value
  stocks). The within-ticker temporal structure and cross-sectional ordering —
  what the models use — are unaffected. A split-only price basis is a V2 option
  if absolute levels ever matter.
- **Undated tickers.** ~24 ADRs / foreign 20-F filers have NaT `filed_date` →
  no ratios (they're absent from the 971). Honest exclusion.
- **GOOGL-type multi-class EPS** divergence as above.
