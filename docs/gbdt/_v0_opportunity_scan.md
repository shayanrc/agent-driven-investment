# v0.1 — Rolling-window opportunity scan (NIFTY 50, +10%)

**Task:** for each NIFTY 50 stock and each horizon `H ∈ {10, 20, 50, 100}` trading days, count rolling-origin events where the maximum `adj_close` over `(t, t+H]` is at least 10% above `adj_close[t]`. Report per-stock and pooled-across-stocks base rates plus first-breach lag percentiles.

**Spec:**
- Universe: NIFTY 50 as of 2026-05-24 (50 tickers; pinned in `configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml`).
- Price field: `adj_close` (bonus + split adjusted; raw `close` would create spurious "events" on corporate-action days).
- Event definition: `max(adj_close in (t, t+H]) ≥ 1.10 · adj_close[t]`.
- Data source: `data/processed.db` via direct SQL (no `data_pipelines.fetch()` dispatch needed).
- Script: [`scripts/gbdt/v0_opportunity_scan.py`](../../scripts/gbdt/v0_opportunity_scan.py).
- Headline JSON: [`results/gbdt/data/_v0_opportunity_scan_data.json`](../../results/gbdt/data/_v0_opportunity_scan_data.json).

## Data coverage

50 / 50 NIFTY 50 tickers had cached data. **All series end 2025-12-31; most start 2020-01-01** (median ≈ 1,492 trading rows per ticker; minimum 578 for newer listings). This is the cached state at the time of the scan — the deeper backfill (the seed agent on `data-seed-nifty-total` branch was incomplete and is being relaunched). When deeper history lands, this scan should be re-run; the v0.1 numbers here are the **5-to-6-year regime** for NIFTY 50 ending Dec-2025.

## Headline table

Aggregated across all 50 NIFTY 50 tickers:

| H (days) | n_stocks | pooled origins | pooled events | pooled base rate | per-stock median | per-stock IQR (q25–q75) | median first-breach lag (days) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 50 | 71,351 | 6,000 | **0.0841** | 0.0665 | 0.048 – 0.117 | 7 |
| 20 | 50 | 70,851 | 15,054 | **0.2125** | 0.1950 | 0.155 – 0.260 | 13 |
| 50 | 50 | 69,351 | 32,713 | **0.4717** | 0.4667 | 0.414 – 0.518 | 23 |
| 100 | 50 | 66,851 | 44,206 | **0.6613** | 0.6688 | 0.605 – 0.709 | 32 |

## Per-stock extremes (sorted by H=10 base rate)

| Rank | Top 5 (richest in 10-day +10% moves) | H=10 | H=100 | Bottom 5 (sparsest) | H=10 | H=100 |
|---:|---|---:|---:|---|---:|---:|
| 1 | NSE:SHRIRAMFIN | 0.246 | 0.798 | NSE:NESTLEIND | 0.018 | 0.448 |
| 2 | NSE:ETERNAL | 0.212 | 0.789 | NSE:TCS | 0.022 | 0.511 |
| 3 | NSE:ADANIENT | 0.207 | 0.760 | NSE:HINDUNILVR | 0.031 | 0.480 |
| 4 | NSE:MAXHEALTH | 0.185 | 0.876 | NSE:ASIANPAINT | 0.036 | 0.572 |
| 5 | NSE:TRENT | 0.165 | 0.841 | NSE:INFY | 0.037 | 0.603 |

## Findings

1. **Strong rarity gradient across horizons.** A +10% breach is rare at H=10 (~8% pooled), one-in-five at H=20, near-coin-flip by H=50 (~47%), and the modal outcome by H=100 (~66%). Equal-spaced horizons produce *very* unequal base rates — this matters for v1 because Brier scores and calibration curves operate on different probability regimes per cell.

2. **First-breach happens early in the window.** Median first-breach lag is ~7 / 13 / 23 / 32 days at H=10 / 20 / 50 / 100. The lag percentiles scale sub-linearly with H — once a +10% rally starts, it tends to register within the first third of the available window. This says the longer horizons are not winning by having "more time for slow moves"; they're winning by having more independent chances at fast moves. A v2 design could lift this insight into a "time-to-event" regression alongside the binary outcome — but that's v2.

3. **Per-stock variance is large — 10× spread at H=10.** SHRIRAMFIN (24.6%) sees a +10% in 10 days more than 13× as often as NESTLEIND (1.8%). The split visible in the top-vs-bottom 5 is roughly **growth/cyclical names rich in events** (Shriram, Adani, Trent, Eternal — high-beta or recovery-phase) vs **defensive consumer/IT names sparse in events** (Nestlé, HUL, TCS, Infy, Asian Paints — low-beta, stable). This is what a NIFTY 50 cross-section over 2020–2025 should look like; the surprise would have been if the spread were narrow.

4. **The cross-stock spread shrinks at longer horizons.** At H=10 the q75/q25 ratio is ~2.4 (0.117 / 0.048); by H=100 it's ~1.2 (0.709 / 0.605). Over a long-enough window, almost every stock is "rich" in 10% events; over a short window, name-specific characteristics dominate.

## What this implies for v1 (`V1_PLAN.md`)

- **Calibration baseline (Stage 9 acceptance criterion #2 — Brier < base-rate).** The base rate to beat for the `up_10_h10` target is ~0.08 pooled. A constant predictor outputting 0.08 has Brier ≈ 0.077. Beating that is non-trivial — feature signal needs to be real. For `up_10_h100` (base rate 0.66) the baseline Brier is 0.224 (much weaker, much harder to beat by a meaningful margin).
- **Per-stock heterogeneity argues against pooled training in v1.** If v1 trains on a single asset (NASDAQ100 per current plan), per-stock heterogeneity isn't an issue. If we later extend to a NIFTY universe, the right unit is per-stock models (or per-cluster), not pooled — the base rates are simply too different.
- **The 18-cell lattice is well-chosen for the rarity gradient.** The 3 horizons (10, 20, 50) span ~6× in base rate at +10%; 4 horizons (adding 100) would extend that to ~8×. The v1 lattice keeps the rarest cell as `+50% in 10 days` (which v0 didn't scan but is structurally rare; safe assumption) and the most-common as `+10% in 50 days` (~47%). Reasonable coverage.

## Caveats / out-of-scope

- **Look-ahead via point-in-time membership.** The universe is "NIFTY 50 as of 2026-05-24" — a stock that was a NIFTY 50 member in 2021 but isn't today is missing, and a stock added recently has its full back-history scanned even for dates when it wasn't a member. Survivorship bias is real; magnitude is unknown without point-in-time membership data (explicitly out of scope per the universe YAML's open question #1).
- **5-to-6-year window dominated by post-COVID recovery + rate-hike cycle.** The cached 2020–2025 window saw above-average market volatility and one of the longer bull runs in recent NSE history. Base rates measured here likely overstate event richness for a normal regime. Re-run when deeper backfill (2010+) lands to test stability across regimes.
- **No transaction costs, no execution model, no PnL.** Per the project-wide anti-rule (see `docs/gbdt/goal.md`'s "What this module is *not*"). This scan counts opportunities; whether they're tradable is a downstream question that doesn't belong in this module.
- **`adj_close`-based event detection.** A target user who'd be relying on `high` prices (intraday max) would see somewhat higher event counts. The v1 plan's target definition uses close-based events; this scan matches that convention for consistency.

## Re-run

```bash
uv run python -m scripts.gbdt.v0_opportunity_scan
```

Updates `results/gbdt/data/_v0_opportunity_scan_data.json` in place. Re-run after deeper-history backfill lands on the NSE cache to test base-rate stability across regimes.
