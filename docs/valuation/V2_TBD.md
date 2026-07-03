# valuation V2 — TBD (deferred follow-ups)

Parking lot for follow-ups discovered during V1 (`V1_PLAN.md`) + its PR review
(#254). Promote to a real `V2_PLAN.md` when a coherent slice is big enough.

## Robustness (from the #254 review)

1. **Defensive duplicate-`fiscal_period_end` guard in `build_ttm_timeline`.** The
   consecutiveness check keys on a `>110`-day gap, so a *duplicate*
   `fiscal_period_end` (e.g. a 10-Q/A amendment, gap = 0 d) would pass and could
   double-count a quarter in the TTM sum. Today this can't happen — the
   `us_fundamentals` cache is PK'd by grid `date` and `dedupe_grid_collisions`
   yields one row per grid date — but a defensive `assert fiscal_period_end`
   is unique (or drop-duplicates keeping the latest `filed_date`) would harden
   the engine against an upstream change. Low risk; cheap.

## Accuracy (accepted V1 tradeoffs; revisit only if a consumer needs absolute levels)

2. **Dividend-adjustment offset on absolute ratio *levels*.** `adj_close` is
   split *and* dividend adjusted, so market-cap-based ratio levels carry a small,
   slowly-varying dividend offset (immaterial for low-yield names; larger for
   high-yield value stocks). The within-ticker temporal structure + cross-sectional
   ordering — what the models use — are unaffected. A split-only price basis
   (needs a raw or split-only-adjusted close from `us_equities`) would fix the
   absolute level. See `goal.md`.

3. **Multi-class EPS divergence** (GOOGL/BRK): `net_income / diluted_shares`
   diverges from per-class reported EPS (V1 validation: GOOGL 6.5%). Per-class
   handling if per-share precision on multi-class names ever matters.

## Reach

4. **Refresh cadence.** V1 is a one-shot full build. A `--since` incremental
   refresh (recompute only dates after the last panel date + any ticker whose
   fundamentals advanced) would keep the panel fresh cheaply, mirroring the
   `/daily-predictions` incremental pattern.

5. **Balance-sheet ratios** (P/B, ROE, leverage, net-debt/EBITDA) once
   `us_fundamentals` gains balance-sheet columns (`data_pipelines` V3_TBD §2).

6. **Wire into gbdt** — the downstream (task #77): expose the panel as an opt-in
   gbdt fundamentals feature family (F18) + A/B whether it beats the sp500
   champions. Its own plan.
