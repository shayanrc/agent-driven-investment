# _273 — F18 fundamentals: second-window (trailing) validation

**Question (the `V1.7_TBD` §1 gate).** `_272` showed F18 fundamentals improve both
sp500 champions on the **date-aligned** 2024-H2 window. A single-window win can be
window-specific (the F17 macro lesson: won trailing `_262/_263`, failed date-aligned
`_264`). So: does the fundamentals win **replicate on a genuinely different window**?

**Design.** Same matched-HP single-fit A/B as `_272` (`base_v2`=`all` vs
`fund`=`all_fundamentals`, identical HP, only the +9 F18 columns differ), but the
specs drop the `split:` block → **default trailing-anchor mode**. Snapshot
2026-07-02 ⇒ **trailing test window = 2026-02-03 → 2026-04-21** (+50/50d) /
**→ 2026-05-27** (+20/25d) — recent, non-overlapping with `_272`'s 2024-H2. Specs:
`sp500_up_{50pct_50d_dd25pct,20pct_25d_dd10pct}_{base_v2,fund}_trail`.

## Results — test R-Precision@K (raw; base_rate for reference)

**+50%/50d** (trailing base_rate 0.0459, 24,300 rows):

| K | base R-p@K | fund R-p@K | Δ |
|---:|---:|---:|---:|
| 1 | 0.1400 | 0.3800 | **+0.2400** |
| 3 | 0.2267 | 0.3733 | **+0.1467** |
| 5 | 0.2600 | 0.3760 | **+0.1160** |
| 10 | 0.2441 | 0.3542 | **+0.1101** |
| 20 | 0.2987 | 0.3681 | **+0.0693** |

AUC 0.8448 → **0.8482** (+0.0035); test Brier 0.04240 → **0.04205** (−0.00035); eval Brier 0.01789 → 0.01808.

**+20%/25d** (trailing base_rate 0.0914, 36,450 rows):

| K | base R-p@K | fund R-p@K | Δ |
|---:|---:|---:|---:|
| 1 | 0.2564 | 0.3077 | **+0.0513** |
| 3 | 0.3034 | 0.2735 | −0.0299 |
| 5 | 0.2949 | 0.3231 | +0.0282 |
| 10 | 0.3167 | 0.3115 | −0.0051 |
| 20 | 0.3270 | 0.3055 | −0.0215 |

AUC 0.7515 → 0.7516 (+0.0001); test Brier 0.07994 → **0.07951** (−0.00043); eval Brier 0.05499 → 0.05476.

All 9 F18 features used in both models (4.5% / 3.5% of tree gain — slightly higher
than date-aligned's 3.0% / 2.3%); revenue-growth ranks lead on trailing
(`fund_rev_ttm_yoy_xs_rank`, `fund_sales_yield_xs_rank`, `fund_earnings_yield`).

## Two-window comparison (does `_272` replicate?)

| signal | +50%/50d date-aligned (`_272`) | +50%/50d trailing (`_273`) | +20%/25d date-aligned | +20%/25d trailing |
|---|---|---|---|---|
| AUC Δ | +0.0066 ✓ | +0.0035 ✓ | +0.0200 ✓ | +0.0001 ✗ |
| test Brier Δ | −0.00035 ✓ | −0.00035 ✓ | −0.00036 ✓ | −0.00043 ✓ |
| R-p@1 | +0.011 ✓ | +0.240 ✓ | +0.150 ✓ | +0.051 ✓ |
| R-p@10 | +0.087 ✓ | +0.110 ✓ | +0.023 ✓ | −0.005 ✗ |
| R-p@20 | +0.092 ✓ | +0.069 ✓ | −0.031 ✗ | −0.022 ✗ |

## Verdict — split

- **+50%/50d: robust two-window fundamentals win.** AUC up, test Brier down, and
  **every** R-p@K up — on *both* the date-aligned and the trailing window. The
  trailing top-of-book gain is large (R-p@1 0.14 → 0.38). This is materially
  stronger evidence than F17 macro ever had (macro failed its second window
  outright). The gate **passes** for this cell.
- **+20%/25d: NOT robust.** Only test Brier (−0.0004) and R-p@1 (+0.05) survive
  both windows; the strong date-aligned mid-K gains (R-p@3 +0.083, R-p@5 +0.042)
  **do not replicate** on trailing (R-p@3 −0.030), and the AUC gain evaporates
  (+0.020 → +0.0001). The gate **does not pass** for this cell — its `_272` edge
  was largely window-specific.

**Net read:** fundamentals are a real, replicating edge on the **long-horizon /
high-threshold** champion (+50%/50d), and a marginal/top-1-only effect on the
short-horizon one (+20%/25d). Consistent with the mechanism — a 50-day-forward
+50% move is a bigger fundamental-quality signal than a 25-day +20% move, which is
more technical/momentum-driven.

## Decision

- **F18 is validated for the +50%/50d champion** (two independent windows). Before
  wiring into `/daily-predictions`, do the **per-arm FS+HP tune** of the `fund`
  arm (`V1.7_TBD` §3) to measure the tuned ceiling vs the deployed champion's
  config — the A/B only measured the *marginal* feature effect at matched
  (single-fit) HP, not a tuned model. A champion-config change remains a human
  decision (`_019`).
- **Hold on +20%/25d** — the fundamentals edge there isn't robust enough to act on.
- **Do NOT blanket-wire F18 into `/daily-predictions`** — the per-cell result is
  heterogeneous; promotion is per-champion, gated on the tune above.

Data: `results/gbdt/data/_273_fundamentals_trailing_data.json`; registry: 4 rows in
`r_precision_at_k.csv`. Prior window: `_272`. Plan: `V1.7_fundamentals_features_plan.md`.
