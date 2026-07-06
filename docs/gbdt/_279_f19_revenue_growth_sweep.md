# _279 — F19 revenue-growth features: 17-cell lattice sweep (V1.8)

**Question.** Does a **revenue-growth** feature family (F19 — rate-of-change of the
top line) add over F18 (valuation *levels*) across the sp500 lattice, on both
backends? Requested by the user: quarterly-YoY %, TTM-YoY %, TTM-QoQ %, and
single-quarter-QoQ % ("if it's bad FS should drop it").

**F19 (12 columns).** Four growth measures, each {level, `_xs_rank`, `_xs_zscore`}
(the F18 idiom), percent form: `fund_rev_q_yoy` (single-quarter YoY, seasonality-
safe), `fund_rev_ttm_yoy_pct` (TTM YoY, the percent twin of F18's log
`fund_rev_ttm_yoy`), `fund_rev_ttm_qoq` (TTM sequential QoQ), `fund_rev_q_qoq`
(single-quarter QoQ). Behind a new **`all_fundamentals2`** token (F1–F16 + F18 +
F19 = 304 cols); `all_fundamentals`/F18 stays byte-identical (`_272`–`_278` intact,
locked by a test). Single-quarter revenue was piped through the valuation panel
(`revenue_q` in `build_ttm_timeline` → the panel → `fund_df`) to feed the quarterly
measures. Causal (C1): `filed_date ≤ t` + strictly-trailing 252/63-td shifts;
verified by the fundamentals-perturbation leakage test.

**Design.** 17 sp500 cells × {xgboost, catboost} × {`all_fundamentals` (F18),
`all_fundamentals2` (F18+F19)} = **68 matched single fits** (default HP,
`max_iterations: 1`, date_aligned `train_start: 2019-01-01`, snapshot 2026-07-06 →
test **2024-07-26 → 2024-12-16**, Q=100). Within a backend the arms differ ONLY in
`candidates`, so the per-cell `f19 − f18` delta is a clean read of the F19
contribution (the `[[project-gbdt-macro-features-f17]]` clean-A/B rule). Tooling:
`scripts/gbdt/{gen_f19_sweep_specs.py, run_f19_sweep.sh}`.

## Window-1 result — F19 delta grid (ΔAUC / ΔR-p@10, ✓ = both up)

Base rate is F18's (identical between arms). `✓` marks cells where F18+F19 improves
**both** AUC and R-Precision@10 over F18 (the deep-book criterion).

| cell (sp500 up) | base | xgb: ΔAUC / ΔR-p@10 | cb: ΔAUC / ΔR-p@10 |
|---|---:|---|---|
| 10pct_10d | 0.070 | +0.007 / +0.006 ✓ | +0.002 / −0.007 |
| 10pct_25d | 0.197 | −0.009 / −0.034 | −0.005 / +0.045 |
| 10pct_50d | 0.335 | +0.047 / +0.016 ✓ | −0.013 / +0.015 |
| 10pct_5d | 0.025 | −0.008 / −0.023 | −0.003 / −0.003 |
| **20pct_100d** | 0.231 | **+0.029 / +0.045 ✓** | **+0.004 / +0.055 ✓** |
| 20pct_10d | 0.009 | −0.004 / −0.009 | +0.003 / −0.030 |
| 20pct_25d | 0.040 | −0.005 / −0.017 | +0.003 / −0.000 |
| 20pct_50d | 0.128 | −0.018 / −0.021 | +0.008 / +0.032 ✓ |
| 20pct_5d | 0.003 | +0.015 / +0.087 ✓ | −0.000 / +0.024 |
| 40pct_100d | 0.060 | +0.006 / −0.000 | +0.002 / +0.010 ✓ |
| 40pct_200d | 0.119 | −0.033 / −0.025 | +0.013 / +0.023 ✓ |
| 40pct_25d | 0.005 | +0.008 / +0.027 ✓ | −0.013 / −0.009 |
| 40pct_50d | 0.020 | −0.010 / −0.094 | +0.014 / −0.011 |
| 50pct_100d | 0.032 | −0.007 / +0.019 | +0.001 / +0.014 ✓ |
| 50pct_200d | 0.071 | −0.009 / +0.006 | +0.003 / −0.004 |
| 50pct_25d | 0.002 | +0.018 / +0.072 ✓ | +0.017 / −0.106 |
| 50pct_50d | 0.010 | −0.020 / +0.058 | −0.005 / −0.122 |

**F19 helps both AUC + R-p@10 on: 6/17 xgb cells, 5/17 cb cells — but the two sets
intersect in exactly ONE cell: `20pct_100d`.** Every other win is single-backend,
which the `_276`/`_278` lesson flags as window-noise-prone. This is the same
"contextually additive, not robust" shape F17-macro (`_264`) and long-horizon F18
(`_278`) showed.

## The one both-backend winner — 20pct/100d (window 1, raw values)

| arm | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgb F18 | 0.5745 | 0.1875 | **0.550** | 0.397 | 0.370 | 0.341 | 0.330 |
| xgb F18+F19 | 0.6032 | **0.1842** | 0.490 | 0.463 | 0.434 | 0.386 | 0.343 |
| cb F18 | 0.6523 | **0.1859** | 0.270 | 0.533 | 0.456 | 0.403 | 0.384 |
| cb F18+F19 | 0.6558 | 0.1879 | 0.220 | **0.597** | **0.530** | **0.458** | **0.419** |

F19 lifts the **deep book (@3–@20) on both backends** (cb @5 0.456→0.530, @10
0.403→0.458; xgb @10 0.341→0.386) + xgb AUC/Brier — but **trades away @1** (xgb
0.55→0.49, cb 0.27→0.22) and cb Brier worsens slightly (+0.002). base_rate 0.231.

## FS kept-set (single-quarter QoQ)

All 12 F19 columns survived in both f19 arms of the winner cell — **but this sweep
is single-fit (`max_iterations: 1`), which does NO importance pruning**, so it
cannot answer "does FS drop the single-quarter QoQ." The requested "if it's bad FS
should drop it" test needs a full FS+HP loop (agent or default) on the F19 pool;
parked in `V1.8_TBD` as the follow-up. What this sweep measures is the *marginal
book/AUC effect of adding the 12 F19 columns*, not their per-column FS survival.

## Context (not recomputed here)

xgb technical-only (`_274` fbase) and cb technical-only (`_278`, 3 cells) baselines
are on the same window; `_278` found cb-base already beats every xgb arm on AUC 6/6
long-horizon cell-windows, and that F18's long-horizon edge failed window-2. F19 on
`20pct/100d` is being tested against that backdrop.

## Window-2 confirmation — the edge does NOT replicate

Re-ran the 4 arms of `20pct/100d` on an independent date_aligned window
(`train_start: 2019-07-01` → test **2025-01-24 → 2025-06-17**, Q=100, base 0.232 —
the `_278` window). Raw values:

| arm | window | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| xgb F18 | w2 | 0.6326 | 0.1718 | 0.350 | 0.393 | 0.416 | 0.425 | 0.427 |
| xgb F18+F19 | w2 | 0.6644 | 0.1688 | 0.440 | 0.437 | 0.388 | 0.395 | 0.385 |
| cb F18 | w2 | 0.7372 | 0.1613 | 0.530 | 0.560 | 0.546 | 0.518 | 0.508 |
| cb F18+F19 | w2 | 0.7192 | 0.1625 | 0.490 | 0.547 | 0.522 | 0.495 | 0.480 |

**The window-1 edge inverts.** On window 1 F19 lifted the deep book (@3–@20) on
both backends and traded away @1. On window 2:
- **xgb:** the book effect **flips sign** — @5/@10/@20 now go *down* (0.416→0.388,
  0.425→0.395, 0.427→0.385) while @1 goes *up* (0.350→0.440). AUC rises both
  windows (+0.032), but the R-p@10 delta is +0.045 (w1) → **−0.030 (w2)**.
- **cb:** F19 now **hurts everything** — AUC −0.018, Brier +0.001, and every K down
  (@10 0.518→0.495). The clean w1 deep-book win (@10 +0.055) reverses to −0.023.

Neither backend clears the deep-book bar on window 2. The one both-backend window-1
winner does not survive.

## Verdict — F19 not adopted

- **No cell shows a robust two-window F19 edge.** F19 (revenue growth) is
  **contextually additive but not robust** — the exact pattern of F17-macro
  (`_264`) and long-horizon F18 (`_278`). The single both-backend window-1 winner
  (`20pct/100d`) failed window-2 replication with a sign-flip on both backends.
- **F19 stays as committed, opt-in infrastructure** (the `all_fundamentals2` token),
  **not promoted.** F18/`all_fundamentals` remains byte-identical; no champion
  change, no `/daily-predictions` change (human decisions, `_019`).
- **The single-quarter-QoQ FS-drop test is unanswered** by this sweep (single-fit
  does no pruning) and parked in `V1.8_TBD` — but with F19 non-robust overall, it's
  a low-priority curiosity, not a blocker.
- Consistent with the standing read: on these sp500 cells the robust lever is the
  **backend** (CatBoost, `_278`), not additional fundamentals feature families.

Registry: 68 window-1 rows (`*_{f18,f19}{xgb,cb}`, mode `single_fit`) + 4 window-2
rows (`*_{f18,f19}{xgb,cb}_w2`).
Specs: `configs/gbdt/experiments/sp500_up_*_{f18,f19}{xgb,cb}.yaml`. Plan:
`V1.8_revenue_growth_features_plan.md`. Prior: `_278` (F18 window-2 + CatBoost),
`_274` (F18 lattice).
