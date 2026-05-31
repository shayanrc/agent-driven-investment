# gbdt experiment — nasdaq100_up_40pct_25d_dd20pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `25`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73600 (independent events ≈ 1502.0; overlap-inflation 49.00×)
- val rows: 36800 (independent events ≈ 751.0; overlap-inflation 49.00×)
- eval rows: 18400 (independent events ≈ 375.5; overlap-inflation 49.00×)
- test rows: 6900 (independent events ≈ 140.8; overlap-inflation 49.00×)
- sample uniqueness weighting: `on` (horizon_days=25)
- positive prevalence (train): 0.015
- positive prevalence (eval): 0.014

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0113 | 0.0071 | -0.0042 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 70/27 |  |
| 1 | 70 | 0.0122 | 0.0072 | -0.0050 | iteration 1 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 1
- iterations run: 2
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -11.124
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0137 | 0.0137 | +0.0000 | 0.0604 | 0.9077 |
| test | 0.0405 | 0.0408 | +0.0003 | 0.2608 | 0.7235 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0139

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1522 | 0.0139 | 343 | 14 | 92 | 251 / 343 / 343 |
| 5 | 0.3008 | 0.0139 | 1144 | 71 | 236 | 331 / 200 / 343 |
| 10 | 0.6047 | 0.0139 | 2144 | 153 | 253 | 342 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0139 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.0139 | 5 | 2 | 5 |
| 10 | 0.4000 | 0.0139 | 10 | 4 | 10 |

### test — n_rows=6900, base_rate=0.0426

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1607 | 0.0426 | 105 | 9 | 56 | 49 / 105 / 105 |
| 5 | 0.2831 | 0.0426 | 405 | 47 | 166 | 83 / 75 / 105 |
| 10 | 0.5060 | 0.0426 | 780 | 127 | 251 | 90 / 75 / 105 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0426 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0426 | 5 | 0 | 5 |
| 10 | 0.3000 | 0.0426 | 10 | 3 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 199 | 7 | 0.0352 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:TSLA | 123 | 0 | 0.0000 |
| NASDAQ:ON | 115 | 0 | 0.0000 |
| NASDAQ:MRVL | 112 | 5 | 0.0446 |
| NASDAQ:MU | 81 | 25 | 0.3086 |
| NASDAQ:MCHP | 66 | 19 | 0.2879 |
| NASDAQ:MDB | 66 | 4 | 0.0606 |
| NASDAQ:TTD | 59 | 0 | 0.0000 |
| NASDAQ:INTC | 50 | 7 | 0.1400 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:TSLA | 123 | 0 | 0.0000 |
| NASDAQ:ON | 115 | 0 | 0.0000 |
| NASDAQ:TTD | 59 | 0 | 0.0000 |
| NASDAQ:TEAM | 45 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 75 | 12 | 0.1600 |
| NASDAQ:MU | 75 | 19 | 0.2533 |
| NASDAQ:TTD | 50 | 0 | 0.0000 |
| NASDAQ:TEAM | 43 | 7 | 0.1628 |
| NASDAQ:INTC | 38 | 8 | 0.2105 |
| NASDAQ:MDB | 30 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:WBD | 21 | 0 | 0.0000 |
| NASDAQ:LRCX | 18 | 0 | 0.0000 |
| NASDAQ:AMD | 12 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TTD | 50 | 0 | 0.0000 |
| NASDAQ:MDB | 30 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:WBD | 21 | 0 | 0.0000 |
| NASDAQ:LRCX | 18 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0139 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0139 | 0.000 |
| 2025Q1 | 109 | 0 | 0.0000 | 0.0139 | 0.000 |
| 2025Q2 | 310 | 32 | 0.1032 | 0.0139 | 7.419 |
| 2025Q3 | 320 | 15 | 0.0469 | 0.0139 | 3.369 |
| 2025Q4 | 310 | 24 | 0.0774 | 0.0139 | 5.565 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0426 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0426 | 0.000 |
| 2025Q4 | 11 | 3 | 0.2727 | 0.0426 | 6.401 |
| 2026Q1 | 305 | 13 | 0.0426 | 0.0426 | 1.000 |
| 2026Q2 | 60 | 31 | 0.5167 | 0.0426 | 12.126 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.2220 | 0.0175 | 0.0457 | `True` |
| test | 6900 | 0.0000 | 0.2220 | 0.0179 | 0.0490 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=11.12); shipped as `isotonic`. Brier vs base-rate: +0.0000 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
