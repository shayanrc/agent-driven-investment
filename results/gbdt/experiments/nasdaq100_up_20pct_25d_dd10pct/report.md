# gbdt experiment — nasdaq100_up_20pct_25d_dd10pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `25`
- max_drawdown: `0.1`
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
- positive prevalence (train): 0.081
- positive prevalence (eval): 0.095

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0601 | 0.0490 | -0.0111 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 70/27 |  |
| 1 | 70 | 0.0599 | 0.0491 | -0.0109 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 47/70 features |  |
| 2 | 47 | 0.0594 | 0.0495 | -0.0100 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -2.343
- Spiegelhalter p: 0.0191

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0784 | 0.0862 | +0.0078 | 0.2646 | 0.8070 |
| test | 0.1010 | 0.1037 | +0.0027 | 0.3553 | 0.7094 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0953

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1392 | 0.0953 | 343 | 27 | 194 | 149 / 343 / 343 |
| 5 | 0.3168 | 0.0953 | 1144 | 269 | 849 | 199 / 200 / 343 |
| 10 | 0.4176 | 0.0953 | 2144 | 542 | 1298 | 283 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0953 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.0953 | 5 | 4 | 5 |
| 10 | 0.6000 | 0.0953 | 10 | 6 | 10 |

### test — n_rows=6900, base_rate=0.1175

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1579 | 0.1175 | 105 | 12 | 76 | 29 / 105 / 105 |
| 5 | 0.3153 | 0.1175 | 405 | 105 | 333 | 53 / 75 / 105 |
| 10 | 0.4559 | 0.1175 | 780 | 243 | 533 | 71 / 75 / 105 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1175 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.1175 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.1175 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 198 | 21 | 0.1061 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:INTC | 122 | 39 | 0.3197 |
| NASDAQ:MRVL | 98 | 40 | 0.4082 |
| NASDAQ:MCHP | 89 | 25 | 0.2809 |
| NASDAQ:AVGO | 84 | 16 | 0.1905 |
| NASDAQ:AMD | 73 | 41 | 0.5616 |
| NASDAQ:MDB | 71 | 19 | 0.2676 |
| NASDAQ:TSLA | 53 | 12 | 0.2264 |
| NASDAQ:TEAM | 38 | 1 | 0.0263 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:TTD | 20 | 0 | 0.0000 |
| NASDAQ:LULU | 7 | 0 | 0.0000 |
| NASDAQ:TEAM | 38 | 1 | 0.0263 |
| NASDAQ:PDD | 10 | 1 | 0.1000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AMD | 58 | 32 | 0.5517 |
| NASDAQ:TTD | 58 | 6 | 0.1034 |
| NASDAQ:TEAM | 57 | 6 | 0.1053 |
| NASDAQ:MSTR | 47 | 11 | 0.2340 |
| NASDAQ:INTC | 44 | 21 | 0.4773 |
| NASDAQ:MDB | 41 | 8 | 0.1951 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MRVL | 20 | 0 | 0.0000 |
| NASDAQ:DDOG | 17 | 12 | 0.7059 |
| NASDAQ:TSLA | 14 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MRVL | 20 | 0 | 0.0000 |
| NASDAQ:TSLA | 14 | 0 | 0.0000 |
| NASDAQ:LULU | 7 | 0 | 0.0000 |
| NASDAQ:TTD | 58 | 6 | 0.1034 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0953 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0953 | 0.000 |
| 2025Q1 | 109 | 5 | 0.0459 | 0.0953 | 0.481 |
| 2025Q2 | 310 | 122 | 0.3935 | 0.0953 | 4.128 |
| 2025Q3 | 320 | 83 | 0.2594 | 0.0953 | 2.721 |
| 2025Q4 | 310 | 59 | 0.1903 | 0.0953 | 1.997 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.1175 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.1175 | 0.000 |
| 2025Q4 | 11 | 5 | 0.4545 | 0.1175 | 3.867 |
| 2026Q1 | 305 | 55 | 0.1803 | 0.1175 | 1.534 |
| 2026Q2 | 60 | 45 | 0.7500 | 0.1175 | 6.381 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.4537 | 0.0724 | 0.0760 | `False` |
| test | 6900 | 0.0000 | 0.4537 | 0.0721 | 0.0738 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=2.34); shipped as `isotonic`. Brier vs base-rate: +0.0078 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
