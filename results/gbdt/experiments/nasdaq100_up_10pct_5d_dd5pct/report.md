# gbdt experiment — nasdaq100_up_10pct_5d_dd5pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `5`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73600 (independent events ≈ 8177.8; overlap-inflation 9.00×)
- val rows: 36800 (independent events ≈ 4088.9; overlap-inflation 9.00×)
- eval rows: 18400 (independent events ≈ 2044.4; overlap-inflation 9.00×)
- test rows: 8740 (independent events ≈ 971.1; overlap-inflation 9.00×)
- sample uniqueness weighting: `on` (horizon_days=5)
- positive prevalence (train): 0.047
- positive prevalence (eval): 0.043

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0371 | 0.0284 | -0.0088 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 115/2 |  |
| 1 | 115 | 0.0371 | 0.0284 | -0.0088 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 85/115 features |  |
| 2 | 85 | 0.0379 | 0.0284 | -0.0095 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 2.314
- Spiegelhalter p: 0.0206

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0379 | 0.0416 | +0.0037 | 0.1488 | 0.8165 |
| test | 0.0631 | 0.0663 | +0.0032 | 0.2339 | 0.7678 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0435

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1455 | 0.0435 | 343 | 24 | 165 | 178 / 343 / 343 |
| 5 | 0.2948 | 0.0435 | 1144 | 143 | 485 | 297 / 200 / 343 |
| 10 | 0.4340 | 0.0435 | 2144 | 263 | 606 | 327 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0435 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.0435 | 5 | 4 | 5 |
| 10 | 0.8000 | 0.0435 | 10 | 8 | 10 |

### test — n_rows=8740, base_rate=0.0714

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2778 | 0.0714 | 125 | 25 | 90 | 35 / 125 / 125 |
| 5 | 0.3819 | 0.0714 | 505 | 139 | 364 | 69 / 95 / 125 |
| 10 | 0.4586 | 0.0714 | 980 | 255 | 556 | 97 / 95 / 125 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0714 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.0714 | 5 | 2 | 5 |
| 10 | 0.2000 | 0.0714 | 10 | 2 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 199 | 22 | 0.1106 |
| NASDAQ:MRVL | 144 | 24 | 0.1667 |
| NASDAQ:ANSS | 143 | 1 | 0.0070 |
| NASDAQ:INTC | 82 | 6 | 0.0732 |
| NASDAQ:TSLA | 81 | 12 | 0.1481 |
| NASDAQ:TTD | 68 | 3 | 0.0441 |
| NASDAQ:AMD | 61 | 9 | 0.1475 |
| NASDAQ:MU | 59 | 19 | 0.3220 |
| NASDAQ:WBD | 59 | 7 | 0.1186 |
| NASDAQ:MDB | 55 | 8 | 0.1455 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TEAM | 15 | 0 | 0.0000 |
| NASDAQ:DXCM | 10 | 0 | 0.0000 |
| NASDAQ:CHTR | 9 | 0 | 0.0000 |
| NASDAQ:CRWD | 7 | 0 | 0.0000 |
| NASDAQ:DDOG | 5 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MU | 74 | 29 | 0.3919 |
| NASDAQ:INTC | 69 | 36 | 0.5217 |
| NASDAQ:TEAM | 66 | 11 | 0.1667 |
| NASDAQ:MSTR | 61 | 9 | 0.1475 |
| NASDAQ:MDB | 48 | 9 | 0.1875 |
| NASDAQ:AMD | 37 | 14 | 0.3784 |
| NASDAQ:DDOG | 32 | 11 | 0.3438 |
| NASDAQ:TTD | 31 | 10 | 0.3226 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MRVL | 20 | 3 | 0.1500 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:CHTR | 10 | 0 | 0.0000 |
| NASDAQ:MSTR | 61 | 9 | 0.1475 |
| NASDAQ:MRVL | 20 | 3 | 0.1500 |
| NASDAQ:TEAM | 66 | 11 | 0.1667 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0435 | 0.000 |
| 2024Q4 | 64 | 1 | 0.0156 | 0.0435 | 0.359 |
| 2025Q1 | 109 | 10 | 0.0917 | 0.0435 | 2.110 |
| 2025Q2 | 310 | 72 | 0.2323 | 0.0435 | 5.342 |
| 2025Q3 | 320 | 30 | 0.0938 | 0.0435 | 2.156 |
| 2025Q4 | 310 | 30 | 0.0968 | 0.0435 | 2.226 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0714 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0714 | 0.000 |
| 2025Q4 | 11 | 4 | 0.3636 | 0.0714 | 5.093 |
| 2026Q1 | 305 | 67 | 0.2197 | 0.0714 | 3.077 |
| 2026Q2 | 160 | 68 | 0.4250 | 0.0714 | 5.953 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.5500 | 0.0420 | 0.0535 | `False` |
| test | 8740 | 0.0000 | 0.5500 | 0.0467 | 0.0532 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=2.31); shipped as `isotonic`. Brier vs base-rate: +0.0037 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
