# gbdt experiment — nasdaq100_up_40pct_50d_dd20pct_agentloop

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `50`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73600 (independent events ≈ 743.4; overlap-inflation 99.00×)
- val rows: 36800 (independent events ≈ 371.7; overlap-inflation 99.00×)
- eval rows: 18400 (independent events ≈ 185.9; overlap-inflation 99.00×)
- test rows: 4600 (independent events ≈ 46.5; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.040
- positive prevalence (eval): 0.060

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 2 | 30 | 0.0324 | 0.0235 | -0.0089 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 1
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -3.798
- Spiegelhalter p: 0.0001

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0511 | 0.0563 | +0.0051 | 0.2073 | 0.8404 |
| test | 0.0498 | 0.0524 | +0.0026 | 0.2191 | 0.6983 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0598

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3116 | 0.0598 | 343 | 62 | 199 | 144 / 343 / 343 |
| 5 | 0.3740 | 0.0598 | 1144 | 276 | 738 | 247 / 200 / 343 |
| 10 | 0.5337 | 0.0598 | 2144 | 538 | 1008 | 314 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0598 | 1 | 1 | 1 |
| 5 | 0.6000 | 0.0598 | 5 | 3 | 5 |
| 10 | 0.3000 | 0.0598 | 10 | 3 | 10 |

### test — n_rows=4600, base_rate=0.0554

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3725 | 0.0554 | 80 | 19 | 51 | 29 / 80 / 80 |
| 5 | 0.4452 | 0.0554 | 280 | 69 | 155 | 63 / 50 / 80 |
| 10 | 0.5187 | 0.0554 | 530 | 111 | 214 | 70 / 50 / 80 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0554 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0554 | 5 | 0 | 5 |
| 10 | 0.2000 | 0.0554 | 10 | 2 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 191 | 11 | 0.0576 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:INTC | 136 | 48 | 0.3529 |
| NASDAQ:MU | 126 | 97 | 0.7698 |
| NASDAQ:MRVL | 125 | 12 | 0.0960 |
| NASDAQ:MDB | 98 | 51 | 0.5204 |
| NASDAQ:TSLA | 87 | 5 | 0.0575 |
| NASDAQ:MCHP | 79 | 5 | 0.0633 |
| NASDAQ:TTD | 69 | 0 | 0.0000 |
| NASDAQ:AVGO | 43 | 25 | 0.5814 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:TTD | 69 | 0 | 0.0000 |
| NASDAQ:NVDA | 16 | 0 | 0.0000 |
| NASDAQ:TSLA | 87 | 5 | 0.0575 |
| NASDAQ:MSTR | 191 | 11 | 0.0576 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 50 | 17 | 0.3400 |
| NASDAQ:INTC | 43 | 28 | 0.6512 |
| NASDAQ:TTD | 43 | 0 | 0.0000 |
| NASDAQ:MU | 39 | 10 | 0.2564 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MRVL | 22 | 4 | 0.1818 |
| NASDAQ:AMD | 16 | 9 | 0.5625 |
| NASDAQ:MDB | 15 | 0 | 0.0000 |
| NASDAQ:TEAM | 14 | 0 | 0.0000 |
| NASDAQ:WBD | 6 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TTD | 43 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:MDB | 15 | 0 | 0.0000 |
| NASDAQ:TEAM | 14 | 0 | 0.0000 |
| NASDAQ:WBD | 6 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0598 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0598 |
| 2025Q1 | 109 | 9 | 0.0826 | 0.0598 |
| 2025Q2 | 310 | 88 | 0.2839 | 0.0598 |
| 2025Q3 | 320 | 117 | 0.3656 | 0.0598 |
| 2025Q4 | 310 | 62 | 0.2000 | 0.0598 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0554 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0554 |
| 2025Q4 | 11 | 5 | 0.4545 | 0.0554 |
| 2026Q1 | 240 | 64 | 0.2667 | 0.0554 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.2653 | 0.0314 | 0.0572 | `False` |
| test | 4600 | 0.0000 | 0.2653 | 0.0366 | 0.0580 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=3.80); shipped as `isotonic`. Brier vs base-rate: +0.0051 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
