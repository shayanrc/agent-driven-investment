# gbdt experiment — sp500_up_40pct_200d_dd20pct_aligned_cbagent

## Spec

- universe: `sp500`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `200`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Data

- tickers in universe: 503
- tickers used: 477
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:CRWD, NASDAQ:DASH, NASDAQ:DDOG, NASDAQ:GEHC, NASDAQ:PLTR, NYSE:CARR, NYSE:COIN, NYSE:CTVA, NYSE:DOW, NYSE:EXE, NYSE:FOX, NYSE:FOXA, NYSE:GEV, NYSE:HOOD, NYSE:KVUE, NYSE:MRNA, NYSE:OTIS, NYSE:Q, NYSE:SNDK, NYSE:SOLV, NYSE:UBER, NYSE:VLTO, NYSE:VRT
- train rows: 381600 (independent events ≈ 956.6; overlap-inflation 398.92×)
- val rows: 190800 (independent events ≈ 478.2; overlap-inflation 399.00×)
- eval rows: 95400 (independent events ≈ 239.1; overlap-inflation 399.00×)
- test rows: 143100 (independent events ≈ 358.6; overlap-inflation 399.00×)
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.212
- positive prevalence (eval): 0.176

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2018-01-01`
- train: `2018-01-02` → `2021-03-08`
- val: `2021-03-09` → `2022-10-06`
- eval: `2022-10-07` → `2023-07-26`
- test: `2023-07-27` → `2024-10-03`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|

## Final checkpoint

- best iteration: 0
- iterations run: 2
- inner stop signal: `agent_should_stop`
- fs_hp_loop callback_mode: `agent_file_protocol`
- tie-break path: `v14_val_flat_eval_rp1` — Val_brier flat: tie set picked by eval R-Precision@1 (V1.4 P1)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -70.773
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1382 | 0.1450 | +0.0068 | 0.4446 | 0.6745 |
| test | 0.1916 | 0.1710 | -0.0206 | 0.6682 | 0.6732 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=95400, base_rate=0.1760

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.8050 | 0.1760 | 200 | 161 | 200 | 0 / 200 / 200 |
| 5 | 0.6390 | 0.1760 | 1000 | 639 | 1000 | 0 / 200 / 200 |
| 10 | 0.5725 | 0.1760 | 2000 | 1145 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1760 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1760 | 5 | 5 | 5 |
| 10 | 0.9000 | 0.1760 | 10 | 9 | 10 |

### test — n_rows=143100, base_rate=0.2189

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5567 | 0.2189 | 300 | 167 | 300 | 0 / 300 / 300 |
| 5 | 0.4927 | 0.2189 | 1500 | 739 | 1500 | 0 / 300 / 300 |
| 10 | 0.4490 | 0.2189 | 3000 | 1347 | 3000 | 0 / 300 / 300 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.2189 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.2189 | 5 | 1 | 5 |
| 10 | 0.2000 | 0.2189 | 10 | 2 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=95400, Q_days=200, base_rate=0.1760

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.8050 | 0.1760 | 200 |
| 3 | 0.6883 | 0.1760 | 200 |
| 5 | 0.6390 | 0.1760 | 200 |
| 10 | 0.5725 | 0.1760 | 200 |
| 20 | 0.5012 | 0.1760 | 200 |

### test — n_rows=143100, Q_days=300, base_rate=0.2189

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.5567 | 0.2189 | 300 |
| 3 | 0.5844 | 0.2189 | 300 |
| 5 | 0.4927 | 0.2189 | 300 |
| 10 | 0.4490 | 0.2189 | 300 |
| 20 | 0.4520 | 0.2189 | 300 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TSLA | 152 | 65 | 0.4276 |
| NYSE:CCL | 120 | 80 | 0.6667 |
| NYSE:NCLH | 87 | 74 | 0.8506 |
| NASDAQ:WBD | 82 | 14 | 0.1707 |
| NASDAQ:AMD | 68 | 60 | 0.8824 |
| NYSE:GNRC | 66 | 26 | 0.3939 |
| NASDAQ:NVDA | 63 | 61 | 0.9683 |
| NYSE:SMCI | 60 | 51 | 0.8500 |
| NYSE:RCL | 55 | 53 | 0.9636 |
| NYSE:CVNA | 47 | 39 | 0.8298 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:PSKY | 24 | 0 | 0.0000 |
| NYSE:ALGN | 21 | 0 | 0.0000 |
| NASDAQ:PYPL | 13 | 0 | 0.0000 |
| NYSE:COHR | 14 | 1 | 0.0714 |
| NASDAQ:WBD | 82 | 14 | 0.1707 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 300 | 233 | 0.7767 |
| NYSE:SMCI | 292 | 131 | 0.4486 |
| NYSE:COHR | 214 | 182 | 0.8505 |
| NYSE:SATS | 182 | 138 | 0.7582 |
| NYSE:PSKY | 170 | 2 | 0.0118 |
| NASDAQ:TSLA | 81 | 18 | 0.2222 |
| NYSE:ALB | 67 | 2 | 0.0299 |
| NYSE:GNRC | 55 | 11 | 0.2000 |
| NASDAQ:WBD | 40 | 1 | 0.0250 |
| NASDAQ:TTD | 22 | 10 | 0.4545 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AMD | 18 | 0 | 0.0000 |
| NYSE:KEY | 11 | 0 | 0.0000 |
| NYSE:XYZ | 8 | 0 | 0.0000 |
| NYSE:CCL | 6 | 0 | 0.0000 |
| NYSE:PSKY | 170 | 2 | 0.0118 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 295 | 224 | 0.7593 | 0.1760 |
| 2023Q1 | 310 | 187 | 0.6032 | 0.1760 |
| 2023Q2 | 310 | 213 | 0.6871 | 0.1760 |
| 2023Q3 | 85 | 15 | 0.1765 | 0.1760 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 230 | 69 | 0.3000 | 0.2189 |
| 2023Q4 | 315 | 203 | 0.6444 | 0.2189 |
| 2024Q1 | 305 | 187 | 0.6131 | 0.2189 |
| 2024Q2 | 315 | 150 | 0.4762 | 0.2189 |
| 2024Q3 | 320 | 123 | 0.3844 | 0.2189 |
| 2024Q4 | 15 | 7 | 0.4667 | 0.2189 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 95400 | 0.0151 | 0.4689 | 0.1248 | 0.0666 | `False` |
| test | 143100 | 0.0000 | 0.3712 | 0.0544 | 0.0385 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=70.77); shipped as `isotonic`. Brier vs base-rate: +0.0068 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
