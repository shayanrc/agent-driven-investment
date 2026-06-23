# gbdt experiment — sp500_up_50pct_200d_dd25pct_aligned_xgbsweep

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `sp500`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `200`
- max_drawdown: `0.25`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 503
- tickers used: 477
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:CRWD, NASDAQ:DASH, NASDAQ:DDOG, NASDAQ:GEHC, NASDAQ:PLTR, NYSE:CARR, NYSE:COIN, NYSE:CTVA, NYSE:DOW, NYSE:EXE, NYSE:FOX, NYSE:FOXA, NYSE:GEV, NYSE:HOOD, NYSE:KVUE, NYSE:MRNA, NYSE:OTIS, NYSE:Q, NYSE:SNDK, NYSE:SOLV, NYSE:UBER, NYSE:VLTO, NYSE:VRT
- train rows: 381600 (independent events ≈ 956.6; overlap-inflation 398.92×)
- val rows: 190800 (independent events ≈ 478.2; overlap-inflation 399.00×)
- eval rows: 95400 (independent events ≈ 239.1; overlap-inflation 399.00×)
- test rows: 143100 (independent events ≈ 358.6; overlap-inflation 399.00×)
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.147
- positive prevalence (eval): 0.107

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
| 0 | 279 | 0.0553 | 0.0720 | 0.0167 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 143/2 |  |
| 1 | 143 | 0.0550 | 0.0722 | 0.0172 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 135/143 features |  |
| 2 | 135 | 0.0542 | 0.0734 | 0.0192 | iteration 2 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`
- tie-break path: `v14_val_flat_eval_rp1` — Val_brier flat: tie set picked by eval R-Precision@1 (V1.4 P1)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -55.078
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0908 | 0.0956 | +0.0048 | 0.3288 | 0.7345 |
| test | 0.1216 | 0.1168 | -0.0048 | 0.5212 | 0.7001 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=95400, base_rate=0.1071

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.7550 | 0.1071 | 200 | 151 | 200 | 0 / 200 / 200 |
| 5 | 0.4780 | 0.1071 | 1000 | 478 | 1000 | 0 / 200 / 200 |
| 10 | 0.4125 | 0.1071 | 2000 | 825 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1071 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1071 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.1071 | 10 | 10 | 10 |

### test — n_rows=143100, base_rate=0.1350

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2967 | 0.1350 | 300 | 89 | 300 | 0 / 300 / 300 |
| 5 | 0.3813 | 0.1350 | 1500 | 572 | 1500 | 0 / 300 / 300 |
| 10 | 0.4140 | 0.1350 | 3000 | 1242 | 3000 | 0 / 300 / 300 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1350 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.1350 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.1350 | 10 | 0 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=95400, Q_days=200, base_rate=0.1071

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.7550 | 0.1071 | 200 |
| 3 | 0.5783 | 0.1071 | 200 |
| 5 | 0.4780 | 0.1071 | 200 |
| 10 | 0.4125 | 0.1071 | 200 |
| 20 | 0.3953 | 0.1071 | 200 |

### test — n_rows=143100, Q_days=300, base_rate=0.1350

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.2967 | 0.1350 | 300 |
| 3 | 0.3689 | 0.1350 | 300 |
| 5 | 0.3813 | 0.1350 | 300 |
| 10 | 0.4140 | 0.1350 | 300 |
| 20 | 0.3977 | 0.1350 | 300 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TTD | 123 | 76 | 0.6179 |
| NASDAQ:WBD | 118 | 14 | 0.1186 |
| NYSE:CCL | 103 | 74 | 0.7184 |
| NASDAQ:TSLA | 86 | 44 | 0.5116 |
| NYSE:CVNA | 71 | 55 | 0.7746 |
| NYSE:NOW | 54 | 40 | 0.7407 |
| NASDAQ:AMD | 44 | 44 | 1.0000 |
| NYSE:AXON | 40 | 1 | 0.0250 |
| NYSE:RCL | 38 | 33 | 0.8684 |
| NASDAQ:PYPL | 35 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:PYPL | 35 | 0 | 0.0000 |
| NYSE:XYZ | 32 | 0 | 0.0000 |
| NYSE:EPAM | 25 | 0 | 0.0000 |
| NYSE:FCX | 21 | 0 | 0.0000 |
| NYSE:LVS | 10 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:PSKY | 214 | 0 | 0.0000 |
| NYSE:SMCI | 158 | 62 | 0.3924 |
| NYSE:CVNA | 153 | 124 | 0.8105 |
| NASDAQ:TSLA | 118 | 42 | 0.3559 |
| NYSE:COHR | 93 | 82 | 0.8817 |
| NYSE:NCLH | 82 | 16 | 0.1951 |
| NYSE:CCL | 75 | 37 | 0.4933 |
| NYSE:SATS | 73 | 55 | 0.7534 |
| NYSE:FSLR | 61 | 18 | 0.2951 |
| NYSE:HAL | 55 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:PSKY | 214 | 0 | 0.0000 |
| NYSE:HAL | 55 | 0 | 0.0000 |
| NYSE:ALGN | 35 | 0 | 0.0000 |
| NYSE:F | 16 | 0 | 0.0000 |
| NYSE:KEY | 11 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 295 | 153 | 0.5186 | 0.1071 |
| 2023Q1 | 310 | 138 | 0.4452 | 0.1071 |
| 2023Q2 | 310 | 180 | 0.5806 | 0.1071 |
| 2023Q3 | 85 | 7 | 0.0824 | 0.1071 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 230 | 46 | 0.2000 | 0.1350 |
| 2023Q4 | 315 | 145 | 0.4603 | 0.1350 |
| 2024Q1 | 305 | 148 | 0.4852 | 0.1350 |
| 2024Q2 | 315 | 147 | 0.4667 | 0.1350 |
| 2024Q3 | 320 | 82 | 0.2562 | 0.1350 |
| 2024Q4 | 15 | 4 | 0.2667 | 0.1350 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 95400 | 0.0000 | 0.2792 | 0.0582 | 0.0638 | `False` |
| test | 143100 | 0.0000 | 0.2792 | 0.0331 | 0.0424 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=55.08); shipped as `isotonic`. Brier vs base-rate: +0.0048 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
