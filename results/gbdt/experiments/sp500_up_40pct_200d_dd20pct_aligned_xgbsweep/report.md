# gbdt experiment — sp500_up_40pct_200d_dd20pct_aligned_xgbsweep

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `sp500`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `200`
- max_drawdown: `0.2`
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
| 0 | 279 | 0.0786 | 0.1153 | 0.0367 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 149/2 |  |
| 1 | 149 | 0.0777 | 0.1176 | 0.0398 | iteration 1 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 0
- iterations run: 2
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`
- tie-break path: `v14_val_flat_eval_rp1` — Val_brier flat: tie set picked by eval R-Precision@1 (V1.4 P1)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -36.630
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1396 | 0.1450 | +0.0054 | 0.4549 | 0.6868 |
| test | 0.1896 | 0.1710 | -0.0186 | 0.6501 | 0.6691 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=95400, base_rate=0.1760

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.7400 | 0.1760 | 200 | 148 | 200 | 0 / 200 / 200 |
| 5 | 0.5690 | 0.1760 | 1000 | 569 | 1000 | 0 / 200 / 200 |
| 10 | 0.5005 | 0.1760 | 2000 | 1001 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1760 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1760 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.1760 | 10 | 10 | 10 |

### test — n_rows=143100, base_rate=0.2189

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2400 | 0.2189 | 300 | 72 | 300 | 0 / 300 / 300 |
| 5 | 0.3800 | 0.2189 | 1500 | 570 | 1500 | 0 / 300 / 300 |
| 10 | 0.4187 | 0.2189 | 3000 | 1256 | 3000 | 0 / 300 / 300 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.2189 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.2189 | 5 | 2 | 5 |
| 10 | 0.3000 | 0.2189 | 10 | 3 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=95400, Q_days=200, base_rate=0.1760

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.7400 | 0.1760 | 200 |
| 3 | 0.6467 | 0.1760 | 200 |
| 5 | 0.5690 | 0.1760 | 200 |
| 10 | 0.5005 | 0.1760 | 200 |
| 20 | 0.4487 | 0.1760 | 200 |

### test — n_rows=143100, Q_days=300, base_rate=0.2189

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.2400 | 0.2189 | 300 |
| 3 | 0.3600 | 0.2189 | 300 |
| 5 | 0.3800 | 0.2189 | 300 |
| 10 | 0.4187 | 0.2189 | 300 |
| 20 | 0.4422 | 0.2189 | 300 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CCL | 131 | 94 | 0.7176 |
| NASDAQ:WBD | 117 | 17 | 0.1453 |
| NASDAQ:TTD | 113 | 78 | 0.6903 |
| NASDAQ:TSLA | 110 | 52 | 0.4727 |
| NYSE:NCLH | 73 | 44 | 0.6027 |
| NYSE:XYZ | 60 | 3 | 0.0500 |
| NASDAQ:META | 59 | 56 | 0.9492 |
| NASDAQ:NVDA | 58 | 58 | 1.0000 |
| NASDAQ:AMD | 55 | 50 | 0.9091 |
| NYSE:CVNA | 51 | 41 | 0.8039 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:IVZ | 23 | 0 | 0.0000 |
| NASDAQ:PYPL | 13 | 0 | 0.0000 |
| NYSE:COHR | 5 | 0 | 0.0000 |
| NYSE:XYZ | 60 | 3 | 0.0500 |
| NASDAQ:WBD | 117 | 17 | 0.1453 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:PSKY | 193 | 14 | 0.0725 |
| NYSE:SATS | 129 | 91 | 0.7054 |
| NASDAQ:TSLA | 123 | 46 | 0.3740 |
| NYSE:SMCI | 98 | 41 | 0.4184 |
| NYSE:NCLH | 81 | 42 | 0.5185 |
| NYSE:COHR | 78 | 39 | 0.5000 |
| NYSE:EQT | 58 | 31 | 0.5345 |
| NYSE:FSLR | 58 | 18 | 0.3103 |
| NYSE:ALGN | 51 | 0 | 0.0000 |
| NYSE:CVNA | 42 | 39 | 0.9286 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ALGN | 51 | 0 | 0.0000 |
| NASDAQ:ON | 38 | 0 | 0.0000 |
| NYSE:APA | 25 | 0 | 0.0000 |
| NYSE:HAL | 25 | 0 | 0.0000 |
| NYSE:SLB | 23 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 295 | 196 | 0.6644 | 0.1760 |
| 2023Q1 | 310 | 183 | 0.5903 | 0.1760 |
| 2023Q2 | 310 | 175 | 0.5645 | 0.1760 |
| 2023Q3 | 85 | 15 | 0.1765 | 0.1760 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 230 | 40 | 0.1739 | 0.2189 |
| 2023Q4 | 315 | 148 | 0.4698 | 0.2189 |
| 2024Q1 | 305 | 124 | 0.4066 | 0.2189 |
| 2024Q2 | 315 | 152 | 0.4825 | 0.2189 |
| 2024Q3 | 320 | 103 | 0.3219 | 0.2189 |
| 2024Q4 | 15 | 3 | 0.2000 | 0.2189 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 95400 | 0.0000 | 0.5000 | 0.1077 | 0.0758 | `False` |
| test | 143100 | 0.0000 | 0.3615 | 0.0602 | 0.0445 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=36.63); shipped as `isotonic`. Brier vs base-rate: +0.0054 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
