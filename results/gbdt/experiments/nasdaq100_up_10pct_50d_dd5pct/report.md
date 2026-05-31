# gbdt experiment — nasdaq100_up_10pct_50d_dd5pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `50`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73600 (independent events ≈ 743.4; overlap-inflation 99.00×)
- val rows: 36800 (independent events ≈ 371.7; overlap-inflation 99.00×)
- eval rows: 18400 (independent events ≈ 185.9; overlap-inflation 99.00×)
- test rows: 4600 (independent events ≈ 46.5; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.366
- positive prevalence (eval): 0.354

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.2184 | 0.2249 | 0.0065 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 22/27 |  |
| 1 | 22 | 0.2195 | 0.2251 | 0.0057 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 14/22 features |  |
| 2 | 14 | 0.2184 | 0.2247 | 0.0063 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -2.451
- Spiegelhalter p: 0.0142

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2216 | 0.2286 | +0.0070 | 0.6353 | 0.5836 |
| test | 0.2041 | 0.1949 | -0.0092 | 0.5993 | 0.4750 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.3538

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.6352 | 0.3538 | 343 | 155 | 244 | 99 / 343 / 343 |
| 5 | 0.4374 | 0.3538 | 1144 | 451 | 1031 | 149 / 200 / 343 |
| 10 | 0.4633 | 0.3538 | 2144 | 921 | 1988 | 154 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.3538 | 1 | 1 | 1 |
| 5 | 0.2000 | 0.3538 | 5 | 1 | 5 |
| 10 | 0.2000 | 0.3538 | 10 | 2 | 10 |

### test — n_rows=4600, base_rate=0.2652

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.6714 | 0.2652 | 80 | 47 | 70 | 10 / 80 / 80 |
| 5 | 0.4424 | 0.2652 | 280 | 119 | 269 | 31 / 50 / 80 |
| 10 | 0.3497 | 0.2652 | 530 | 178 | 509 | 33 / 50 / 80 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.2652 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.2652 | 5 | 4 | 5 |
| 10 | 0.4000 | 0.2652 | 10 | 4 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADI | 200 | 103 | 0.5150 |
| NASDAQ:ADBE | 194 | 25 | 0.1289 |
| NASDAQ:AAPL | 179 | 98 | 0.5475 |
| NASDAQ:ANSS | 143 | 44 | 0.3077 |
| NASDAQ:INTC | 125 | 76 | 0.6080 |
| NASDAQ:ADP | 88 | 1 | 0.0114 |
| NASDAQ:AMAT | 81 | 44 | 0.5432 |
| NASDAQ:ADSK | 52 | 14 | 0.2692 |
| NASDAQ:AMD | 39 | 25 | 0.6410 |
| NASDAQ:MSTR | 15 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 15 | 0 | 0.0000 |
| NASDAQ:ADP | 88 | 1 | 0.0114 |
| NASDAQ:ADBE | 194 | 25 | 0.1289 |
| NASDAQ:ADSK | 52 | 14 | 0.2692 |
| NASDAQ:ANSS | 143 | 44 | 0.3077 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADI | 50 | 32 | 0.6400 |
| NASDAQ:AMAT | 49 | 21 | 0.4286 |
| NASDAQ:AMD | 44 | 22 | 0.5000 |
| NASDAQ:AMZN | 44 | 14 | 0.3182 |
| NASDAQ:ADSK | 43 | 9 | 0.2093 |
| NASDAQ:ANSS | 29 | 19 | 0.6552 |
| NASDAQ:ASML | 7 | 0 | 0.0000 |
| NASDAQ:AAPL | 6 | 0 | 0.0000 |
| NASDAQ:ADBE | 6 | 0 | 0.0000 |
| NASDAQ:AZN | 1 | 1 | 1.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ASML | 7 | 0 | 0.0000 |
| NASDAQ:AAPL | 6 | 0 | 0.0000 |
| NASDAQ:ADBE | 6 | 0 | 0.0000 |
| NASDAQ:ADSK | 43 | 9 | 0.2093 |
| NASDAQ:AMZN | 44 | 14 | 0.3182 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 22 | 0.7097 | 0.3538 | 2.006 |
| 2024Q4 | 64 | 22 | 0.3438 | 0.3538 | 0.972 |
| 2025Q1 | 109 | 2 | 0.0183 | 0.3538 | 0.052 |
| 2025Q2 | 310 | 169 | 0.5452 | 0.3538 | 1.541 |
| 2025Q3 | 320 | 135 | 0.4219 | 0.3538 | 1.192 |
| 2025Q4 | 310 | 101 | 0.3258 | 0.3538 | 0.921 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 17 | 1.0000 | 0.2652 | 3.770 |
| 2025Q3 | 12 | 2 | 0.1667 | 0.2652 | 0.628 |
| 2025Q4 | 11 | 5 | 0.4545 | 0.2652 | 1.714 |
| 2026Q1 | 240 | 95 | 0.3958 | 0.2652 | 1.492 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.2886 | 0.5625 | 0.3881 | 0.0668 | `False` |
| test | 4600 | 0.2886 | 0.5625 | 0.3499 | 0.0321 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=2.45); shipped as `isotonic`. Brier vs base-rate: +0.0070 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
