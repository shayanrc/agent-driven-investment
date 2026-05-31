# gbdt experiment — nasdaq100_up_50pct_25d_dd25pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `25`
- max_drawdown: `0.25`
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
- positive prevalence (train): 0.007
- positive prevalence (eval): 0.007

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0052 | 0.0038 | -0.0014 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 25/27 |  |
| 1 | 25 | 0.0062 | 0.0037 | -0.0025 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 16/25 features |  |
| 2 | 16 | 0.0059 | 0.0038 | -0.0021 | iteration 2 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -4.741
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0071 | 0.0068 | -0.0003 | 0.0350 | 0.8969 |
| test | 0.0287 | 0.0283 | -0.0004 | 0.1935 | 0.7607 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0068

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0469 | 0.0068 | 343 | 3 | 64 | 279 / 343 / 343 |
| 5 | 0.3571 | 0.0068 | 1144 | 45 | 126 | 342 / 200 / 343 |
| 10 | 0.5794 | 0.0068 | 2144 | 73 | 126 | 343 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0068 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0068 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0068 | 10 | 0 | 10 |

### test — n_rows=6900, base_rate=0.0291

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3077 | 0.0291 | 105 | 16 | 52 | 53 / 105 / 105 |
| 5 | 0.4101 | 0.0291 | 405 | 57 | 139 | 88 / 75 / 105 |
| 10 | 0.5450 | 0.0291 | 780 | 109 | 200 | 97 / 75 / 105 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0291 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0291 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0291 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 147 | 3 | 0.0204 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:AMD | 132 | 18 | 0.1364 |
| NASDAQ:TSLA | 94 | 3 | 0.0319 |
| NASDAQ:AVGO | 89 | 1 | 0.0112 |
| NASDAQ:INTC | 85 | 8 | 0.0941 |
| NASDAQ:MU | 75 | 11 | 0.1467 |
| NASDAQ:MCHP | 63 | 0 | 0.0000 |
| NASDAQ:CRWD | 58 | 0 | 0.0000 |
| NASDAQ:WBD | 51 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MCHP | 63 | 0 | 0.0000 |
| NASDAQ:CRWD | 58 | 0 | 0.0000 |
| NASDAQ:WBD | 51 | 0 | 0.0000 |
| NASDAQ:MRVL | 46 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MU | 75 | 17 | 0.2267 |
| NASDAQ:AMD | 59 | 10 | 0.1695 |
| NASDAQ:MSTR | 45 | 5 | 0.1111 |
| NASDAQ:INTC | 42 | 9 | 0.2143 |
| NASDAQ:TEAM | 33 | 0 | 0.0000 |
| NASDAQ:MDB | 30 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:CRWD | 28 | 6 | 0.2143 |
| NASDAQ:DDOG | 24 | 8 | 0.3333 |
| NASDAQ:LRCX | 15 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TEAM | 33 | 0 | 0.0000 |
| NASDAQ:MDB | 30 | 0 | 0.0000 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:LRCX | 15 | 0 | 0.0000 |
| NASDAQ:TTD | 6 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0068 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0068 | 0.000 |
| 2025Q1 | 109 | 0 | 0.0000 | 0.0068 | 0.000 |
| 2025Q2 | 310 | 7 | 0.0226 | 0.0068 | 3.297 |
| 2025Q3 | 320 | 21 | 0.0656 | 0.0068 | 9.583 |
| 2025Q4 | 310 | 17 | 0.0548 | 0.0068 | 8.008 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0291 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0291 | 0.000 |
| 2025Q4 | 11 | 2 | 0.1818 | 0.0291 | 6.242 |
| 2026Q1 | 305 | 9 | 0.0295 | 0.0291 | 1.013 |
| 2026Q2 | 60 | 46 | 0.7667 | 0.0291 | 26.318 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.3065 | 0.0065 | 0.0257 | `True` |
| test | 6900 | 0.0000 | 0.3065 | 0.0051 | 0.0210 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=4.74); shipped as `isotonic`. Brier vs base-rate: -0.0003 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
