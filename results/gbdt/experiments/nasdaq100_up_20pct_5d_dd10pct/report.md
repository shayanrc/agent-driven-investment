# gbdt experiment — nasdaq100_up_20pct_5d_dd10pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `5`
- max_drawdown: `0.1`
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
- positive prevalence (train): 0.007
- positive prevalence (eval): 0.006

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0050 | 0.0046 | -0.0004 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 103/2 |  |
| 1 | 103 | 0.0052 | 0.0046 | -0.0006 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 77/103 features |  |
| 2 | 77 | 0.0052 | 0.0046 | -0.0006 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 5.651
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0056 | 0.0057 | +0.0001 | 0.0295 | 0.8563 |
| test | 0.0127 | 0.0128 | +0.0001 | 0.0660 | 0.7558 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0057

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0758 | 0.0057 | 343 | 5 | 66 | 277 / 343 / 343 |
| 5 | 0.3131 | 0.0057 | 1144 | 31 | 99 | 341 / 200 / 343 |
| 10 | 0.5048 | 0.0057 | 2144 | 53 | 105 | 343 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0057 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0057 | 5 | 0 | 5 |
| 10 | 0.2000 | 0.0057 | 10 | 2 | 10 |

### test — n_rows=8740, base_rate=0.0129

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1556 | 0.0129 | 125 | 7 | 45 | 80 / 125 / 125 |
| 5 | 0.3909 | 0.0129 | 505 | 43 | 110 | 120 / 95 / 125 |
| 10 | 0.5044 | 0.0129 | 980 | 57 | 113 | 125 / 95 / 125 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0129 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0129 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0129 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 198 | 1 | 0.0051 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MRVL | 134 | 3 | 0.0224 |
| NASDAQ:TSLA | 125 | 9 | 0.0720 |
| NASDAQ:MU | 116 | 4 | 0.0345 |
| NASDAQ:ON | 75 | 1 | 0.0133 |
| NASDAQ:AMD | 67 | 1 | 0.0149 |
| NASDAQ:MCHP | 62 | 8 | 0.1290 |
| NASDAQ:AVGO | 52 | 3 | 0.0577 |
| NASDAQ:TTD | 38 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:TTD | 38 | 0 | 0.0000 |
| NASDAQ:LULU | 25 | 0 | 0.0000 |
| NASDAQ:CHTR | 23 | 0 | 0.0000 |
| NASDAQ:MDB | 21 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 94 | 6 | 0.0638 |
| NASDAQ:MU | 92 | 10 | 0.1087 |
| NASDAQ:TEAM | 67 | 6 | 0.0896 |
| NASDAQ:INTC | 54 | 13 | 0.2407 |
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:TTD | 26 | 2 | 0.0769 |
| NASDAQ:LRCX | 24 | 1 | 0.0417 |
| NASDAQ:CSGP | 22 | 0 | 0.0000 |
| NASDAQ:AMD | 20 | 2 | 0.1000 |
| NASDAQ:AMAT | 17 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 29 | 0 | 0.0000 |
| NASDAQ:CSGP | 22 | 0 | 0.0000 |
| NASDAQ:AMAT | 17 | 0 | 0.0000 |
| NASDAQ:MDB | 12 | 0 | 0.0000 |
| NASDAQ:KLAC | 7 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.0057 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.0057 | 0.000 |
| 2025Q1 | 109 | 4 | 0.0367 | 0.0057 | 6.431 |
| 2025Q2 | 310 | 20 | 0.0645 | 0.0057 | 11.306 |
| 2025Q3 | 320 | 5 | 0.0156 | 0.0057 | 2.738 |
| 2025Q4 | 310 | 2 | 0.0065 | 0.0057 | 1.131 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q2 | 17 | 0 | 0.0000 | 0.0129 | 0.000 |
| 2025Q3 | 12 | 0 | 0.0000 | 0.0129 | 0.000 |
| 2025Q4 | 11 | 2 | 0.1818 | 0.0129 | 14.063 |
| 2026Q1 | 305 | 7 | 0.0230 | 0.0129 | 1.775 |
| 2026Q2 | 160 | 34 | 0.2125 | 0.0129 | 16.436 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.3333 | 0.0072 | 0.0167 | `True` |
| test | 8740 | 0.0000 | 0.3333 | 0.0082 | 0.0189 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=5.65); shipped as `isotonic`. Brier vs base-rate: +0.0001 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
