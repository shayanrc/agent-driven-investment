# gbdt experiment — nasdaq100_up_40pct_100d_dd20pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `100`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73309 (independent events ≈ 368.8; overlap-inflation 198.79×)
- val rows: 36800 (independent events ≈ 184.9; overlap-inflation 199.00×)
- eval rows: 18400 (independent events ≈ 92.5; overlap-inflation 199.00×)
- test rows: 18400 (independent events ≈ 92.5; overlap-inflation 199.00×)
- sample uniqueness weighting: `on` (horizon_days=100)
- positive prevalence (train): 0.131
- positive prevalence (eval): 0.089

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2019-01-01`
- train: `2019-01-02` → `2022-03-04`
- val: `2022-03-07` → `2023-10-06`
- eval: `2023-10-09` → `2024-07-25`
- test: `2024-07-26` → `2025-05-13`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0724 | 0.0956 | 0.0232 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 62/27 |  |
| 1 | 62 | 0.0733 | 0.0981 | 0.0248 | iteration 1 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 0
- iterations run: 2
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -27.368
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0786 | 0.0812 | +0.0027 | 0.2925 | 0.7919 |
| test | 0.0771 | 0.0876 | +0.0105 | 0.2709 | 0.8380 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0892

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2800 | 0.0892 | 200 | 56 | 200 | 0 / 200 / 200 |
| 5 | 0.2907 | 0.0892 | 1000 | 250 | 860 | 65 / 200 / 200 |
| 10 | 0.3576 | 0.0892 | 2000 | 438 | 1225 | 150 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0892 | 1 | 1 | 1 |
| 5 | 0.6000 | 0.0892 | 5 | 3 | 5 |
| 10 | 0.5000 | 0.0892 | 10 | 5 | 10 |

### test — n_rows=18400, base_rate=0.0971

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2280 | 0.0971 | 200 | 44 | 193 | 7 / 200 / 200 |
| 5 | 0.4732 | 0.0971 | 1000 | 353 | 746 | 77 / 200 / 200 |
| 10 | 0.5135 | 0.0971 | 2000 | 664 | 1293 | 107 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0971 | 1 | 0 | 1 |
| 5 | 0.6000 | 0.0971 | 5 | 3 | 5 |
| 10 | 0.7000 | 0.0971 | 10 | 7 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:WBD | 176 | 11 | 0.0625 |
| NASDAQ:MSTR | 175 | 88 | 0.5029 |
| NASDAQ:DDOG | 163 | 21 | 0.1288 |
| NASDAQ:MDB | 136 | 15 | 0.1103 |
| NASDAQ:TSLA | 112 | 35 | 0.3125 |
| NASDAQ:TEAM | 90 | 14 | 0.1556 |
| NASDAQ:MRVL | 78 | 61 | 0.7821 |
| NASDAQ:TTD | 48 | 2 | 0.0417 |
| NASDAQ:ZS | 10 | 0 | 0.0000 |
| NASDAQ:DXCM | 4 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ZS | 10 | 0 | 0.0000 |
| NASDAQ:TTD | 48 | 2 | 0.0417 |
| NASDAQ:WBD | 176 | 11 | 0.0625 |
| NASDAQ:MDB | 136 | 15 | 0.1103 |
| NASDAQ:DDOG | 163 | 21 | 0.1288 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MDB | 187 | 46 | 0.2460 |
| NASDAQ:MRVL | 145 | 78 | 0.5379 |
| NASDAQ:AMD | 141 | 1 | 0.0071 |
| NASDAQ:MSTR | 118 | 59 | 0.5000 |
| NASDAQ:TSLA | 86 | 60 | 0.6977 |
| NASDAQ:MU | 69 | 21 | 0.3043 |
| NASDAQ:CRWD | 67 | 28 | 0.4179 |
| NASDAQ:NVDA | 62 | 22 | 0.3548 |
| NASDAQ:INTC | 45 | 11 | 0.2444 |
| NASDAQ:WBD | 25 | 22 | 0.8800 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ON | 21 | 0 | 0.0000 |
| NASDAQ:TEAM | 7 | 0 | 0.0000 |
| NASDAQ:AMD | 141 | 1 | 0.0071 |
| NASDAQ:PDD | 20 | 2 | 0.1000 |
| NASDAQ:INTC | 45 | 11 | 0.2444 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 127 | 0.4379 | 0.0892 |
| 2024Q1 | 305 | 55 | 0.1803 | 0.0892 |
| 2024Q2 | 315 | 42 | 0.1333 | 0.0892 |
| 2024Q3 | 90 | 26 | 0.2889 | 0.0892 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 139 | 0.6043 | 0.0971 |
| 2024Q4 | 320 | 103 | 0.3219 | 0.0971 |
| 2025Q1 | 300 | 24 | 0.0800 | 0.0971 |
| 2025Q2 | 150 | 87 | 0.5800 | 0.0971 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.3064 | 0.0403 | 0.0561 | `False` |
| test | 18400 | 0.0000 | 1.0000 | 0.0555 | 0.0629 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=27.37); shipped as `isotonic`. Brier vs base-rate: +0.0027 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
