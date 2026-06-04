# gbdt experiment — nasdaq100_up_50pct_100d_dd25pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `100`
- max_drawdown: `0.25`
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
- positive prevalence (train): 0.085
- positive prevalence (eval): 0.052

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
| 0 | 279 | 0.0497 | 0.0553 | 0.0056 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 59/27 |  |
| 1 | 59 | 0.0526 | 0.0573 | 0.0047 | iteration 1 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 1
- iterations run: 2
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -46.350
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0482 | 0.0496 | +0.0014 | 0.2100 | 0.8048 |
| test | 0.0588 | 0.0635 | +0.0047 | 0.2237 | 0.8700 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.0524

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3636 | 0.0524 | 200 | 68 | 187 | 13 / 200 / 200 |
| 5 | 0.3454 | 0.0524 | 1000 | 220 | 637 | 131 / 200 / 200 |
| 10 | 0.3451 | 0.0524 | 2000 | 284 | 823 | 175 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0524 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.0524 | 5 | 4 | 5 |
| 10 | 0.9000 | 0.0524 | 10 | 9 | 10 |

### test — n_rows=18400, base_rate=0.0682

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3409 | 0.0682 | 200 | 60 | 176 | 24 / 200 / 200 |
| 5 | 0.4858 | 0.0682 | 1000 | 326 | 671 | 84 / 200 / 200 |
| 10 | 0.5534 | 0.0682 | 2000 | 575 | 1039 | 153 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0682 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.0682 | 5 | 2 | 5 |
| 10 | 0.5000 | 0.0682 | 10 | 5 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:WBD | 176 | 1 | 0.0057 |
| NASDAQ:MSTR | 171 | 94 | 0.5497 |
| NASDAQ:TSLA | 151 | 59 | 0.3907 |
| NASDAQ:DDOG | 148 | 20 | 0.1351 |
| NASDAQ:MDB | 118 | 4 | 0.0339 |
| NASDAQ:MRVL | 84 | 26 | 0.3095 |
| NASDAQ:PDD | 43 | 0 | 0.0000 |
| NASDAQ:TEAM | 38 | 0 | 0.0000 |
| NASDAQ:ZS | 37 | 15 | 0.4054 |
| NASDAQ:ON | 24 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:PDD | 43 | 0 | 0.0000 |
| NASDAQ:TEAM | 38 | 0 | 0.0000 |
| NASDAQ:ON | 24 | 0 | 0.0000 |
| NASDAQ:WBD | 176 | 1 | 0.0057 |
| NASDAQ:MDB | 118 | 4 | 0.0339 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MDB | 183 | 27 | 0.1475 |
| NASDAQ:MRVL | 158 | 71 | 0.4494 |
| NASDAQ:MSTR | 158 | 87 | 0.5506 |
| NASDAQ:AMD | 80 | 0 | 0.0000 |
| NASDAQ:ON | 68 | 0 | 0.0000 |
| NASDAQ:TSLA | 59 | 50 | 0.8475 |
| NASDAQ:WBD | 56 | 27 | 0.4821 |
| NASDAQ:INTC | 53 | 0 | 0.0000 |
| NASDAQ:NVDA | 48 | 25 | 0.5208 |
| NASDAQ:MU | 47 | 13 | 0.2766 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AMD | 80 | 0 | 0.0000 |
| NASDAQ:ON | 68 | 0 | 0.0000 |
| NASDAQ:INTC | 53 | 0 | 0.0000 |
| NASDAQ:DXCM | 22 | 0 | 0.0000 |
| NASDAQ:PDD | 13 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 94 | 0.3241 | 0.0524 |
| 2024Q1 | 305 | 46 | 0.1508 | 0.0524 |
| 2024Q2 | 315 | 60 | 0.1905 | 0.0524 |
| 2024Q3 | 90 | 20 | 0.2222 | 0.0524 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 137 | 0.5957 | 0.0682 |
| 2024Q4 | 320 | 84 | 0.2625 | 0.0682 |
| 2025Q1 | 300 | 31 | 0.1033 | 0.0682 |
| 2025Q2 | 150 | 74 | 0.4933 | 0.0682 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.2578 | 0.0209 | 0.0490 | `True` |
| test | 18400 | 0.0000 | 0.6875 | 0.0192 | 0.0426 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=46.35); shipped as `isotonic`. Brier vs base-rate: +0.0014 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
