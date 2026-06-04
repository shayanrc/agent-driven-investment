# gbdt experiment — nasdaq100_up_10pct_200d_dd5pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `200`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 100
- tickers used: 89
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:CRWD, NASDAQ:DASH, NASDAQ:DDOG, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PDD, NASDAQ:PLTR
- train rows: 71149 (independent events ≈ 178.6; overlap-inflation 398.36×)
- val rows: 35600 (independent events ≈ 89.2; overlap-inflation 399.00×)
- eval rows: 17800 (independent events ≈ 44.6; overlap-inflation 399.00×)
- test rows: 26700 (independent events ≈ 66.9; overlap-inflation 399.00×)
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.466
- positive prevalence (eval): 0.498

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
| 0 | 279 | 0.2471 | 0.2512 | 0.0040 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 10/27 |  |
| 1 | 10 | 0.2470 | 0.2522 | 0.0052 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 10/10 features |  |
| 2 | 10 | 0.2470 | 0.2522 | 0.0052 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 21.525
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2678 | 0.2500 | -0.0178 | 0.7301 | 0.4987 |
| test | 0.2509 | 0.2459 | -0.0051 | 0.6955 | 0.5000 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=17800, base_rate=0.4978

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4650 | 0.4978 | 200 | 93 | 200 | 0 / 200 / 200 |
| 5 | 0.5221 | 0.4978 | 1000 | 520 | 996 | 2 / 200 / 200 |
| 10 | 0.4748 | 0.4978 | 2000 | 941 | 1982 | 3 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.4978 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.4978 | 5 | 4 | 5 |
| 10 | 0.4000 | 0.4978 | 10 | 4 | 10 |

### test — n_rows=26700, base_rate=0.4357

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4900 | 0.4357 | 300 | 147 | 300 | 0 / 300 / 300 |
| 5 | 0.4180 | 0.4357 | 1500 | 627 | 1500 | 0 / 300 / 300 |
| 10 | 0.4502 | 0.4357 | 3000 | 1350 | 2999 | 1 / 300 / 300 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.4357 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.4357 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.4357 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AAPL | 200 | 108 | 0.5400 |
| NASDAQ:ADBE | 200 | 124 | 0.6200 |
| NASDAQ:ADI | 198 | 91 | 0.4596 |
| NASDAQ:ADP | 196 | 111 | 0.5663 |
| NASDAQ:ADSK | 175 | 82 | 0.4686 |
| NASDAQ:EXC | 25 | 0 | 0.0000 |
| NASDAQ:CCEP | 4 | 4 | 1.0000 |
| NASDAQ:TMUS | 2 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:EXC | 25 | 0 | 0.0000 |
| NASDAQ:ADI | 198 | 91 | 0.4596 |
| NASDAQ:ADSK | 175 | 82 | 0.4686 |
| NASDAQ:AAPL | 200 | 108 | 0.5400 |
| NASDAQ:ADP | 196 | 111 | 0.5663 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AAPL | 300 | 147 | 0.4900 |
| NASDAQ:ADBE | 300 | 70 | 0.2333 |
| NASDAQ:ADI | 300 | 108 | 0.3600 |
| NASDAQ:ADP | 300 | 180 | 0.6000 |
| NASDAQ:ADSK | 300 | 122 | 0.4067 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ADBE | 300 | 70 | 0.2333 |
| NASDAQ:ADI | 300 | 108 | 0.3600 |
| NASDAQ:ADSK | 300 | 122 | 0.4067 |
| NASDAQ:AAPL | 300 | 147 | 0.4900 |
| NASDAQ:ADP | 300 | 180 | 0.6000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 295 | 141 | 0.4780 | 0.4978 |
| 2023Q1 | 310 | 142 | 0.4581 | 0.4978 |
| 2023Q2 | 310 | 208 | 0.6710 | 0.4978 |
| 2023Q3 | 85 | 29 | 0.3412 | 0.4978 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 230 | 51 | 0.2217 | 0.4357 |
| 2023Q4 | 315 | 153 | 0.4857 | 0.4357 |
| 2024Q1 | 305 | 96 | 0.3148 | 0.4357 |
| 2024Q2 | 315 | 157 | 0.4984 | 0.4357 |
| 2024Q3 | 320 | 161 | 0.5031 | 0.4357 |
| 2024Q4 | 15 | 9 | 0.6000 | 0.4357 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 17800 | 0.3646 | 0.4000 | 0.3647 | 0.0015 | `True` |
| test | 26700 | 0.3646 | 0.3646 | 0.3646 | 0.0000 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=21.53); shipped as `isotonic`. Brier vs base-rate: -0.0178 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
