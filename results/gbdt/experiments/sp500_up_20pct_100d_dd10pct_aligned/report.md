# gbdt experiment — sp500_up_20pct_100d_dd10pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `sp500`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `100`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 503
- tickers used: 486
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:PLTR, NYSE:CARR, NYSE:COIN, NYSE:EXE, NYSE:GEV, NYSE:HOOD, NYSE:KVUE, NYSE:OTIS, NYSE:Q, NYSE:SNDK, NYSE:SOLV, NYSE:VLTO
- train rows: 388173 (independent events ≈ 1952.1; overlap-inflation 198.85×)
- val rows: 194400 (independent events ≈ 976.9; overlap-inflation 199.00×)
- eval rows: 97200 (independent events ≈ 488.4; overlap-inflation 199.00×)
- test rows: 97200 (independent events ≈ 488.4; overlap-inflation 199.00×)
- sample uniqueness weighting: `on` (horizon_days=100)
- positive prevalence (train): 0.294
- positive prevalence (eval): 0.347

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
| 0 | 279 | 0.1728 | 0.1991 | 0.0263 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 37/27 |  |
| 1 | 37 | 0.1757 | 0.2016 | 0.0259 | iteration 1 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 1
- iterations run: 2
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -58.453
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2851 | 0.2265 | -0.0585 | 1.0287 | 0.6199 |
| test | 0.1632 | 0.1676 | +0.0044 | 0.5123 | 0.6855 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=97200, base_rate=0.3469

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3700 | 0.3469 | 200 | 74 | 200 | 0 / 200 / 200 |
| 5 | 0.3610 | 0.3469 | 1000 | 361 | 1000 | 0 / 200 / 200 |
| 10 | 0.4255 | 0.3469 | 2000 | 851 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.3469 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.3469 | 5 | 2 | 5 |
| 10 | 0.5000 | 0.3469 | 10 | 5 | 10 |

### test — n_rows=97200, base_rate=0.2129

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3800 | 0.2129 | 200 | 76 | 200 | 0 / 200 / 200 |
| 5 | 0.4470 | 0.2129 | 1000 | 447 | 1000 | 0 / 200 / 200 |
| 10 | 0.4090 | 0.2129 | 2000 | 818 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.2129 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.2129 | 5 | 2 | 5 |
| 10 | 0.7000 | 0.2129 | 10 | 7 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TSLA | 159 | 59 | 0.3711 |
| NASDAQ:WBD | 143 | 11 | 0.0769 |
| NASDAQ:DDOG | 99 | 29 | 0.2929 |
| NYSE:CVNA | 84 | 54 | 0.6429 |
| NYSE:ALB | 82 | 3 | 0.0366 |
| NASDAQ:PANW | 74 | 52 | 0.7027 |
| NYSE:NCLH | 70 | 20 | 0.2857 |
| NYSE:COHR | 57 | 32 | 0.5614 |
| NASDAQ:ON | 55 | 21 | 0.3818 |
| NYSE:ALGN | 35 | 23 | 0.6571 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:DELL | 14 | 0 | 0.0000 |
| NYSE:MRNA | 14 | 0 | 0.0000 |
| NYSE:PSKY | 10 | 0 | 0.0000 |
| NYSE:ALB | 82 | 3 | 0.0366 |
| NASDAQ:WBD | 143 | 11 | 0.0769 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AVGO | 149 | 60 | 0.4027 |
| NASDAQ:CRWD | 115 | 75 | 0.6522 |
| NASDAQ:DXCM | 104 | 33 | 0.3173 |
| NASDAQ:TSLA | 91 | 48 | 0.5275 |
| NASDAQ:AMD | 87 | 17 | 0.1954 |
| NASDAQ:INTC | 81 | 33 | 0.4074 |
| NYSE:VRT | 81 | 50 | 0.6173 |
| NYSE:ALB | 55 | 22 | 0.4000 |
| NASDAQ:MU | 40 | 16 | 0.4000 |
| NYSE:CVNA | 35 | 27 | 0.7714 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:PSKY | 21 | 0 | 0.0000 |
| NASDAQ:CTAS | 9 | 0 | 0.0000 |
| NASDAQ:NVDA | 9 | 0 | 0.0000 |
| NASDAQ:LRCX | 7 | 0 | 0.0000 |
| NYSE:SMCI | 28 | 4 | 0.1429 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 135 | 0.4655 | 0.3469 |
| 2024Q1 | 305 | 62 | 0.2033 | 0.3469 |
| 2024Q2 | 315 | 149 | 0.4730 | 0.3469 |
| 2024Q3 | 90 | 15 | 0.1667 | 0.3469 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 135 | 0.5870 | 0.2129 |
| 2024Q4 | 320 | 164 | 0.5125 | 0.2129 |
| 2025Q1 | 300 | 45 | 0.1500 | 0.2129 |
| 2025Q2 | 150 | 103 | 0.6867 | 0.2129 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 97200 | 0.0000 | 0.2848 | 0.0926 | 0.0492 | `True` |
| test | 97200 | 0.0000 | 0.4558 | 0.1272 | 0.0640 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=58.45); shipped as `isotonic`. Brier vs base-rate: -0.0585 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
