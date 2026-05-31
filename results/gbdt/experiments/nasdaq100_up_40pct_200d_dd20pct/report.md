# gbdt experiment — nasdaq100_up_40pct_200d_dd20pct

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=200 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `200`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73600 (independent events ≈ 184.5; overlap-inflation 399.00×)
- val rows: 36800 (independent events ≈ 92.2; overlap-inflation 399.00×)
- eval rows: 9200 (independent events ≈ 23.1; overlap-inflation 399.00×)
- test rows: 0
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.256
- positive prevalence (eval): 0.304

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1530 | 0.1447 | -0.0083 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 25/27 |  |
| 1 | 25 | 0.1582 | 0.1453 | -0.0129 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 18/25 features |  |
| 2 | 18 | 0.1545 | 0.1445 | -0.0101 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -9.384
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1959 | 0.2115 | +0.0156 | 0.5865 | 0.6936 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=9200, base_rate=0.3037

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.8800 | 0.3037 | 201 | 88 | 100 | 101 / 201 / 201 |
| 5 | 0.6760 | 0.3037 | 601 | 338 | 500 | 101 / 100 / 201 |
| 10 | 0.6570 | 0.3037 | 1101 | 657 | 1000 | 101 / 100 / 201 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.3037 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.3037 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.3037 | 10 | 10 | 10 |

### test — n_rows=0, base_rate=n/a

_segment empty — no picks._

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 100 | 0 | 0.0000 |
| NASDAQ:INTC | 100 | 88 | 0.8800 |
| NASDAQ:MSTR | 83 | 11 | 0.1325 |
| NASDAQ:AVGO | 60 | 52 | 0.8667 |
| NASDAQ:MDB | 34 | 34 | 1.0000 |
| NASDAQ:MRVL | 33 | 30 | 0.9091 |
| NASDAQ:MU | 26 | 26 | 1.0000 |
| NASDAQ:NVDA | 25 | 25 | 1.0000 |
| NASDAQ:PDD | 24 | 10 | 0.4167 |
| NASDAQ:DXCM | 22 | 4 | 0.1818 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 100 | 0 | 0.0000 |
| NASDAQ:TEAM | 22 | 0 | 0.0000 |
| NASDAQ:MSTR | 83 | 11 | 0.1325 |
| NASDAQ:DXCM | 22 | 4 | 0.1818 |
| NASDAQ:PDD | 24 | 10 | 0.4167 |

### test

_no picks._

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.3037 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.3037 | 0.000 |
| 2025Q1 | 66 | 26 | 0.3939 | 0.3037 | 1.297 |
| 2025Q2 | 310 | 219 | 0.7065 | 0.3037 | 2.326 |
| 2025Q3 | 130 | 93 | 0.7154 | 0.3037 | 2.356 |

### test

_no picks._

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 9200 | 0.0000 | 1.0000 | 0.3508 | 0.1232 | `False` |
| test | 0 | n/a | n/a | n/a | n/a | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=9.38); shipped as `isotonic`. Brier vs base-rate: +0.0156 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
