# gbdt experiment — nasdaq100_up_10pct_200d_dd5pct

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=200 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
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
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73600 (independent events ≈ 184.5; overlap-inflation 399.00×)
- val rows: 36800 (independent events ≈ 92.2; overlap-inflation 399.00×)
- eval rows: 9200 (independent events ≈ 23.1; overlap-inflation 399.00×)
- test rows: 0
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.429
- positive prevalence (eval): 0.430

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.2302 | 0.2463 | 0.0161 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 22/27 |  |
| 1 | 22 | 0.2356 | 0.2461 | 0.0104 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 17/22 features |  |
| 2 | 17 | 0.2307 | 0.2457 | 0.0150 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -3.035
- Spiegelhalter p: 0.0024

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2477 | 0.2451 | -0.0026 | 0.6885 | 0.5097 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=9200, base_rate=0.4299

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5685 | 0.4299 | 201 | 83 | 146 | 55 / 201 / 201 |
| 5 | 0.5131 | 0.4299 | 601 | 274 | 534 | 106 / 100 / 201 |
| 10 | 0.4975 | 0.4299 | 1101 | 500 | 1005 | 108 / 100 / 201 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.4299 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.4299 | 5 | 4 | 5 |
| 10 | 0.7000 | 0.4299 | 10 | 7 | 10 |

### test — n_rows=0, base_rate=n/a

_segment empty — no picks._

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 100 | 46 | 0.4600 |
| NASDAQ:ADSK | 59 | 22 | 0.3729 |
| NASDAQ:ADP | 54 | 10 | 0.1852 |
| NASDAQ:AAPL | 43 | 17 | 0.3953 |
| NASDAQ:ADBE | 42 | 12 | 0.2857 |
| NASDAQ:ADI | 40 | 14 | 0.3500 |
| NASDAQ:CTAS | 32 | 1 | 0.0312 |
| NASDAQ:AMD | 20 | 16 | 0.8000 |
| NASDAQ:INTC | 18 | 13 | 0.7222 |
| NASDAQ:BIIB | 16 | 14 | 0.8750 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:DDOG | 10 | 0 | 0.0000 |
| NASDAQ:CTSH | 6 | 0 | 0.0000 |
| NASDAQ:CTAS | 32 | 1 | 0.0312 |
| NASDAQ:ADP | 54 | 10 | 0.1852 |
| NASDAQ:MU | 5 | 1 | 0.2000 |

### test

_no picks._

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 24 | 0.7742 | 0.4299 | 1.801 |
| 2024Q4 | 64 | 22 | 0.3438 | 0.4299 | 0.800 |
| 2025Q1 | 66 | 0 | 0.0000 | 0.4299 | 0.000 |
| 2025Q2 | 310 | 163 | 0.5258 | 0.4299 | 1.223 |
| 2025Q3 | 130 | 65 | 0.5000 | 0.4299 | 1.163 |

### test

_no picks._

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 9200 | 0.2804 | 0.5833 | 0.4547 | 0.0529 | `False` |
| test | 0 | n/a | n/a | n/a | n/a | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=3.04); shipped as `isotonic`. Brier vs base-rate: -0.0026 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
