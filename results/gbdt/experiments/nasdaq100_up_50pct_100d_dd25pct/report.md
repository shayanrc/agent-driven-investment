# gbdt experiment — nasdaq100_up_50pct_100d_dd25pct

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=100 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
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
- train rows: 73600 (independent events ≈ 369.8; overlap-inflation 199.00×)
- val rows: 36800 (independent events ≈ 184.9; overlap-inflation 199.00×)
- eval rows: 18400 (independent events ≈ 92.5; overlap-inflation 199.00×)
- test rows: 0
- sample uniqueness weighting: `on` (horizon_days=100)
- positive prevalence (train): 0.068
- positive prevalence (eval): 0.113

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0456 | 0.0459 | 0.0002 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 47/27 |  |
| 1 | 47 | 0.0461 | 0.0457 | -0.0004 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 37/47 features |  |
| 2 | 37 | 0.0462 | 0.0455 | -0.0007 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -4.287
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0909 | 0.1002 | +0.0094 | 0.3574 | 0.7888 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.1130

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4150 | 0.1130 | 343 | 83 | 200 | 143 / 343 / 343 |
| 5 | 0.3946 | 0.1130 | 1144 | 393 | 996 | 146 / 200 / 343 |
| 10 | 0.4633 | 0.1130 | 2144 | 826 | 1783 | 242 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1130 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.1130 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.1130 | 10 | 0 | 10 |

### test — n_rows=0, base_rate=n/a

_segment empty — no picks._

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AVGO | 160 | 63 | 0.3937 |
| NASDAQ:INTC | 150 | 108 | 0.7200 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:TSLA | 116 | 10 | 0.0862 |
| NASDAQ:MU | 94 | 94 | 1.0000 |
| NASDAQ:MSTR | 92 | 0 | 0.0000 |
| NASDAQ:MRVL | 82 | 28 | 0.3415 |
| NASDAQ:MDB | 46 | 32 | 0.6957 |
| NASDAQ:AMD | 45 | 41 | 0.9111 |
| NASDAQ:NVDA | 43 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MSTR | 92 | 0 | 0.0000 |
| NASDAQ:NVDA | 43 | 0 | 0.0000 |
| NASDAQ:PDD | 34 | 0 | 0.0000 |
| NASDAQ:DXCM | 31 | 0 | 0.0000 |

### test

_no picks._

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.1130 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.1130 | 0.000 |
| 2025Q1 | 109 | 10 | 0.0917 | 0.1130 | 0.812 |
| 2025Q2 | 310 | 159 | 0.5129 | 0.1130 | 4.539 |
| 2025Q3 | 320 | 130 | 0.4062 | 0.1130 | 3.595 |
| 2025Q4 | 310 | 94 | 0.3032 | 0.1130 | 2.684 |

### test

_no picks._

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.8667 | 0.0765 | 0.1108 | `False` |
| test | 0 | n/a | n/a | n/a | n/a | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=4.29); shipped as `isotonic`. Brier vs base-rate: +0.0094 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
