# gbdt experiment — nasdaq100_up_20pct_100d_dd10pct

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=100 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `100`
- max_drawdown: `0.1`
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
- positive prevalence (train): 0.305
- positive prevalence (eval): 0.321

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1945 | 0.1993 | 0.0048 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 24/27 |  |
| 1 | 24 | 0.1945 | 0.1990 | 0.0045 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 19/24 features |  |
| 2 | 19 | 0.1946 | 0.1985 | 0.0040 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -8.038
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2125 | 0.2180 | +0.0055 | 0.6129 | 0.6248 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.3210

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.6450 | 0.3210 | 343 | 129 | 200 | 143 / 343 / 343 |
| 5 | 0.4356 | 0.3210 | 1144 | 433 | 994 | 147 / 200 / 343 |
| 10 | 0.4399 | 0.3210 | 2144 | 863 | 1962 | 152 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.3210 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.3210 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.3210 | 10 | 10 | 10 |

### test — n_rows=0, base_rate=n/a

_segment empty — no picks._

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MCHP | 123 | 45 | 0.3659 |
| NASDAQ:MSTR | 101 | 0 | 0.0000 |
| NASDAQ:ADI | 88 | 58 | 0.6591 |
| NASDAQ:MRVL | 86 | 42 | 0.4884 |
| NASDAQ:AAPL | 78 | 73 | 0.9359 |
| NASDAQ:AVGO | 60 | 33 | 0.5500 |
| NASDAQ:MU | 41 | 33 | 0.8049 |
| NASDAQ:TTD | 41 | 11 | 0.2683 |
| NASDAQ:FAST | 35 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MSTR | 101 | 0 | 0.0000 |
| NASDAQ:FAST | 35 | 0 | 0.0000 |
| NASDAQ:ADBE | 34 | 0 | 0.0000 |
| NASDAQ:NXPI | 26 | 0 | 0.0000 |

### test

_no picks._

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.3210 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.3210 | 0.000 |
| 2025Q1 | 109 | 0 | 0.0000 | 0.3210 | 0.000 |
| 2025Q2 | 310 | 207 | 0.6677 | 0.3210 | 2.080 |
| 2025Q3 | 320 | 113 | 0.3531 | 0.3210 | 1.100 |
| 2025Q4 | 310 | 113 | 0.3645 | 0.3210 | 1.135 |

### test

_no picks._

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.6744 | 0.3551 | 0.1057 | `False` |
| test | 0 | n/a | n/a | n/a | n/a | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=8.04); shipped as `isotonic`. Brier vs base-rate: +0.0055 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
