# gbdt experiment — nasdaq100_up_40pct_100d_dd20pct

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=100 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
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
- train rows: 73600 (independent events ≈ 369.8; overlap-inflation 199.00×)
- val rows: 36800 (independent events ≈ 184.9; overlap-inflation 199.00×)
- eval rows: 18400 (independent events ≈ 92.5; overlap-inflation 199.00×)
- test rows: 0
- sample uniqueness weighting: `on` (horizon_days=100)
- positive prevalence (train): 0.109
- positive prevalence (eval): 0.158

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0725 | 0.0726 | 0.0001 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 54/27 |  |
| 1 | 54 | 0.0708 | 0.0724 | 0.0016 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 46/54 features |  |
| 2 | 46 | 0.0720 | 0.0723 | 0.0003 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -2.797
- Spiegelhalter p: 0.0052

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1177 | 0.1327 | +0.0150 | 0.3748 | 0.7928 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=18400, base_rate=0.1575

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5800 | 0.1575 | 343 | 116 | 200 | 143 / 343 / 343 |
| 5 | 0.4689 | 0.1575 | 1144 | 467 | 996 | 145 / 200 / 343 |
| 10 | 0.4741 | 0.1575 | 2144 | 935 | 1972 | 153 / 200 / 343 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1575 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1575 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.1575 | 10 | 10 | 10 |

### test — n_rows=0, base_rate=n/a

_segment empty — no picks._

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 175 | 16 | 0.0914 |
| NASDAQ:INTC | 173 | 138 | 0.7977 |
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:MRVL | 130 | 47 | 0.3615 |
| NASDAQ:AVGO | 126 | 63 | 0.5000 |
| NASDAQ:MCHP | 71 | 22 | 0.3099 |
| NASDAQ:AMD | 51 | 51 | 1.0000 |
| NASDAQ:MU | 42 | 41 | 0.9762 |
| NASDAQ:MDB | 40 | 40 | 1.0000 |
| NASDAQ:TTD | 36 | 10 | 0.2778 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 143 | 0 | 0.0000 |
| NASDAQ:LULU | 18 | 0 | 0.0000 |
| NASDAQ:TEAM | 12 | 0 | 0.0000 |
| NASDAQ:PDD | 32 | 2 | 0.0625 |
| NASDAQ:TSLA | 22 | 2 | 0.0909 |

### test

_no picks._

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.1575 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.1575 | 0.000 |
| 2025Q1 | 109 | 11 | 0.1009 | 0.1575 | 0.641 |
| 2025Q2 | 310 | 199 | 0.6419 | 0.1575 | 4.076 |
| 2025Q3 | 320 | 143 | 0.4469 | 0.1575 | 2.837 |
| 2025Q4 | 310 | 114 | 0.3677 | 0.1575 | 2.335 |

### test

_no picks._

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 18400 | 0.0000 | 0.4318 | 0.1212 | 0.1147 | `False` |
| test | 0 | n/a | n/a | n/a | n/a | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=2.80); shipped as `isotonic`. Brier vs base-rate: +0.0150 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
