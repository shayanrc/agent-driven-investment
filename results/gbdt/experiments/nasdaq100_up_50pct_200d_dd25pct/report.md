# gbdt experiment — nasdaq100_up_50pct_200d_dd25pct

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=200 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `200`
- max_drawdown: `0.25`
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
- positive prevalence (train): 0.176
- positive prevalence (eval): 0.240

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1114 | 0.1070 | -0.0045 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 29/27 |  |
| 1 | 29 | 0.1141 | 0.1084 | -0.0057 | iteration 1 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 1
- iterations run: 2
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -8.466
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1588 | 0.1826 | +0.0237 | 0.4924 | 0.7607 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=9200, base_rate=0.2403

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.9300 | 0.2403 | 201 | 93 | 100 | 101 / 201 / 201 |
| 5 | 0.6680 | 0.2403 | 601 | 334 | 500 | 101 / 100 / 201 |
| 10 | 0.6280 | 0.2403 | 1101 | 628 | 1000 | 101 / 100 / 201 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.2403 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.2403 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.2403 | 10 | 10 | 10 |

### test — n_rows=0, base_rate=n/a

_segment empty — no picks._

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 100 | 0 | 0.0000 |
| NASDAQ:AVGO | 100 | 87 | 0.8700 |
| NASDAQ:INTC | 100 | 95 | 0.9500 |
| NASDAQ:AMD | 90 | 86 | 0.9556 |
| NASDAQ:DXCM | 42 | 0 | 0.0000 |
| NASDAQ:CRWD | 40 | 1 | 0.0250 |
| NASDAQ:TTD | 32 | 0 | 0.0000 |
| NASDAQ:AMAT | 24 | 24 | 1.0000 |
| NASDAQ:LRCX | 20 | 20 | 1.0000 |
| NASDAQ:MSTR | 15 | 7 | 0.4667 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:ANSS | 100 | 0 | 0.0000 |
| NASDAQ:DXCM | 42 | 0 | 0.0000 |
| NASDAQ:TTD | 32 | 0 | 0.0000 |
| NASDAQ:LULU | 8 | 0 | 0.0000 |
| NASDAQ:PDD | 8 | 0 | 0.0000 |

### test

_no picks._

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2024Q3 | 31 | 0 | 0.0000 | 0.2403 | 0.000 |
| 2024Q4 | 64 | 0 | 0.0000 | 0.2403 | 0.000 |
| 2025Q1 | 66 | 26 | 0.3939 | 0.2403 | 1.639 |
| 2025Q2 | 310 | 240 | 0.7742 | 0.2403 | 3.221 |
| 2025Q3 | 130 | 68 | 0.5231 | 0.2403 | 2.177 |

### test

_no picks._

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 9200 | 0.0000 | 1.0000 | 0.2691 | 0.1338 | `False` |
| test | 0 | n/a | n/a | n/a | n/a | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=8.47); shipped as `isotonic`. Brier vs base-rate: +0.0237 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
