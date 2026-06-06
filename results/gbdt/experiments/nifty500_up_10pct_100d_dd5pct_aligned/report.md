# gbdt experiment — nifty500_up_10pct_100d_dd5pct_aligned

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=100 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `100`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 500
- tickers used: 376
- tickers excluded: NSE:AADHARHFC, NSE:ABDL, NSE:ABLBL, NSE:ABSLAMC, NSE:ACMESOLAR, NSE:ACUTAAS, NSE:AEGISVOPAK, NSE:AFCONS, NSE:AIIL, NSE:ANANDRATHI, NSE:ANGELONE, NSE:ANTHEM, NSE:ANURAS, NSE:APTUS, NSE:ATHERENERG, NSE:AWL, NSE:BAJAJHFL, NSE:BELRISE, NSE:BHARTIHEXA, NSE:BIKAJI, NSE:BLUEJET, NSE:CAMS, NSE:CANHLIFE, NSE:CARTRADE, NSE:CHOICEIN, NSE:CLEAN, NSE:COHANCE, NSE:CONCORDBIO, NSE:CPPLUS, NSE:CRAFTSMAN, NSE:DATAPATTNS, NSE:DELHIVERY, NSE:DEVYANI, NSE:DOMS, NSE:EMCURE, NSE:EMMVEE, NSE:ENRIN, NSE:ETERNAL, NSE:FIRSTCRY, NSE:FIVESTAR, NSE:GLAND, NSE:GODIGIT, NSE:GROWW, NSE:HDBFS, NSE:HEXT, NSE:HOMEFIRST, NSE:HONASA, NSE:HYUNDAI, NSE:ICICIAMC, NSE:IGIL, NSE:IKS, NSE:INDGN, NSE:ITCHOTELS, NSE:JAINREC, NSE:JIOFIN, NSE:JSWCEMENT, NSE:JSWINFRA, NSE:JUBLINGREA, NSE:JYOTICNC, NSE:KALYANKJIL, NSE:KAYNES, NSE:KFINTECH, NSE:KIMS, NSE:LATENTVIEW, NSE:LENSKART, NSE:LGEINDIA, NSE:LICI, NSE:LLOYDSME, NSE:LODHA, NSE:MANKIND, NSE:MAPMYINDIA, NSE:MAXHEALTH, NSE:MAZDOCK, NSE:MEDANTA, NSE:MEESHO, NSE:MSUMI, NSE:NETWEB, NSE:NIVABUPA, NSE:NSLNISP, NSE:NTPCGREEN, NSE:NUVAMA, NSE:NUVOCO, NSE:NYKAA, NSE:OLAELEC, NSE:ONESOURCE, NSE:PARADEEP, NSE:PAYTM, NSE:PINELABS, NSE:PIRAMALFIN, NSE:POLICYBZR, NSE:POWERINDIA, NSE:PPLPHARMA, NSE:PREMIERENE, NSE:PTCIL, NSE:PWL, NSE:RAILTEL, NSE:RAINBOW, NSE:RRKABEL, NSE:SAGILITY, NSE:SAILIFE, NSE:SAPPHIRE, NSE:SBFC, NSE:SBICARD, NSE:SHRIRAMFIN, NSE:SHYAMMETL, NSE:SIGNATURE, NSE:SONACOMS, NSE:STARHEALTH, NSE:SUMICHEM, NSE:SWIGGY, NSE:SYRMA, NSE:TATACAP, NSE:TATATECH, NSE:TBOTEK, NSE:TEGA, NSE:TENNIND, NSE:THELEELA, NSE:TMCV, NSE:TRAVELFOOD, NSE:URBANCO, NSE:UTIAMC, NSE:VIJAYA, NSE:VMM, NSE:WAAREEENER
- train rows: 300996 (independent events ≈ 1514.8; overlap-inflation 198.71×)
- val rows: 150755 (independent events ≈ 757.6; overlap-inflation 199.00×)
- eval rows: 75907 (independent events ≈ 381.4; overlap-inflation 199.00×)
- test rows: 37600 (independent events ≈ 188.9; overlap-inflation 199.00×)
- sample uniqueness weighting: `on` (horizon_days=100)
- positive prevalence (train): 0.436
- positive prevalence (eval): 0.491

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2019-01-01`
- train: `2019-01-01` → `2022-03-29`
- val: `2022-03-30` → `2023-11-09`
- eval: `2023-11-10` → `2024-09-03`
- test: `2024-09-04` → `2025-01-27`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1539 | 0.3056 | 0.1516 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 191/2 |  |
| 1 | 191 | 0.1548 | 0.3047 | 0.1498 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 186/191 features;  |  |
| 2 | 186 | 0.1548 | 0.3047 | 0.1498 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 246.159
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2556 | 0.2499 | -0.0056 | 0.7084 | 0.5036 |
| test | 0.2416 | 0.1676 | -0.0740 | 0.6763 | 0.5081 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.4911

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5099 | 0.4911 | 202 | 103 | 202 | 0 / 202 / 202 |
| 5 | 0.4248 | 0.4911 | 1010 | 429 | 1010 | 0 / 202 / 202 |
| 10 | 0.4441 | 0.4911 | 2020 | 897 | 2020 | 0 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.4911 | 1 | 0 | 1 |
| 5 | 0.6000 | 0.4911 | 5 | 3 | 5 |
| 10 | 0.6000 | 0.4911 | 10 | 6 | 10 |

### test — n_rows=37600, base_rate=0.2129

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1700 | 0.2129 | 100 | 17 | 100 | 0 / 100 / 100 |
| 5 | 0.1980 | 0.2129 | 500 | 99 | 500 | 0 / 100 / 100 |
| 10 | 0.1970 | 0.2129 | 1000 | 197 | 1000 | 0 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.2129 | 1 | 1 | 1 |
| 5 | 0.2000 | 0.2129 | 5 | 1 | 5 |
| 10 | 0.1000 | 0.2129 | 10 | 1 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:360ONE | 118 | 72 | 0.6102 |
| NSE:ADANIENSOL | 85 | 33 | 0.3882 |
| NSE:ACE | 75 | 37 | 0.4933 |
| NSE:AAVAS | 62 | 17 | 0.2742 |
| NSE:3MINDIA | 60 | 17 | 0.2833 |
| NSE:AARTIIND | 56 | 21 | 0.3750 |
| NSE:ABB | 56 | 14 | 0.2500 |
| NSE:ADANIGREEN | 50 | 24 | 0.4800 |
| NSE:ABREL | 43 | 14 | 0.3256 |
| NSE:ABBOTINDIA | 36 | 20 | 0.5556 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:AFFLE | 11 | 0 | 0.0000 |
| NSE:HAL | 6 | 0 | 0.0000 |
| NSE:IRCON | 6 | 0 | 0.0000 |
| NSE:BHEL | 11 | 1 | 0.0909 |
| NSE:ABFRL | 5 | 1 | 0.2000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:360ONE | 45 | 15 | 0.3333 |
| NSE:3MINDIA | 31 | 0 | 0.0000 |
| NSE:MOTHERSON | 26 | 0 | 0.0000 |
| NSE:AARTIIND | 23 | 2 | 0.0870 |
| NSE:SUZLON | 20 | 0 | 0.0000 |
| NSE:AAVAS | 18 | 13 | 0.7222 |
| NSE:OIL | 18 | 4 | 0.2222 |
| NSE:GALLANTT | 15 | 3 | 0.2000 |
| NSE:GAIL | 14 | 4 | 0.2857 |
| NSE:CEMPRO | 13 | 5 | 0.3846 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:3MINDIA | 31 | 0 | 0.0000 |
| NSE:MOTHERSON | 26 | 0 | 0.0000 |
| NSE:SUZLON | 20 | 0 | 0.0000 |
| NSE:MMTC | 11 | 0 | 0.0000 |
| NSE:ABB | 9 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 104 | 0.6118 | 0.4911 |
| 2024Q1 | 310 | 141 | 0.4548 | 0.4911 |
| 2024Q2 | 305 | 131 | 0.4295 | 0.4911 |
| 2024Q3 | 225 | 53 | 0.2356 | 0.4911 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 14 | 0.1474 | 0.2129 |
| 2024Q4 | 310 | 75 | 0.2419 | 0.2129 |
| 2025Q1 | 95 | 10 | 0.1053 | 0.2129 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0000 | 0.5588 | 0.4397 | 0.0613 | `False` |
| test | 37600 | 0.3090 | 0.5776 | 0.4850 | 0.0082 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=246.16); shipped as `isotonic`. Brier vs base-rate: -0.0056 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
