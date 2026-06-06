# gbdt experiment — nifty500_up_30pct_100d_dd15pct_aligned

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=100 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `30`
- horizon_days: `100`
- max_drawdown: `0.15`
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
- positive prevalence (train): 0.314
- positive prevalence (eval): 0.339

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
| 0 | 279 | 0.1055 | 0.2691 | 0.1636 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 162/2 |  |
| 1 | 162 | 0.1058 | 0.2713 | 0.1655 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 150/162 features;  |  |
| 2 | 150 | 0.1052 | 0.2720 | 0.1668 | iteration 2 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 308.909
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2246 | 0.2240 | -0.0006 | 0.6445 | 0.5330 |
| test | 0.1393 | 0.0638 | -0.0755 | 0.4641 | 0.5503 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.3388

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5396 | 0.3388 | 202 | 109 | 202 | 0 / 202 / 202 |
| 5 | 0.4851 | 0.3388 | 1010 | 490 | 1010 | 0 / 202 / 202 |
| 10 | 0.4965 | 0.3388 | 2020 | 1003 | 2020 | 0 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.3388 | 1 | 1 | 1 |
| 5 | 0.6000 | 0.3388 | 5 | 3 | 5 |
| 10 | 0.3000 | 0.3388 | 10 | 3 | 10 |

### test — n_rows=37600, base_rate=0.0685

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3000 | 0.0685 | 100 | 30 | 100 | 0 / 100 / 100 |
| 5 | 0.1940 | 0.0685 | 500 | 97 | 500 | 0 / 100 / 100 |
| 10 | 0.1805 | 0.0685 | 1000 | 180 | 997 | 1 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0685 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0685 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0685 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ACE | 102 | 45 | 0.4412 |
| NSE:GVT&D | 69 | 34 | 0.4928 |
| NSE:ADANIPOWER | 58 | 56 | 0.9655 |
| NSE:ABREL | 52 | 51 | 0.9808 |
| NSE:SUZLON | 50 | 44 | 0.8800 |
| NSE:GALLANTT | 38 | 5 | 0.1316 |
| NSE:PRESTIGE | 38 | 9 | 0.2368 |
| NSE:ADANIPORTS | 36 | 1 | 0.0278 |
| NSE:ADANIGREEN | 30 | 2 | 0.0667 |
| NSE:INOXWIND | 27 | 25 | 0.9259 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:BHEL | 18 | 0 | 0.0000 |
| NSE:SOBHA | 13 | 0 | 0.0000 |
| NSE:HUDCO | 10 | 0 | 0.0000 |
| NSE:GABRIEL | 9 | 0 | 0.0000 |
| NSE:KPRMILL | 9 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:GALLANTT | 58 | 35 | 0.6034 |
| NSE:TARIL | 37 | 5 | 0.1351 |
| NSE:HEG | 36 | 11 | 0.3056 |
| NSE:APARINDS | 29 | 0 | 0.0000 |
| NSE:ADANIGREEN | 26 | 0 | 0.0000 |
| NSE:GRSE | 23 | 0 | 0.0000 |
| NSE:WOCKPHARMA | 22 | 0 | 0.0000 |
| NSE:JUBLPHARMA | 20 | 0 | 0.0000 |
| NSE:KEC | 19 | 3 | 0.1579 |
| NSE:DEEPAKFERT | 14 | 3 | 0.2143 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:APARINDS | 29 | 0 | 0.0000 |
| NSE:ADANIGREEN | 26 | 0 | 0.0000 |
| NSE:GRSE | 23 | 0 | 0.0000 |
| NSE:WOCKPHARMA | 22 | 0 | 0.0000 |
| NSE:JUBLPHARMA | 20 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 114 | 0.6706 | 0.3388 |
| 2024Q1 | 310 | 197 | 0.6355 | 0.3388 |
| 2024Q2 | 305 | 126 | 0.4131 | 0.3388 |
| 2024Q3 | 225 | 53 | 0.2356 | 0.3388 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 22 | 0.2316 | 0.0685 |
| 2024Q4 | 310 | 53 | 0.1710 | 0.0685 |
| 2025Q1 | 95 | 22 | 0.2316 | 0.0685 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0000 | 0.3967 | 0.3095 | 0.0628 | `False` |
| test | 37600 | 0.1113 | 0.3967 | 0.3431 | 0.0292 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=308.91); shipped as `isotonic`. Brier vs base-rate: -0.0006 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
