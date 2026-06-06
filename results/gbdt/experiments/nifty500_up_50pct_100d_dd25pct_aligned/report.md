# gbdt experiment — nifty500_up_50pct_100d_dd25pct_aligned

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=100 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `100`
- max_drawdown: `0.25`
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
- positive prevalence (train): 0.170
- positive prevalence (eval): 0.163

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
| 0 | 279 | 0.0632 | 0.1580 | 0.0948 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 156/2 |  |
| 1 | 156 | 0.0635 | 0.1576 | 0.0941 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 150/156 features;  |  |
| 2 | 150 | 0.0635 | 0.1576 | 0.0941 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 271.364
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1336 | 0.1362 | +0.0026 | 0.4397 | 0.6092 |
| test | 0.0428 | 0.0154 | -0.0274 | 0.2150 | 0.7271 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.1627

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3416 | 0.1627 | 202 | 69 | 202 | 0 / 202 / 202 |
| 5 | 0.2218 | 0.1627 | 1010 | 224 | 1010 | 0 / 202 / 202 |
| 10 | 0.2276 | 0.1627 | 2020 | 459 | 2017 | 3 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1627 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.1627 | 5 | 1 | 5 |
| 10 | 0.1000 | 0.1627 | 10 | 1 | 10 |

### test — n_rows=37600, base_rate=0.0156

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0156 | 100 | 0 | 100 | 0 / 100 / 100 |
| 5 | 0.0024 | 0.0156 | 500 | 1 | 411 | 41 / 100 / 100 |
| 10 | 0.0275 | 0.0156 | 1000 | 15 | 545 | 87 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0156 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0156 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0156 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:APARINDS | 132 | 29 | 0.2197 |
| NSE:ABREL | 122 | 41 | 0.3361 |
| NSE:ACE | 110 | 23 | 0.2091 |
| NSE:ADANIENSOL | 79 | 6 | 0.0759 |
| NSE:ADANIPOWER | 76 | 21 | 0.2763 |
| NSE:BHEL | 52 | 12 | 0.2308 |
| NSE:360ONE | 51 | 1 | 0.0196 |
| NSE:ADANIGREEN | 51 | 6 | 0.1176 |
| NSE:BEML | 51 | 33 | 0.6471 |
| NSE:ABFRL | 49 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ABFRL | 49 | 0 | 0.0000 |
| NSE:ABB | 28 | 0 | 0.0000 |
| NSE:ATGL | 22 | 0 | 0.0000 |
| NSE:BSE | 21 | 0 | 0.0000 |
| NSE:AMBER | 13 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ABREL | 100 | 0 | 0.0000 |
| NSE:360ONE | 90 | 0 | 0.0000 |
| NSE:ABFRL | 71 | 0 | 0.0000 |
| NSE:ADANIGREEN | 47 | 0 | 0.0000 |
| NSE:ACE | 46 | 0 | 0.0000 |
| NSE:ADANIPOWER | 43 | 0 | 0.0000 |
| NSE:ADANIENSOL | 36 | 1 | 0.0278 |
| NSE:AEGISLOG | 26 | 0 | 0.0000 |
| NSE:ABCAPITAL | 16 | 0 | 0.0000 |
| NSE:ABB | 14 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:ABREL | 100 | 0 | 0.0000 |
| NSE:360ONE | 90 | 0 | 0.0000 |
| NSE:ABFRL | 71 | 0 | 0.0000 |
| NSE:ADANIGREEN | 47 | 0 | 0.0000 |
| NSE:ACE | 46 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 64 | 0.3765 | 0.1627 |
| 2024Q1 | 310 | 112 | 0.3613 | 0.1627 |
| 2024Q2 | 305 | 48 | 0.1574 | 0.1627 |
| 2024Q3 | 225 | 0 | 0.0000 | 0.1627 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 0 | 0.0000 | 0.0156 |
| 2024Q4 | 310 | 1 | 0.0032 | 0.0156 |
| 2025Q1 | 95 | 0 | 0.0000 | 0.0156 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0000 | 0.2318 | 0.1548 | 0.0637 | `False` |
| test | 37600 | 0.0070 | 0.2318 | 0.1749 | 0.0574 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=271.36); shipped as `isotonic`. Brier vs base-rate: +0.0026 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
