# gbdt experiment — nifty500_up_20pct_100d_dd10pct_aligned

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=100 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `100`
- max_drawdown: `0.1`
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
- positive prevalence (train): 0.403
- positive prevalence (eval): 0.452

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
| 0 | 279 | 0.1317 | 0.3237 | 0.1920 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 166/2 |  |
| 1 | 166 | 0.1296 | 0.3249 | 0.1953 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 155/166 features;  |  |
| 2 | 155 | 0.1283 | 0.3169 | 0.1886 | iteration 2 from FS+HP callback :: inner_stop=cap | cap |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `cap`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 318.486
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2524 | 0.2477 | -0.0048 | 0.6991 | 0.4775 |
| test | 0.2132 | 0.1155 | -0.0978 | 0.6190 | 0.5080 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.4519

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.7525 | 0.4519 | 202 | 152 | 202 | 0 / 202 / 202 |
| 5 | 0.4881 | 0.4519 | 1010 | 493 | 1010 | 0 / 202 / 202 |
| 10 | 0.4936 | 0.4519 | 2020 | 997 | 2020 | 0 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.4519 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.4519 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.4519 | 10 | 10 | 10 |

### test — n_rows=37600, base_rate=0.1332

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3500 | 0.1332 | 100 | 35 | 100 | 0 / 100 / 100 |
| 5 | 0.2080 | 0.1332 | 500 | 104 | 500 | 0 / 100 / 100 |
| 10 | 0.1720 | 0.1332 | 1000 | 172 | 1000 | 0 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1332 | 1 | 0 | 1 |
| 5 | 0.6000 | 0.1332 | 5 | 3 | 5 |
| 10 | 0.6000 | 0.1332 | 10 | 6 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:360ONE | 139 | 111 | 0.7986 |
| NSE:ABREL | 122 | 106 | 0.8689 |
| NSE:ACE | 75 | 25 | 0.3333 |
| NSE:3MINDIA | 68 | 15 | 0.2206 |
| NSE:ABB | 68 | 51 | 0.7500 |
| NSE:AARTIIND | 58 | 27 | 0.4655 |
| NSE:AAVAS | 44 | 9 | 0.2045 |
| NSE:ACC | 41 | 6 | 0.1463 |
| NSE:ADANIENSOL | 36 | 18 | 0.5000 |
| NSE:KPIL | 36 | 1 | 0.0278 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:INDIACEM | 13 | 0 | 0.0000 |
| NSE:PHOENIXLTD | 6 | 0 | 0.0000 |
| NSE:KPIL | 36 | 1 | 0.0278 |
| NSE:SOBHA | 20 | 1 | 0.0500 |
| NSE:ADANIGREEN | 15 | 1 | 0.0667 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:360ONE | 55 | 16 | 0.2909 |
| NSE:TARIL | 35 | 21 | 0.6000 |
| NSE:GALLANTT | 28 | 16 | 0.5714 |
| NSE:KPIL | 27 | 0 | 0.0000 |
| NSE:TECHNOE | 27 | 0 | 0.0000 |
| NSE:ABREL | 26 | 0 | 0.0000 |
| NSE:ANANTRAJ | 26 | 17 | 0.6538 |
| NSE:3MINDIA | 20 | 0 | 0.0000 |
| NSE:AARTIIND | 18 | 0 | 0.0000 |
| NSE:ABB | 18 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:KPIL | 27 | 0 | 0.0000 |
| NSE:TECHNOE | 27 | 0 | 0.0000 |
| NSE:ABREL | 26 | 0 | 0.0000 |
| NSE:3MINDIA | 20 | 0 | 0.0000 |
| NSE:AARTIIND | 18 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 131 | 0.7706 | 0.4519 |
| 2024Q1 | 310 | 185 | 0.5968 | 0.4519 |
| 2024Q2 | 305 | 121 | 0.3967 | 0.4519 |
| 2024Q3 | 225 | 56 | 0.2489 | 0.4519 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 7 | 0.0737 | 0.1332 |
| 2024Q4 | 310 | 84 | 0.2710 | 0.1332 |
| 2025Q1 | 95 | 13 | 0.1368 | 0.1332 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0635 | 0.4898 | 0.4290 | 0.0520 | `False` |
| test | 37600 | 0.0635 | 0.4898 | 0.4439 | 0.0341 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=318.49); shipped as `isotonic`. Brier vs base-rate: -0.0048 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
