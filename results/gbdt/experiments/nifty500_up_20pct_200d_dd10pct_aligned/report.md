# gbdt experiment — nifty500_up_20pct_200d_dd10pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `200`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 500
- tickers used: 357
- tickers excluded: NSE:360ONE, NSE:AADHARHFC, NSE:AAVAS, NSE:ABDL, NSE:ABLBL, NSE:ABSLAMC, NSE:ACMESOLAR, NSE:ACUTAAS, NSE:ADANIGREEN, NSE:AEGISVOPAK, NSE:AFCONS, NSE:AFFLE, NSE:AIIL, NSE:ANANDRATHI, NSE:ANGELONE, NSE:ANTHEM, NSE:ANURAS, NSE:APTUS, NSE:ATGL, NSE:ATHERENERG, NSE:AWL, NSE:BAJAJHFL, NSE:BELRISE, NSE:BHARTIHEXA, NSE:BIKAJI, NSE:BLUEJET, NSE:CAMS, NSE:CANHLIFE, NSE:CARTRADE, NSE:CHALET, NSE:CHOICEIN, NSE:CLEAN, NSE:COHANCE, NSE:CONCORDBIO, NSE:CPPLUS, NSE:CRAFTSMAN, NSE:CREDITACC, NSE:DALBHARAT, NSE:DATAPATTNS, NSE:DELHIVERY, NSE:DEVYANI, NSE:DOMS, NSE:EMCURE, NSE:EMMVEE, NSE:ENRIN, NSE:ETERNAL, NSE:FIRSTCRY, NSE:FIVESTAR, NSE:FLUOROCHEM, NSE:FORCEMOT, NSE:GLAND, NSE:GODIGIT, NSE:GROWW, NSE:GRSE, NSE:HDBFS, NSE:HDFCAMC, NSE:HEXT, NSE:HOMEFIRST, NSE:HONASA, NSE:HYUNDAI, NSE:ICICIAMC, NSE:IGIL, NSE:IKS, NSE:INDGN, NSE:INDIAMART, NSE:IRCON, NSE:IRCTC, NSE:ITCHOTELS, NSE:JAINREC, NSE:JIOFIN, NSE:JSWCEMENT, NSE:JSWINFRA, NSE:JUBLINGREA, NSE:JYOTICNC, NSE:KALYANKJIL, NSE:KAYNES, NSE:KFINTECH, NSE:KIMS, NSE:KPITTECH, NSE:LATENTVIEW, NSE:LENSKART, NSE:LGEINDIA, NSE:LICI, NSE:LLOYDSME, NSE:LODHA, NSE:MANKIND, NSE:MAPMYINDIA, NSE:MAXHEALTH, NSE:MAZDOCK, NSE:MEDANTA, NSE:MEESHO, NSE:MSUMI, NSE:NETWEB, NSE:NIVABUPA, NSE:NSLNISP, NSE:NTPCGREEN, NSE:NUVAMA, NSE:NUVOCO, NSE:NYKAA, NSE:OLAELEC, NSE:ONESOURCE, NSE:PARADEEP, NSE:PAYTM, NSE:PINELABS, NSE:PIRAMALFIN, NSE:POLICYBZR, NSE:POLYCAB, NSE:POWERINDIA, NSE:PPLPHARMA, NSE:PREMIERENE, NSE:PTCIL, NSE:PWL, NSE:RAILTEL, NSE:RAINBOW, NSE:RITES, NSE:RRKABEL, NSE:RVNL, NSE:SAGILITY, NSE:SAILIFE, NSE:SAPPHIRE, NSE:SBFC, NSE:SBICARD, NSE:SHRIRAMFIN, NSE:SHYAMMETL, NSE:SIGNATURE, NSE:SONACOMS, NSE:STARHEALTH, NSE:SUMICHEM, NSE:SWIGGY, NSE:SYRMA, NSE:TATACAP, NSE:TATATECH, NSE:TBOTEK, NSE:TEGA, NSE:TENNIND, NSE:THELEELA, NSE:TMCV, NSE:TRAVELFOOD, NSE:URBANCO, NSE:UTIAMC, NSE:VIJAYA, NSE:VMM, NSE:WAAREEENER
- train rows: 286254 (independent events ≈ 719.8; overlap-inflation 397.66×)
- val rows: 143505 (independent events ≈ 359.7; overlap-inflation 399.00×)
- eval rows: 71387 (independent events ≈ 178.9; overlap-inflation 399.00×)
- test rows: 107804 (independent events ≈ 270.2; overlap-inflation 399.00×)
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.403
- positive prevalence (eval): 0.640

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2018-01-01`
- train: `2018-01-01` → `2021-03-30`
- val: `2021-03-31` → `2022-11-14`
- eval: `2022-11-15` → `2023-09-01`
- test: `2023-09-04` → `2024-11-21`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1310 | 0.2513 | 0.1203 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 155/2 |  |
| 1 | 155 | 0.1294 | 0.2500 | 0.1206 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 142/155 features;  |  |
| 2 | 142 | 0.1307 | 0.2487 | 0.1180 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 116.513
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.3271 | 0.2304 | -0.0968 | 0.8554 | 0.4503 |
| test | 0.2795 | 0.2499 | -0.0296 | 0.7566 | 0.4186 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=71387, base_rate=0.6401

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.6850 | 0.6401 | 200 | 137 | 200 | 0 / 200 / 200 |
| 5 | 0.6770 | 0.6401 | 1000 | 677 | 1000 | 0 / 200 / 200 |
| 10 | 0.6650 | 0.6401 | 2000 | 1330 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.6401 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.6401 | 5 | 2 | 5 |
| 10 | 0.7000 | 0.6401 | 10 | 7 | 10 |

### test — n_rows=107804, base_rate=0.4892

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3775 | 0.4892 | 302 | 114 | 302 | 0 / 302 / 302 |
| 5 | 0.4073 | 0.4892 | 1510 | 615 | 1510 | 0 / 302 / 302 |
| 10 | 0.4175 | 0.4892 | 3020 | 1261 | 3020 | 0 / 302 / 302 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.4892 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.4892 | 5 | 1 | 5 |
| 10 | 0.1000 | 0.4892 | 10 | 1 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:3MINDIA | 63 | 62 | 0.9841 |
| NSE:IREDA | 57 | 16 | 0.2807 |
| NSE:AARTIIND | 43 | 3 | 0.0698 |
| NSE:ABB | 39 | 34 | 0.8718 |
| NSE:CGPOWER | 39 | 38 | 0.9744 |
| NSE:KEI | 37 | 37 | 1.0000 |
| NSE:CHOLAHLDNG | 33 | 30 | 0.9091 |
| NSE:SUPREMEIND | 25 | 24 | 0.9600 |
| NSE:ALKEM | 21 | 21 | 1.0000 |
| NSE:SARDAEN | 21 | 20 | 0.9524 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:PHOENIXLTD | 14 | 0 | 0.0000 |
| NSE:BAJAJHLDNG | 10 | 0 | 0.0000 |
| NSE:M&MFIN | 10 | 0 | 0.0000 |
| NSE:BIOCON | 8 | 0 | 0.0000 |
| NSE:PIDILITIND | 5 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:POWERGRID | 67 | 0 | 0.0000 |
| NSE:CUMMINSIND | 40 | 0 | 0.0000 |
| NSE:3MINDIA | 39 | 25 | 0.6410 |
| NSE:ABB | 39 | 16 | 0.4103 |
| NSE:EMAMILTD | 39 | 0 | 0.0000 |
| NSE:IPCALAB | 39 | 35 | 0.8974 |
| NSE:NESTLEIND | 35 | 0 | 0.0000 |
| NSE:AARTIIND | 34 | 13 | 0.3824 |
| NSE:SOBHA | 34 | 14 | 0.4118 |
| NSE:COCHINSHIP | 33 | 9 | 0.2727 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:POWERGRID | 67 | 0 | 0.0000 |
| NSE:CUMMINSIND | 40 | 0 | 0.0000 |
| NSE:EMAMILTD | 39 | 0 | 0.0000 |
| NSE:NESTLEIND | 35 | 0 | 0.0000 |
| NSE:NATCOPHARM | 33 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 170 | 71 | 0.4176 | 0.6401 |
| 2023Q1 | 310 | 194 | 0.6258 | 0.6401 |
| 2023Q2 | 300 | 267 | 0.8900 | 0.6401 |
| 2023Q3 | 220 | 145 | 0.6591 | 0.6401 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 95 | 49 | 0.5158 | 0.4892 |
| 2023Q4 | 305 | 216 | 0.7082 | 0.4892 |
| 2024Q1 | 310 | 129 | 0.4161 | 0.4892 |
| 2024Q2 | 305 | 159 | 0.5213 | 0.4892 |
| 2024Q3 | 320 | 46 | 0.1437 | 0.4892 |
| 2024Q4 | 175 | 16 | 0.0914 | 0.4892 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 71387 | 0.0000 | 0.6658 | 0.3448 | 0.0509 | `False` |
| test | 107804 | 0.0000 | 0.6449 | 0.3443 | 0.0437 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=116.51); shipped as `isotonic`. Brier vs base-rate: -0.0968 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
