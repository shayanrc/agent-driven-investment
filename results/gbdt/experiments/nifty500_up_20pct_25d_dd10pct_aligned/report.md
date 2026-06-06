# gbdt experiment — nifty500_up_20pct_25d_dd10pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `25`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 500
- tickers used: 376
- tickers excluded: NSE:AADHARHFC, NSE:ABDL, NSE:ABLBL, NSE:ABSLAMC, NSE:ACMESOLAR, NSE:ACUTAAS, NSE:AEGISVOPAK, NSE:AFCONS, NSE:AIIL, NSE:ANANDRATHI, NSE:ANGELONE, NSE:ANTHEM, NSE:ANURAS, NSE:APTUS, NSE:ATHERENERG, NSE:AWL, NSE:BAJAJHFL, NSE:BELRISE, NSE:BHARTIHEXA, NSE:BIKAJI, NSE:BLUEJET, NSE:CAMS, NSE:CANHLIFE, NSE:CARTRADE, NSE:CHOICEIN, NSE:CLEAN, NSE:COHANCE, NSE:CONCORDBIO, NSE:CPPLUS, NSE:CRAFTSMAN, NSE:DATAPATTNS, NSE:DELHIVERY, NSE:DEVYANI, NSE:DOMS, NSE:EMCURE, NSE:EMMVEE, NSE:ENRIN, NSE:ETERNAL, NSE:FIRSTCRY, NSE:FIVESTAR, NSE:GLAND, NSE:GODIGIT, NSE:GROWW, NSE:HDBFS, NSE:HEXT, NSE:HOMEFIRST, NSE:HONASA, NSE:HYUNDAI, NSE:ICICIAMC, NSE:IGIL, NSE:IKS, NSE:INDGN, NSE:ITCHOTELS, NSE:JAINREC, NSE:JIOFIN, NSE:JSWCEMENT, NSE:JSWINFRA, NSE:JUBLINGREA, NSE:JYOTICNC, NSE:KALYANKJIL, NSE:KAYNES, NSE:KFINTECH, NSE:KIMS, NSE:LATENTVIEW, NSE:LENSKART, NSE:LGEINDIA, NSE:LICI, NSE:LLOYDSME, NSE:LODHA, NSE:MANKIND, NSE:MAPMYINDIA, NSE:MAXHEALTH, NSE:MAZDOCK, NSE:MEDANTA, NSE:MEESHO, NSE:MSUMI, NSE:NETWEB, NSE:NIVABUPA, NSE:NSLNISP, NSE:NTPCGREEN, NSE:NUVAMA, NSE:NUVOCO, NSE:NYKAA, NSE:OLAELEC, NSE:ONESOURCE, NSE:PARADEEP, NSE:PAYTM, NSE:PINELABS, NSE:PIRAMALFIN, NSE:POLICYBZR, NSE:POWERINDIA, NSE:PPLPHARMA, NSE:PREMIERENE, NSE:PTCIL, NSE:PWL, NSE:RAILTEL, NSE:RAINBOW, NSE:RRKABEL, NSE:SAGILITY, NSE:SAILIFE, NSE:SAPPHIRE, NSE:SBFC, NSE:SBICARD, NSE:SHRIRAMFIN, NSE:SHYAMMETL, NSE:SIGNATURE, NSE:SONACOMS, NSE:STARHEALTH, NSE:SUMICHEM, NSE:SWIGGY, NSE:SYRMA, NSE:TATACAP, NSE:TATATECH, NSE:TBOTEK, NSE:TEGA, NSE:TENNIND, NSE:THELEELA, NSE:TMCV, NSE:TRAVELFOOD, NSE:URBANCO, NSE:UTIAMC, NSE:VIJAYA, NSE:VMM, NSE:WAAREEENER
- train rows: 300996 (independent events ≈ 6144.8; overlap-inflation 48.98×)
- val rows: 150755 (independent events ≈ 3076.6; overlap-inflation 49.00×)
- eval rows: 75907 (independent events ≈ 1549.1; overlap-inflation 49.00×)
- test rows: 37600 (independent events ≈ 767.3; overlap-inflation 49.00×)
- sample uniqueness weighting: `on` (horizon_days=25)
- positive prevalence (train): 0.158
- positive prevalence (eval): 0.150

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
| 0 | 279 | 0.0713 | 0.0992 | 0.0279 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 205/2 |  |
| 1 | 205 | 0.0724 | 0.0994 | 0.0269 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 197/205 features;  |  |
| 2 | 197 | 0.0718 | 0.0998 | 0.0280 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 67.099
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1283 | 0.1273 | -0.0011 | 0.4391 | 0.6270 |
| test | 0.0465 | 0.0425 | -0.0040 | 0.2011 | 0.7224 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.1496

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2673 | 0.1496 | 202 | 54 | 202 | 0 / 202 / 202 |
| 5 | 0.2277 | 0.1496 | 1010 | 230 | 1010 | 0 / 202 / 202 |
| 10 | 0.2198 | 0.1496 | 2020 | 444 | 2020 | 0 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1496 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.1496 | 5 | 2 | 5 |
| 10 | 0.4000 | 0.1496 | 10 | 4 | 10 |

### test — n_rows=37600, base_rate=0.0445

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2100 | 0.0445 | 100 | 21 | 100 | 0 / 100 / 100 |
| 5 | 0.2547 | 0.0445 | 500 | 123 | 483 | 12 / 100 / 100 |
| 10 | 0.2297 | 0.0445 | 1000 | 192 | 836 | 38 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0445 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.0445 | 5 | 2 | 5 |
| 10 | 0.2000 | 0.0445 | 10 | 2 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:GVT&D | 84 | 30 | 0.3571 |
| NSE:FACT | 59 | 8 | 0.1356 |
| NSE:ADANIENSOL | 54 | 9 | 0.1667 |
| NSE:SUZLON | 40 | 16 | 0.4000 |
| NSE:TARIL | 35 | 10 | 0.2857 |
| NSE:APARINDS | 33 | 4 | 0.1212 |
| NSE:GRSE | 32 | 1 | 0.0312 |
| NSE:INOXWIND | 31 | 10 | 0.3226 |
| NSE:RVNL | 29 | 18 | 0.6207 |
| NSE:NEWGEN | 27 | 4 | 0.1481 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:JPPOWER | 17 | 0 | 0.0000 |
| NSE:ELECON | 16 | 0 | 0.0000 |
| NSE:BDL | 14 | 0 | 0.0000 |
| NSE:NLCINDIA | 12 | 0 | 0.0000 |
| NSE:OIL | 10 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:GALLANTT | 54 | 10 | 0.1852 |
| NSE:TARIL | 41 | 19 | 0.4634 |
| NSE:COCHINSHIP | 39 | 4 | 0.1026 |
| NSE:RPOWER | 36 | 9 | 0.2500 |
| NSE:BSE | 25 | 15 | 0.6000 |
| NSE:NEULANDLAB | 22 | 6 | 0.2727 |
| NSE:WOCKPHARMA | 20 | 4 | 0.2000 |
| NSE:HEG | 19 | 0 | 0.0000 |
| NSE:ITI | 19 | 13 | 0.6842 |
| NSE:AMBER | 18 | 8 | 0.4444 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:HEG | 19 | 0 | 0.0000 |
| NSE:ELECON | 12 | 0 | 0.0000 |
| NSE:BBTC | 9 | 0 | 0.0000 |
| NSE:IRFC | 9 | 0 | 0.0000 |
| NSE:PCBL | 8 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 45 | 0.2647 | 0.1496 |
| 2024Q1 | 310 | 44 | 0.1419 | 0.1496 |
| 2024Q2 | 305 | 109 | 0.3574 | 0.1496 |
| 2024Q3 | 225 | 32 | 0.1422 | 0.1496 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 21 | 0.2211 | 0.0445 |
| 2024Q4 | 310 | 88 | 0.2839 | 0.0445 |
| 2025Q1 | 95 | 14 | 0.1474 | 0.0445 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0000 | 0.2277 | 0.0847 | 0.0512 | `False` |
| test | 37600 | 0.0000 | 0.3117 | 0.1129 | 0.0495 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=67.10); shipped as `isotonic`. Brier vs base-rate: -0.0011 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
