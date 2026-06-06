# gbdt experiment — nifty500_up_20pct_10d_dd10pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `nifty500`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `10`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 500
- tickers used: 376
- tickers excluded: NSE:AADHARHFC, NSE:ABDL, NSE:ABLBL, NSE:ABSLAMC, NSE:ACMESOLAR, NSE:ACUTAAS, NSE:AEGISVOPAK, NSE:AFCONS, NSE:AIIL, NSE:ANANDRATHI, NSE:ANGELONE, NSE:ANTHEM, NSE:ANURAS, NSE:APTUS, NSE:ATHERENERG, NSE:AWL, NSE:BAJAJHFL, NSE:BELRISE, NSE:BHARTIHEXA, NSE:BIKAJI, NSE:BLUEJET, NSE:CAMS, NSE:CANHLIFE, NSE:CARTRADE, NSE:CHOICEIN, NSE:CLEAN, NSE:COHANCE, NSE:CONCORDBIO, NSE:CPPLUS, NSE:CRAFTSMAN, NSE:DATAPATTNS, NSE:DELHIVERY, NSE:DEVYANI, NSE:DOMS, NSE:EMCURE, NSE:EMMVEE, NSE:ENRIN, NSE:ETERNAL, NSE:FIRSTCRY, NSE:FIVESTAR, NSE:GLAND, NSE:GODIGIT, NSE:GROWW, NSE:HDBFS, NSE:HEXT, NSE:HOMEFIRST, NSE:HONASA, NSE:HYUNDAI, NSE:ICICIAMC, NSE:IGIL, NSE:IKS, NSE:INDGN, NSE:ITCHOTELS, NSE:JAINREC, NSE:JIOFIN, NSE:JSWCEMENT, NSE:JSWINFRA, NSE:JUBLINGREA, NSE:JYOTICNC, NSE:KALYANKJIL, NSE:KAYNES, NSE:KFINTECH, NSE:KIMS, NSE:LATENTVIEW, NSE:LENSKART, NSE:LGEINDIA, NSE:LICI, NSE:LLOYDSME, NSE:LODHA, NSE:MANKIND, NSE:MAPMYINDIA, NSE:MAXHEALTH, NSE:MAZDOCK, NSE:MEDANTA, NSE:MEESHO, NSE:MSUMI, NSE:NETWEB, NSE:NIVABUPA, NSE:NSLNISP, NSE:NTPCGREEN, NSE:NUVAMA, NSE:NUVOCO, NSE:NYKAA, NSE:OLAELEC, NSE:ONESOURCE, NSE:PARADEEP, NSE:PAYTM, NSE:PINELABS, NSE:PIRAMALFIN, NSE:POLICYBZR, NSE:POWERINDIA, NSE:PPLPHARMA, NSE:PREMIERENE, NSE:PTCIL, NSE:PWL, NSE:RAILTEL, NSE:RAINBOW, NSE:RRKABEL, NSE:SAGILITY, NSE:SAILIFE, NSE:SAPPHIRE, NSE:SBFC, NSE:SBICARD, NSE:SHRIRAMFIN, NSE:SHYAMMETL, NSE:SIGNATURE, NSE:SONACOMS, NSE:STARHEALTH, NSE:SUMICHEM, NSE:SWIGGY, NSE:SYRMA, NSE:TATACAP, NSE:TATATECH, NSE:TBOTEK, NSE:TEGA, NSE:TENNIND, NSE:THELEELA, NSE:TMCV, NSE:TRAVELFOOD, NSE:URBANCO, NSE:UTIAMC, NSE:VIJAYA, NSE:VMM, NSE:WAAREEENER
- train rows: 300996 (independent events ≈ 15843.8; overlap-inflation 19.00×)
- val rows: 150755 (independent events ≈ 7934.5; overlap-inflation 19.00×)
- eval rows: 75907 (independent events ≈ 3995.1; overlap-inflation 19.00×)
- test rows: 37600 (independent events ≈ 1978.9; overlap-inflation 19.00×)
- sample uniqueness weighting: `on` (horizon_days=10)
- positive prevalence (train): 0.051
- positive prevalence (eval): 0.038

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
| 0 | 279 | 0.0264 | 0.0238 | -0.0026 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 205/2 |  |
| 1 | 205 | 0.0259 | 0.0237 | -0.0022 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 196/205 features |  |
| 2 | 196 | 0.0256 | 0.0239 | -0.0017 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 18.271
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0364 | 0.0367 | +0.0003 | 0.1590 | 0.6976 |
| test | 0.0134 | 0.0136 | +0.0002 | 0.0670 | 0.7972 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=75907, base_rate=0.0381

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1634 | 0.0381 | 202 | 33 | 202 | 0 / 202 / 202 |
| 5 | 0.1223 | 0.0381 | 1010 | 120 | 981 | 18 / 202 / 202 |
| 10 | 0.1293 | 0.0381 | 2020 | 224 | 1732 | 73 / 202 / 202 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0381 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0381 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0381 | 10 | 0 | 10 |

### test — n_rows=37600, base_rate=0.0138

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1170 | 0.0138 | 100 | 11 | 94 | 6 / 100 / 100 |
| 5 | 0.1296 | 0.0138 | 500 | 42 | 324 | 62 / 100 / 100 |
| 10 | 0.1467 | 0.0138 | 1000 | 65 | 443 | 85 / 100 / 100 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0138 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.0138 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.0138 | 10 | 0 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:TARIL | 78 | 18 | 0.2308 |
| NSE:RVNL | 44 | 6 | 0.1364 |
| NSE:IFCI | 40 | 7 | 0.1750 |
| NSE:SUZLON | 38 | 0 | 0.0000 |
| NSE:COCHINSHIP | 36 | 12 | 0.3333 |
| NSE:ADANIENSOL | 34 | 4 | 0.1176 |
| NSE:FACT | 33 | 6 | 0.1818 |
| NSE:IREDA | 33 | 9 | 0.2727 |
| NSE:FORCEMOT | 29 | 3 | 0.1034 |
| NSE:ITI | 28 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:SUZLON | 38 | 0 | 0.0000 |
| NSE:ITI | 28 | 0 | 0.0000 |
| NSE:GRSE | 17 | 0 | 0.0000 |
| NSE:RPOWER | 15 | 0 | 0.0000 |
| NSE:APARINDS | 13 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:TARIL | 61 | 3 | 0.0492 |
| NSE:COCHINSHIP | 51 | 6 | 0.1176 |
| NSE:GRSE | 48 | 6 | 0.1250 |
| NSE:IDEA | 41 | 7 | 0.1707 |
| NSE:PCBL | 24 | 0 | 0.0000 |
| NSE:DEEPAKFERT | 21 | 0 | 0.0000 |
| NSE:GRAVITA | 21 | 0 | 0.0000 |
| NSE:BSE | 18 | 1 | 0.0556 |
| NSE:RPOWER | 16 | 9 | 0.5625 |
| NSE:WOCKPHARMA | 16 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NSE:PCBL | 24 | 0 | 0.0000 |
| NSE:DEEPAKFERT | 21 | 0 | 0.0000 |
| NSE:GRAVITA | 21 | 0 | 0.0000 |
| NSE:WOCKPHARMA | 16 | 0 | 0.0000 |
| NSE:IFCI | 15 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 170 | 35 | 0.2059 | 0.0381 |
| 2024Q1 | 310 | 33 | 0.1065 | 0.0381 |
| 2024Q2 | 305 | 47 | 0.1541 | 0.0381 |
| 2024Q3 | 225 | 5 | 0.0222 | 0.0381 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 95 | 7 | 0.0737 | 0.0138 |
| 2024Q4 | 310 | 27 | 0.0871 | 0.0138 |
| 2025Q1 | 95 | 8 | 0.0842 | 0.0138 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 75907 | 0.0000 | 0.1549 | 0.0208 | 0.0179 | `True` |
| test | 37600 | 0.0000 | 0.1964 | 0.0216 | 0.0184 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=18.27); shipped as `isotonic`. Brier vs base-rate: +0.0003 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
