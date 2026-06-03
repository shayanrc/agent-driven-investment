# gbdt experiment — russell1000_up_20pct_50d_dd10pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `50`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 711200 (independent events ≈ 7183.8; overlap-inflation 99.00×)
- val rows: 355600 (independent events ≈ 3591.9; overlap-inflation 99.00×)
- eval rows: 177800 (independent events ≈ 1796.0; overlap-inflation 99.00×)
- test rows: 44450 (independent events ≈ 449.0; overlap-inflation 99.00×)
- sample uniqueness weighting: `on` (horizon_days=50)
- positive prevalence (train): 0.167
- positive prevalence (eval): 0.181

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1192 | 0.1165 | -0.0027 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 34/27 |  |
| 1 | 34 | 0.1195 | 0.1163 | -0.0032 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 30/34 features |  |
| 2 | 30 | 0.1197 | 0.1166 | -0.0031 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -5.327
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1337 | 0.1481 | +0.0143 | 0.4216 | 0.7426 |
| test | 0.1232 | 0.1296 | +0.0064 | 0.4032 | 0.6816 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.1808

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2696 | 0.1808 | 216 | 55 | 204 | 12 / 216 / 216 |
| 5 | 0.3695 | 0.1808 | 1017 | 371 | 1004 | 16 / 200 / 216 |
| 10 | 0.4212 | 0.1808 | 2017 | 844 | 2004 | 16 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1808 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1808 | 5 | 5 | 5 |
| 10 | 0.9000 | 0.1808 | 10 | 9 | 10 |

### test — n_rows=44450, base_rate=0.1530

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3800 | 0.1530 | 66 | 19 | 50 | 16 / 66 / 66 |
| 5 | 0.3520 | 0.1530 | 266 | 88 | 250 | 16 / 50 / 66 |
| 10 | 0.3260 | 0.1530 | 516 | 163 | 500 | 16 / 50 / 66 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1530 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1530 | 5 | 5 | 5 |
| 10 | 0.7000 | 0.1530 | 10 | 7 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 185 | 27 | 0.1459 |
| NYSE:ASTS | 173 | 86 | 0.4971 |
| NYSE:CVNA | 79 | 27 | 0.3418 |
| NASDAQ:TSLA | 67 | 23 | 0.3433 |
| NASDAQ:MRVL | 56 | 7 | 0.1250 |
| NYSE:CAR | 47 | 3 | 0.0638 |
| NASDAQ:MDB | 34 | 20 | 0.5882 |
| NYSE:VKTX | 34 | 21 | 0.6176 |
| NYSE:SMCI | 30 | 11 | 0.3667 |
| NYSE:MRNA | 29 | 13 | 0.4483 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CAR | 47 | 3 | 0.0638 |
| NASDAQ:MRVL | 56 | 7 | 0.1250 |
| NYSE:ENPH | 8 | 1 | 0.1250 |
| NASDAQ:MSTR | 185 | 27 | 0.1459 |
| NASDAQ:INTC | 20 | 3 | 0.1500 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CLF | 40 | 2 | 0.0500 |
| NASDAQ:MSTR | 31 | 8 | 0.2581 |
| NYSE:ASTS | 25 | 18 | 0.7200 |
| NYSE:COHR | 17 | 13 | 0.7647 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:ENPH | 16 | 15 | 0.9375 |
| NYSE:ELF | 13 | 4 | 0.3077 |
| NASDAQ:MRVL | 11 | 5 | 0.4545 |
| NASDAQ:TTD | 10 | 0 | 0.0000 |
| NASDAQ:MDB | 8 | 1 | 0.1250 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NASDAQ:TTD | 10 | 0 | 0.0000 |
| NYSE:INSP | 8 | 0 | 0.0000 |
| NYSE:BRBR | 7 | 0 | 0.0000 |
| NYSE:CVNA | 6 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q1 | 77 | 14 | 0.1818 | 0.1808 |
| 2025Q2 | 310 | 188 | 0.6065 | 0.1808 |
| 2025Q3 | 320 | 99 | 0.3094 | 0.1808 |
| 2025Q4 | 310 | 70 | 0.2258 | 0.1808 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2025Q4 | 26 | 5 | 0.1923 | 0.1530 |
| 2026Q1 | 240 | 83 | 0.3458 | 0.1530 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 1.0000 | 0.1709 | 0.0899 | `False` |
| test | 44450 | 0.0000 | 0.3441 | 0.1727 | 0.0873 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=5.33); shipped as `isotonic`. Brier vs base-rate: +0.0143 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
