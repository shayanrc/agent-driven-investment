# gbdt experiment — russell1000_up_50pct_200d_dd25pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `200`
- max_drawdown: `0.25`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 858
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:CRWD, NASDAQ:DASH, NASDAQ:DDOG, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:ASTS, NYSE:AUR, NYSE:AVTR, NYSE:BAM, NYSE:BEPC, NYSE:BILL, NYSE:BIRK, NYSE:BJ, NYSE:BLSH, NYSE:BRBR, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CHWY, NYSE:CLVT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:CTVA, NYSE:DJT, NYSE:DKNG, NYSE:DOCS, NYSE:DOW, NYSE:DT, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ELAN, NYSE:ESAB, NYSE:ESTC, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FOX, NYSE:FOXA, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:LYFT, NYSE:MDLN, NYSE:MP, NYSE:MRNA, NYSE:MRP, NYSE:NCNO, NYSE:NET, NYSE:NIQ, NYSE:NU, NYSE:NVST, NYSE:OGN, NYSE:ONON, NYSE:ONTO, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:PINS, NYSE:PSN, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TIGO, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:TW, NYSE:U, NYSE:UBER, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VRT, NYSE:VSNT, NYSE:WFRD, NYSE:XP, NYSE:YETI, NYSE:ZM
- train rows: 685740 (independent events ≈ 1721.4; overlap-inflation 398.37×)
- val rows: 343200 (independent events ≈ 860.2; overlap-inflation 399.00×)
- eval rows: 171600 (independent events ≈ 430.1; overlap-inflation 399.00×)
- test rows: 257400 (independent events ≈ 645.1; overlap-inflation 399.00×)
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.181
- positive prevalence (eval): 0.125

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2018-01-01`
- train: `2018-01-02` → `2021-03-08`
- val: `2021-03-09` → `2022-10-06`
- eval: `2022-10-07` → `2023-07-26`
- test: `2023-07-27` → `2024-10-03`

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0969 | 0.0811 | -0.0158 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 52/27 |  |
| 1 | 52 | 0.0956 | 0.0823 | -0.0132 | iteration 1 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 0
- iterations run: 2
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -142.749
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1020 | 0.1093 | +0.0073 | 0.3547 | 0.7218 |
| test | 0.1376 | 0.1288 | -0.0088 | 0.5254 | 0.6968 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=171600, base_rate=0.1249

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.7450 | 0.1249 | 200 | 149 | 200 | 0 / 200 / 200 |
| 5 | 0.5710 | 0.1249 | 1000 | 571 | 1000 | 0 / 200 / 200 |
| 10 | 0.5340 | 0.1249 | 2000 | 1068 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1249 | 1 | 0 | 1 |
| 5 | 0.2000 | 0.1249 | 5 | 1 | 5 |
| 10 | 0.2000 | 0.1249 | 10 | 2 | 10 |

### test — n_rows=257400, base_rate=0.1518

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.7367 | 0.1518 | 300 | 221 | 300 | 0 / 300 / 300 |
| 5 | 0.5387 | 0.1518 | 1500 | 808 | 1500 | 0 / 300 / 300 |
| 10 | 0.4660 | 0.1518 | 3000 | 1398 | 3000 | 0 / 300 / 300 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.1518 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.1518 | 5 | 0 | 5 |
| 10 | 0.2000 | 0.1518 | 10 | 2 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 143 | 113 | 0.7902 |
| NYSE:GME | 130 | 6 | 0.0462 |
| NASDAQ:MDB | 127 | 106 | 0.8346 |
| NYSE:CELH | 61 | 49 | 0.8033 |
| NYSE:SMCI | 58 | 54 | 0.9310 |
| NYSE:W | 51 | 29 | 0.5686 |
| NYSE:VKTX | 45 | 35 | 0.7778 |
| NASDAQ:TEAM | 44 | 19 | 0.4318 |
| NYSE:ROKU | 40 | 39 | 0.9750 |
| NYSE:NTRA | 37 | 13 | 0.3514 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ENPH | 34 | 0 | 0.0000 |
| NYSE:CAR | 10 | 0 | 0.0000 |
| NYSE:WAL | 7 | 0 | 0.0000 |
| NYSE:OKTA | 6 | 0 | 0.0000 |
| NYSE:GME | 130 | 6 | 0.0462 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 291 | 237 | 0.8144 |
| NYSE:VKTX | 209 | 81 | 0.3876 |
| NYSE:SMCI | 203 | 63 | 0.3103 |
| NYSE:SMMT | 199 | 114 | 0.5729 |
| NASDAQ:MSTR | 194 | 136 | 0.7010 |
| NYSE:QXO | 110 | 82 | 0.7455 |
| NYSE:W | 90 | 26 | 0.2889 |
| NYSE:GME | 82 | 34 | 0.4146 |
| NYSE:MPT | 37 | 18 | 0.4865 |
| NYSE:ROKU | 22 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ROKU | 22 | 0 | 0.0000 |
| NYSE:RNG | 17 | 0 | 0.0000 |
| NASDAQ:MDB | 17 | 1 | 0.0588 |
| NYSE:W | 90 | 26 | 0.2889 |
| NYSE:SMCI | 203 | 63 | 0.3103 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 295 | 143 | 0.4847 | 0.1249 |
| 2023Q1 | 310 | 199 | 0.6419 | 0.1249 |
| 2023Q2 | 310 | 205 | 0.6613 | 0.1249 |
| 2023Q3 | 85 | 24 | 0.2824 | 0.1249 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 230 | 86 | 0.3739 | 0.1518 |
| 2023Q4 | 315 | 247 | 0.7841 | 0.1518 |
| 2024Q1 | 305 | 171 | 0.5607 | 0.1518 |
| 2024Q2 | 315 | 134 | 0.4254 | 0.1518 |
| 2024Q3 | 320 | 164 | 0.5125 | 0.1518 |
| 2024Q4 | 15 | 6 | 0.4000 | 0.1518 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 171600 | 0.0000 | 0.6923 | 0.0771 | 0.0821 | `False` |
| test | 257400 | 0.0000 | 0.3570 | 0.0353 | 0.0375 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=142.75); shipped as `isotonic`. Brier vs base-rate: +0.0073 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
