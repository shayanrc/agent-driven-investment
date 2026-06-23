# gbdt experiment — russell1000_up_40pct_200d_dd20pct_aligned_cbagent

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `200`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Data

- tickers in universe: 1002
- tickers used: 858
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:CRWD, NASDAQ:DASH, NASDAQ:DDOG, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:ASTS, NYSE:AUR, NYSE:AVTR, NYSE:BAM, NYSE:BEPC, NYSE:BILL, NYSE:BIRK, NYSE:BJ, NYSE:BLSH, NYSE:BRBR, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CHWY, NYSE:CLVT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:CTVA, NYSE:DJT, NYSE:DKNG, NYSE:DOCS, NYSE:DOW, NYSE:DT, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ELAN, NYSE:ESAB, NYSE:ESTC, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FOX, NYSE:FOXA, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:LYFT, NYSE:MDLN, NYSE:MP, NYSE:MRNA, NYSE:MRP, NYSE:NCNO, NYSE:NET, NYSE:NIQ, NYSE:NU, NYSE:NVST, NYSE:OGN, NYSE:ONON, NYSE:ONTO, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:PINS, NYSE:PSN, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TIGO, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:TW, NYSE:U, NYSE:UBER, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VRT, NYSE:VSNT, NYSE:WFRD, NYSE:XP, NYSE:YETI, NYSE:ZM
- train rows: 685740 (independent events ≈ 1721.4; overlap-inflation 398.37×)
- val rows: 343200 (independent events ≈ 860.2; overlap-inflation 399.00×)
- eval rows: 171600 (independent events ≈ 430.1; overlap-inflation 399.00×)
- test rows: 257400 (independent events ≈ 645.1; overlap-inflation 399.00×)
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.242
- positive prevalence (eval): 0.195

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

## Final checkpoint

- best iteration: 0
- iterations run: 2
- inner stop signal: `agent_should_stop`
- fs_hp_loop callback_mode: `agent_file_protocol`
- tie-break path: `v14_val_flat_eval_rp1` — Val_brier flat: tie set picked by eval R-Precision@1 (V1.4 P1)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -128.806
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1496 | 0.1570 | +0.0075 | 0.4785 | 0.6856 |
| test | 0.2038 | 0.1806 | -0.0232 | 0.6926 | 0.6576 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=171600, base_rate=0.1951

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.8050 | 0.1951 | 200 | 161 | 200 | 0 / 200 / 200 |
| 5 | 0.6670 | 0.1951 | 1000 | 667 | 1000 | 0 / 200 / 200 |
| 10 | 0.5595 | 0.1951 | 2000 | 1119 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1951 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1951 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.1951 | 10 | 10 | 10 |

### test — n_rows=257400, base_rate=0.2365

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5667 | 0.2365 | 300 | 170 | 300 | 0 / 300 / 300 |
| 5 | 0.5147 | 0.2365 | 1500 | 772 | 1500 | 0 / 300 / 300 |
| 10 | 0.4553 | 0.2365 | 3000 | 1366 | 3000 | 0 / 300 / 300 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.2365 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.2365 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.2365 | 10 | 10 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=171600, Q_days=200, base_rate=0.1951

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.8050 | 0.1951 | 200 |
| 3 | 0.7167 | 0.1951 | 200 |
| 5 | 0.6670 | 0.1951 | 200 |
| 10 | 0.5595 | 0.1951 | 200 |
| 20 | 0.4905 | 0.1951 | 200 |

### test — n_rows=257400, Q_days=300, base_rate=0.2365

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.5667 | 0.2365 | 300 |
| 3 | 0.5478 | 0.2365 | 300 |
| 5 | 0.5147 | 0.2365 | 300 |
| 10 | 0.4553 | 0.2365 | 300 |
| 20 | 0.4222 | 0.2365 | 300 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 183 | 109 | 0.5956 |
| NASDAQ:MDB | 167 | 120 | 0.7186 |
| NASDAQ:TEAM | 92 | 71 | 0.7717 |
| NASDAQ:TTD | 78 | 69 | 0.8846 |
| NYSE:CCL | 70 | 35 | 0.5000 |
| NYSE:GME | 66 | 4 | 0.0606 |
| NASDAQ:NFLX | 55 | 45 | 0.8182 |
| NASDAQ:MRVL | 52 | 49 | 0.9423 |
| NYSE:SMCI | 49 | 46 | 0.9388 |
| NASDAQ:AMD | 43 | 43 | 1.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:DOCU | 14 | 0 | 0.0000 |
| NYSE:CZR | 11 | 0 | 0.0000 |
| NYSE:OKTA | 8 | 0 | 0.0000 |
| NYSE:GME | 66 | 4 | 0.0606 |
| NYSE:NTRA | 5 | 1 | 0.2000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 272 | 178 | 0.6544 |
| NYSE:CVNA | 270 | 211 | 0.7815 |
| NYSE:SMCI | 211 | 87 | 0.4123 |
| NYSE:SMMT | 138 | 76 | 0.5507 |
| NYSE:W | 121 | 27 | 0.2231 |
| NYSE:MPT | 70 | 11 | 0.1571 |
| NYSE:VKTX | 66 | 49 | 0.7424 |
| NYSE:GME | 62 | 22 | 0.3548 |
| NYSE:ROKU | 54 | 0 | 0.0000 |
| NYSE:ZION | 46 | 28 | 0.6087 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ROKU | 54 | 0 | 0.0000 |
| NYSE:ENPH | 19 | 0 | 0.0000 |
| NYSE:RNG | 13 | 0 | 0.0000 |
| NYSE:SIRI | 5 | 0 | 0.0000 |
| NYSE:RH | 9 | 1 | 0.1111 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 295 | 221 | 0.7492 | 0.1951 |
| 2023Q1 | 310 | 216 | 0.6968 | 0.1951 |
| 2023Q2 | 310 | 213 | 0.6871 | 0.1951 |
| 2023Q3 | 85 | 17 | 0.2000 | 0.1951 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 230 | 79 | 0.3435 | 0.2365 |
| 2023Q4 | 315 | 231 | 0.7333 | 0.2365 |
| 2024Q1 | 305 | 180 | 0.5902 | 0.2365 |
| 2024Q2 | 315 | 115 | 0.3651 | 0.2365 |
| 2024Q3 | 320 | 160 | 0.5000 | 0.2365 |
| 2024Q4 | 15 | 7 | 0.4667 | 0.2365 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 171600 | 0.0000 | 0.3816 | 0.1267 | 0.0932 | `False` |
| test | 257400 | 0.0000 | 0.3816 | 0.0627 | 0.0482 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=128.81); shipped as `isotonic`. Brier vs base-rate: +0.0075 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
