# gbdt experiment — russell1000_up_50pct_200d_dd25pct_aligned_cbagent

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `200`
- max_drawdown: `0.25`
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

## Final checkpoint

- best iteration: 0
- iterations run: 2
- inner stop signal: `agent_should_stop`
- fs_hp_loop callback_mode: `agent_file_protocol`
- tie-break path: `v14_val_flat_eval_rp1` — Val_brier flat: tie set picked by eval R-Precision@1 (V1.4 P1)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -142.263
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1022 | 0.1093 | +0.0071 | 0.3557 | 0.7239 |
| test | 0.1383 | 0.1288 | -0.0095 | 0.5519 | 0.7032 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=171600, base_rate=0.1249

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.9000 | 0.1249 | 200 | 180 | 200 | 0 / 200 / 200 |
| 5 | 0.6980 | 0.1249 | 1000 | 698 | 1000 | 0 / 200 / 200 |
| 10 | 0.5700 | 0.1249 | 2000 | 1140 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1249 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.1249 | 5 | 2 | 5 |
| 10 | 0.4000 | 0.1249 | 10 | 4 | 10 |

### test — n_rows=257400, base_rate=0.1518

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.7433 | 0.1518 | 300 | 223 | 300 | 0 / 300 / 300 |
| 5 | 0.5667 | 0.1518 | 1500 | 850 | 1500 | 0 / 300 / 300 |
| 10 | 0.4713 | 0.1518 | 3000 | 1414 | 3000 | 0 / 300 / 300 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1518 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1518 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.1518 | 10 | 10 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=171600, Q_days=200, base_rate=0.1249

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.9000 | 0.1249 | 200 |
| 3 | 0.6867 | 0.1249 | 200 |
| 5 | 0.6980 | 0.1249 | 200 |
| 10 | 0.5700 | 0.1249 | 200 |
| 20 | 0.4600 | 0.1249 | 200 |

### test — n_rows=257400, Q_days=300, base_rate=0.1518

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.7433 | 0.1518 | 300 |
| 3 | 0.5867 | 0.1518 | 300 |
| 5 | 0.5667 | 0.1518 | 300 |
| 10 | 0.4713 | 0.1518 | 300 |
| 20 | 0.4013 | 0.1518 | 300 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 185 | 135 | 0.7297 |
| NASDAQ:MDB | 111 | 103 | 0.9279 |
| NASDAQ:TEAM | 87 | 45 | 0.5172 |
| NYSE:W | 83 | 58 | 0.6988 |
| NYSE:GME | 74 | 1 | 0.0135 |
| NASDAQ:TTD | 72 | 69 | 0.9583 |
| NYSE:CVNA | 71 | 56 | 0.7887 |
| NASDAQ:NFLX | 54 | 46 | 0.8519 |
| NYSE:SMCI | 51 | 50 | 0.9804 |
| NYSE:ROKU | 31 | 31 | 1.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:DXCM | 6 | 0 | 0.0000 |
| NYSE:NTRA | 6 | 0 | 0.0000 |
| NYSE:GME | 74 | 1 | 0.0135 |
| NYSE:SMMT | 13 | 1 | 0.0769 |
| NYSE:QXO | 24 | 12 | 0.5000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 300 | 241 | 0.8033 |
| NYSE:SMCI | 252 | 85 | 0.3373 |
| NASDAQ:MSTR | 243 | 182 | 0.7490 |
| NYSE:SMMT | 222 | 139 | 0.6261 |
| NYSE:VKTX | 133 | 61 | 0.4586 |
| NYSE:W | 132 | 25 | 0.1894 |
| NYSE:GME | 96 | 47 | 0.4896 |
| NYSE:QXO | 85 | 68 | 0.8000 |
| NYSE:ROKU | 16 | 0 | 0.0000 |
| NYSE:MPT | 11 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ROKU | 16 | 0 | 0.0000 |
| NYSE:MPT | 11 | 0 | 0.0000 |
| NYSE:W | 132 | 25 | 0.1894 |
| NYSE:SMCI | 252 | 85 | 0.3373 |
| NYSE:VKTX | 133 | 61 | 0.4586 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2022Q4 | 295 | 211 | 0.7153 | 0.1249 |
| 2023Q1 | 310 | 227 | 0.7323 | 0.1249 |
| 2023Q2 | 310 | 236 | 0.7613 | 0.1249 |
| 2023Q3 | 85 | 24 | 0.2824 | 0.1249 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q3 | 230 | 97 | 0.4217 | 0.1518 |
| 2023Q4 | 315 | 266 | 0.8444 | 0.1518 |
| 2024Q1 | 305 | 156 | 0.5115 | 0.1518 |
| 2024Q2 | 315 | 137 | 0.4349 | 0.1518 |
| 2024Q3 | 320 | 185 | 0.5781 | 0.1518 |
| 2024Q4 | 15 | 9 | 0.6000 | 0.1518 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 171600 | 0.0000 | 0.8000 | 0.0741 | 0.0770 | `False` |
| test | 257400 | 0.0000 | 0.3624 | 0.0329 | 0.0344 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=142.26); shipped as `isotonic`. Brier vs base-rate: +0.0071 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
