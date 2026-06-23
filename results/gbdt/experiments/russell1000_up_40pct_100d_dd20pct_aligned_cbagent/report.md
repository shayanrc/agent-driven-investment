# gbdt experiment — russell1000_up_40pct_100d_dd20pct_aligned_cbagent

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `100`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `agent_file_protocol`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 708264 (independent events ≈ 3564.0; overlap-inflation 198.73×)
- val rows: 355600 (independent events ≈ 1786.9; overlap-inflation 199.00×)
- eval rows: 177800 (independent events ≈ 893.5; overlap-inflation 199.00×)
- test rows: 177800 (independent events ≈ 893.5; overlap-inflation 199.00×)
- sample uniqueness weighting: `on` (horizon_days=100)
- positive prevalence (train): 0.121
- positive prevalence (eval): 0.101

## Segment windows

- split mode: `date_aligned`
- train_start anchor: `2019-01-01`
- train: `2019-01-02` → `2022-03-04`
- val: `2022-03-07` → `2023-10-06`
- eval: `2023-10-09` → `2024-07-25`
- test: `2024-07-26` → `2025-05-13`

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
- Spiegelhalter Z: -69.155
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0914 | 0.0907 | -0.0007 | 0.3670 | 0.7566 |
| test | 0.0677 | 0.0733 | +0.0056 | 0.2496 | 0.8184 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.1009

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.8200 | 0.1009 | 200 | 164 | 200 | 0 / 200 / 200 |
| 5 | 0.6420 | 0.1009 | 1000 | 642 | 1000 | 0 / 200 / 200 |
| 10 | 0.5395 | 0.1009 | 2000 | 1079 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1009 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1009 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.1009 | 10 | 10 | 10 |

### test — n_rows=177800, base_rate=0.0796

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.4500 | 0.0796 | 200 | 90 | 200 | 0 / 200 / 200 |
| 5 | 0.4190 | 0.0796 | 1000 | 419 | 1000 | 0 / 200 / 200 |
| 10 | 0.3926 | 0.0796 | 2000 | 782 | 1992 | 5 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0796 | 1 | 0 | 1 |
| 5 | 0.6000 | 0.0796 | 5 | 3 | 5 |
| 10 | 0.7000 | 0.0796 | 10 | 7 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=177800, Q_days=200, base_rate=0.1009

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.8200 | 0.1009 | 200 |
| 3 | 0.6833 | 0.1009 | 200 |
| 5 | 0.6420 | 0.1009 | 200 |
| 10 | 0.5395 | 0.1009 | 200 |
| 20 | 0.3870 | 0.1009 | 200 |

### test — n_rows=177800, Q_days=200, base_rate=0.0796

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.4500 | 0.0796 | 200 |
| 3 | 0.4083 | 0.0796 | 200 |
| 5 | 0.4190 | 0.0796 | 200 |
| 10 | 0.3920 | 0.0796 | 200 |
| 20 | 0.3337 | 0.0796 | 200 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 200 | 177 | 0.8850 |
| NYSE:ASTS | 196 | 106 | 0.5408 |
| NASDAQ:MSTR | 182 | 98 | 0.5385 |
| NYSE:QXO | 104 | 87 | 0.8365 |
| NYSE:SMCI | 72 | 63 | 0.8750 |
| NYSE:GME | 66 | 19 | 0.2879 |
| NYSE:SMMT | 64 | 29 | 0.4531 |
| NYSE:MPT | 51 | 5 | 0.0980 |
| NYSE:NET | 31 | 30 | 0.9677 |
| NYSE:LYFT | 20 | 20 | 1.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:MPT | 51 | 5 | 0.0980 |
| NYSE:GME | 66 | 19 | 0.2879 |
| NYSE:SMMT | 64 | 29 | 0.4531 |
| NASDAQ:MSTR | 182 | 98 | 0.5385 |
| NYSE:ASTS | 196 | 106 | 0.5408 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 194 | 92 | 0.4742 |
| NYSE:ASTS | 191 | 58 | 0.3037 |
| NYSE:CVNA | 103 | 92 | 0.8932 |
| NYSE:GME | 90 | 38 | 0.4222 |
| NYSE:ELF | 60 | 0 | 0.0000 |
| NASDAQ:TSLA | 59 | 22 | 0.3729 |
| NYSE:CHWY | 39 | 19 | 0.4872 |
| NYSE:SMCI | 37 | 2 | 0.0541 |
| NYSE:FTAI | 36 | 26 | 0.7222 |
| NYSE:COHR | 34 | 2 | 0.0588 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ELF | 60 | 0 | 0.0000 |
| NYSE:ENPH | 20 | 0 | 0.0000 |
| NYSE:MRNA | 7 | 0 | 0.0000 |
| NYSE:SMCI | 37 | 2 | 0.0541 |
| NYSE:COHR | 34 | 2 | 0.0588 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 220 | 0.7586 | 0.1009 |
| 2024Q1 | 305 | 171 | 0.5607 | 0.1009 |
| 2024Q2 | 315 | 198 | 0.6286 | 0.1009 |
| 2024Q3 | 90 | 53 | 0.5889 | 0.1009 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 133 | 0.5783 | 0.0796 |
| 2024Q4 | 320 | 93 | 0.2906 | 0.0796 |
| 2025Q1 | 300 | 75 | 0.2500 | 0.0796 |
| 2025Q2 | 150 | 118 | 0.7867 | 0.0796 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.3765 | 0.0253 | 0.0414 | `True` |
| test | 177800 | 0.0000 | 0.3765 | 0.0357 | 0.0539 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=69.15); shipped as `isotonic`. Brier vs base-rate: -0.0007 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
