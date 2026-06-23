# gbdt experiment — russell1000_up_40pct_100d_dd20pct_aligned_resnap

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `100`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `default`

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
| 0 | 279 | 0.0741 | 0.0570 | -0.0171 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 52/27 |  |
| 1 | 52 | 0.0738 | 0.0567 | -0.0171 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 47/52 features |  |
| 2 | 47 | 0.0735 | 0.0557 | -0.0177 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`
- tie-break path: `v14_val_flat_eval_rp1` — Val_brier flat: tie set picked by eval R-Precision@1 (V1.4 P1)

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -54.445
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0910 | 0.0907 | -0.0003 | 0.3584 | 0.7612 |
| test | 0.0679 | 0.0733 | +0.0054 | 0.2498 | 0.8210 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.1009

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.7100 | 0.1009 | 200 | 142 | 200 | 0 / 200 / 200 |
| 5 | 0.5370 | 0.1009 | 1000 | 537 | 1000 | 0 / 200 / 200 |
| 10 | 0.4610 | 0.1009 | 2000 | 922 | 2000 | 0 / 200 / 200 |

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
| 1 | 0.5500 | 0.0796 | 200 | 110 | 200 | 0 / 200 / 200 |
| 5 | 0.4470 | 0.0796 | 1000 | 447 | 1000 | 0 / 200 / 200 |
| 10 | 0.3835 | 0.0796 | 2000 | 764 | 1992 | 5 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0796 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.0796 | 5 | 2 | 5 |
| 10 | 0.2000 | 0.0796 | 10 | 2 | 10 |

## R-Precision@K (canonical macro)

Per-day fixed K, **macro-averaged** across days with ``R_q > 0``: ``R-Precision@K = (1/Q) · Σ r_q / min(K, R_q)`` where ``R_q`` = positives that day, ``r_q`` = positives caught in top-K, sorted by ``(p_calibrated desc, ticker asc)`` stable mergesort. This is the cross-cell headline (matches ``results/gbdt/data/r_precision_at_k.csv``) — distinct from the Top-K block's ``per_day.p_at_k`` above, which is micro-aggregated (both forms are mathematically valid; macro is canonical for cross-cell comparison). See ``.claude/memories/project-r-precision-methodology.md``.

### eval — n_rows=177800, Q_days=200, base_rate=0.1009

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.7100 | 0.1009 | 200 |
| 3 | 0.5983 | 0.1009 | 200 |
| 5 | 0.5370 | 0.1009 | 200 |
| 10 | 0.4610 | 0.1009 | 200 |
| 20 | 0.3910 | 0.1009 | 200 |

### test — n_rows=177800, Q_days=200, base_rate=0.0796

| k | R-Precision@k | base_rate | Q_days |
|---|---|---|---|
| 1 | 0.5500 | 0.0796 | 200 |
| 3 | 0.5167 | 0.0796 | 200 |
| 5 | 0.4470 | 0.0796 | 200 |
| 10 | 0.3820 | 0.0796 | 200 |
| 20 | 0.3407 | 0.0796 | 200 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 197 | 107 | 0.5431 |
| NYSE:SMMT | 151 | 89 | 0.5894 |
| NYSE:SMCI | 149 | 38 | 0.2550 |
| NYSE:CVNA | 144 | 131 | 0.9097 |
| NYSE:GME | 110 | 43 | 0.3909 |
| NYSE:ASTS | 87 | 38 | 0.4368 |
| NYSE:W | 66 | 26 | 0.3939 |
| NYSE:VKTX | 31 | 31 | 1.0000 |
| NYSE:BILL | 15 | 5 | 0.3333 |
| NYSE:LYFT | 15 | 9 | 0.6000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:MPT | 6 | 0 | 0.0000 |
| NYSE:SMCI | 149 | 38 | 0.2550 |
| NYSE:BILL | 15 | 5 | 0.3333 |
| NYSE:GME | 110 | 43 | 0.3909 |
| NYSE:W | 66 | 26 | 0.3939 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:GME | 177 | 80 | 0.4520 |
| NASDAQ:MSTR | 174 | 90 | 0.5172 |
| NYSE:CVNA | 158 | 103 | 0.6519 |
| NYSE:VKTX | 88 | 14 | 0.1591 |
| NYSE:SMCI | 67 | 7 | 0.1045 |
| NYSE:ASTS | 63 | 29 | 0.4603 |
| NYSE:W | 59 | 8 | 0.1356 |
| NYSE:SMMT | 52 | 43 | 0.8269 |
| NASDAQ:TSLA | 34 | 19 | 0.5588 |
| NYSE:CELH | 20 | 8 | 0.4000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ENPH | 7 | 0 | 0.0000 |
| NYSE:SMCI | 67 | 7 | 0.1045 |
| NYSE:W | 59 | 8 | 0.1356 |
| NYSE:ELF | 20 | 3 | 0.1500 |
| NYSE:VKTX | 88 | 14 | 0.1591 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 177 | 0.6103 | 0.1009 |
| 2024Q1 | 305 | 157 | 0.5148 | 0.1009 |
| 2024Q2 | 315 | 159 | 0.5048 | 0.1009 |
| 2024Q3 | 90 | 44 | 0.4889 | 0.1009 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 137 | 0.5957 | 0.0796 |
| 2024Q4 | 320 | 90 | 0.2812 | 0.0796 |
| 2025Q1 | 300 | 117 | 0.3900 | 0.0796 |
| 2025Q2 | 150 | 103 | 0.6867 | 0.0796 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.3630 | 0.0266 | 0.0426 | `True` |
| test | 177800 | 0.0000 | 0.3630 | 0.0350 | 0.0521 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=54.45); shipped as `isotonic`. Brier vs base-rate: -0.0003 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
