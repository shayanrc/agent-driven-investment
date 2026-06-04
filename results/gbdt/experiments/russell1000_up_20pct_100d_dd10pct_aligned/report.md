# gbdt experiment — russell1000_up_20pct_100d_dd10pct_aligned

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `100`
- max_drawdown: `0.1`
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
- positive prevalence (train): 0.311
- positive prevalence (eval): 0.357

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
| 0 | 279 | 0.1763 | 0.1808 | 0.0045 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 41/27 |  |
| 1 | 41 | 0.1723 | 0.1785 | 0.0061 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 37/41 features |  |
| 2 | 37 | 0.1704 | 0.1834 | 0.0130 | iteration 2 from FS+HP callback :: inner_stop=degradation | degradation |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `degradation`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -80.640
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2705 | 0.2296 | -0.0409 | 0.8163 | 0.5884 |
| test | 0.1697 | 0.1790 | +0.0093 | 0.5161 | 0.7026 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.3573

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5750 | 0.3573 | 200 | 115 | 200 | 0 / 200 / 200 |
| 5 | 0.5200 | 0.3573 | 1000 | 520 | 1000 | 0 / 200 / 200 |
| 10 | 0.4315 | 0.3573 | 2000 | 863 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.3573 | 1 | 1 | 1 |
| 5 | 0.4000 | 0.3573 | 5 | 2 | 5 |
| 10 | 0.3000 | 0.3573 | 10 | 3 | 10 |

### test — n_rows=177800, base_rate=0.2336

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2400 | 0.2336 | 200 | 48 | 200 | 0 / 200 / 200 |
| 5 | 0.3710 | 0.2336 | 1000 | 371 | 1000 | 0 / 200 / 200 |
| 10 | 0.3540 | 0.2336 | 2000 | 708 | 2000 | 0 / 200 / 200 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.2336 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.2336 | 5 | 5 | 5 |
| 10 | 0.8000 | 0.2336 | 10 | 8 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CVNA | 192 | 108 | 0.5625 |
| NYSE:SMCI | 167 | 66 | 0.3952 |
| NASDAQ:MSTR | 127 | 67 | 0.5276 |
| NYSE:SMMT | 114 | 68 | 0.5965 |
| NYSE:ASTS | 80 | 68 | 0.8500 |
| NYSE:NCLH | 61 | 36 | 0.5902 |
| NASDAQ:DDOG | 50 | 7 | 0.1400 |
| NYSE:MPT | 40 | 32 | 0.8000 |
| NYSE:GME | 29 | 12 | 0.4138 |
| NYSE:DKNG | 17 | 10 | 0.5882 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TSLA | 16 | 0 | 0.0000 |
| NYSE:OKTA | 9 | 0 | 0.0000 |
| NASDAQ:TEAM | 6 | 0 | 0.0000 |
| NASDAQ:WBD | 6 | 0 | 0.0000 |
| NYSE:XYZ | 6 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 195 | 53 | 0.2718 |
| NYSE:SMMT | 173 | 63 | 0.3642 |
| NASDAQ:MSTR | 155 | 58 | 0.3742 |
| NYSE:CVNA | 145 | 61 | 0.4207 |
| NYSE:W | 55 | 29 | 0.5273 |
| NASDAQ:TSLA | 45 | 3 | 0.0667 |
| NYSE:VKTX | 38 | 4 | 0.1053 |
| NYSE:GME | 31 | 27 | 0.8710 |
| NYSE:CL | 30 | 0 | 0.0000 |
| NYSE:AGCO | 20 | 17 | 0.8500 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CL | 30 | 0 | 0.0000 |
| NYSE:SMCI | 16 | 0 | 0.0000 |
| NYSE:ONTO | 10 | 0 | 0.0000 |
| NYSE:TRGP | 9 | 0 | 0.0000 |
| NASDAQ:TSLA | 45 | 3 | 0.0667 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``P@5`` falls toward ``base_rate`` or below. ``lift`` omitted from the table by project reporting convention.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2023Q4 | 290 | 186 | 0.6414 | 0.3573 |
| 2024Q1 | 305 | 158 | 0.5180 | 0.3573 |
| 2024Q2 | 315 | 132 | 0.4190 | 0.3573 |
| 2024Q3 | 90 | 44 | 0.4889 | 0.3573 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate |
|---|---|---|---|---|
| 2024Q3 | 230 | 114 | 0.4957 | 0.2336 |
| 2024Q4 | 320 | 110 | 0.3438 | 0.2336 |
| 2025Q1 | 300 | 61 | 0.2033 | 0.2336 |
| 2025Q2 | 150 | 86 | 0.5733 | 0.2336 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.4596 | 0.1445 | 0.0600 | `False` |
| test | 177800 | 0.0000 | 0.9048 | 0.1766 | 0.0639 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=80.64); shipped as `isotonic`. Brier vs base-rate: -0.0409 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
