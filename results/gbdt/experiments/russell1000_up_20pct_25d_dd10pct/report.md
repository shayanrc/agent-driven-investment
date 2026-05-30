# gbdt experiment — russell1000_up_20pct_25d_dd10pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `20`
- horizon_days: `25`
- max_drawdown: `0.1`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 711200 (independent events ≈ 14514.3; overlap-inflation 49.00×)
- val rows: 355600 (independent events ≈ 7257.1; overlap-inflation 49.00×)
- eval rows: 177800 (independent events ≈ 3628.6; overlap-inflation 49.00×)
- test rows: 66675 (independent events ≈ 1360.7; overlap-inflation 49.00×)
- sample uniqueness weighting: `on` (horizon_days=25)
- positive prevalence (train): 0.073
- positive prevalence (eval): 0.073

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0581 | 0.0478 | -0.0102 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 57/27 |  |
| 1 | 57 | 0.0578 | 0.0478 | -0.0099 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 50/57 features |  |
| 2 | 50 | 0.0581 | 0.0478 | -0.0103 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -4.678
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0610 | 0.0676 | +0.0066 | 0.2177 | 0.8170 |
| test | 0.0861 | 0.0916 | +0.0054 | 0.3042 | 0.7297 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0729

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2800 | 0.0729 | 216 | 56 | 200 | 16 / 216 / 216 |
| 5 | 0.3383 | 0.0729 | 1017 | 338 | 999 | 17 / 200 / 216 |
| 10 | 0.3494 | 0.0729 | 2017 | 688 | 1969 | 24 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0729 | 1 | 1 | 1 |
| 5 | 0.2000 | 0.0729 | 5 | 1 | 5 |
| 10 | 0.2000 | 0.0729 | 10 | 2 | 10 |

### test — n_rows=66675, base_rate=0.1019

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3333 | 0.1019 | 91 | 25 | 75 | 16 / 91 / 91 |
| 5 | 0.3787 | 0.1019 | 391 | 142 | 375 | 16 / 75 / 91 |
| 10 | 0.3800 | 0.1019 | 766 | 285 | 750 | 16 / 75 / 91 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1019 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.1019 | 5 | 5 | 5 |
| 10 | 0.8000 | 0.1019 | 10 | 8 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 200 | 86 | 0.4300 |
| NYSE:SMMT | 180 | 44 | 0.2444 |
| NASDAQ:MSTR | 154 | 22 | 0.1429 |
| NYSE:SMCI | 150 | 58 | 0.3867 |
| NYSE:RH | 72 | 29 | 0.4028 |
| NYSE:VKTX | 49 | 4 | 0.0816 |
| NYSE:MRNA | 47 | 16 | 0.3404 |
| NYSE:W | 31 | 18 | 0.5806 |
| NYSE:FTAI | 27 | 19 | 0.7037 |
| NYSE:QXO | 26 | 17 | 0.6538 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CAR | 8 | 0 | 0.0000 |
| NYSE:CVNA | 14 | 1 | 0.0714 |
| NYSE:VKTX | 49 | 4 | 0.0816 |
| NASDAQ:MSTR | 154 | 22 | 0.1429 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 75 | 25 | 0.3333 |
| NYSE:LITE | 74 | 50 | 0.6757 |
| NYSE:SRPT | 70 | 19 | 0.2714 |
| NYSE:SMMT | 49 | 13 | 0.2653 |
| NYSE:VKTX | 37 | 6 | 0.1622 |
| NYSE:COHR | 21 | 15 | 0.7143 |
| NASDAQ:MSTR | 16 | 6 | 0.3750 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:ENPH | 6 | 0 | 0.0000 |
| NYSE:INSP | 6 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:ENPH | 6 | 0 | 0.0000 |
| NYSE:INSP | 6 | 0 | 0.0000 |
| NYSE:W | 6 | 0 | 0.0000 |
| NYSE:VKTX | 37 | 6 | 0.1622 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 1 | 0.0130 | 0.0729 | 0.178 |
| 2025Q2 | 310 | 165 | 0.5323 | 0.0729 | 7.299 |
| 2025Q3 | 320 | 82 | 0.2562 | 0.0729 | 3.514 |
| 2025Q4 | 310 | 90 | 0.2903 | 0.0729 | 3.981 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q4 | 26 | 3 | 0.1154 | 0.1019 | 1.132 |
| 2026Q1 | 305 | 113 | 0.3705 | 0.1019 | 3.634 |
| 2026Q2 | 60 | 26 | 0.4333 | 0.1019 | 4.251 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 0.5321 | 0.0686 | 0.0627 | `False` |
| test | 66675 | 0.0000 | 0.3939 | 0.0727 | 0.0645 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=4.68); shipped as `isotonic`. Brier vs base-rate: +0.0066 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
