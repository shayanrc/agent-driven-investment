# gbdt experiment — russell1000_up_10pct_10d_dd5pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `10`
- max_drawdown: `0.05`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 711200 (independent events ≈ 37431.6; overlap-inflation 19.00×)
- val rows: 355600 (independent events ≈ 18715.8; overlap-inflation 19.00×)
- eval rows: 177800 (independent events ≈ 9357.9; overlap-inflation 19.00×)
- test rows: 80010 (independent events ≈ 4211.1; overlap-inflation 19.00×)
- sample uniqueness weighting: `on` (horizon_days=10)
- positive prevalence (train): 0.101
- positive prevalence (eval): 0.093

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0802 | 0.0684 | -0.0119 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 60/27 |  |
| 1 | 60 | 0.0809 | 0.0685 | -0.0125 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 47/60 features |  |
| 2 | 47 | 0.0803 | 0.0684 | -0.0119 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: -6.989
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0769 | 0.0846 | +0.0077 | 0.2847 | 0.7863 |
| test | 0.1043 | 0.1098 | +0.0055 | 0.3533 | 0.6954 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0933

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.2941 | 0.0933 | 216 | 60 | 204 | 12 / 216 / 216 |
| 5 | 0.3340 | 0.0933 | 1017 | 335 | 1003 | 17 / 200 / 216 |
| 10 | 0.3320 | 0.0933 | 2017 | 660 | 1988 | 20 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0933 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.0933 | 5 | 4 | 5 |
| 10 | 0.9000 | 0.0933 | 10 | 9 | 10 |

### test — n_rows=80010, base_rate=0.1256

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3556 | 0.1256 | 106 | 32 | 90 | 16 / 106 / 106 |
| 5 | 0.3444 | 0.1256 | 466 | 155 | 450 | 16 / 90 / 106 |
| 10 | 0.3344 | 0.1256 | 916 | 301 | 900 | 16 / 90 / 106 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.1256 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.1256 | 5 | 4 | 5 |
| 10 | 0.8000 | 0.1256 | 10 | 8 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 172 | 70 | 0.4070 |
| NASDAQ:MSTR | 145 | 22 | 0.1517 |
| NYSE:MRNA | 106 | 34 | 0.3208 |
| NYSE:RH | 81 | 24 | 0.2963 |
| NYSE:SMCI | 61 | 14 | 0.2295 |
| NYSE:ENPH | 59 | 14 | 0.2373 |
| NYSE:SMMT | 41 | 17 | 0.4146 |
| NYSE:VKTX | 40 | 14 | 0.3500 |
| NYSE:CVNA | 32 | 9 | 0.2812 |
| NYSE:SRPT | 27 | 8 | 0.2963 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TTD | 7 | 0 | 0.0000 |
| NYSE:CELH | 5 | 0 | 0.0000 |
| NYSE:VFC | 7 | 1 | 0.1429 |
| NASDAQ:MSTR | 145 | 22 | 0.1517 |
| NYSE:SMCI | 61 | 14 | 0.2295 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 87 | 24 | 0.2759 |
| NYSE:COHR | 65 | 39 | 0.6000 |
| NYSE:LITE | 51 | 26 | 0.5098 |
| NASDAQ:MSTR | 50 | 19 | 0.3800 |
| NYSE:INSP | 30 | 4 | 0.1333 |
| NYSE:ELF | 26 | 6 | 0.2308 |
| NYSE:SRPT | 26 | 6 | 0.2308 |
| NYSE:BRBR | 23 | 3 | 0.1304 |
| NYSE:SMMT | 20 | 7 | 0.3500 |
| NYSE:AL | 16 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CORT | 9 | 0 | 0.0000 |
| NYSE:MRNA | 5 | 0 | 0.0000 |
| NYSE:VKTX | 5 | 0 | 0.0000 |
| NYSE:BRBR | 23 | 3 | 0.1304 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 24 | 0.3117 | 0.0933 | 3.339 |
| 2025Q2 | 310 | 132 | 0.4258 | 0.0933 | 4.562 |
| 2025Q3 | 320 | 101 | 0.3156 | 0.0933 | 3.382 |
| 2025Q4 | 310 | 78 | 0.2516 | 0.0933 | 2.696 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q4 | 26 | 4 | 0.1538 | 0.1256 | 1.225 |
| 2026Q1 | 305 | 107 | 0.3508 | 0.1256 | 2.794 |
| 2026Q2 | 135 | 44 | 0.3259 | 0.1256 | 2.596 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0000 | 1.0000 | 0.1162 | 0.1055 | `False` |
| test | 80010 | 0.0000 | 0.3776 | 0.1084 | 0.0783 | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=6.99); shipped as `isotonic`. Brier vs base-rate: +0.0077 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
