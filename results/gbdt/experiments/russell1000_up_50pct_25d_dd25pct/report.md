# gbdt experiment — russell1000_up_50pct_25d_dd25pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `50`
- horizon_days: `25`
- max_drawdown: `0.25`
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
- positive prevalence (train): 0.005
- positive prevalence (eval): 0.004

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.0042 | 0.0027 | -0.0015 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 34/27 |  |
| 1 | 34 | 0.0045 | 0.0027 | -0.0018 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 27/34 features |  |
| 2 | 27 | 0.0044 | 0.0027 | -0.0017 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `native`
- Spiegelhalter Z: 1.439
- Spiegelhalter p: 0.1503

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.0037 | 0.0040 | +0.0002 | 0.0195 | 0.9300 |
| test | 0.0087 | 0.0086 | -0.0000 | 0.0426 | 0.8798 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.0040

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3641 | 0.0040 | 216 | 67 | 184 | 32 / 216 / 216 |
| 5 | 0.2060 | 0.0040 | 1017 | 110 | 534 | 179 / 200 / 216 |
| 10 | 0.2289 | 0.0040 | 2017 | 141 | 616 | 205 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0040 | 1 | 1 | 1 |
| 5 | 0.6000 | 0.0040 | 5 | 3 | 5 |
| 10 | 0.4000 | 0.0040 | 10 | 4 | 10 |

### test — n_rows=66675, base_rate=0.0087

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.0270 | 0.0087 | 91 | 2 | 74 | 17 / 91 / 91 |
| 5 | 0.1447 | 0.0087 | 391 | 44 | 304 | 47 / 75 / 91 |
| 10 | 0.1790 | 0.0087 | 766 | 80 | 447 | 71 / 75 / 91 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0087 | 1 | 0 | 1 |
| 5 | 0.4000 | 0.0087 | 5 | 2 | 5 |
| 10 | 0.2000 | 0.0087 | 10 | 2 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 200 | 67 | 0.3350 |
| NYSE:SMMT | 199 | 20 | 0.1005 |
| NYSE:SMCI | 145 | 2 | 0.0138 |
| NYSE:SRPT | 139 | 7 | 0.0504 |
| NASDAQ:MSTR | 133 | 3 | 0.0226 |
| NYSE:VKTX | 69 | 2 | 0.0290 |
| NYSE:LITE | 36 | 8 | 0.2222 |
| NYSE:RH | 29 | 0 | 0.0000 |
| NYSE:QXO | 28 | 1 | 0.0357 |
| NYSE:AL | 16 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:RH | 29 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CAR | 9 | 0 | 0.0000 |
| NYSE:SMCI | 145 | 2 | 0.0138 |
| NASDAQ:MSTR | 133 | 3 | 0.0226 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:ASTS | 75 | 2 | 0.0267 |
| NYSE:LITE | 75 | 23 | 0.3067 |
| NYSE:CIEN | 48 | 5 | 0.1042 |
| NYSE:WDC | 39 | 11 | 0.2821 |
| NYSE:SRPT | 36 | 0 | 0.0000 |
| NYSE:COHR | 34 | 1 | 0.0294 |
| NYSE:SMMT | 27 | 0 | 0.0000 |
| NASDAQ:MSTR | 25 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CORT | 10 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:SRPT | 36 | 0 | 0.0000 |
| NYSE:SMMT | 27 | 0 | 0.0000 |
| NASDAQ:MSTR | 25 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:CORT | 10 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 10 | 0.1299 | 0.0040 | 32.707 |
| 2025Q2 | 310 | 42 | 0.1355 | 0.0040 | 34.120 |
| 2025Q3 | 320 | 30 | 0.0938 | 0.0040 | 23.610 |
| 2025Q4 | 310 | 28 | 0.0903 | 0.0040 | 22.747 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q4 | 26 | 2 | 0.0769 | 0.0087 | 8.828 |
| 2026Q1 | 305 | 33 | 0.1082 | 0.0087 | 12.417 |
| 2026Q2 | 60 | 9 | 0.1500 | 0.0087 | 17.214 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.0001 | 0.6972 | 0.0044 | 0.0193 | `True` |
| test | 66675 | 0.0001 | 0.6168 | 0.0050 | 0.0177 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: native-passable (|z|=1.44<2). Brier vs base-rate: +0.0002 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
