# gbdt experiment — russell1000_up_10pct_50d_dd5pct

## Warnings

- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `50`
- max_drawdown: `0.05`
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
- positive prevalence (train): 0.346
- positive prevalence (eval): 0.352

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.2160 | 0.2248 | 0.0088 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 14/27 |  |
| 1 | 14 | 0.2163 | 0.2249 | 0.0086 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 13/14 features |  |
| 2 | 13 | 0.2172 | 0.2249 | 0.0077 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 0
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `native`
- Spiegelhalter Z: -0.660
- Spiegelhalter p: 0.5090

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2234 | 0.2281 | +0.0047 | 0.6384 | 0.6092 |
| test | 0.2126 | 0.2095 | -0.0031 | 0.6167 | 0.5112 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.3519

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.3902 | 0.3519 | 216 | 80 | 205 | 11 / 216 / 216 |
| 5 | 0.3821 | 0.3519 | 1017 | 384 | 1005 | 16 / 200 / 216 |
| 10 | 0.3840 | 0.3519 | 2017 | 770 | 2005 | 16 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.3519 | 1 | 1 | 1 |
| 5 | 0.8000 | 0.3519 | 5 | 4 | 5 |
| 10 | 0.9000 | 0.3519 | 10 | 9 | 10 |

### test — n_rows=44450, base_rate=0.2988

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.1200 | 0.2988 | 66 | 6 | 50 | 16 / 66 / 66 |
| 5 | 0.1720 | 0.2988 | 266 | 43 | 250 | 16 / 50 / 66 |
| 10 | 0.2340 | 0.2988 | 516 | 117 | 500 | 16 / 50 / 66 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.2988 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.2988 | 5 | 0 | 5 |
| 10 | 0.2000 | 0.2988 | 10 | 2 | 10 |

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:DXCM | 58 | 9 | 0.1552 |
| NASDAQ:AVGO | 53 | 43 | 0.8113 |
| NASDAQ:FANG | 52 | 22 | 0.4231 |
| NASDAQ:MSTR | 49 | 4 | 0.0816 |
| NYSE:GME | 37 | 27 | 0.7297 |
| NASDAQ:AMD | 35 | 15 | 0.4286 |
| NASDAQ:MCHP | 35 | 6 | 0.1714 |
| NYSE:CHWY | 35 | 28 | 0.8000 |
| NYSE:SMCI | 35 | 13 | 0.3714 |
| NASDAQ:LULU | 32 | 10 | 0.3125 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CLVT | 22 | 0 | 0.0000 |
| NYSE:VST | 16 | 0 | 0.0000 |
| NASDAQ:FAST | 15 | 0 | 0.0000 |
| NYSE:LW | 13 | 0 | 0.0000 |
| NYSE:MANH | 13 | 0 | 0.0000 |

### test

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:BILL | 23 | 2 | 0.0870 |
| NASDAQ:MRVL | 22 | 8 | 0.3636 |
| NASDAQ:TSLA | 18 | 0 | 0.0000 |
| NASDAQ:MCHP | 16 | 1 | 0.0625 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NASDAQ:CRWD | 15 | 1 | 0.0667 |
| NYSE:BBY | 15 | 0 | 0.0000 |
| NYSE:AXON | 14 | 0 | 0.0000 |
| NYSE:CVNA | 13 | 1 | 0.0769 |
| NASDAQ:NVDA | 12 | 0 | 0.0000 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:TSLA | 18 | 0 | 0.0000 |
| NYSE:AL | 16 | 0 | 0.0000 |
| NYSE:BBY | 15 | 0 | 0.0000 |
| NYSE:AXON | 14 | 0 | 0.0000 |
| NASDAQ:NVDA | 12 | 0 | 0.0000 |

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 10 | 0.1299 | 0.3519 | 0.369 |
| 2025Q2 | 310 | 197 | 0.6355 | 0.3519 | 1.806 |
| 2025Q3 | 320 | 94 | 0.2938 | 0.3519 | 0.835 |
| 2025Q4 | 310 | 83 | 0.2677 | 0.3519 | 0.761 |

### test

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q4 | 26 | 5 | 0.1923 | 0.2988 | 0.644 |
| 2026Q1 | 240 | 38 | 0.1583 | 0.2988 | 0.530 |

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.3047 | 0.4287 | 0.3683 | 0.0295 | `True` |
| test | 44450 | 0.3047 | 0.4019 | 0.3533 | 0.0292 | `True` |

## Per-experiment verdict (algorithmic readout)

Calibration: native-passable (|z|=0.66<2). Brier vs base-rate: +0.0047 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
