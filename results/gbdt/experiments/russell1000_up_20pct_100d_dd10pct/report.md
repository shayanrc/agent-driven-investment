# gbdt experiment — russell1000_up_20pct_100d_dd10pct

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=100 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
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
- train rows: 711200 (independent events ≈ 3573.9; overlap-inflation 199.00×)
- val rows: 355600 (independent events ≈ 1786.9; overlap-inflation 199.00×)
- eval rows: 177800 (independent events ≈ 893.5; overlap-inflation 199.00×)
- test rows: 0
- sample uniqueness weighting: `on` (horizon_days=100)
- positive prevalence (train): 0.287
- positive prevalence (eval): 0.309

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1837 | 0.1992 | 0.0155 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 24/27 |  |
| 1 | 24 | 0.1839 | 0.1991 | 0.0152 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 19/24 features |  |
| 2 | 19 | 0.1833 | 0.1990 | 0.0156 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `native`
- Spiegelhalter Z: 1.434
- Spiegelhalter p: 0.1517

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2018 | 0.2136 | +0.0118 | 0.5904 | 0.6653 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=177800, base_rate=0.3092

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.5637 | 0.3092 | 216 | 115 | 204 | 12 / 216 / 216 |
| 5 | 0.5727 | 0.3092 | 1017 | 575 | 1004 | 16 / 200 / 216 |
| 10 | 0.5220 | 0.3092 | 2017 | 1046 | 2004 | 16 / 200 / 216 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.3092 | 1 | 0 | 1 |
| 5 | 0.0000 | 0.3092 | 5 | 0 | 5 |
| 10 | 0.0000 | 0.3092 | 10 | 0 | 10 |

### test — n_rows=0, base_rate=n/a

_segment empty — no picks._

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NYSE:CORT | 114 | 67 | 0.5877 |
| NYSE:RH | 102 | 62 | 0.6078 |
| NYSE:SMCI | 95 | 52 | 0.5474 |
| NYSE:SMMT | 92 | 27 | 0.2935 |
| NYSE:SRPT | 67 | 41 | 0.6119 |
| NASDAQ:MSTR | 66 | 0 | 0.0000 |
| NYSE:ONTO | 51 | 39 | 0.7647 |
| NYSE:FTAI | 50 | 36 | 0.7200 |
| NYSE:ENPH | 39 | 13 | 0.3333 |
| NYSE:ASTS | 35 | 17 | 0.4857 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:MSTR | 66 | 0 | 0.0000 |
| NYSE:QRVO | 13 | 0 | 0.0000 |
| NYSE:MPT | 9 | 0 | 0.0000 |
| NYSE:PENN | 9 | 0 | 0.0000 |
| NYSE:AL | 16 | 4 | 0.2500 |

### test

_no picks._

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 16 | 0.2078 | 0.3092 | 0.672 |
| 2025Q2 | 310 | 234 | 0.7548 | 0.3092 | 2.442 |
| 2025Q3 | 320 | 174 | 0.5437 | 0.3092 | 1.759 |
| 2025Q4 | 310 | 151 | 0.4871 | 0.3092 | 1.576 |

### test

_no picks._

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 177800 | 0.2097 | 0.4962 | 0.3087 | 0.0674 | `False` |
| test | 0 | n/a | n/a | n/a | n/a | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: native-passable (|z|=1.43<2). Brier vs base-rate: +0.0118 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
