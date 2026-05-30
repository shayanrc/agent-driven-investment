# gbdt experiment — russell1000_up_40pct_200d_dd20pct

## Warnings

- **test_split**: Test segment expected to be EMPTY: horizon_days=200 >= split.test_rows=100, so every ticker's trailing 100 rows have NaN targets (forward window incomplete). headline_test will be {} and predictions/test.csv will be header-only. Eval segment is still measured. (threshold=100)
- **hp_search**: HP search disabled in sweep mode (max_iter=3 < threshold=5); the FS+HP loop ran feature-selection only — see issue #32.

## Spec

- universe: `russell1000`
- direction: `up`
- threshold_pct: `40`
- horizon_days: `200`
- max_drawdown: `0.2`
- fs_hp_loop callback_mode: `default`

## Data

- tickers in universe: 1002
- tickers used: 889
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR, NYSE:ACI, NYSE:AFRM, NYSE:ALAB, NYSE:ALGM, NYSE:AMTM, NYSE:APG, NYSE:AS, NYSE:AUR, NYSE:BAM, NYSE:BEPC, NYSE:BIRK, NYSE:BLSH, NYSE:BROS, NYSE:BSY, NYSE:CAI, NYSE:CARR, NYSE:CART, NYSE:CAVA, NYSE:CBC, NYSE:CCC, NYSE:CERT, NYSE:CNM, NYSE:CNXC, NYSE:COIN, NYSE:CPNG, NYSE:CR, NYSE:CRCL, NYSE:DJT, NYSE:DOCS, NYSE:DTM, NYSE:DUOL, NYSE:DV, NYSE:ECG, NYSE:ESAB, NYSE:EXE, NYSE:FIGR, NYSE:FOUR, NYSE:FRMI, NYSE:GEV, NYSE:GLIBA, NYSE:GLIBK, NYSE:GTLB, NYSE:GTM, NYSE:GXO, NYSE:HAYW, NYSE:HOOD, NYSE:INGM, NYSE:IOT, NYSE:KD, NYSE:KRMN, NYSE:KVUE, NYSE:LCID, NYSE:LINE, NYSE:LLYVA, NYSE:LLYVK, NYSE:LOAR, NYSE:MDLN, NYSE:MP, NYSE:MRP, NYSE:NCNO, NYSE:NIQ, NYSE:NU, NYSE:OGN, NYSE:ONON, NYSE:OTIS, NYSE:OWL, NYSE:PATH, NYSE:PCOR, NYSE:Q, NYSE:QS, NYSE:RAL, NYSE:RBLX, NYSE:RBRK, NYSE:RDDT, NYSE:REYN, NYSE:RIVN, NYSE:RKLB, NYSE:RKT, NYSE:ROIV, NYSE:RPRX, NYSE:RVMD, NYSE:RYAN, NYSE:S, NYSE:SAIL, NYSE:SARO, NYSE:SFD, NYSE:SHC, NYSE:SN, NYSE:SNDK, NYSE:SNOW, NYSE:SOFI, NYSE:SOLS, NYSE:SOLV, NYSE:TEM, NYSE:TLN, NYSE:TOST, NYSE:TPG, NYSE:U, NYSE:UHAL-B, NYSE:UWMC, NYSE:VGNT, NYSE:VIK, NYSE:VLTO, NYSE:VNT, NYSE:VSNT, NYSE:WFRD
- train rows: 711200 (independent events ≈ 1782.5; overlap-inflation 398.98×)
- val rows: 355600 (independent events ≈ 891.2; overlap-inflation 399.00×)
- eval rows: 88900 (independent events ≈ 222.8; overlap-inflation 399.00×)
- test rows: 0
- sample uniqueness weighting: `on` (horizon_days=200)
- positive prevalence (train): 0.222
- positive prevalence (eval): 0.265

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.1440 | 0.1552 | 0.0111 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 16/27 |  |
| 1 | 16 | 0.1455 | 0.1553 | 0.0098 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 14/16 features |  |
| 2 | 14 | 0.1439 | 0.1551 | 0.0112 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 2
- iterations run: 3
- inner stop signal: `plateau`
- fs_hp_loop callback_mode: `default`

## Calibration

- method requested: `conditional_isotonic`
- decision: `isotonic`
- Spiegelhalter Z: 11.925
- Spiegelhalter p: 0.0000

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.1763 | 0.1947 | +0.0185 | 0.5317 | 0.7446 |

## Top-K precision (per-day + global)

Per-day: pick the top-K rows by ``p_calibrated`` each date, pool across days. ``P@k = sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))`` where ``R(d)`` is the count of positives on day ``d`` — the ``min(R(d), k)`` denominator is the achievable-positives count (mandatory; see ``.claude/memories/project-r-precision-methodology.md``). ``n_denom = sum_d(min(R(d), k))``; ``n_days_R<k`` = days with fewer than ``k`` positives. Global: top-K by score across the whole segment, denominator ``min(k, total_positives)``. ``base_rate`` = unweighted segment positive prevalence (compare P@k to base_rate directly; lift omitted from the table by project reporting convention).

### eval — n_rows=88900, base_rate=0.2649

Per-day:

| k | P@k | base_rate | n_picks | n_positives | n_denom | days_R<k / days_full_k / days_total |
|---|---|---|---|---|---|---|
| 1 | 0.9245 | 0.2649 | 116 | 98 | 106 | 10 / 116 / 116 |
| 5 | 0.6581 | 0.2649 | 517 | 333 | 506 | 16 / 100 / 116 |
| 10 | 0.6561 | 0.2649 | 1017 | 660 | 1006 | 16 / 100 / 116 |

Global (top-K across entire segment):

| k | P@k | base_rate | n_picks | n_positives | n_denom |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.2649 | 1 | 1 | 1 |
| 5 | 1.0000 | 0.2649 | 5 | 5 | 5 |
| 10 | 1.0000 | 0.2649 | 10 | 10 | 10 |

### test — n_rows=0, base_rate=n/a

_segment empty — no picks._

## Per-ticker hit-rate when picked (k=5)

Aggregates per-day top-5 picks by ticker. Top 10 most-picked + bottom 5 most-anti-predictive (when picked at least once) shown; the full table is in ``metrics.json::segment_diagnostics.<seg>.per_ticker_hit_rate.rows``.

### eval

Top-10 by n_picks:

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:AVGO | 100 | 92 | 0.9200 |
| NASDAQ:AMD | 82 | 82 | 1.0000 |
| NASDAQ:INTC | 70 | 64 | 0.9143 |
| NASDAQ:FAST | 49 | 0 | 0.0000 |
| NASDAQ:CRWD | 42 | 20 | 0.4762 |
| NASDAQ:LRCX | 30 | 30 | 1.0000 |
| NASDAQ:DXCM | 26 | 3 | 0.1154 |
| NASDAQ:AMAT | 23 | 23 | 1.0000 |
| NASDAQ:LULU | 23 | 0 | 0.0000 |
| NYSE:AL | 16 | 6 | 0.3750 |

Bottom-5 by hit_rate (n_picks ≥ 5):

| ticker | n_picks | n_positives | hit_rate |
|---|---|---|---|
| NASDAQ:FAST | 49 | 0 | 0.0000 |
| NASDAQ:LULU | 23 | 0 | 0.0000 |
| NASDAQ:MCHP | 5 | 0 | 0.0000 |
| NASDAQ:DXCM | 26 | 3 | 0.1154 |
| NASDAQ:MU | 8 | 1 | 0.1250 |

### test

_no picks._

## Per-quarter P@5 stability

P@5 grouped by calendar quarter. ``base_rate`` is the segment-wide positive prevalence (constant across rows); regime-dependent collapse shows as a quarter where ``lift`` falls toward 1.0 or below.

### eval

| quarter | n_picks | n_positives | P@5 | base_rate | lift |
|---|---|---|---|---|---|
| 2025Q1 | 77 | 22 | 0.2857 | 0.2649 | 1.078 |
| 2025Q2 | 310 | 230 | 0.7419 | 0.2649 | 2.801 |
| 2025Q3 | 130 | 81 | 0.6231 | 0.2649 | 2.352 |

### test

_no picks._

## Prediction-range diagnostics

Distribution of ``p_calibrated`` per segment. ``flag_low_separation = true`` when ``std < low_separation_threshold`` — a sign the model's predictions cluster so tightly that ranking is noise.

| segment | n_rows | min | max | mean | std | flag_low_separation |
|---|---|---|---|---|---|---|
| eval | 88900 | 0.0989 | 0.6000 | 0.3242 | 0.1242 | `False` |
| test | 0 | n/a | n/a | n/a | n/a | `False` |

## Per-experiment verdict (algorithmic readout)

Calibration: required isotonic (|z|=11.93); shipped as `isotonic`. Brier vs base-rate: +0.0185 (model beats baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
