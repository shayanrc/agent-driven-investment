# gbdt experiment — nasdaq100_up_10pct_100d_dd5pct

## Spec

- universe: `nasdaq100`
- direction: `up`
- threshold_pct: `10`
- horizon_days: `100`
- max_drawdown: `0.05`

## Data

- tickers in universe: 100
- tickers used: 92
- tickers excluded: NASDAQ:ABNB, NASDAQ:APP, NASDAQ:ARM, NASDAQ:CEG, NASDAQ:DASH, NASDAQ:GEHC, NASDAQ:GFS, NASDAQ:PLTR
- train rows: 73600 (independent events ≈ 369.8; overlap-inflation 199.00×)
- val rows: 36800 (independent events ≈ 184.9; overlap-inflation 199.00×)
- eval rows: 18400 (independent events ≈ 92.5; overlap-inflation 199.00×)
- test rows: 0
- sample uniqueness weighting: `on` (horizon_days=100)
- positive prevalence (train): 0.418
- positive prevalence (eval): 0.398

## Iteration history

| iter | n_features | train Brier | val Brier | gap | rationale | inner_stop |
|---|---|---|---|---|---|---|
| 0 | 279 | 0.2258 | 0.2444 | 0.0186 | iteration 0 — full feature pool, default HPs :: algorithmic fallback: kept 33/27 |  |
| 1 | 33 | 0.2301 | 0.2434 | 0.0133 | iteration 1 from FS+HP callback :: algorithmic fallback: kept 20/33 features |  |
| 2 | 20 | 0.2306 | 0.2440 | 0.0134 | iteration 2 from FS+HP callback :: inner_stop=plateau | plateau |

## Final checkpoint

- best iteration: 1
- iterations run: 3
- inner stop signal: `plateau`

## Calibration

- method requested: `conditional_isotonic`
- decision: `native`
- Spiegelhalter Z: 1.972
- Spiegelhalter p: 0.0486

![reliability](figs/reliability_diagram.png)

## Headline metrics

| segment | Brier | base-rate Brier | improvement | LogLoss | AUC |
|---|---|---|---|---|---|
| eval | 0.2456 | 0.2396 | -0.0060 | 0.6843 | 0.4880 |

## Per-experiment verdict (algorithmic readout)

Calibration: native-passable (|z|=1.97<2). Brier vs base-rate: -0.0060 (worse than baseline).

> NOTE: this verdict is generated from the metrics by a simple rule (see ``report._algorithmic_verdict``); it is NOT an automated pass/fail gate. The user reads the artifact and decides whether the cell ships.
