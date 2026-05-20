# V3.5.1 — per-failure weight inspection

Canonical run: `runs/analog_mc/20260520T045525Z` (76 folds, 66×5 weight grid).

Weights are `[w_z20, w_z50, w_z200]` (short, medium, long horizon).

## Cross-fold reference

| Stat | w_z20 | w_z50 | w_z200 | n_eff |
|---|---:|---:|---:|---:|
| median | 0.100 | 0.100 | 0.151 | 150.0 |
| mean | 0.310 | 0.287 | 0.403 | 132.4 |

Folds with **w_z20 > 0.5** (short-horizon dominant): **21/76**.
Folds with **w_z20 < 0.2** (long-horizon dominant): **44/76**.

## Failure anchors

| Anchor | Fold | Weights `[w20,w50,w200]` | n_eff | val_crps | test_crps | realized 60d |
|---|---:|---|---:|---:|---:|---:|
| 2010-04-23 | 42 | [0.900, 0.100, 0.000] | 150 | 0.04965 | 0.03759 | -10.4% |
| 2001-10-02 | 24 | [0.000, 0.100, 0.900] | 50 | 0.15238 | 0.10850 | +38.6% |
| 2018-10-08 | 60 | [0.090, 0.675, 0.235] | 150 | 0.03286 | 0.04193 | -12.7% |
| 2020-03-16 | 63 | [0.000, 0.000, 1.000] | 150 | 0.07541 | 0.07333 | +43.8% |
| 2026-02-19 | 75 | [0.977, 0.000, 0.023] | 150 | 0.01799 | 0.02306 | +17.5% |

## Control anchors

| Anchor | Fold | Weights `[w20,w50,w200]` | n_eff | val_crps | test_crps | realized 60d |
|---|---:|---|---:|---:|---:|---:|
| 1991-03-26 | 2 | [0.408, 0.001, 0.591] | 150 | 0.06083 | 0.02971 | -1.6% |
| 2010-11-10 | 43 | [0.883, 0.000, 0.117] | 150 | 0.04017 | 0.02793 | +7.4% |
| 2012-03-14 | 46 | [0.000, 0.000, 1.000] | 150 | 0.03147 | 0.02971 | -5.5% |
| 2025-07-02 | 74 | [0.994, 0.000, 0.006] | 150 | 0.04285 | 0.02179 | +8.2% |
| 2017-06-01 | 57 | [0.000, 1.000, 0.000] | 150 | 0.02241 | 0.01738 | +0.1% |

## Distribution stats

| | failure | control | cross-fold |
|---|---:|---:|---:|
| w_z20 mean | 0.394 | 0.457 | 0.310 |
| w_z20 median | 0.090 | 0.408 | 0.100 |
| w_z200 mean | 0.431 | 0.343 | 0.403 |

## Verdict

**Weights heterogeneous across failures.** No single tuning regime explains all 5 misses — this isn't a search-myopia problem in isolation. **Non-finding for v4 reshape — does not change priorities.** Proceed to V3.5.2.
