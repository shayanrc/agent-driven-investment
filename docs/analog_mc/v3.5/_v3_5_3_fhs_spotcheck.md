# V3.5.3 — GARCH-FHS spot-check vs v2.4

FHS = fit GARCH(1,1) on causal returns → simulate 1000 σ-paths → i.i.d. draw 1000 residual sequences from `r_t/σ_t` pool. Compare 50/90 band coverage and terminal band widths against v2.4.

## Coverage (days of 60 inside band)

| Anchor | Group | Realized 60d | v2.4 50/60 | **v2.4 90/60** | FHS 50/60 | **FHS 90/60** | Δ90 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2010-04-23 | failure | -10.4% | 3 | **27** | 2 | **9** | -18 |
| 2001-10-02 | failure | +38.6% | 6 | **44** | 0 | **24** | -20 |
| 2018-10-08 | failure | -12.7% | 4 | **31** | 3 | **30** | -1 |
| 2020-03-16 | failure | +43.8% | 6 | **38** | 59 | **60** | +22 |
| 2026-02-19 | failure | +17.5% | 32 | **41** | 29 | **60** | +19 |
| 1991-03-26 | control | -1.6% | 48 | **60** | 48 | **60** | +0 |
| 2010-11-10 | control | +7.4% | 47 | **56** | 49 | **56** | +0 |
| 2012-03-14 | control | -5.5% | 28 | **55** | 33 | **60** | +5 |
| 2025-07-02 | control | +8.2% | 59 | **60** | 58 | **60** | +0 |
| 2017-06-01 | control | +0.1% | 43 | **60** | 44 | **60** | +0 |

## Terminal (day-60) band widths (price-relative %)

| Anchor | v2.4 50% | v2.4 90% | FHS 50% | FHS 90% | 90% ratio FHS/v24 |
|---|---:|---:|---:|---:|---:|
| 2010-04-23 | 9.3% | 25.9% | 10.9% | 27.0% | 1.04× |
| 2001-10-02 | 43.0% | 118.0% | 25.4% | 64.0% | 0.54× |
| 2018-10-08 | 8.5% | 21.4% | 11.7% | 30.1% | 1.41× |
| 2020-03-16 | 40.9% | 80.5% | 59.5% | 162.5% | 2.02× |
| 2026-02-19 | 10.5% | 29.4% | 13.0% | 32.6% | 1.11× |
| 1991-03-26 | 15.5% | 46.9% | 14.4% | 34.2% | 0.73× |
| 2010-11-10 | 9.4% | 25.4% | 10.9% | 28.7% | 1.13× |
| 2012-03-14 | 10.7% | 23.7% | 12.9% | 31.9% | 1.35× |
| 2025-07-02 | 12.9% | 34.4% | 11.6% | 31.1% | 0.90× |
| 2017-06-01 | 7.1% | 19.4% | 10.5% | 25.8% | 1.33× |

## Verdict

- v2.4 catches **0/5** failures at 90%-band ≥45/60.
- FHS catches **2/5** failures at 90%-band ≥45/60.
- FHS produces a wider 90%-band terminal width than v2.4 in **4/5** failure anchors.

**Mixed FHS result** (2/5 caught, 4/5 wider). FHS gives partial improvement on some anchors but isn't a clear winner. A1 worth evaluating formally but not promoting ahead of B1 on this evidence alone.
