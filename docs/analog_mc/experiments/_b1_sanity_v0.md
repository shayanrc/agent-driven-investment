# B1 sanity v0 — per-failure isolated comparison

Direct per-anchor comparison of B1-on vs B1-off, holding everything else fixed (same fold-selected weights/n_eff from the canonical v2.4 run). Search-time effect is NOT included — this is the isolated correction-magnitude diagnostic from step 7 of the B1 build order.

Canonical run: `runs/analog_mc/20260520T045525Z` · 5 failure + 5 control anchors from V3.5_RESULTS.

## B1 correction magnitudes

| Anchor | Group | Realized | Matcher E[60d] | B1 pred | Correction | Per-day | Clamp |
|---|---|---:|---:|---:|---:|---:|---|
| 2010-04-23 | failure | -10.4% | +3.5% | +1.1% | -2.35% | -3.97bp | — |
| 2001-10-02 | failure | +38.6% | +11.4% | +20.9% | +8.53% | +13.64bp | — |
| 2018-10-08 | failure | -12.7% | +2.1% | +2.1% | +0.06% | +0.11bp | — |
| 2020-03-16 | failure | +43.8% | +5.4% | +6.2% | +0.70% | +1.16bp | — |
| 2026-02-19 | failure | +17.5% | +4.7% | +7.5% | +2.67% | +4.39bp | — |
| 1991-03-26 | control | -1.6% | +7.0% | +1.0% | -5.62% | -9.64bp | — |
| 2010-11-10 | control | +7.4% | +5.2% | +4.5% | -0.66% | -1.10bp | — |
| 2012-03-14 | control | -5.5% | +2.7% | -5.2% | -7.75% | -13.44bp | — |
| 2025-07-02 | control | +8.2% | +4.3% | +1.7% | -2.53% | -4.27bp | — |
| 2017-06-01 | control | +0.1% | +5.3% | +3.7% | -1.51% | -2.54bp | — |

## Coverage and CRPS deltas

| Anchor | Group | v2.4 CRPS | B1 CRPS | Δ CRPS | Δ CRPS rel | v24 90/60 | B1 90/60 | Δ 90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2010-04-23 | failure | 0.07027 | 0.06160 | -0.00868 | -12.35% | 28 | 34 | +6 |
| 2001-10-02 | failure | 0.11689 | 0.09038 | -0.02650 | -22.67% | 39 | 51 | +12 |
| 2018-10-08 | failure | 0.06346 | 0.06391 | +0.00044 | +0.70% | 32 | 31 | -1 |
| 2020-03-16 | failure | 0.17662 | 0.18588 | +0.00926 | +5.24% | 37 | 26 | -11 |
| 2026-02-19 | failure | 0.05048 | 0.04394 | -0.00654 | -12.96% | 41 | 45 | +4 |
| 1991-03-26 | control | 0.02354 | 0.02287 | -0.00067 | -2.84% | 60 | 60 | +0 |
| 2010-11-10 | control | 0.01556 | 0.01592 | +0.00035 | +2.28% | 56 | 56 | +0 |
| 2012-03-14 | control | 0.03280 | 0.01665 | -0.01615 | -49.23% | 57 | 60 | +3 |
| 2025-07-02 | control | 0.01750 | 0.02134 | +0.00383 | +21.89% | 60 | 60 | +0 |
| 2017-06-01 | control | 0.01278 | 0.01194 | -0.00084 | -6.61% | 60 | 60 | +0 |

## Aggregate

| Group | v2.4 mean CRPS | B1 mean CRPS | Δ rel | v2.4 mean 90/60 | B1 mean 90/60 |
|---|---:|---:|---:|---:|---:|
| Failure (5) | 0.09555 | 0.08914 | -6.70% | 35.4 | 37.4 |
| Control (5) | 0.02044 | 0.01774 | -13.19% | 58.6 | 59.2 |

## Sanity verdict

- **Clamps fired**: 0/10 anchors.
- **Correction sign matches realized direction**: 6/10 anchors total, 4/5 failures.
- **Failure CRPS change (isolated)**: -6.70%.
- **Control CRPS change (isolated)**: -13.19%.

**Caveat.** This is the *isolated* effect — the matcher's weights/n_eff are held at v2.4-selected values. The full canonical B1 run re-searches weights with B1 active (decision D10), which may strengthen or weaken these numbers. This sanity output is a sign-of-life + magnitude check, not a promotion decision.
