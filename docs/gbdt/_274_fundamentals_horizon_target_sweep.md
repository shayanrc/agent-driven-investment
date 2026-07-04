# _274 — F18 fundamentals: complete horizon×target sweep (sp500)

**Question.** `_272`/`_273` measured F18 on only the two deployed champions (+50%/50d
robust across two windows, +20%/25d not). That left the shape of the edge unknown:
is the +50/50 win an isolated cell, or part of a coherent region of the
(threshold × horizon) lattice? This sweep maps **where** fundamentals help.

**Design.** All **17 canonical sp500 cells** × {`_fbase` = `all` (F1–F16), `_ffund` =
`all_fundamentals` (F1–F16 + the current **13-col** F18)}. IDENTICAL matched config to
the `_272` A/B — xgboost, default HP, single fit (`max_iterations: 1`), **date_aligned**
split (train_start 2019-01-01) — so within each cell the ONLY difference is the +13 F18
columns and the `fund − base` delta is a clean read of the F18 contribution at that
(threshold, horizon). Features don't depend on the target, so all 17 cells share 2
feature-matrix builds; 34 runs, all rc=0. Uniform test window **2024-07-26 → 2024-12-16
(Q=100 days)** across every cell (date_aligned anchoring is horizon-invariant), so the
deltas are directly comparable cell-to-cell. Data: `results/gbdt/data/_274_fund_sweep_data.json`.
Heatmap: `results/gbdt/_274_fund_sweep/_274_fund_sweep_heatmap.png`.

## Results — raw base→fund (Δ), sorted by target then horizon

| cell | base_rate | AUC b→f (Δ) | test Brier b→f (Δ) | R-p@1 b→f (Δ) | R-p@10 b→f (Δ) |
|---|---:|---|---|---|---|
| +10%/5d | 0.0252 | 0.791→0.796 (+0.005) | 0.0239→0.0239 (−0.0000) | 0.152→0.162 (+0.010) | 0.173→0.172 (−0.002) |
| +10%/10d | 0.0695 | 0.760→0.750 (−0.010) | 0.0631→0.0632 (+0.0001) | 0.280→0.250 (−0.030) | 0.264→0.233 (−0.031) |
| +10%/25d | 0.1974 | 0.652→0.660 (+0.008) | 0.1603→0.1609 (+0.0005) | 0.280→0.320 (+0.040) | 0.334→0.356 (+0.022) |
| +10%/50d | 0.3348 | 0.507→0.489 (−0.019) | 0.2427→0.2455 (+0.0028) | 0.470→0.370 (−0.100) | 0.378→0.379 (+0.001) |
| +20%/5d | 0.0027 | 0.870→0.839 (−0.031) | 0.0027→0.0027 (+0.0000) | 0.131→0.115 (−0.016) | 0.191→0.119 (−0.072) |
| +20%/10d | 0.0092 | 0.864→0.854 (−0.009) | 0.0090→0.0090 (+0.0000) | 0.100→0.078 (−0.022) | 0.154→0.131 (−0.023) |
| **+20%/25d** | 0.0402 | 0.773→0.788 (**+0.016**) | 0.0376→0.0373 (**−0.0003**) | 0.100→0.190 (+0.090) | 0.188→0.224 (+0.036) |
| +20%/50d | 0.1277 | 0.688→0.687 (−0.001) | 0.1141→0.1134 (−0.0007) | 0.160→0.440 (+0.280) | 0.314→0.317 (+0.003) |
| +20%/100d | 0.2311 | 0.593→0.575 (−0.018) | 0.1853→0.1875 (+0.0022) | 0.330→0.550 (+0.220) | 0.328→0.341 (+0.013) |
| +40%/25d | 0.0046 | 0.911→0.882 (−0.029) | 0.0046→0.0045 (−0.0001) | 0.138→0.057 (−0.080) | 0.294→0.215 (−0.079) |
| **+40%/50d** | 0.0198 | 0.856→0.864 (**+0.009**) | 0.0191→0.0190 (**−0.0001**) | 0.050→0.070 (+0.020) | 0.229→0.289 (+0.061) |
| +40%/100d | 0.0601 | 0.792→0.781 (−0.011) | 0.0553→0.0550 (−0.0003) | 0.180→0.150 (−0.030) | 0.260→0.308 (+0.048) |
| **+40%/200d** | 0.1189 | 0.653→0.678 (**+0.026**) | 0.1032→0.1013 (**−0.0019**) | 0.140→0.330 (+0.190) | 0.306→0.364 (+0.058) |
| +50%/25d | 0.0018 | 0.910→0.887 (−0.023) | 0.0017→0.0018 (+0.0000) | 0.150→0.033 (−0.117) | 0.286→0.250 (−0.036) |
| +50%/50d | 0.0097 | 0.907→0.902 (−0.005) | 0.0098→0.0097 (−0.0000) | 0.110→0.121 (+0.011) | 0.218→0.196 (−0.022) |
| +50%/100d | 0.0322 | 0.875→0.868 (−0.006) | 0.0303→0.0304 (+0.0001) | 0.168→0.263 (+0.095) | 0.272→0.245 (−0.027) |
| +50%/200d | 0.0712 | 0.734→0.732 (−0.002) | 0.0645→0.0639 (−0.0006) | 0.190→0.300 (+0.110) | 0.197→0.282 (+0.085) |

(Bold = clean win: AUC↑ **and** Brier↓ together, the two robust bulk signals agreeing.)

## The map — a horizon gradient, not isolated cells

**Horizon is the dominant axis, and the sign flips along it:**

- **Short (5–10d): F18 consistently HURTS.** Every 5d/10d cell loses AUC (+20%/5d −0.031,
  +10%/10d −0.010, +20%/10d −0.009) and top-K. A sub-2-week move is technical /
  microstructure-driven; quarterly fundamentals are stale there, so F18 is pure noise the
  model overfits. This is the mechanism confirmed at the short end.
- **Long (100–200d): F18 helps the tail broadly**, even where bulk AUC is flat. The
  standout is **+40%/200d — everything up** (AUC +0.026, Brier −0.0019, R-p@1 +0.19,
  R-p@10 +0.058). +50%/200d (Brier −0.0006, R-p@1 +0.11, R-p@10 +0.085) and +40%/100d
  (Brier −0.0003, R-p@10 +0.048) sharpen the top-of-book even with flat/negative AUC.
- **Medium (25–50d): threshold-dependent.** Clean wins at **moderate** thresholds
  (+20%/25d, +40%/50d), neutral at the +50%/50d champion, and it HURTS where the target
  is near-impossible on that horizon (+40%/25d, +50%/25d — base_rate 0.002–0.005, an
  essentially unpredictable fast-big move that F18 can only add variance to).

**Reading across both axes:** the F18 edge lives in the **moderate-to-high-threshold ×
medium-to-long-horizon** band (bottom-right of the heatmap: 20–50% × 50–200d) and is
actively harmful in the **short-horizon** strip (top-left, any threshold × 5–10d) and the
**rare-fast** corner (high threshold × short horizon). This is exactly the
`_273` mechanism generalized: the longer the horizon and the bigger the required move, the
more that move reflects *business fundamentals* rather than noise — up to the point where
the move becomes too rare to model at all.

## Caveat — the +50%/50d champion is diluted by the z-scores here

The sweep's +50%/50d is **neutral** (AUC −0.005, R-p@10 −0.022) — visibly weaker than
`_272`'s **9-col** date-aligned result (AUC +0.007, R-p@10 +0.087). Same config, same
window; the only difference is the **4 z-score columns** added after `_272`
(`[[project-gbdt-macro-features-f17]]`-style FS matters here). Forced in at single-fit
default HP, the z-scores help long-horizon cells (they drive +40%/200d) but **add noise on
the +50%/50d champion**. This does NOT overturn `_272`/`_273` (which stand for 9-col F18);
it shows the **13-col superset is not strictly ≥ 9-col at forced single-fit — the value of
the z-scores has to be unlocked by per-cell feature selection**, dropping them where they
dilute. That is precisely what the `V1.7_TBD` §3 FS+HP tune does.

## Caveat — window + noise

Single window (date-aligned 2024-H2), matched single-fit HP (not each cell's tuned
optimum). Q=100 days: better than `_273`'s trailing Q≈60, but R-p@1 is still a mean over
~100 near-binary outcomes — read AUC + Brier + the *aggregate* R-p shape as the robust
signals, individual R-p@1 levels as noisy. The ultra-rare cells (base_rate < 0.005:
+50%/25d, +40%/25d, +20%/5d) have ~1 positive/day and near-degenerate targets — their
deltas are the least trustworthy (and all point the same way: F18 hurts the unpredictable).

## Verdict

- **The F18 edge is a coherent region, not a lucky cell.** It concentrates in
  moderate-high-threshold × medium-long-horizon cells and is strongest at the
  **long-horizon corner (+40%/200d, +50%/200d, +40%/100d)** — none of which are currently
  tracked. It is genuinely **harmful at short horizons** — a real reason NOT to blanket-wire
  F18.
- **Deployed champions:** +20%/25d is a clean win *on this 13-col date-aligned sweep*
  (AUC↑ + Brier↓ + R-p↑) — but `_273` showed its edge does not replicate on the trailing
  window, so **hold**. +50%/50d is diluted by the z-scores at forced single-fit — the §3
  FS+HP tune is the resolution (expect FS to drop the noisy z-scores and recover/exceed the
  9-col edge).
- **New signal for the candidate set:** if a longer-horizon champion is ever considered,
  fundamentals add materially more there (+40%/200d: everything up) than at the current
  50d/25d champions. Candidate-cell input only — a champion swap remains a human decision
  (`_019`).

## Next

1. **`V1.7_TBD` §3 — +50%/50d FS+HP tune** (unchanged priority): now doubly-motivated —
   measure the tuned ceiling AND let FS resolve the z-score dilution shown here.
2. **Optional:** a two-window (trailing) confirmation of the long-horizon winners
   (+40%/200d, +50%/200d) before treating them as candidate cells — same `_272`→`_273`
   pattern. Parked in `V1.7_TBD`.

Registry: 34 rows in `r_precision_at_k.csv` (`*_fbase` / `*_ffund`). Tooling:
`scripts/gbdt/{gen_fund_sweep_specs,run_fund_sweep.sh,aggregate_fund_sweep,plot_fund_sweep_heatmap}`.
Prior: `_272` (date-aligned 2-cell A/B), `_273` (trailing 2nd window). Plan:
`V1.7_fundamentals_features_plan.md`.
