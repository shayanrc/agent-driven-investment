# _283 — calendar2 (F21) matched A/B: preliminary verdict

**Plan:** `docs/gbdt/V1.9_calendar2_features_plan.md`. **Feature:** F21 calendar2 family
(opt-in tokens `all_calendar2`, `all_fundamentals_calendar2`). **Question:** does adding
month-of-quarter + quarter-of-year (cyclically encoded, 4 cols) improve top-K skill over
the technical baseline when run faithfully through the calibrated / uniqueness-weighted
runner?

## Verdict (one window — preliminary)

**Not a clean win, and top-of-book NEGATIVE. NOT promoted.** On this single window
(test 2024-07-26 → 2024-12-16), +cal2 **degrades R-Precision@1 on all three cells**
(−0.125, −0.040, −0.130) while modestly *helping* the wider book (R-p@10 up on all three;
R-p@20 up on two). AUC is roughly flat (+0.027, +0.007, −0.001). So F21 **redistributes**
skill away from the very top pick toward the mid-book rather than adding it — the opposite
of what a concentrated top-K strategy wants.

This does not reproduce the standalone importance signal (`moq_sin` ranked 16/297 by gain,
+cal2 lifted test R-p@K on a representative window). Under the faithful calibrated/weighted
runner the top-tail effect flips sign — consistent with the pre-branch note that the
standalone fit did not reproduce the runner's model. **Per the F17-macro (`_264`) /
F18-long-horizon (`_278`) / F19 (`_279`) / F20-VWAP (`_281`) precedent, one window is not a
verdict** — but here even the single window is unfavorable at the top of book, so the prior
going into a second window is weak.

## Setup

Matched **single-fit** A/B (default HP, `max_iterations: 1`, `callback_mode: default`,
date_aligned `train_start 2019-01-01`, snapshot 2026-07-06 — directly comparable to the
`_279`/`_281` window), **xgboost**. Two arms per cell differing ONLY in
`features.candidates` — `all` (base, 279 cols) vs `all_calendar2` (283 cols) — so the
per-cell delta is a clean read of the F21 contribution (the
`[[project-gbdt-macro-features-f17]]` matched-HP rule; never compare arms via the auto-loop,
which tunes each arm independently). Three representative **nasdaq100** cells (the VWAP-
lattice cells). Panel 647,669 rows / 92 tickers; all metrics are the **test** segment.
Tooling: `scripts/gbdt/{gen_cal2_ab_specs.py, run_cal2_ab.sh, aggregate_cal2_ab.py}`;
specs under `configs/gbdt/experiments/cal2_ab/`.

## Results

Test-segment metrics (raw metric + base rate; base = `all`, cal2 = `all_calendar2`):

| cell | arm | base_rate | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| +50% / 25d | base | 0.0102 | 0.9087 | 0.4844 | 0.5599 | 0.6615 | 0.8099 | 0.8490 |
| +50% / 25d | cal2 | 0.0102 | 0.9358 | 0.3594 | 0.5625 | 0.6484 | 0.8646 | 0.9089 |
| +20% / 50d | base | 0.1700 | 0.7209 | 0.4400 | 0.4433 | 0.4380 | 0.3837 | 0.4368 |
| +20% / 50d | cal2 | 0.1700 | 0.7280 | 0.4000 | 0.4467 | 0.4300 | 0.4074 | 0.4808 |
| +40% / 200d | base | 0.1566 | 0.7482 | 0.5900 | 0.4800 | 0.4360 | 0.3891 | 0.5180 |
| +40% / 200d | cal2 | 0.1566 | 0.7475 | 0.4600 | 0.5233 | 0.4460 | 0.4031 | 0.5046 |

Delta (cal2 − base):

| cell | ΔAUC | ΔR-p@1 | ΔR-p@3 | ΔR-p@5 | ΔR-p@10 | ΔR-p@20 |
|---|--:|--:|--:|--:|--:|--:|
| +50% / 25d | +0.0271 | **−0.1250** | +0.0026 | −0.0130 | +0.0547 | +0.0599 |
| +20% / 50d | +0.0071 | **−0.0400** | +0.0033 | −0.0080 | +0.0237 | +0.0440 |
| +40% / 200d | −0.0008 | **−0.1300** | +0.0433 | +0.0100 | +0.0140 | −0.0134 |

## Reading

- **R-p@1 down on 3/3** (−0.04 to −0.13) is the headline. The single top pick — what the
  deployed `TopKDailyKellyLabelExit` concentrates on most — gets *worse* with F21. The one
  large AUC move (+50%/25d, +0.027) coincides with the largest R-p@1 *drop* (−0.125):
  better bulk ranking, flatter top tail — the same "bulk-metric gain masks top-book
  regression" pattern flagged for L2-family regularization in `_276` and the anti-AUC
  val_brier lesson.
- **Wider book modestly up** (R-p@10 +3/3; R-p@20 +2/3). If anything F21 is a mid-book
  feature, not a top-1 feature. But the mid-book gains are within the noise band seen for
  every rejected opt-in family (F17/F20), and they do not offset the top-1 loss for a
  concentrated strategy.
- **No structural signature.** A real calendar effect would light up the *same* K-band
  consistently; instead the sign pattern is ragged across cells (e.g. R-p@20 +0.060 /
  +0.044 / −0.013).

## Status

**F21 NOT promoted; no champion or `/daily-predictions` change.** F21 stays an opt-in token,
byte-identical to `all` when unused (the default `all` pool is unchanged at 279 cols;
`all_calendar2` = 283). Artifacts: 6 `metrics.json` under
`results/gbdt/experiments/nasdaq100_up_*_cal2ab_{base,cal2}`; 6 rows appended to
`results/gbdt/data/r_precision_at_k.csv`. **Follow-up (the actual gate):** an independent
date-aligned second window — but given R-p@1 fell on 3/3 here, a champion swap is not on the
table without a clear top-of-book reversal. Related: `_264` (macro non-replication),
`_278`/`_279` (F18/F19 long-horizon fails), `_281` (VWAP non-replication).
