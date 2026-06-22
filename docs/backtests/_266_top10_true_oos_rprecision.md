# _266 — Top-10 (by R-p@3) re-scored on true post-test-set OOS

**Headline:** For the 10 highest R-Precision@3 cells in the registry, each model was
re-scored on data **strictly after its own published test window** (forward bars with
complete labels) — the strict generalization check. Two results dominate: (1) **only
4/10 are reproducible** — the 5 highest-ranked *aligned-200d* cells (russell1000 ×3,
sp500 ×2) fail infer's faithfulness self-check (their training feature-matrix cache is
gone, so rebuilt features diverge from the saved `test.csv` by 0.03–0.29 ≫ 1e-4) and one
cell is absent from the checkout; (2) where reproducible (the nasdaq 50d cells),
**true-OOS R-p is positive but well below the in-test number, and wide-K collapses to
the base rate** (cell5 @10 0.285 ≈ base 0.284). The two 25d cells look strong but sit on
**15 labelable days** — too noisy to trust.

## Setup

- `scripts/backtests/true_oos_rprecision.py`: per cell → `infer_fresh_predictions`
  (now backend-dispatched: XGBoost `.ubj` / CatBoost `.cbm`) scores dates after
  `test_end`; `gbdt.targets.build_target` attaches realized labels (NaN past the
  complete-label boundary); R-Precision@K on the labelable OOS days only, canonical
  `min(K, R_q)` denominator + (p desc, ticker asc) tie-break.
- Top-10 by R-Precision@3 from the 164-row registry; `--end 2026-06-20`.
- A model's true-OOS window = `(test_end, today − horizon trading days]`. 200d cells
  (test_end 2024-10-03) would have ~11 months labelable; 50d/25d cells only weeks.

## Results

| cell | split | true-OOS window | days | base_rate | test R-p@3 | OOS @1 | @3 | @5 | @10 |
|---|---|---|---|---|---|---|---|---|---|
| nasdaq cell5 reval_regen (50d) | trailing | 2025-12-29 → 2026-04-02 | 66 | 0.284 | 0.756 | 0.606 | 0.495 | 0.392 | 0.285 |
| nasdaq cell5b b_acceptance (50d) | trailing | 2026-03-13 → 2026-04-02 | 15 | 0.356 | 0.638 | 0.667 | 0.422 | 0.467 | 0.300 |
| nasdaq 10/25d b_acceptance | trailing | 2026-04-20 → 2026-05-08 | 15 | 0.328 | 0.526 | 0.933 | 0.844 | 0.707 | 0.620 |
| nasdaq 10/25d | trailing | 2026-04-20 → 2026-05-08 | 15 | 0.328 | 0.526 | 0.933 | 0.867 | 0.693 | 0.680 |
| russell1000 +50%/200d aligned | aligned | — | — | — | 0.642 | self-check abort (not reproducible) | | | |
| russell1000 +40%/200d aligned | aligned | — | — | — | 0.570 | self-check abort | | | |
| sp500 +50%/200d aligned | aligned | — | — | — | 0.569 | self-check abort | | | |
| russell1000 +50%/200d aligned_agent_v14p1 | aligned | — | — | — | 0.559 | self-check abort | | | |
| sp500 +40%/200d aligned | aligned | — | — | — | 0.556 | self-check abort | | | |
| nasdaq cell5 reval (non-regen) | trailing | — | — | — | 0.538 | absent from main checkout | | | |

## Reading

- **Reproducibility gap (the load-bearing finding).** The aligned-200d sweep cells were
  trained against a per-universe feature-matrix cache that no longer exists on disk
  (built in transient worktrees). `infer_fresh_predictions` rebuilds features on the raw
  panel and **aborts** when the reproduced `p_raw` can't match the saved `test.csv` — the
  right call (better no score than an untrustworthy one). So the registry's *best* cells
  could not be validated forward at all. This is a tooling/artifact-persistence
  limitation, not a model verdict.
- **For the reproducible 50d cells, true-OOS skill is real but lower than in-test.**
  cell5 OOS @3 0.495 vs its in-test 0.756 (still 1.74× the 0.284 OOS base), but @10 0.285
  is essentially the base rate — **no wide-K edge survives** into the forward window. The
  top-of-book (@1 ≈ 0.6) holds up better than the body.
- **25d cells: high but unreliable.** @1 0.93 on only 15 labelable days (≈3 weeks) — a
  handful of lucky days dominate; not a stable estimate.

## Verdict

In-test R-Precision **overstates** forward-OOS skill for the reproducible cells (the
edge concentrates at the very top and decays to the base rate by @10), and the
highest-ranked aligned cells **can't be forward-validated** until their training feature
caches are reconstructed. Recommendation: **persist the per-universe feature-matrix cache
for any championed/deployed cell** so forward-OOS re-scoring stays possible, and read the
registry's in-test R-p as optimistic relative to true forward performance.

## Artifacts

- Data: `results/backtests/data/_266_true_oos_data.json` (all 10 cells, statuses + R-p@K)
- Tooling: `scripts/backtests/true_oos_rprecision.py`;
  `scripts/backtests/infer_fresh_predictions.py` (CatBoost dispatch)
- Builds on `_265` (date-aligned matched refits of the trailing champions)
