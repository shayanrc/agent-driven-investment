# gbdt — Sample-uniqueness weighting (LdP §4.4)

**Default behavior**: gbdt experiments apply López de Prado §4.4 sample-uniqueness weights to every training/scoring row. Opt-out via `target.uniqueness_weighting: false` in the spec — reproduces the pre-PR overlap-naive baseline.

## Why this exists

gbdt labels every `(ticker, date)` row with "did this ticker breach +X% within H days while keeping drawdown ≤ Y%?". The `H`-day forward window overlaps across consecutive entry dates — adjacent rows share `H − 1` of their `H` future bars and the same outcome event labels them identically.

Without correction:
1. **Effective sample size inflated** — model sees ~`2H` as many correlated examples as it should.
2. **Train/val/eval leakage** — `H`-day windows straddle fold boundaries, putting one event in multiple folds.
3. **Loss-weighted bias** — periods with overlapping positives dominate the gradient.

Sweep exp #1 (nasdaq100 +10% / 100d / dd5%) made this concrete: prevalence 42.4% in the training panel vs 19.7% in non-overlapping EDA — a 2.15× inflation that vanished after uniqueness weighting.

## What it does

`src/gbdt/uniqueness.py::compute_uniqueness_weights(panel, horizon)` returns a per-row weight in `(0, 1]`:

- Interior row (positions `[H-1, N-H]` in a contiguous single-ticker series): weight `1 / (2H − 1)`.
- Edge rows: weight `1 / (overlap_count + 1)` where `overlap_count = min(i, H-1) + min(N-1-i, H-1)`.
- Two tickers' rows never overlap (intra-ticker only).
- `horizon = 1` is a no-op (forward windows of length 1 share no bars).

`Σw ≈ N / (2H − 1)` is the natural "number of independent forward events." Kish's ESS `(Σw)² / Σw²` is reported separately and measures the variance penalty from non-uniform weighting (insensitive to uniform scaling).

## Where it integrates

- `gbdt.targets.build_target` is unchanged — labels are still as-before.
- `__main__.run_experiment` computes weights once after target build, passes them through `walk_forward_train(sample_weights=...)`.
- `gbdt.train._gather_segment` carves weights in parallel with X/y per fold.
- `gbdt.model.GBDTModel.fit` accepts `train_weight=` / `val_weight=` → CatBoost `Pool(weight=...)`. Early-stopping signal is the weighted val Brier.
- `gbdt.diagnostics.build_diagnostic_bundle` reports weighted train/val Brier + weighted Spiegelhalter Z + weighted val prevalence + per-segment ESS.
- `__main__._compute_headline` emits weighted `brier / log_loss / roc_auc` and keeps unweighted twins (`*_unweighted`) for cross-check.
- `metrics.json::sample_uniqueness.effective_sample_size_per_fold` carries `(ess_kish, sum_weights, n_rows, overlap_inflation_ratio)` per segment.
- `predictions/<seg>.csv` gains a `sample_weight` column.

## Affected experiments

Any sweep that ran pre-PR has biased prevalence + headline metrics. Expect:
- `headline_eval.brier` (now weighted) to differ from previously-reported numbers.
- Top features may shift — index-vol features were artificially boosted by overlap noise.
- `effective_sample_size_per_fold` is the canonical "how much info is here" number.

The first re-runs in scope are: Exp 1 (the original pilot) and sweep #1 (nasdaq100 +10%/100d/dd5%, the bug-surfacing cell).

## Tests

`tests/gbdt/test_uniqueness.py` — 18 tests covering the weight formula, ESS, weighted Brier/AUC/Spiegelhalter, multi-ticker independence, and the `horizon=1` no-op.

## References

- López de Prado, *Advances in Financial Machine Learning*, §4.4 (uniqueness via co-occurrence) + §4.5 (closed-form approximation).
- Original report of the bias: sweep exp #1 metrics + EDA in `docs/gbdt/_v0_opportunity_scan*.md`.
- Implementation PR: this branch (`gbdt-uniqueness-weights`).
