# _265 — Trailing champions refit on the date-aligned split (matched)

**Headline:** The 5 trailing-split models that the backtests track (the deployed sp500
champions + the nasdaq cell5 / ndx40 models) were re-fit with their **exact recipe**
(backend + HP overrides + pruned feature set) on the **date-aligned** split
(`train_start 2019-01-01`, test **2024-07-26 → 2024-12-16**) — only the split changed.
**Top-1 R-Precision drops sharply on the 2024-H2 window for every cell** (e.g.
sp500 +50%/50d @1 0.640 → 0.110; ndx +10%/50d 0.800 → 0.28/0.25), while wide-K is
mixed — ndx +40%/50d is actually *stronger* at K≥10 on the aligned window (@20 0.563 →
0.926). This is "same recipe, different OOS regime," not strictly a worse model, but it
reaffirms that the trailing champions' strong top-of-book skill is **window-dependent**
(the recent trailing windows are favourable), consistent with the macro-program lesson.

## Setup

- **Matched refit** (`backend.hp_starting` = the champion's resolved HP, `features` =
  its exact set, `fs_hp_loop.max_iterations: 1` → single fit). Only `split.mode`
  changes (trailing → `date_aligned`). Isolates the split effect — the matched-HP A/B
  methodology of `_262`.
- Generator: `scripts/gbdt/gen_aligned_champion_specs.py` (provenance + reproducible).
  Pruned nasdaq cells build `candidates: all` and **`exclude` the dropped columns** —
  the feature builder takes family tokens / `all`, not individual column names, so a
  pinned column list must go through `exclude` (the complement: 249 for ndx40, 89 for
  cell5).
- 5 models: sp500 +20%/25d & +50%/50d (champion = all 279 feat, mcw=10 → identical
  recipe to the existing `_daswbase`), ndx +40%/50d (mix: 30 feat, mcw5/cs0.5/γ0.5),
  ndx +10%/50d ×2 (reval: 190 feat, depth-2/η0.1/500-tree; bacc: 190 feat,
  depth-2/η0.1/200-tree). All xgboost, conditional_isotonic. `--snapshot-end 2026-06-20`.
- **Caveat:** the nasdaq cache had 11 tickers stale (>20d), so the ndx refits train on
  **92 tickers** (not the full 100). sp500 uses 486.

## Results — trailing champion vs date-aligned twin (raw R-Precision@K + base rate)

| cell | split | test_end | base_rate | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|---|---|---|
| sp500 +20%/25d | trailing | 2026-04-17 | 0.089 | 0.413 | 0.431 | 0.405 | 0.403 | 0.360 |
| sp500 +20%/25d | date-aligned | 2024-12-16 | 0.040 | 0.170 | 0.257 | 0.228 | 0.221 | 0.286 |
| sp500 +50%/50d | trailing | 2026-03-12 | 0.026 | 0.640 | 0.427 | 0.320 | 0.346 | 0.428 |
| sp500 +50%/50d | date-aligned | 2024-12-16 | 0.010 | 0.110 | 0.084 | 0.126 | 0.195 | 0.260 |
| ndx +40%/50d (mix) | trailing | 2026-03-12 | 0.055 | 0.667 | 0.490 | 0.467 | 0.522 | 0.563 |
| ndx +40%/50d (mix) | date-aligned | 2024-12-16 | 0.037 | 0.333 | 0.313 | 0.367 | 0.735 | 0.926 |
| ndx +10%/50d (reval) | trailing | 2025-12-26 | 0.338 | 0.800 | 0.756 | 0.716 | 0.618 | 0.663 |
| ndx +10%/50d (reval) | date-aligned | 2024-12-16 | 0.371 | 0.280 | 0.207 | 0.304 | 0.321 | 0.364 |
| ndx +10%/50d (bacc) | trailing | 2026-03-12 | 0.265 | 0.800 | 0.638 | 0.586 | 0.513 | 0.503 |
| ndx +10%/50d (bacc) | date-aligned | 2024-12-16 | 0.371 | 0.250 | 0.250 | 0.270 | 0.278 | 0.363 |

@1 change (trailing → date-aligned): sp500/20 −0.24, sp500/50 −0.53, ndx40 −0.33,
ndx10-reval −0.52, ndx10-bacc −0.55 — uniformly large at the top. ndx +40%/50d inverts
at wide K (@20 +0.36).

## Reading

- The two windows differ in **regime and base rate** (the date-aligned 2024-H2 test has
  a *lower* positive prevalence for the sp500 cells, *higher* for ndx +10%), so the
  cell-for-cell deltas aren't apples-to-apples — but that's the point: a model whose
  top-1 skill is real out-of-sample should not collapse this much on an independent
  window. It does, at K=1, for all five.
- **ndx +40%/50d** keeps (and improves) wide-K precision on the aligned window — its
  edge is broader-tail, less top-1-concentrated, and travels better across regimes.
- sp500 champmatch reproduces the existing `_daswbase` recipe (all feats, mcw=10); the
  rows match within float precision — a useful consistency check.

## Verdict

The trailing champions' headline top-1 numbers (which the backtests and
`/daily-predictions` showcase) are **partly window-specific** — on the date-aligned
2024-H2 window the same recipes are much weaker at K=1. This does not retract the
deployed champions (they remain the validated trailing models), but it argues for
judging them on **multiple independent windows**, and it sets up the next step:
re-scoring the registry's best cells on **true post-test-set OOS** data (`_266`).

## Artifacts

- Registry: 5 `*_aligned_*match` rows in `results/gbdt/data/r_precision_at_k.csv`
- Specs: `configs/gbdt/experiments/*_aligned_{champmatch,mixmatch,revalmatch,baccmatch}.yaml`
- Generator: `scripts/gbdt/gen_aligned_champion_specs.py`
