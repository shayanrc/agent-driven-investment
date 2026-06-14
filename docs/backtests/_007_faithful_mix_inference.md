# _007: faithful inference for `_mix` cells + ndx40 rolling validation

## TL;DR

`_006` couldn't roll the strong rare-event cells because inference on `_mix`
cells **failed the faithfulness self-check** (ndx40 max_abs_diff 3.3e-2). Root
cause, traced exactly: a provider **gap-fill during the cache refresh inserted a
single historical bar — `NASDAQ:AZN` on `2026-02-09`** — that the model never
trained on. For path-dependent features (the stateful F16 `*_outside_band`
running counts + cross-sectional ranks/z-scores), that one row perturbs every
downstream value, shifting p_raw by 3.3e-2. The plain cell-5 models matched
(1.5e-8) because their features are path-independent.

**Fix:** align the inference panel to the model's training row-set — drop
post-training gap-fill rows (recovered from the cell's cached universe
feature-matrix index, whose rows ARE the training panel). Dropping AZN 2026-02-09
restores **max_abs_diff = 1.36e-08** (exact reproduction). The strict self-check
is retained as the validator; it now passes.

**Payoff — ndx40_mix is now rollable, and it shows a positive rolling edge** (the
first rare-event cell to survive the `_006` gate):

| ndx40_mix (rank/equal, full extended OOS 2025-06-05→2026-06-12) | value |
|---|---|
| Strategy total return | **+121.7%** (max-DD −9.6%) |
| NDX buy-and-hold | +37.5% |
| Rolling 50-day windows beating NDX | **57%** (24 / 42) |
| Median 50-day excess | **+6.6%** (p25 −4.4%, p75 +29.1%, max +47.9%) |
| Entries over full OOS | 11 |

Contrast the cell-5 family (`_006`): 38–46% of windows beat NDX, median excess
−0.6%/−4.4% (no edge). **ndx40_mix is genuinely different** — a positive,
right-skewed rolling excess. But see caveats: only **11 entries**, one cell, one
(bull-market) OOS window.

## Root cause (traced)

1. `_006` self-check: ndx40 inference reproduced test.csv to only 3.3e-2 (>1e-4 → abort).
2. The cell's **cached** universe feature-matrix reproduces test.csv to **1.36e-8** — so
   it IS the training matrix; my *rebuild* was the divergent one.
3. Divergent features were all standardized/ranked/banded
   (`vol_xs_zscore_*`, `realized_vol_zscore_200`, `*_outside_band_*`, `dollar_move_xs_rank`);
   raw levels matched.
4. Panel diff: my refreshed panel had **645098** historical rows vs training's **645097** —
   exactly **one extra row: `NASDAQ:AZN` `2026-02-09`** (a gap-fill the provider backfilled
   during the refresh). `_signed_days_outside_band_one` is a running count along each ticker's
   series, so an inserted row shifts all subsequent values; cross-sectional features at that
   date shift too.
5. Dropping that one row → self-check **1.36e-8**. Confirmed.

## Fix

`infer_fresh_predictions.py`: `_training_panel_index()` locates the cell's cached
universe feature-matrix (same universe, index ⊇ the cell's test rows; smallest snapshot
covering them = the training matrix), reads its index cheaply (index-only parquet read).
`build_scores(..., align_panel=True)` (default) drops panel rows dated ≤ the training
snapshot that are **absent from the training index** (the gap-fills), then builds features —
faithful historical reproduction + a consistent fresh extension. Fresh rows (> snapshot) are
untouched. `--no-align` escape hatch; if no training matrix is found, it proceeds on the raw
panel and the strict self-check still guards. Generalized `universe` to read from the cell
spec (was hardcoded nasdaq100), so sp500/russell1000 cells work once their caches exist.

**Regression:** the plain cell-5 model (`revalidation_regen`) still self-checks **PASSED
(faithful, ≤1e-4)** after the change — no regression (its test rows predate the AZN
backfill, so its reproduction is unaffected either way).

## Methodology

ndx40 rolled exactly as `_006`: one full-OOS rank/equal (c=1.0) back-test over
test.csv + faithful fresh predictions, then rolling 50-day excess-return distribution
vs NDX at stride 5. Calibrator fit on the cell's VAL.

## Caveats

- **C1: 11 entries.** The +40%/50d/dd20 label fires rarely; the full-OOS +121.7% rests on
  ~11 picks hitting large (+40% target) moves in a tech bull run. Small N, high variance —
  the wide rolling spread (p25 −4.4% to p75 +29.1%) reflects this. **Not an alpha claim** on
  one cell / one window.
- **C2: overlapping windows** (stride 5 < 50) → autocorrelated; 57%/median +6.6% are
  descriptive, not iid-significant.
- **C3: the fix depends on the cached training matrix being present.** If a cell's cache was
  evicted, alignment can't run and inference falls back to the (possibly divergent) raw panel
  — the self-check then re-aborts, correctly. Re-running the cell repopulates the cache.
- **C4: bull-market tailwind**; C5: zero costs / DD-not-bounded — as prior memos.

## Reproducibility

- Branch `backtests-v12-faithful-mix-inference`.
- `uv run python -m scripts.backtests.infer_fresh_predictions --cell <_mix cell> --out <fresh.csv>`
  (alignment automatic; prints the dropped gap-fill rows).
- `uv run python -m scripts.backtests.run_rolling_validation --cell <cell> --fresh <fresh.csv> --out <dir> --name <n>`
- ndx40 artifacts under `results/backtests/_007_mix_fresh/`.

## Open questions / follow-ups

- **Roll the remaining strong cells**: sp500 +50% (AUC 0.90), sp500 +20%, russell1000 +50% —
  needs per-universe cache refresh + (now-generalized) inference; then roll.
- **More OOS for ndx40**: 11 entries is thin; a longer/rolling-forward OOS as the cache ages
  would tighten the distribution.
- **Persist panel snapshots** (or store the training row-set in the artifact) so faithful
  inference doesn't depend on the live feature cache surviving.
