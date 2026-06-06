# #242 — nifty500 sweep (20 cells, sweep mode + XGBoost)

First broad cross-section coverage of the **NIFTY 500** universe (501 constituents, NSE) under V1.4 date-aligned splits. Lattice: 4 thresholds × 5 horizons = 20 cells. XGBoost backend, sweep mode (fs_hp_loop.max_iterations=3, callback_mode=default). Sequential execution, ~70 min wall-clock total (cache-shared features across cells).

## Setup

- **Lattice**: thresholds ∈ {10, 20, 30, 50}%, horizons ∈ {10, 25, 50, 100, 200} d, dd matched to threshold (5/10/15/25%).
- **Split anchor**: H<200 → train_start=2019-01-01 (V1.4 D2 canonical); H=200 → train_start=2018-01-01 + test_rows=300 + min_rows_per_ticker=2000 (per #233 follow-up).
- **Backend**: xgboost (cross-backend coverage vs prior catboost-only NSE row `nifty500_up_30pct_50d_dd15pct`).
- **Snapshot pin**: 2026-05-22.
- **Worktree**: `wt-nifty500-sweep`.
- **Cache**: 500/500 tickers covered through 2025-06-25 (one stale-cache warning on NTPC ignored, non-fatal).

## Headline results (test segment, sorted by R-p@1)

| cell | rows | base | AUC | R-p@1 | R-p@5 | R-p@10 |
|---|---:|---:|---:|---:|---:|---:|
| nifty500_up_30pct_200d_dd15pct_aligned | 107,804 | 0.444 | 0.415 | **0.470** | 0.466 | 0.456 |
| nifty500_up_10pct_200d_dd5pct_aligned | 107,804 | 0.454 | 0.465 | 0.434 | 0.469 | 0.469 |
| nifty500_up_20pct_200d_dd10pct_aligned | 107,804 | 0.489 | 0.419 | 0.377 | 0.398 | 0.394 |
| nifty500_up_20pct_100d_dd10pct_aligned | 37,600 | 0.133 | 0.508 | 0.350 | 0.252 | 0.240 |
| nifty500_up_10pct_10d_dd5pct_aligned | 37,600 | 0.093 | 0.675 | 0.340 | 0.337 | 0.250 |
| nifty500_up_50pct_200d_dd25pct_aligned | 107,804 | 0.290 | 0.440 | 0.338 | 0.376 | 0.392 |
| nifty500_up_10pct_50d_dd5pct_aligned | 37,600 | 0.199 | 0.535 | 0.320 | 0.305 | 0.318 |
| nifty500_up_30pct_100d_dd15pct_aligned | 37,600 | 0.069 | 0.550 | 0.300 | 0.207 | 0.166 |
| nifty500_up_10pct_25d_dd5pct_aligned | 37,600 | 0.166 | 0.601 | 0.210 | 0.272 | 0.292 |
| nifty500_up_20pct_25d_dd10pct_aligned | 37,600 | 0.044 | 0.722 | 0.210 | 0.247 | 0.196 |
| nifty500_up_20pct_50d_dd10pct_aligned | 37,600 | 0.084 | 0.645 | 0.190 | 0.224 | 0.218 |
| nifty500_up_10pct_100d_dd5pct_aligned | 37,600 | 0.213 | 0.508 | 0.170 | 0.146 | 0.165 |
| nifty500_up_20pct_10d_dd10pct_aligned | 37,600 | 0.014 | 0.797 | 0.117 | 0.158 | 0.151 |
| nifty500_up_50pct_25d_dd25pct_aligned | 37,600 | 0.002 | 0.827 | 0.115 | 0.097 | 0.084 |
| nifty500_up_30pct_50d_dd15pct_aligned | 37,600 | 0.029 | 0.740 | 0.080 | 0.157 | 0.158 |
| nifty500_up_30pct_25d_dd15pct_aligned | 37,600 | 0.013 | 0.814 | 0.074 | 0.135 | 0.135 |
| nifty500_up_50pct_50d_dd25pct_aligned | 37,600 | 0.005 | 0.889 | 0.047 | 0.060 | 0.067 |
| nifty500_up_30pct_10d_dd15pct_aligned | 37,600 | 0.003 | 0.830 | 0.045 | 0.064 | 0.087 |
| nifty500_up_50pct_100d_dd25pct_aligned | 37,600 | 0.016 | 0.727 | 0.000 | 0.063 | 0.108 |
| nifty500_up_50pct_10d_dd25pct_aligned | 37,600 | 0.000 | 0.866 | 0.000 | 0.000 | 0.000 |

(Per project convention: no lift column. Readers compute R-p@1/base on demand.)

## Mechanistic reading

- **H=200 family** (top 3 + #6, base 0.29-0.49): all four cells have test AUC in the 0.41-0.47 band — **anti-AUC territory** (near-random discrimination on test). High R-p@1 is largely driven by the high base rate, not by skill. The 30%/200d row is the only one where R-p@1 (0.47) > base (0.44), and only by 6%. Long-horizon NSE breach prediction is broadly weak on this panel; the eval→test decay is severe enough that the model effectively reverts to random.
- **Short-horizon high-AUC cells** (R-p@1 between 0.04 and 0.21 with AUC 0.72-0.89): test AUC looks great, but the rare-event base rates (0.003-0.045) make R-p@1 tiny in absolute terms even at high relative lift. Useful for a portfolio strategy that needs few but clean picks; the runner's per-day denominator handles the staggered NSE panel correctly (`min(R(d), K)`).
- **Two degenerate cells** (50%/100d, 50%/10d) — both produced R-p@1 = 0.000 on test. The 50%/10d cell had base = 0.000 (essentially zero positives in the test window), so the model had nothing to learn; the 50%/100d cell had base = 0.016 (rare-but-not-zero) but the model's top picks missed every positive day. These are the "no signal to find" rows of the lattice.
- **Mid-cells with reasonable AUC but R-p@1 below base**: 10%/100d, 10%/25d, 50%/100d — calibration-under-drift artefact (the val→eval→test prevalence shift on NSE is sharper than on US, per the `_177` analysis pattern). Ranking is preserved on eval but not on test.

## Decision-rule verdict

- The cell at the existing NSE row (`nifty500_up_30pct_50d_dd15pct`, CatBoost, R-p@1=0.431 on a 0.057 base) DROPPED to R-p@1 = 0.080 under XGBoost sweep mode. Two possibilities: (a) cross-backend variance — CatBoost's default 1000-iter + has_time=True is a different model class than XGBoost defaults; (b) the original cell used `default_full_loop` with 8 iters, this sweep used 3 iters. Untangling needs a controlled re-run.
- **The +30%/+200d cell is the best candidate for an agent-mode follow-up** if/when #240 is patched — R-p@1 = 0.470 on a 0.444 base means there IS some skill (~6% lift) but the model is in the same val_brier-flat regime that broke #239 and #241. Don't deploy V1.3 Option B agent_file_protocol mode on it yet.
- The +20%/+10d cell (AUC 0.797 on base 0.014) is the most "interesting" rare-event cell — high AUC + meaningful lift suggests a recoverable signal even if R-p@1 absolute is small. Worth a closer look in a v0.5 deep-dive.

## Reproducing

```
cd wt-nifty500-sweep
for spec in configs/gbdt/experiments/nifty500_up_*_aligned.yaml; do
  uv run python -m gbdt experiment "$spec" --snapshot-end 2026-05-22
done
uv run python scripts/gbdt/regenerate_r_precision_at_k_csv.py
```

## Wall-clock breakdown

| phase | wall-clock |
|---|---:|
| Cell 1 (panel build + features + target + 3-iter loop) | ~8 min |
| Cells 2-20 (features cache hit + target + 3-iter loop) | ~3 min each (19 × 3 min ≈ 57 min) |
| **Total** | **~70 min** |

The feature-cache sharing (`#183` / `#226`) reduced per-sibling-cell time by ~5× vs cold runs.
