# project_mgmt/archives/

Frozen historical artifacts that were sitting in the main checkout's working directory or in worktrees, moved here so the repo root stays clean. Nothing in this tree is consumed by any module's runtime — these are reference materials only.

## Layout

```
project_mgmt/archives/
├── pr-reviews/              # review notes written before opening past PRs
├── run-logs/                # raw run-time logs from past experiments + data ops
└── biased_baselines/        # gbdt experiment results from before the sample-uniqueness fix (PR #18); kept as before/after comparators for task #113's re-runs
```

## `pr-reviews/`

Per-PR review notes that informed each PR's body or follow-up commits. The review *outcome* lives in the PR's GitHub comments + merge commit message; these are the working notes that produced those outcomes.

- `review_pr2.md` — PR #2 (v1-skills → main)
- `review_pr3.md` — PR #3 (docs-meta-refresh → main)
- `review_pr4.md` — PR #4 (gbdt-v1 → main)
- `review_v5_a2.md` — V5.A.2 review on the `v5-experiments` branch (not yet merged)
- `review_v5_a2_followup.md` — V5.A.2 cleanup follow-up

## `run-logs/`

Raw stdout/stderr logs from past walk-forward runs and data-pipeline seeds. Kept because the analytical conclusions (e.g., A2.1 corrwindow window-length sweep verdict, NIFTY 50 per-ticker fetch outcomes) aren't trivially reproducible from the cache state alone.

- `a2_sanity_20260520T213355Z.log` — A2.1 corrwindow sanity v0 sweep over L ∈ {10,20,60,100}; ends with verdict "Best window length on failures: L=100" (-17.15% failure CRPS).
- `a2_corrwindow_canonical_20260521T114727Z.log` — A2.1 canonical run (76 folds, mean test_crps=0.06141).
- `_a2_corrwindow_v0_aborted.log` — same family, aborted at fold 36/76.
- `b1_canonical_20260520T212218Z.log` — B1 canonical run (76 folds, mean test_crps=0.05021).
- `b5_joint_canonical_20260521T174024Z.log` — B5 joint canonical run (76 folds, mean test_crps=0.06417).
- `canonical_E2_Ds30.log` — E2 Ds30 canonical run (76 folds, mean test_crps=0.04750).
- `nifty50_deep_20260525T053039Z.log` — NIFTY 50 deep-history seed (PR #6 data ops trail, per-ticker fetch outcomes including gap-detection + provider-fallback diagnostics).
- `nse_nifty50_seed.log` — failed NIFTY 50 re-seed attempt (D8-immutability + UNIQUE-constraint collisions); kept as a snapshot of what to NOT do.

## `biased_baselines/`

Pre-uniqueness-fix gbdt experiment results. The sample-uniqueness weighting fix landed in PR #18 (LdP §4.4 — overlapping-label weight discounting). Experiments run **before** that fix had inflated effective sample sizes and biased calibration. Task #113 re-runs the affected experiments; these directories are the **BEFORE state** kept for direct before/after comparison.

- `nifty50_up_20pct_50d_dd10pct_pre_uniqueness_fix/` — Exp 1 baseline (ran 2026-05-26 12:00 on `wt-exp-nifty50-up20-50d-v2`).
- `nasdaq100_up_10pct_100d_dd5pct_pre_uniqueness_fix/` — Sweep #1 baseline (ran 2026-05-26 20:26 on `wt-sweep-runner`).

Each contains the full per-experiment artifact set (`spec.yaml`, `features.yaml`, `hp.yaml`, `metrics.json`, `iterations.jsonl`, `model.cbm`, `calibration.pkl`, `figs/`, `predictions/`, `report.md`). When #113 produces the post-fix re-runs, compare on: AUC, weighted Brier, calibration (Spiegelhalter z), Kish ESS, fold-uniqueness distribution.

## Retention policy

These are write-once, read-rarely. Don't delete without a deliberate decision — the cost to keep them is bytes, the cost to regenerate is hours of compute (for the run-logs) or carries methodology drift (for the biased baselines, which can't be regenerated at all once the post-fix runs land).
