# R-Precision@K canonical registry

`r_precision_at_k.csv` is the cross-cell R-Precision@K registry for gbdt experiments. One row per completed experiment with a non-empty `predictions/test.csv`. Recomputed from scratch — does not trust any per-run `metrics.json::p_at_k` cached field (those use the legacy micro form on pre-2026-06-01 artifacts).

## Formula

```
R-Precision@K  =  (1 / Q)  ·  Σ_{q=1..Q}  r_q / min(K, R_q)
```

where for each test day q:
- **R_q** = number of actual positives that day (`y_true == 1`)
- **r_q** = positives caught in the top-K picks on day q, sorted by `p_calibrated` descending (tie-break: `ticker` ascending, stable mergesort — matches the runner's per-day P@k convention in `src/gbdt/topk_diagnostics.py`)
- **Q** = number of days with `R_q > 0` (days with no positives are skipped — `min(K, 0)` is ill-defined and the day contributes no information)
- **K** is a fixed integer; the standard set is **{1, 3, 5, 10, 20}**

Per-day ratio `r_q / min(K, R_q)` is bounded in [0, 1]:
- On days where `R_q ≥ K`, the denominator is `K` and the ratio is plain precision-at-K.
- On days where `R_q < K`, the denominator is `R_q` — penalising the model for not catching `K` positives when only `R_q` exist would mis-normalise. The achievable ceiling stays at 1.

Aggregation is **macro**: mean of per-day ratios, equal weight per day. This matches the question *"how reliable is the model on a typical day?"*. Each trading day is one decision occasion.

## Columns

| column | meaning |
|---|---|
| `experiment` | Experiment directory name under `results/gbdt/experiments/<name>/` |
| `rows` | Row count in `predictions/test.csv` |
| `Q_days` | Number of test days with R_q > 0 (the denominator-Q in the formula) |
| `base_rate` | `df["y_true"].mean()` — overall positive prevalence in the test set |
| `AUC` | `sklearn.metrics.roc_auc_score(y_true, p_calibrated)` |
| `R_precision_at_1` | R-Precision@K at K=1 (macro form, the formula above) |
| `R_precision_at_3` | … at K=3 |
| `R_precision_at_5` | … at K=5 |
| `R_precision_at_10` | … at K=10 |
| `R_precision_at_20` | … at K=20 |
| `mode` | Training regime: `sweep` / `default_full_loop` / `agent_file_protocol` / `agentloop_legacy` (added 2026-06-05) |
| `n_iterations_run` | Realized FS+HP iteration count from `<artifact>/iterations.jsonl` (or sidecar JSON for pruned `_agentloop*` cells; blank when neither carries it) |
| `backend` | `xgboost` / `catboost` — read from `<artifact>/spec.yaml::backend.library`, defaulting to `catboost` when omitted (matches the runner's default) |
| `train_start`, `train_end`, `val_start`, `val_end`, `eval_start`, `eval_end`, `test_start`, `test_end` | V1.4 P3 calendar-date columns — universe-anchored segment bounds |

Rows are sorted by AUC descending for readability — re-sort as needed.

### Mode classifier rules

Primary (from `<artifact>/spec.yaml`):
- `backend.fs_hp_loop.callback_mode == "agent_file_protocol"` → `agent_file_protocol`
- `callback_mode == "default"` (or absent) AND `max_iterations <= 3` → `sweep`
- `callback_mode == "default"` (or absent) AND `max_iterations >= 4` → `default_full_loop`

Fallback (cell-name suffix — only when the artifact dir is gone; never overrides primary):
- `_agentloop*` → `agentloop_legacy`
- `_aligned` → `sweep`
- `_pilot` → `default_full_loop`
- `_b_acceptance_agent` → `agent_file_protocol`; `_b_acceptance` (alone) → `default_full_loop`
- `_xgb_acceptance` / `_acceptance` → `agent_file_protocol`
- `_phase8` / `_catboost_phase8` → `default_full_loop`
- no suffix → `sweep`

A static dispatch table in `scripts/gbdt/regenerate_r_precision_at_k_csv.py::_PRUNED_AGENTLOOP_FALLBACK` pins the 13 known pruned `_agentloop*` cells to their reported iteration counts (from `results/gbdt/data/_195`, `_222`, `_223` sidecars). The fallback only applies when no sibling worktree carries the artifact dir.

## Lift (computed on demand, not stored)

Per CLAUDE.md § Reporting conventions, lift columns are not stored in tables — they compress two pieces of information into one number and lose the base rate scale. Compute on demand:

```
lift_at_K  =  R_precision_at_K  /  base_rate
```

Lift is fine in narrative prose ("nasdaq H=25 R-Precision@10 was 1.86× base rate"); just not as a checked-in column.

## Compound signal/null rule (CLAUDE.md § What not to do — gbdt)

- **AUC ∈ [0.45, 0.55] AND R-Precision@10 lift < 1.2×** → null signal flagged
- **AUC ∈ [0.45, 0.55] AND R-Precision@10 lift > 1.5×** → top-tail signal hidden by AUC; investigate the prediction-extreme regime, don't dismiss

These thresholds were originally calibrated against the legacy weighted R-precision metric. They remain serviceable for R-Precision@10 (verified on the H=25 4-cell corpus in `docs/gbdt/_138_h25_cross_market_combined.md`), but as more cells accumulate they may want recalibration. The compound *form* (AUC + top-tail metric) is the durable lesson, not the specific threshold.

## Relationship to the prior "weighted R-precision" (now legacy)

Pre-2026-06-01, the project used a single "weighted R-precision":

```
weighted R-precision  =  Σ_q (positives caught in top R_q) / Σ_q R_q
                     =  Σ_q r_q^{K=R_q} / Σ_q R_q
```

— per-day **variable** K (always K = R(d)), with **micro** (sum/sum) aggregation. **This is a different metric**, not a special case of R-Precision@K at K=R(d). For most cells the two land within ~30% of each other; the direction of divergence depends on whether high-R(d) days are easier or harder for the model.

Memos written before 2026-06-01 quote the legacy form in their body narratives; each one has an appended "R-Precision@K (current methodology)" section that re-states the same cells under the current metric. The legacy `*_r_precision_*` fields in `results/gbdt/data/_<id>_data.json` files record the legacy form.

## Regenerating this CSV

```bash
uv run python -m scripts.gbdt.regenerate_r_precision_at_k_csv
```

The script scans both the current checkout and sibling worktrees (`/mnt/122CEE982CEE765F/Workspace/wt-*/`), picks the freshest `predictions/test.csv` per experiment name (by mtime), computes the columns above, and writes back to this CSV. Tolerant of corrupted-worktree I/O errors (skips with warning).

## See also

- `.claude/memories/project-r-precision-methodology.md` — full project methodology + history of the 2026-06-01 rename
- `scripts/gbdt/compute_r_precision.py` — per-cell post-hoc computation (emits both R-Precision@K + legacy weighted R-precision)
- `scripts/gbdt/regenerate_r_precision_at_k_csv.py` — registry regeneration
- `CLAUDE.md` § Reporting conventions + § What not to do — gbdt — the rules this CSV backs
