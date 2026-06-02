# _211 — Cell-5 manual FS+HP tuning (XGBoost): beating sweep R-Precision@1 by +24%

**Branch**: `gbdt-211-cell5-manual-tuning-memo`.
**Date**: 2026-06-02.
**Cell**: nasdaq100 +10%/50d/dd5% — the anti-AUC strong-top-1 anomaly cell deferred from `_195`.
**Canonical metrics**: `results/gbdt/data/r_precision_at_k.csv`.

## Headline

Manual XGBoost tuning **beats sweep R-Precision@1 by +0.157 (+24% relative)** on cell-5:

| Backend / track | Test R-p@1 | Test AUC | Test base_rate |
|---|---|---|---|
| Sweep (CatBoost defaults) | 0.6714 | 0.4750 | 0.2652 |
| Agent loop (mcw=10, γ=1) | 0.5429 | 0.4871 | 0.2652 |
| **Manual tuned (XGBoost)** | **0.8286** | 0.4717 | 0.2652 |
| Baseline (XGBoost defaults, cliff FS) | 0.3429 | 0.5201 | 0.2652 |

Sweep test R-Precision@1 lift = 2.53×; manual tuned lift = **3.12×**. Test AUC stays sub-null
(0.47–0.52 across all configs — the anti-AUC anomaly: ranking signal lives in the
prediction-extreme top tail only).

The agent loop's heavy regularization (`mcw=10 + γ=1`) actively HURT versus baseline on
top-1 (0.543 vs sweep 0.671 = −19%). The manual track found that **light regularization
+ small model (≤24 leaves) + aggressive FS (top-130) + slow-LR-with-early-stopping** beats
both.

## What won

```python
# Final winning config — total wall-clock to fit: ~4 seconds, best_iter=6 trees
hp = {
    "max_depth": 2,
    "colsample_bytree": 0.4,
    "scale_pos_weight": 1.0,   # ! NOT 2.0 (the val_brier optimum hurt R-p@1)
    "gamma": 0.0,              # ! NOT > 0 (any γ ≥ 0.5 walks toward the degenerate sink — see § "Degenerate sink")
    "eta": 0.1,
    "n_estimators": 500,
    "early_stopping_rounds": 30,
}
# + FS: top-130 features by iter_0 importance (cliff cut alone, 190, was second-tier)
# best_iter = 6 trees of max_depth=2 → ≤ 24 leaves total
```

The model is essentially a **smart prior + 6 tiny corrections**. Train Brier ≈ val Brier ≈
trivial-constant baseline (0.224), meaning most predictions are at the base rate. The 6
trees worth of small perturbations are what carry the top-1 ranking signal — and they are
sharp enough to push test R-p@1 to 0.829.

## How we got here: the FS → response-curve → mix-and-match methodology

Eight script-driven batches; each fit takes ~3–10 s on hist+nj=8 + cached features. Outside
the loop entirely because the loop's val_brier objective walks toward a degenerate constant
on this anti-AUC cell — see § "Why the loop fails on this cell shape" below.

### Stage 1 — FS cliff cut (1 fit)

iter_0 with XGBoost defaults: train_brier=0.096, val_brier=0.263, gap=0.167 (severe overfit).
Importance distribution had a **clean cliff at 0.01**: 89 features at exactly 0 importance,
0 in the (0.001, 0.01) band, continuous distribution above 0.01.

Dropped the 89 sub-floor features → byte-identical model (XGBoost colsample=1.0 considers
all features at every split; importance=0 features are never selected, so dropping them
doesn't change predictions). This establishes the cliff-cut FS keep-list (190 features) as
the working set for HP exploration.

### Stage 2 — single-knob response curves (~30 fits)

Mapped val_brier response across 8 knobs:

| Knob | Values tested | Δval at best | Sweet spot | Curve shape |
|---|---|---|---|---|
| `gamma` | 0, 0.5, 1, 2, 5, 7, 10, 15, 20, 50, 100 | −0.038 (γ=10) | DEGENERATE at γ≥20 — converges to constant predictor (val=0.226) |
| `alpha` (L1) | 0, 0.5, 1, 5, 10, 15, 20, 30, 50, 100 | −0.038 (α=20) | DEGENERATE at α≥30 — same sink |
| `max_depth` | 2, 3, 4, 6, 8, 10 | −0.012 (d=3) | inverted-U; floor at 3 |
| `n_estimators` (no ES) | 100, 500, 1000, 2000, 5000 | 0 at default 100; > 500 catastrophic overfit |
| `scale_pos_weight` | 0.5, 1, 2, 4 | −0.011 (spw=2) | inverted-U at 2.0 |
| `colsample_bytree` | 0.3, 0.5, 0.7, 1 | −0.009 (cs=0.5) | mild dip |
| `max_bin` | 64, 128, 256, 512, 1024 | −0.006 (128) | shallow minimum |
| `subsample` | 0.5, 0.7, 1 | −0.005 (subs=0.7) | very mild |

**The gamma / alpha "winners" are a trap.** Both curves descend monotonically to the
no-splits-pass-the-threshold regime where the booster emits the prior probability for
every row. val_brier hits ~0.226 (the weighted base-rate baseline) at γ≥20 or α≥30 — that
is the trivial constant predictor, not a real model. Optimizing on val_brier alone walks
right into this sink.

### Stage 3 — mix-and-match around non-degenerate winners (~25 fits)

Combined structurally different knob families (depth + col-sample + class-weight + tree-count
+ split-quality). Winner of this stage:

```python
backbone_lean = {"max_depth": 2, "colsample_bytree": 0.4,
                 "scale_pos_weight": 2.0, "n_estimators": 50, "gamma": 0.0}
# val R-p@1 = 0.5297 (lift 1.55× on val base 0.343)
```

### Stage 4 — eval validation flipped the leaderboard (~10 fits, +eval scoring)

Scored top val candidates on the eval segment too. The val_brier ordering reshuffled
dramatically:

- val R-p@1 winner `lean` → eval rank 6.
- val R-p@1 rank 10 `lean+FS=top130` → eval rank 2.
- The new eval-rank-1: `lean + eta=0.1 + n_estimators=500 + ES=30` — slow-LR + early-stopping,
  which we had originally dismissed because its val_brier was unimpressive.

**Val R-p@1 is unreliable as the optimization signal on this cell.** It moves with noise
because Q_days × 1 pick/day = small sample of ranking events; val_brier averaged over
36,800 rows is paradoxically a better predictor of eval R-p@1 than val R-p@1 itself.

### Stage 5 — eta+ES neighborhood probe (~18 fits)

Confirmed `eta ∈ [0.1, 0.12]` is the eta sweet spot; ES window invariant across {20, 30, 40, 50}
(best_iter found inside 17 trees of any of those windows); deeper trees with slow-LR (d=3, d=4)
hurt R-p@1; dropping `scale_pos_weight` (set to 1.0) **helped on eval** — best_iter dropped
to 3 trees and eval R-p@1 jumped to 0.6270.

### Stage 6 — FS refinement (~15 fits)

FS sweep around top-130: top-100, top-110, top-120, top-130, top-140, top-150, top-160. Best:
**top-130 + champ-spw** at eval R-p@1=0.7377 (lift 2.09× on eval base 0.354). best_iter=6.

### Stage 7 — 1-shot TEST shootout (8 configs)

After exhausting val + eval, scored 8 candidates ONCE on the held-out test segment. The eval
ranking carried over reasonably to test for the top 2:

| Test rank | Config | nf | Test R-p@1 | Test R-p@5 | Test R-p@10 | vs sweep R-p@1 |
|---|---|---|---|---|---|---|
| 1 | champ-spw + FS=top130 (winner) | 130 | **0.8286** | 0.5264 | 0.4749 | **+0.157 (+24%)** |
| 2 | champ + FS=top150 | 150 | 0.5571 | 0.5100 | 0.4795 | −0.114 |
| 3 | loop_final (mcw=10, γ=1) | 190 | 0.5429 | 0.4521 | 0.4599 | −0.129 |
| 4 | lean (val R-p@1 winner) | 190 | 0.5286 | 0.4293 | 0.4335 | −0.143 |
| 5 | champ-spw + FS=cliff | 190 | 0.5143 | 0.5436 | 0.5234 | −0.157 |
| 6 | champ + FS=top130 | 130 | 0.4286 | 0.4029 | 0.4186 | −0.243 |
| 7-8 | champ + FS=cliff / baseline | 190 | 0.3429 | 0.4521 / 0.4286 | 0.4485 / 0.4243 | −0.329 |

**The winner sweeps top-1** by +24%, but **loses to sweep at K=5 and K=10** (test@5: 0.526 vs
0.569 = −7.5%; test@10: 0.475 vs 0.515 = −7.7%). The model is **K=1-specialized** — it makes
the day's top pick with high precision but its lift dilutes across the broader top-K window
versus sweep's CatBoost-default deeper model.

## Why the loop fails on this cell shape

Cell-5 is the canonical **anti-AUC anomaly** (CLAUDE.md "What not to do — gbdt", PR #28 +
memo #138 rule): test AUC=0.475 (in the [0.45, 0.55] null band) yet sweep R-Precision@10
lift = 1.94× (well above the 1.5× "investigate" threshold). The ranking signal lives in
the **prediction-extreme top tail only**; the bulk of predictions are noise around the base
rate.

Three load-bearing problems for the agent loop on this cell:

1. **val_brier has a degenerate global minimum at the constant predictor.** The booster
   minimizes val_brier by predicting the weighted base rate everywhere (val=0.224 ≈ 0.226
   baseline). The L2 grid (`mcw=10`) and gamma/alpha tracks all push the model in that
   direction. The loop's L1 tie-break (gap + Z) actively SELECTS the most-regularized iter
   inside the band, compounding the harm.

2. **val R-p@1 is unreliable.** Q_days × 1 pick/day gives a tiny effective sample. Configs
   ranked #1 by val R-p@1 land at eval rank 6; configs ranked #10 land at eval rank 2.
   The loop has no eval-segment scoring during iteration; it commits on val signals only.

3. **The win is a tiny model** (6 trees of depth 2, ≤ 24 leaves total). The loop's HP
   defaults (`n_estimators` implicit at hundreds-of-trees, eta=0.3, no ES) over-fit to
   match val_brier — exactly the opposite of what we want for top-1.

The manual track sidestepped (1) by scoring **val_brier AND val R-p@K AND eval R-p@K** every
iteration, then trusting eval R-p@K when the others disagreed. It sidestepped (2) by holding
val signals as candidate-generators and eval R-p@K as the oracle. It sidestepped (3) by
trying tiny-model HP regimes (eta=0.1 + ES + small n_estimators) that the loop's defaults
never reach.

## What this means for the playbook

Rules **10, 11, 12** added to `.claude/memories/project-gbdt-tuning-playbook.md` (shipped
in this PR). Summary:

- **Rule 10 (eval-as-oracle on anti-AUC cells)**: When test AUC is in [0.45, 0.55] AND
  R-Precision@10 lift > 1.5×, score val_brier AND val R-p@K AND eval R-p@K, and treat eval
  R-p@K as the oracle if val signals disagree.
- **Rule 11 (FS → response-curve → mix → eval-validation manual methodology)**: The five
  stages above are the reusable manual-tuning recipe. Use it when the loop's val_brier
  objective is misaligned (see rule 10) or when you suspect an HP regime the loop's
  defaults can't reach (eta+ES; tiny-model n_estimators).
- **Rule 12 (tiny models can dominate on top-1)**: 6 trees of depth 2 (≤ 24 leaves) beat
  sweep's CatBoost defaults (1000 iters, depth 6, ~64,000 leaves max) by 24% on R-p@1.
  Don't dismiss tiny configs because their val_brier is "close to baseline" — that's a
  feature on anti-AUC cells where most predictions should be at the prior.

## CLAUDE.md updates

Two new bullets in "What not to do — gbdt":

- **val_brier has a degenerate global minimum at the constant predictor on anti-AUC cells**
  — don't ship the val_brier winner without scoring R-Precision@K too.
- **Don't dismiss tiny models on top-1 metrics** — 6 trees of depth 2 beat sweep on cell-5
  R-p@1 by 24%; standard "more capacity / more rounds" intuition is wrong on anti-AUC cells.

## What's open

- **Rules 10/11/12 are derived from n=1 cell (cell-5).** Replicate the manual methodology
  on cells 1+3 (sp500 strong-top-1 cells where the agent loop also under-performed) to
  verify the methodology transfers; expect different HP optima per rule 8 (no transferable
  recipe across cell shapes).
- **Wire eval R-p@K into the agent loop iteration bundle.** Currently only val signals
  drive the loop. Per-iteration eval scoring would let the loop apply rule 10 in-flight.
  Feasibility: predictions are emitted at finalization only; would need a per-iter eval
  scoring path in the runner.
- **Audit the L1 tie-break (#187) on anti-AUC cells.** It selects the lowest-gap iter
  among val_brier ties; on cell-5 (and likely on other anti-AUC cells) "lowest gap" is the
  config closest to the degenerate constant predictor. Refit the tie-break rule with eval
  R-p@K as the tiebreaker on flagged cells.

## Artifacts

All raw batch JSONs under `/mnt/122CEE982CEE765F/Workspace/wt-cell5-agentloop/results/gbdt/experiments/`:

| Stage | Script | JSON |
|---|---|---|
| 1 — FS cliff cut | (loop iter_0 in `nasdaq100_up_10pct_50d_dd5pct_manual/`) | — |
| 2 — single-knob curves | `/tmp/cell5_response_curves.py`, `/tmp/cell5_extend_curves.py` | `_cell5_response_curves.json` |
| 3 — mix discovery | `/tmp/cell5_mix_rpk.py`, `/tmp/cell5_mix_batch.py`, `/tmp/cell5_mix_batch2.py`, `/tmp/cell5_mix_batch3.py`, `/tmp/cell5_mix_batch4_eta.py` | `_cell5_mix_rpk_results.json`, `_cell5_mix_batch{1..4}_*.json` |
| 4 — eval validation | `/tmp/cell5_mix_batch5_fs_eval.py` | `_cell5_mix_batch5_fs_eval_results.json` |
| 5 — eta+ES probe | `/tmp/cell5_mix_batch6_eta01.py` | `_cell5_mix_batch6_eta01_results.json` |
| 6 — FS refinement | `/tmp/cell5_mix_batch7_final.py` | `_cell5_mix_batch7_final_results.json` |
| 7 — 1-shot test | `/tmp/cell5_test_shootout.py` | `_cell5_test_shootout_results.json` |

JSON sidecar (machine-readable headline): `results/gbdt/data/_211_cell5_manual_tuning_xgb_data.json`.
Canonical R-Precision@K row (test segment): `results/gbdt/data/r_precision_at_k.csv` —
`nasdaq100_up_10pct_50d_dd5pct_manual_xgb`.
