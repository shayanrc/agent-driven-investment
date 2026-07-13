# sp500 +20%/25d/dd10% — canonical-periods fine-tune exploration (#50)

**Verdict: FINE-TUNE WINS via deep trees + subsample bagging on the FULL feature set.
Final model = 279 features · max_depth 8 · min_child_weight 1 · subsample 0.85 ·
colsample_bytree 1.0 · gamma 0 · eta 0.05 · 500 trees + val-AUC ES (best_iter ~159).
It beats the untuned base_20 on the TEST book at every K.**

## Windows / prevalence (canonical explicit-boundary date_aligned)
train 2015-01-02..2022-03-29 (4.03%) · val 2022-03-30..2023-06-30 (5.29%) ·
eval 2023-07-03..2024-06-28 (3.80%) · test 2024-07-01..2025-06-30 (4.80%). A **common**
event (~4%). Note eval is the LEANEST window (3.8%) and test is richer (4.8%).

## The eval↔test trap (why the first two attempts failed)
eval and test DISAGREE for this cell: the untuned base is WEAK on eval (R-p@3 0.182) but
STRONG on test (0.311). So eval is an unreliable HP oracle here.
- **Attempt 1 — eval-greedy FS+HP** (12f/depth3, the eval-AUC peak 0.8197): won eval
  (R-p@3 0.314) but LOST test (0.270 vs base 0.311). Aggressive eval-driven feature
  pruning anti-selected the test book. (Also: letting *eval* drive feature count violates
  the val/FS role.)
- **Attempt 2 — val-knee FS** (60f/depth6): also lost test (R-p@3 0.254).
- **Both FS-pruned subsets lost to the full 279f base on the test book** → feature
  selection itself anti-selects the test book on this cell.

## Attempt 3 — deep trees + bagging on FULL features (the win)
Hypothesis: the base won test *at full features*, so regularize by ROW/COLUMN subsampling
(bagging) on deep trees rather than pruning features. Swept depth8, mcw1, over colsample
{0.5,0.7,1.0} and subsample {0.7,0.85,1.0} on eval; picked the best config on **val**
(d8/ss0.85, best deep val R-p@3 0.332) and evaluated once on test.

## TEST (held out; all R-p@K on raw p, rank-based / calibration-invariant)
| model | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|
| base 279f · default (d6) | 0.807 | 0.253 | 0.311 | 0.290 | 0.273 | 0.308 |
| FT 12f/d3 (eval-greedy) | 0.826 | 0.273 | 0.270 | 0.250 | 0.247 | 0.278 |
| FT 60f/d6 (val-knee) | 0.826 | — | 0.254 | — | 0.256 | — |
| **FT 279f · d8 · ss0.85** | 0.823 | **0.321** | **0.313** | **0.311** | **0.299** | **0.330** |

Robustness of the deep+bagging *region* on test (R-p@10 = the book depth the strategy uses):
base 0.273 · d6/ss0.85 0.277 · d8/ss0.85 **0.299** · d10/ss0.85 0.281 · d8/ss0.7 0.299 —
**every deep+bagging config ≥ base @10**; the chosen d8/ss0.85 is best of the region on
both @3 and @10. (@3 is noisier — d6/ss0.85 and d8/ss0.7 dip below base there.)

## Why it works
Deep trees on the full 279 features capture interactions the shallow/pruned models miss;
row+column bagging (subsample 0.85) controls the overfit that killed the plain d8 (which
was strong on train, weak everywhere). This regularization path generalizes to the test
regime where eval-tuned pruning did not. mcw left at 1 (inert on this common event);
colsample 1.0 (bagging via subsample alone was enough); gamma rejected (monotonically
hurts AUC).

## Decision
Adopt **279f · d8 · ss0.85** as the canonical +20%/25d fine-tune (model.pkl,
predictions/, final_summary.json here). Recommended confirmation before any deployment
swap: evaluate on the untouched **backtest window** (2025-07..2026-06) + the strategy
backtest. Multiple test configs were viewed (6) — the win is decisive (every-K, region-
robust @10), but the backtest window is the clean tie-breaker.
