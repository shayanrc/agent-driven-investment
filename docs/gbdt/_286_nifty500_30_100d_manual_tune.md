# _286 — nifty500 30/100d fund-arm HP tuning: no robust win (single-fit stands)

**Verdict: no HP configuration robustly beats the single-fit on a principled
val+eval basis. The single-fit stands.** The top-K signal on this cell is
noise-dominated at the eval-window scale. Consistent with the fragile-edge read
of the whole nifty500 F18 work (`_285`).

## Context

`#29` aimed to finetune the two F18-powered nifty500 cells (30/100d, 20/100d)
surfaced by `_285`. The agent-protocol FS+HP loop **anti-selected the test book**
(eval R-p@1 0.733 → **test 0.151** vs single-fit 0.283) — the `_276` lesson
repeating. Pivoted to disciplined manual sweeps judged **only on val/eval** (test
sealed), using **min(val, eval)** at each K as the robustness metric.

Single-fit baseline (30/100d ffund, sealed test): AUC 0.614, R-p@1 0.283,
R-p@3 0.285, R-p@10 0.297, base_rate 0.143.

## Sweeps (all 30/100d ffund, min(val,eval) as the robustness metric)

| sweep | finding |
|---|---|
| **depth {2–8}, 100t** | val_rp3 flat; depth-2 best min_rp1 (0.579) but eval-noisy; default depth-6 *worst* on the robust metric (val-overfit) |
| **subsample {0.5–0.9}, 100t** | ss=0.8 spiked (min_rp1 0.676) — but a **100-tree artifact**: collapsed to 0.334 at 51 trees under ES |
| **1000 trees + ES50** | all early-stop at ~51 trees (eta 0.3); ES **under-trains the top-K tail** vs fixed 100t (subsample=0.8 edge did not survive) |
| **geometric depth search** (ss=0.8, 1000+ES) | "peak" depth-5 is an **eval-noise spike** (val_rp3 flat, std 0.018; eval std 0.081; depth-5 eval_rp1 0.787 a lone outlier). No robust depth |
| **depth-5 HP** (mcw/cs/ss/λ/γ) | none beats base (min_rp3 0.532); ss=0.9 ~tie |
| **eta {0.3/0.1/0.05/0.03}** | **strict no-op** — eta is rank-invariant under isotonic calibration (Spearman 1.0). See [[project-gbdt-eta-rank-invariant]] |

## Bankable methodology

- **val/eval-agreement (min) as the robustness filter.** val and eval have near-
  identical prevalence (0.333 / 0.342), so a config that wins one window but not
  the other (e.g. depth-3 eval 0.649 / val 0.519) is **instability, not a base-
  rate artifact** — a window-luck flag that predicts test anti-selection.
- **Test regime shift caps transferable performance.** test prevalence 0.143 ≪
  val/eval ~0.34 (the 2024-09→2025-07 lower-momentum regime) — absolute test
  R-p@K is structurally lower than val/eval for any config.
- **Two xgboost-backend facts surfaced here:** [[project-gbdt-xgboost-tree-count]]
  (n_estimators=100 default) + [[project-gbdt-eta-rank-invariant]] (eta no-op on
  rank metrics).

## 20/100d

Not separately tuned. The 30/100d negative result + eta rank-invariance + the
fragile-edge read make a per-cell HP win unlikely; deferred rather than pursued.

## Disposition

`#29` closed. No champion swap, no promotion. Follow-up energy redirected to
`#28` (sp500 regime + F18 sweep — the independent second-market replication of
the "F18 helps at 100d" hypothesis).
