# nasdaq100 +40%/50d/dd20% — canonical-periods fine-tune exploration (#51)

**Verdict: the untuned baseline (all 279 features · default HP d6) is the model.
No fine-tune config cleanly beats it — every regularizer trades the (already
excellent) top-of-book @1 for marginal deep-book gains.**

## Windows / prevalence (canonical explicit-boundary date_aligned)
train 156,290 (2.31%) · val 27,090 (3.07%) · eval 21,500 (2.22%) · test (3.00%).
A **rare** event (~2–3%, like #49). Same eval↔test inversion as sp500_20:
baseline eval R-p@1 0.555 but the runner's default-loop metrics showed 0.068 — see below.

## Baseline methodology fix (applies to all cells)
The runner's `metrics.json` base metrics are NOT a clean bar: the default loop applies
matched-sweep HP + conditional-isotonic calibration and (on rare/anti-AUC cells) can sit
at a weak point. The correct control is `final_fit` at DEFAULT HP (all 279f, d6, mcw1,
val-AUC ES, raw p) — the same code path as the FT candidates. For nasdaq the proper
baseline (test rp@3 0.462, rp@10 0.719) is much STRONGER than the runner metric (0.379 /
0.679). **Always compare to the controlled `final_fit all/d6` baseline, not metrics.json.**

## Proper baseline (all/d6) TEST — the bar
auc 0.9396 · R-p@ 1/3/5/10/20 = **0.564 / 0.462 / 0.500 / 0.719 / 0.956**  (base rate 0.030).
Note best_iter=11: val-AUC ES stops early (the cell is high-signal; few trees suffice).

## HP exploration (val+eval, one knob at a time)
FS was inert at default HP (trajectory rounds 279→93 byte-identical — the greedy fit
ignores redundant features), so pruning can't help; the lever is HP. Tried both the #49
path (mcw regularization) and the #50 path (deep + row/col bagging):

| config | test AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|
| **baseline all/d6** | 0.940 | **0.564** | 0.462 | 0.500 | 0.719 | 0.956 |
| d8 · ss0.85 | 0.941 | 0.475 | 0.472 | 0.495 | 0.726 | 0.966 |
| d6 · mcw10 | 0.930 | 0.469 | 0.416 | 0.443 | 0.662 | 0.955 |

On val+eval, EVERY bagging/reg config lowered @1 (val@1 0.29–0.35 vs base 0.41) while
lifting @10 — val correctly predicted the @1 sacrifice. d6/mcw10 looked best on val
(v@1 0.452, v@10 0.740) but LOST on test (eval↔test inversion). d8/ss0.85 is the only
config that beats base on the deep book (@3/@10/@20 + AUC) but it drops @1 by 0.089.

## Why base wins here (vs sp500_20 where deep+bagging won every K)
The starting point differs: sp500_20's base @1 was a mediocre 0.253 (room to grow →
bagging lifted it to 0.321). nasdaq's base @1 is already an exceptional 0.564, so
subsampling can only smooth the sharp top *down*. Book gains from d8/ss0.85 are tiny
(@10 +0.007, @3 +0.010) and don't offset the @1 loss (−0.089). The models are equivalent
on the book (±0.01); base clearly wins @1.

## Decision
Adopt the **baseline all/d6** as the canonical nasdaq100 +40%/50d model (saved here as
model.pkl / predictions / final_summary). d8/ss0.85 is documented as a book-oriented
alternative (marginally better @10/@20/AUC, worse @1) — the untouched **backtest window**
would break the tie if a book-vs-top call is needed before deployment.
