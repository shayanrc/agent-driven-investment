# russell1000 +40%/100d/dd20% — canonical-periods fine-tune exploration (#52)

**Verdict: FINE-TUNE WINS via deep trees + strong row/col bagging on full features.
Final model = 279 features · max_depth 8 · min_child_weight 1 · subsample 0.7 ·
colsample_bytree 0.7 · gamma 0 · eta 0.05 · 500 trees + val-AUC ES. Beats the controlled
baseline at EVERY K on test, decisively at the top (@1 +0.124).**

## Windows / prevalence
815 tickers kept (min_rows gate, of 1002). train 1.48M · val 257k · eval 204k · test 204k.
Prevalence ~**8–9%** (a COMMON event). Cache reaches 2025-11-19 → the 100d test labels
(test_end 2025-06-30 + 100 td ≈ 2025-11-19) are just barely complete; test = 203,750
labeled rows (815 × 250 td). (The runner's own headline_test is empty — its trailing-anchor
`test_rows` heuristic misfires on horizon-100 — but `final_fit`'s explicit-boundary carve is
correct; ignore runner metrics, use the controlled baseline.)

## Controlled baseline (all/d6) TEST — the bar
auc 0.807 · R-p@ 1/3/5/10/20 = 0.208 / 0.288 / 0.334 / 0.356 / 0.362.
Base @1 is LOW (0.208) → room to grow (the sp500_20 situation, not nasdaq's maxed 0.564).

## HP exploration (val+eval)
Common + low-base-@1 → deep+bagging is the indicated path. Swept depth {6,8,10} × bagging
{ss0.85 / ss0.7+cs0.7}. Deep+bagging lifted the book on BOTH windows including @1
(val@1 0.375→~0.53, eval@1 0.272→0.31) — val+eval agreed. Best on val: d8·ss0.7·cs0.7
(val@5 0.397, @10 0.360, eval AUC 0.779).

## TEST (held out; raw p)
| model | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|
| baseline all/d6 | 0.807 | 0.208 | 0.288 | 0.334 | 0.356 | 0.362 |
| d8 · ss0.85 | 0.811 | 0.272 | 0.320 | 0.360 | 0.347 | 0.366 |
| **d8 · ss0.7 · cs0.7** | 0.810 | **0.332** | **0.351** | **0.396** | **0.362** | **0.381** |

The val-selected d8·ss0.7·cs0.7 wins every K; stronger bagging (ss0.7+cs0.7) beat the
milder ss0.85 (which dipped at @10). Clean win, val-selected + test-confirmed.

## Decision
Adopt **279f · d8 · ss0.7 · cs0.7** as the canonical russell1000 +40%/100d model.
Third confirmation of the pattern: **common event + low base @1 ⇒ deep trees + bagging on
the full feature set beats the default**, and the bagging strength is itself a knob (here
0.7/0.7 > 0.85/1.0).
