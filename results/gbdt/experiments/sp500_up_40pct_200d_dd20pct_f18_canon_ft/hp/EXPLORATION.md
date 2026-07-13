# sp500 +40%/200d/dd20% F18 fundamentals — canonical fine-tune exploration (#54)

**Verdict: the baseline (all 292 features [279 technical + 13 F18] · default HP d6) is the
model. Deep+bagging loses at EVERY K — the base top-of-book is already exceptional.**

## Setup
Feature token `all_fundamentals` → 292 cols (279 technical + 13 F18 point-in-time
valuation-ratio cols; needs `results/valuation/data/valuation_panel.parquet`, built here:
2.58M rows / 971 tickers). Prevalence ~**17%** (COMMON). 200d horizon; sp500 cache current
to 2026-04-16 → 117,000 labeled test rows (full window).

## Controlled baseline (all/d6) TEST — the bar
auc 0.7509 · R-p@ 1/3/5/10/20 = **0.628 / 0.517 / 0.505 / 0.454 / 0.395** (base rate 0.170).
Base @1 = 0.628 — exceptional.

## HP (val+eval) → deep+bagging tried (common event)
On val, bagging lifted @3–@20 modestly but lowered @1 (base val@1 0.302 best). Took the two
val-best book configs to test.

## TEST
| model | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|
| **baseline all/d6** | 0.751 | **0.628** | **0.517** | **0.505** | **0.454** | **0.395** |
| d8 · ss0.7 · cs0.7 | 0.750 | 0.576 | 0.429 | 0.402 | 0.413 | 0.363 |
| d10 · ss0.7 · cs0.7 | 0.744 | 0.520 | 0.445 | 0.430 | 0.419 | 0.385 |

Baseline wins EVERY K. Unlike the other common cells (#50/#52/#53 where deep+bagging won),
here it loses everywhere.

## Why (refines the heuristic)
Common event alone doesn't guarantee a deep+bagging win — it's **base @1 headroom** that
decides. sp500 large-caps + the F18 fundamentals signal give a compact d6 model an already-
maxed top (@1 0.628, like nasdaq #51's 0.564); row/col bagging only averages that sharp top
away. Deep+bagging wins when base @1 is LOW (sp500_20 0.253, russell_40_100 0.208); it loses
when base @1 is already high (nasdaq 0.564, this cell 0.628), regardless of prevalence.

## Decision
Adopt the **baseline all/d6** (292f incl. F18). Note per CLAUDE.md the F18 edge failed
two-window replication and is NOT promoted — this is the canonical-period F18 model for
forward comparison, not a deployment swap.
