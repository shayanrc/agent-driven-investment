# russell1000 +50%/200d/dd25% — canonical-periods fine-tune exploration (#53)

**Verdict: FINE-TUNE (deep + strong bagging) adopted — wins 4 of 5 K's + AUC on test;
loses only @1. Final model = 279 features · d8 · mcw1 · ss0.7 · cs0.7 · gamma0 · eta0.05.**

## Setup
Prevalence ~**12–17%** (COMMON; +50% over 200 days is frequent in the rally-heavy train
window). 200d horizon → needed the cache seeded forward to 2026-04+ (russell was stale at
2025-11-19; `data_pipelines seed … --end 2026-07-06` fixed it → test = 203,750 labeled rows).

## Controlled baseline (all/d6) TEST
auc 0.7765 · R-p@ 1/3/5/10/20 = 0.516 / 0.497 / 0.457 / 0.422 / 0.386 (base rate 0.120).

## HP (val+eval) → deep+bagging (common event)
d8·ss0.7·cs0.7 was the clear val winner (val@1 0.508 vs base 0.371, val@3 0.439 vs 0.370,
best eval@3 0.403 / @10 0.446) — same shape as the #52 winner.

## TEST
| model | AUC | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---|---|---|---|---|
| baseline all/d6 | 0.777 | **0.516** | 0.497 | 0.457 | 0.422 | 0.386 |
| **FT d8·ss0.7·cs0.7** | 0.779 | 0.396 | **0.511** | **0.526** | **0.459** | **0.408** |

FT wins @3/@5/@10/@20 + AUC (decisively @5, +0.069); loses @1 (0.396 vs 0.516). val@1
(0.508) mispredicted test @1 — the eval↔test inversion strikes @1 again.

## Decision
Adopt **279f · d8 · ss0.7 · cs0.7** (book-oriented; for a top-K K>1 strategy the @3–@20 book
advantage outweighs the @1 loss). The untouched backtest window (2025-07→2026-06, now
labeled after seeding) is the tie-breaker for the @1-vs-book call before deployment.
Common-event → deep+bagging pattern holds for the 4th time (#50, #52, #53, and F18 pending).
