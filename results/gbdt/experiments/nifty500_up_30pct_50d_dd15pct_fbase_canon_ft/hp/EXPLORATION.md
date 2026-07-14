# nifty500 +30%/50d/dd15% fbase (technical) — canonical finetune (V1.10, task #55)

**Verdict: BASELINE STANDS (`all / d6`).** No finetune adopted — deep+bagging trades a
severe @1 loss for razor-thin deep-book gains that don't clear the recipe's book bar. The
technical champion is the untuned controlled baseline.

## Windows / prevalence
Canonical explicit-boundary split (train 2015 · val 2022-03-30 · eval 2023-07-01 ·
**test 2024-07-01 → 2025-06-30** · backtest 2025-07 → 2026-06).
- test base_rate **0.072**, eval base_rate 0.205 — **eval↔test prevalence inverts** →
  selected on val, confirmed on test. 283 technical features (F1–F16 + F21 calendar2).

## Controlled baseline (the bar) — `final_fit FEATS=all HP=d6/mcw1/ss1/cs1/g0/eta0.05`
test AUC **0.663** (genuinely high) · R-p@1/3/5/10/20 = **0.242 / 0.203 / 0.190 / 0.187 / 0.233**.

## Path tried — deep+bagging grid (recipe §3), selected on val
Val lifted strongly (0.35–0.45 across K) but **did NOT transfer to test** (the val window's
higher prevalence inflated it). Val-best two on test:

| config | test @1 | @3 | @5 | @10 | @20 |
|---|--:|--:|--:|--:|--:|
| **baseline all/d6 (KEPT)** | **0.242** | 0.203 | 0.190 | 0.187 | 0.233 |
| d6 ss0.85 cs1.0 | 0.161 | 0.208 | 0.197 | 0.213 | 0.251 |
| d10 ss0.85 cs0.7 | 0.117 | 0.163 | 0.188 | 0.169 | 0.211 |

## Verdict
d6/ss0.85/cs1.0 nominally beats @3–@20 but by noise-level margins (+0.005/+0.007/+0.026/+0.018)
while **collapsing @1** (0.242→0.161, −0.081). Per recipe §"real determinant" (base @1 mid →
FT trades @1 for the deeper book), the trade isn't worth it here: the @3/@5 gains are within
noise and the @1 loss is large. **Baseline stands** (the #49/#51/#54 outcome). A top-10/20-only
strategy could prefer d6/ss0.85/cs1.0 (@10 +0.026, @20 +0.018), but the default read is baseline.

## Caveats
- The high test AUC (0.663) says the baseline has real signal; deep+bagging just redistributes
  it away from the spike without a net book win.
- NOT promoted / no nifty500 `/daily-predictions` path — parity/research result.
