# nifty500 +10%/25d/dd5% ffund — canonical finetune (V1.11, task #60)

**Verdict: ADOPT `d8 / mcw1 / ss0.7 / cs0.7`** — beats the controlled baseline at EVERY K.

Canonical split; test base_rate 0.248, eval 0.380 (eval↔test inversion → selected on val).
Controlled baseline (`all/d6`): test R-p@1/3/5/10/20 = 0.229/0.238/0.259/0.266/0.282 —
weak @1 (below base rate) + rising book = the deep+bagging "adopt" profile (recipe §"real
determinant"). 16-config sweep on val; val-best d8/ss0.7/cs0.7 → test
**0.341/0.300/0.313/0.307/0.311** (every K up, @1 +0.112). The #50/#52 outcome.
Backtest 2025-07→2026-07: +11.6% (basket +6.2%), raw top-K +37.4%, target/DD 29/49.
Full tables: `docs/gbdt/_290_nifty500_fund_cells_v111.md`.
