# nifty500 +20%/100d/dd10% ffund — canonical finetune (V1.11, task #60)

**Verdict: BASELINE STANDS (`all/d6`)** — no finetune adopted.

Canonical split; test base_rate 0.245, eval 0.437 (eval↔test inversion → selected on val).
Controlled baseline: test R-p@1/3/5/10/20 = **0.454**/0.371/0.341/0.320/0.308 — high @1
(top already maxed). Val-best FTs (d10/ss0.7/cs0.7: 0.337/0.339/0.317/0.304/0.310;
d6/ss0.7/cs0.7: 0.386/0.329/0.314/0.322/0.318) collapse @1 without a book win — the
#51/#54 outcome; the strong val lift was prevalence-inflated and didn't transfer.
The saved model is the baseline config. Backtest 2025-07→2026-07: +9.6% (basket +6.2%),
raw top-K +6.7%, target/DD 26/37. Full tables: `docs/gbdt/_290_nifty500_fund_cells_v111.md`.
