# nifty500 +50%/200d/dd25% ffund (F18-IN) — canonical finetune (V1.10, task #55)

**Verdict: ADOPT `d10 / mcw1 / ss0.7 / cs1.0`.** Deep+bagging on the full de-confounded
feature set beats the controlled baseline BOOK at every K≥3 while holding @1 — the
de-confounded F18-IN edge survives the finetune on the held-out test window.

## Windows / prevalence
Canonical explicit-boundary split: train 2015-01-01 · val 2022-03-30 · eval 2023-07-01 ·
**test 2024-07-01 → 2025-06-30** · backtest 2025-07 → 2026-06.
- test base_rate **0.121**, eval base_rate 0.437 — **eval↔test prevalence inverts** (eval 3.6× hotter),
  so eval is an unreliable HP oracle → **selected on val, confirmed on test**.
- 293 features incl. the 10 F18-IN valuation columns (train-window fund NaN 17% post-backfill,
  down from the invalidated _285's 48–58%; see `[[project-in-fundamentals-coverage-cliff]]`).

## Controlled baseline (the bar) — `final_fit FEATS=all HP=d6/mcw1/ss1/cs1/g0/eta0.05`
test AUC 0.589 · R-p@1/3/5/10/20 = **0.414 / 0.340 / 0.293 / 0.246 / 0.230** (base_rate 0.121).

## Path tried — deep+bagging grid (recipe §3), selected on val
16 configs: depth{6,8,10} × ss{0.7,0.85} × cs{0.7,1.0} (mcw1) + mcw{5,10} variants.
Val-best two taken to test:

| config | test @1 | @3 | @5 | @10 | @20 |
|---|--:|--:|--:|--:|--:|
| baseline all/d6 | 0.414 | 0.340 | 0.293 | 0.246 | 0.230 |
| **d10 ss0.7 cs1.0 (ADOPTED)** | 0.406 | **0.347** | **0.329** | **0.289** | **0.260** |
| d6 ss0.85 cs0.7 | 0.345 | 0.299 | 0.252 | 0.226 | 0.216 |

## Verdict
`d10/ss0.7/cs1.0` beats the baseline **book** at @3/@5/@10/@20 (+0.007/+0.036/+0.043/+0.030)
while @1 is flat (−0.008, noise). Adopt for a top-K (K≥3) strategy. This is a cleaner outcome
than #53 russell_50_200 (same 50%/200d shape, which lost @1) — the de-confounded F18-IN
signal both preserves the spike and thickens the deeper book.

## Caveats
- **Single test window.** The F17/F18 history (sp500 F18 _279/_280, macro _264, _285 itself)
  is single-window fund wins that failed replication. The backtest (2025-07→2026-06) is the
  independent-window check — see `hp/` backtest logs / the memo.
- **200d room:** backtest-window entries after ~2025-09 lack a full 200d forward path → their
  target labels truncate (the #53 caveat). Backtest return is on the labelable subset.
- NOT promoted / no `/daily-predictions` deploy path for nifty500 today — parity/research result.
