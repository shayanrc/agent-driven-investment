# #241 — r1k +50%/200d agent-mode (V1.3 Option B, XGBoost) UNDERPERFORMS sweep R-p@1=0.737

**Verdict: FAIL.** Test R-p@1 = 0.577 (sweep baseline 0.737, **−22% loss**). Strong 2nd data point for the #240 L1 tie-break bug — the runner's val_brier-best selection shipped a strictly-dominated iter that the lex oracle would have rejected.

Companion to `_239_r1k_up40_100d_agent_mode.md` (1st data point: same bug, −12% loss on a different cell). This run was driven parent-side after the dispatched sub-agent stalled at the combine pause; the iter loop ran 3 iters before agent emitted `should_stop=true`.

## Headline metrics (test segment, 200 days, base 0.152)

| metric | sweep (CatBoost) | agent (iter_2 shipped) | true best (iter_0, NOT shipped) | Δ vs sweep |
|---|---:|---:|---:|---:|
| eval R-p@1 | (not on main) | 0.680 | **0.865** | — |
| **test R-p@1** | **0.737** | **0.577** | est. ≈0.74 | **−22%** |
| test R-p@5 | 0.539 | 0.491 | — | −9% |
| test R-p@10 | 0.466 | 0.473 | — | flat |
| test AUC | 0.697 | 0.724 | — | +4% |

## Scout phase — apex at gamma=0.1

Phase A: features 28 min + FS-prefit (143/279 kept) + scout (35/35 in 24 min) = ~52 min. Scout per-knob lex winners:

| knob | best value | eval R-p@1 | val_brier |
|---|---:|---:|---:|
| **gamma** | **0.1** | **0.865** | 0.0870 |
| eta | 0.1 | 0.760 | 0.0817 |
| alpha | 0.1 | 0.760 | 0.0911 |
| scale_pos_weight | 4.52 (SqrtBal) | 0.775 | 0.2090 ★ degenerate |
| max_depth | 6 (default) | 0.730 | 0.0883 |
| colsample_bytree | 1.0 (default) | 0.730 | 0.0883 |
| min_child_weight | 1 (default) | 0.730 | 0.0883 |
| subsample | 1.0 (default) | 0.730 | 0.0883 |

Sharp gamma response: gamma=0 → R-p@1=0.730, gamma=0.05 → 0.785, **gamma=0.1 → 0.865 (apex)**, gamma=0.2 → 0.525, gamma=0.5 → 0.585, gamma=1.0 → 0.640.

## Combine phase — 30 mix configs

Speed-biased: 3 single-knob anchors + 12 gamma=0.1 mixes + 4 gamma-neighborhood probes (0.05, 0.2) + 3 tiny-model probes + 2 lex-zeroth ablations + 6 misc. Lex winners (R-p@1 desc):

| # | label | eval R-p@1 | R-p@3 | R-p@5 | val_brier |
|---|---|---:|---:|---:|---:|
| **1** | **anchor_gamma_0.1** | **0.865** | 0.732 | 0.645 | 0.0870 ← picked |
| 26 | mix_gamma0.1_csb0.7 | 0.850 | 0.720 | 0.648 | 0.0866 |
| 16 | mix_eta0.01_alpha0.1 | 0.835 | 0.680 | 0.677 | 0.0731 |
| 14 | mix_gamma0.05_eta0.01 | 0.830 | 0.798 | 0.742 | 0.0730 |
| 0 | lex_zeroth_with_spw | 0.775 | 0.632 | 0.583 | 0.2039 |

Lex zeroth lost: scale_pos_weight=4.52 ballooned val_brier to 0.20 (overweighted). Combine = ~22 min, ~45 s/fit on 257K-row panel.

## Iter loop — 3 iters

| iter | hp | val_brier | train_val_gap | eval R-p@1 | eval R-p@3 | eval R-p@5 |
|---|---|---:|---:|---:|---:|---:|
| **0** | gamma=0.1 (combine winner) | 0.0870 | +0.0094 | **0.865** ← TRUE BEST | 0.732 | 0.645 |
| 1 | gamma=0.08 | 0.0878 | +0.0125 | 0.805 | 0.680 | 0.604 |
| **2** | **gamma=0.12** | **0.0847** ★ best val_brier | +0.0064 | **0.680** ← shipped | 0.632 | 0.574 |

Mapped gamma apex with 3 bracket probes; gamma=0.1 is a narrow peak. Other knobs were exhaustively tested in combine (eta, alpha, max_depth, colsample, mcw, subsample, spw — all degrade gamma=0.1's R-p@1 by 5-50%). Further iter loop exploration would just re-confirm.

`should_stop=true` at iter_2.

## The #240 bug — val_brier-flat selection ships dominated iter

**val_brier range across iters: |0.0878 − 0.0847| = 0.0031 < tie_band 0.005.** L1 entered tie-break mode and picked iter_2:
- iter_2 has the **lowest val_brier (0.0847)** of the three
- iter_2 has the **smallest train_val_gap (+0.0064)**
- iter_2 has eval R-p@1 = **0.680 (the WORST of the three)**

iter_0 has eval R-p@1 = 0.865 (+0.185 over iter_2). The lex oracle would rank iter_0 > iter_1 > iter_2 unambiguously. The runner's L1 picked the LEX-WORST iter.

**This is a stronger #240 data point than #239** because the magnitude is larger: #239 lost 0.125 eval R-p@1 to L1; this run lost 0.185 eval R-p@1. Test impact: −0.16 R-p@1 (0.737 sweep → 0.577 shipped).

## Wall-clock breakdown

| phase | wall-clock |
|---|---:|
| Phase A (features + FS-prefit + scout) | ~52 min |
| Phase A re-run + combine fits (after schema bug) | ~22 min |
| iter_0 (combine winner re-fit) | ~3 min |
| iter_1 + iter_2 + finalize | ~5 min |
| **Total** | **~82 min** |

(Plus ~75 min lost to sub-agent stall + schema error during my schema-violation re-write — both unrelated to the cell.)

## Secondary issues observed

- `metrics.json::loop.n_iterations_run = 0` (3 iters actually ran) — same as #239
- `iterations.jsonl` is empty (0 lines) — should have logged 3 rows
- `top_k_metrics.per_day` only emits K ∈ {1, 5, 10} on this run — K=3 and K=20 absent (#239 had all 5)
- Schema requires `combine_decision.configs[i].hp` wrapping; the README/EXPERIMENT_SPEC.md should call this out (initial parent submission was flat dict)

## Open question

The fundamental issue: **on healthy-AUC strong-top-1 cells with narrow per-cell HP optima, the iter loop's val_brier-best selection is anti-correlated with the lex oracle.** This is now confirmed on 2 cells. Recommend prioritizing #240 fix:
- (a) When val_brier range < tie_band, fall back to **eval R-p@1-best** iter (not L1 gap+Z)
- (b) Or: always L1-tie-break by **eval R-p@K weighted sum** (matches lex oracle priority)
- (c) Or: emit a HARD WARNING in finalize when L1 picks an iter with eval R-p@1 < max(eval R-p@1 across iters) − 0.05; let the agent override

Until #240 is patched, V1.3 Option B agent mode is **unsafe to deploy on strong-top-1 cells**. The combine phase machinery is doing real work (R-p@1=0.865 was found from scout+combine alone), but finalization throws it away.
