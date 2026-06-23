# _271 — CatBoost agent-tune of russell1000 +40%/100d (the sole table-wide extension)

**Question (`_270` follow-up):** after the top-10 CatBoost sweeps were all cbagent-covered (`_269` + `_270`),
`russell1000_up_40pct_100d_dd20pct` was the **only remaining cell anywhere in the registry** where the
pattern still held — a CatBoost sweep (#18 by R-p@3) whose XGBoost agent had lost to it, with no CatBoost
agent-tune. Does tuning the robust backend beat the sweep here? **Answer: no — a clean negative. The
CatBoost agent-tune loses the headline metrics decisively (@1 −0.100, @3 −0.108 vs the sweep) on a
catastrophic eval→test decay (−0.370 @1). The ≤3-iter CatBoost sweep wins.**

## The bar replicates (resnap, like `_270`)

russell cells are `_266`-faithfulness-suspect, so per the `_270` discipline the sweep bar was re-run on the
current vintage (`russell1000_up_40pct_100d_dd20pct_aligned_resnap`, byte-identical spec, `--snapshot-end
2026-06-20`). **The resnap reproduces the sweep to the decimal** (@1 0.550, @3 0.517, @5 0.447, @10 0.382,
@20 0.341) — the bar is trustworthy, so the negative verdict holds against a verified sweep, not a stale one.

## Results — CatBoost-agent vs the (verified) sweep vs XGBoost agent (test R-Precision@K + base_rate)

| model | @1 | @3 | @5 | @10 | @20 | base | AUC |
|---|---|---|---|---|---|---|---|
| CatBoost sweep (resnap = #18, verified) | **0.550** | **0.517** | **0.447** | 0.382 | 0.341 | 0.080 | 0.79 |
| **CatBoost agent** | 0.450 | 0.408 | 0.419 | **0.392** | 0.334 | 0.080 | 0.818 |
| XGBoost agent | 0.485 | 0.407 | 0.431 | 0.379 | 0.338 | 0.080 | — |

CatBoost-agent vs the sweep per K: **@1 −0.100, @3 −0.108, @5 −0.028, @10 +0.010, @20 −0.007** — it loses
@1/@3/@5/@20 and only edges @10. vs the *XGBoost* agent it merely ties @3 (0.408 vs 0.407) and loses @1
(0.450 vs 0.485) — so switching to the robust backend did **not even reliably beat the prior losing agent**
here. Test window 2024-07-26 → 2025-05-13 (test_end 2025-05-13), base 0.080, 200 days. Higher test AUC
(0.818) than the sweep but worse top-of-book — AUC and R-p@1/@3 diverge, the familiar anti-AUC pattern.

## Eval→test decay (the cause — worse than `_270`)

| metric | eval | test | decay |
|---|---|---|---|
| @1 | 0.820 | 0.450 | **−0.370** |
| @3 | 0.683 | 0.408 | −0.275 |

The combine pick had a stellar eval (@1 0.820, 8.1× base) that **collapsed −0.370 into the test window** —
more than double the decay of `_270`'s russell/50 (−0.157). The eval window (2023-10→2024-07, base 0.101)
simply did not generalize to the test window (2024-07→2025-05, base 0.080); no in-loop lever addresses
window decay. The sub-agent did the right things — chose a **broad, L2-regularized, non-subsampled** combine
config (`alpha05_only`) precisely to avoid the `_270` "@3 band lost when you subsample" failure, and a
slower-lr smoother in iter_1 *regressed* eval so it was rejected — and it still decayed. This is not a
tuning miss; it is the objective.

## Reading — the cbagent-extension thread, closed

Across every cell where the pattern applied, agent-tuning CatBoost is **opportunistic, not dependable**:

| cell | horizon | CatBoost-agent vs sweep | memo |
|---|---|---|---|
| sp500 +50%/200d | H=200 | **WIN** (+0.300 @1) | `_269` |
| sp500 +40%/200d | H=200 | loss @1 (−0.170), won @3 | `_269` |
| russell +40%/200d | H=200 | loss @1 (−0.173), won @3 | `_269` |
| russell +50%/200d (#1) | H=200 | marginal mixed (+0.007 @1, −0.055 @3) | `_270` |
| **russell +40%/100d (#18)** | **H=100** | **clear loss (−0.100 @1, −0.108 @3)** | **`_271`** |

One big win, two @1-losses-but-@3-wins, one marginal mixed, one clear loss — outcome set by eval→test decay
direction, which the agent cannot see. The pattern is **not horizon-specific** (the lone H=100 extension
behaves like the H=200 cells). The single-window eval-R-p objective is the bottleneck across both backends
and both horizons.

## Verdict

**Negative.** The CatBoost ≤3-iter sweep is the robust default at this cell, as everywhere. Agent-tuning
CatBoost did not beat it (or even the prior losing XGBoost agent) at @1/@3. This closes the table-wide
cbagent-extension thread: there is **no remaining pattern-match to try**, and the consistent lesson is that
the fix must be the **objective (multi-window / decay-penalized eval, V1.5_TBD #2)**, not the backend.

## Recommendation

- **Keep the CatBoost sweep as the default** for this cell (and all H=100/H=200 cells). Do not promote.
- **Promote V1.5_TBD #2 (multi-window eval objective) before any further agent-tuning** — it is now
  demonstrated across 5 cells and 2 horizons as the binding constraint.
- Investigation #26 (russell/50 @3) can still proceed for the FS-prefit mechanism, but `_271` reinforces
  that the dominant term is window decay, not the knob.

## Artifacts

- Cells committed: `70b1b6b` (cbagent), `24174b1` (sweep resnap + specs); registry rows
  `russell1000_up_40pct_100d_dd20pct_aligned_{cbagent,resnap}`.
- Sidecar: `results/gbdt/data/_271_russell40_100d_cbagent_data.json`.
