# _276 — agent-driven FS+HP loop on the top-3 F18 fundamentals models

**Question.** `_275` showed the default loop's ceiling is shallow (2–3 iterations, one
l2 step, degradation-gated). What does the full agent-driven loop
(`callback_mode: agent_file_protocol`) find on the same three cells — and do
in-loop gains transfer to the test book?

**Design.** `<cell>_ffundagent`: `all_fundamentals` (292 cols), date_aligned matching
`_274`/`_275` (snapshot 2026-07-02 → test **2024-07-26 → 2024-12-16, Q=100**),
xgboost, `max_iterations: 16`, runner gates neutralized so the agent manages
exploration (`plateau_threshold: 0.0001` — the #204 workaround; `degradation_gate:
0.25`; `tie_band: 0.0` = strict val-Brier argmin at finalization). iter 0 = default
HP on the full pool ≡ the `_274` ffund single-fit (in-loop anchor). The agent read
each iteration's bundle (val Brier, gap, per-iter **eval R-Precision@K**) and wrote
the FS/HP decisions; every rationale is in `loop/iter_<N>_decision.json` +
`iterations.jsonl`.

## What the loops explored

| cell | iters | families explored | val argmin config | stop |
|---|---|---|---|---|
| +40%/200d | 10 | FS cliff-cut, L2 {0,4.5,10,20}, depth {2,3,4,6}, colsample 0.5, mcw 3, mcw×L2 | 155c, d6, λ10 (iter 6) | agent |
| +20%/100d | 3 | FS cliff-cut, L2 {0,4.5} on cut pool | **full-pool default (iter 0)** | agent |
| +20%/50d | 4 | pure-HP on full pool: L2 {0,4.5,10}, depth 4 | 292c, d6, λ4.5 (iter 1) | agent |

+40%/200d also got a follow-up 1-shot (`_ffundagent8`): the loop's **eval-book
dominator** (155c, λ10, mcw 3 — eval R-p@1 0.815, @3–@20 all best-in-loop), which
the val-argmin finalizer had passed over, re-emitted via a 2-iteration mini-run so
the untouched test window could adjudicate the val-vs-eval disagreement.

## Results — test window, raw values (base_rate for reference)

**+40%/200d** (base_rate 0.1189):

| model | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ffund (single-fit) | 0.678 | 0.1013 | 0.330 | 0.327 | 0.338 | **0.364** | **0.368** |
| ffundtune (`_275` loop) | 0.658 | 0.1015 | **0.580** | **0.460** | **0.414** | 0.359 | 0.335 |
| ffundagent (loop pick: λ10) | **0.679** | 0.1020 | 0.200 | 0.250 | 0.348 | 0.333 | 0.311 |
| ffundagent8 (λ10+mcw3) | 0.674 | 0.1019 | 0.250 | 0.203 | 0.276 | 0.259 | 0.277 |

In-loop, the two agent configs dominated: λ10 held the val argmin (0.16129, vs
0.16375 for the λ4.5 family) **and** eval R-p@1 0.795; λ10+mcw3 dominated the eval
book at every K (@1 0.815, @3 0.638, @5 0.644, @10 0.626). On test both collapse
below the untuned single-fit's book. The finalizer's re-fit is faithful (val
0.16129 reproduced exactly) — the collapse is the configs', not an artifact.

**+20%/100d** (base_rate 0.2311): every explored direction (cliff-cut, L2 on the cut
pool) degraded val; the loop finalized on **iter 0** and the test book is
value-identical to the ffund single-fit (0.550 / 0.397 / 0.370 / 0.341 / 0.331,
AUC 0.575). Convergent with `_275` (whose default loop also failed to beat
single-fit here) — two methodologies, same verdict.

**+20%/50d** (base_rate 0.1277):

| model | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ffund (single-fit) | 0.687 | 0.1134 | **0.440** | **0.393** | **0.340** | **0.317** | **0.292** |
| ffundagent (λ4.5 full-pool) | **0.703** | **0.1122** | 0.210 | 0.287 | 0.294 | 0.292 | 0.278 |

The agent found a genuinely new config (`_275` never screened pure-L2 without an FS
cut): val Brier −8% (0.11731 → 0.10788), and on test both AUC (+0.016) and Brier
improve — yet the test book is worse at **every** K.

## The finding — in-loop signals anti-select on the test book

Three cells, three instances of the same failure mode, under three different
guises:

1. **+40%/200d:** val argmin AND eval-book dominance both picked configs that
   collapse on the test book (eval→test ordering at top-1 was nearly inverted).
2. **+20%/100d:** the only cell where the loop *refused* to displace iter 0 — and
   the only cell whose test book was preserved.
3. **+20%/50d:** a real val gain that transferred to test *bulk* metrics (AUC↑,
   Brier↓) while the test *book* regressed at every K.

Mechanistic read: on these long-horizon fundamentals cells, L2-style
regularization improves bulk ranking/calibration but **flattens the extreme
prediction tail that top-K trading actually consumes**. Val Brier, test AUC, and
test Brier are all bulk metrics — none of them see the tail. This generalizes the
V1.3 anti-AUC lesson (val_brier misalignment) to cells that are NOT anti-AUC
flagged (AUCs here are 0.57–0.70): **Brier/AUC-guided config selection is
structurally misaligned with the top-book operating metric on tail-consuming
cells**, whichever loop drives it.

The corollary cuts at `_275` too: its +40%/200d headline (`ffundtune` R-p@1 0.580)
was itself selected via val on the same windows from a (3-config) screen. It has
more test-side support than the agent configs (its test book beat single-fit at
@1–@5), but until it replicates on an independent window it should be treated as a
candidate, not a result.

## Secondary observations

- **FS kept all 13 F18 columns everywhere.** Both cliff-cuts (cells 1+2, 137/125
  dropped) removed only technical features — consistent with `_275`; the
  fundamentals family carries gain wherever it's offered.
- **+20%/100d and +20%/50d want the full 292-col pool.** Every cut degraded val
  (and in `_275`, val + test). Only +40%/200d tolerates (and per `_275`'s test read,
  may benefit from) a cliff-cut.
- The protocol itself worked flawlessly across 17 launch/resume cycles + 2
  finalizations (faithful re-fits, exact decision application, no state
  corruption) — the machinery is production-ready even where the tuning result is
  negative.

## Verdict

- **Best-known configs are unchanged: the plain ffund single-fits on all three
  cells** (+40%/200d's `_275` ffundtune remains a *candidate* pending the trailing
  confirmation). No agent-loop config is adopted.
- **The `V1.7_TBD` §5 trailing confirmation is now the critical path** and must
  carry BOTH +40%/200d arms (ffund single-fit AND `_275` ffundtune) — if ffundtune's
  top-book edge is a val-window selection artifact, the trailing window will show
  it.
- Process rule going forward (added to CLAUDE.md): don't select configs by
  val-Brier/AUC/bulk-Brier improvements on top-K cells without a held-out *book*
  check — bulk metrics don't see the tail.

Registry: 4 rows (`*_ffundagent` ×3, `*_ffundagent8`), mode `agent_file_protocol`.
Specs: `configs/gbdt/experiments/*_ffundagent{,8}.yaml`. Full per-iteration
decision trail: `results/gbdt/experiments/<cell>_ffundagent/loop/` +
`iterations.jsonl`. Prior: `_275` (default loop), `_274` (lattice), `_272`/`_273`
(champion A/Bs). Plan: `V1.7_fundamentals_features_plan.md`.
