# _275 — full FS+HP loop on the top-3 F18 fundamentals models

**Question.** `_274` measured F18 at forced single-fit default HP. What does the
standard FS+HP loop add on the top fundamentals cells — and does FS drop the
z-score columns where they dilute (the `_274` hypothesis)?

**Cell selection.** Top 3 `_ffund` models by R-p@3 (the leaderboard sort) **within
the `_274` F18-edge band** (20–50% × 50–200d): **+20%/100d**, **+20%/50d**,
**+40%/200d**. Two higher-raw-R-p@3 cells were passed over deliberately:
`+10%/50d` (0.417) is a diagnosed-null cell per the compound rule (AUC 0.489 ∈
[0.46, 0.54], R-p@10 lift 0.379/0.335 = 1.13 < 1.2 — its R-p@3 is base-rate, not
skill), and `+10%/25d` (0.350) sits outside the F18-edge band with a marginal
fundamentals delta (`_274`: ΔAUC +0.008 on a 0.197 base-rate cell) — tuning it
would measure the technical model, not the fundamentals contribution.

**Design.** `<cell>_ffundtune`: `features.candidates: all_fundamentals` (292 cols =
279 technical + 13 F18), default auto-callback loop (`callback_mode: default`,
`max_iterations: 8`, plateau 0.005, degradation gate 0.01 — the champbase
convention), **date_aligned** split matching `_274`, snapshot 2026-07-02 → the same
test window **2024-07-26 → 2024-12-16 (Q=100)**, so tuned vs single-fit is
same-window comparable. All 3 runs rc=0 (~9–10 min each).

## What the loop actually did

| cell | iters | best iter | winning config | stop |
|---|---|---|---|---|
| +20%/100d | 2 | 1 | 161 feats (FS cut), `l2_leaf_reg: 4.5` | degradation |
| +20%/50d | 2 | **0** | **full 292 pool, default HP** (= the `_274` single-fit) | degradation |
| +40%/200d | 3 | 1 | 141 feats (FS cut), `l2_leaf_reg: 4.5` | degradation |

The degradation gate fired after 1–2 FS cuts in every cell — the loop explored one
l2 step and no other HP. **This is the default loop's ceiling, not an agent-driven
tune's** (the callback's algorithmic-fallback FS is a blunt importance cliff-cut).

## Results — test window, raw values (base_rate for reference)

**+20%/100d** (base_rate 0.2311):

| model | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fbase (single-fit) | 0.593 | 0.1853 | 0.330 | 0.310 | 0.328 | 0.328 | 0.318 |
| ffund (single-fit) | 0.575 | 0.1875 | **0.550** | **0.397** | **0.370** | **0.341** | **0.331** |
| ffundtune (loop) | 0.588 | 0.1869 | 0.380 | 0.327 | 0.338 | 0.324 | 0.322 |

**+20%/50d** (base_rate 0.1277): loop reverted to iter 0 ⇒ **ffundtune ≡ ffund**
(AUC 0.687, Brier 0.1134, R-p@1 0.440, @3 0.393, @5 0.340, @10 0.317, @20 0.292).
The FS cut to 150 degraded val Brier; the full 292-col single-fit stands.

**+40%/200d** (base_rate 0.1189):

| model | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fbase (single-fit) | 0.653 | 0.1032 | 0.140 | 0.293 | 0.296 | 0.306 | 0.310 |
| ffund (single-fit) | **0.678** | 0.1013 | 0.330 | 0.327 | 0.338 | **0.364** | **0.368** |
| ffundtune (loop) | 0.658 | 0.1015 | **0.580** | **0.460** | **0.414** | 0.359 | 0.335 |

## Findings

1. **+40%/200d: the tune is a large top-of-book win.** The 141-feature + l2 4.5
   model nearly doubles R-p@1 over the single-fit fund arm (0.33 → **0.58**, 4.9×
   the 0.119 base rate) and lifts @3/@5 strongly (+0.13/+0.08), at flat Brier and
   R-p@10. The `_274` standout cell gets materially better top-K when FS trims the
   technical pool around the fundamentals. Bulk AUC dips (0.678 → 0.658) — the loop
   trades bulk ranking for tail sharpness, which is what a top-K portfolio uses.
2. **+20%/50d: the single-fit was already the optimum** the default loop could
   find — every FS cut degraded val Brier. No change.
3. **+20%/100d: the loop's val-based pick is worse on test top-K** (R-p@1 0.55 →
   0.38 vs the single-fit fund arm). Iter 1 won on the val tie-band, but the test
   window disagrees. The **single-fit ffund remains this cell's best model**; the
   tune result is recorded, not adopted.
4. **FS never dropped a fundamentals column.** In all 3 cells the importance
   cliff-cut removed only technical features — **all 13 F18 columns (including all
   4 z-scores) survived every cut** (292→161, →150, →141). The `_274` hypothesis
   ("FS will drop the diluting z-scores") is wrong at the importance level: the
   z-scores carry gain everywhere; dilution (where it exists, e.g. +50%/50d) is not
   visible to importance-based pruning. Resolving that needs targeted ablation
   (9-col vs 13-col arms), not the auto-loop.

## Caveats

Single window (date-aligned 2024-H2), Q=100 — R-p@1 levels are noisy; the @1/@3/@5
*pattern* (all three up strongly on +40/200) is more robust than any single K. The
loop stopped after 2–3 iterations in every cell, so the HP envelope beyond one l2
step is unexplored — an agent-driven (`agent_file_protocol`) tune remains the way
to push further (`V1.7_TBD` §3 for the +50%/50d champion is unchanged). The
per-cell verdict "tuned vs single-fit" uses the same window the loop's val segment
precedes — no leakage, but also only one test regime.

## Verdict

- **+40%/200d is now the strongest fundamentals cell on record** (R-p@1 0.58 /
  @3 0.46 / @5 0.41 at base 0.119, AUC 0.658, Brier 0.1015) — untracked, long-horizon.
  Two-window (trailing) confirmation before any candidate-cell consideration is
  already parked as `V1.7_TBD` §5; this strengthens the case for running it.
- **+20%/50d and +20%/100d keep their single-fit fund arms** as best-known configs.
- No champion change, no `/daily-predictions` change (human decisions, `_019`).

Registry: 3 `*_ffundtune` rows (`default_full_loop`). Specs:
`configs/gbdt/experiments/*_ffundtune.yaml`; runner: `scripts/gbdt/run_f18_top3_tune.sh`.
Prior: `_274` (lattice sweep), `_272`/`_273` (champion A/Bs). Plan:
`V1.7_fundamentals_features_plan.md`.
