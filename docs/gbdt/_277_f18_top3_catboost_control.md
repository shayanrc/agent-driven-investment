# _277 — CatBoost control: the same top-3 F18 cells, different backend

**Question.** `_276` found in-loop-selected configs anti-select the test book under
xgboost. Is that an xgboost property or general — and how does CatBoost itself
compare on these cells? All prior fundamentals arms (`_272`–`_276`) were xgboost.

**Design.** Same protocol, cells, windows, and 292-col `all_fundamentals` pool as
`_276` (date_aligned, snapshot 2026-07-02 → test **2024-07-26 → 2024-12-16, Q=100**),
with `backend.library: catboost` (`has_time=True` pinned, C6). `_276` lesson
applied: a **fixed 2-probe screen per cell** (`l2_leaf_reg` 3→10, `depth` 6→4 —
same set on all cells for comparability), then stop. Where the loop's val argmin
displaced iter 0, the CatBoost-defaults arm was re-emitted via a 2-launch mini-run
(`*_ffundcbagent0`) so **both** cb arms get a test read. 6 test-scored artifacts.

## In-loop behaviour — CatBoost barely overfits

| cell | iter-0 gap (xgb `_276`) | iter-0 val (xgb best tuned val) | l2 10 | depth 4 |
|---|---|---|---|---|
| +40%/200d | +0.026 (+0.103) | 0.15057 (0.16129) | val↓ book↓ | **val argmin** + book held |
| +20%/100d | +0.030 (+0.100) | 0.20240 (0.20288) | val↓ book↓ | **val argmin** (−8.7%), eval top-book crashed |
| +20%/50d | +0.004 (+0.057) | 0.10335 (0.10788) | val↓ book~ | **val argmin** + book held |

Ordered boosting self-regularizes: gaps are 3–14× smaller than xgboost's, and
**CatBoost's default val beats or effectively ties xgboost's best *tuned* val on all
three cells** (the +20%/100d margin is 0.0005, vs xgb's untuned iter-0 there).
`l2_leaf_reg` 10 was rejected on val on all three (no xgboost-style "val up, tail
down" ambiguity — val and book mostly agreed in-loop). Depth 4 took the val argmin
everywhere and finalized as each loop's pick.

## Results — test window, raw values (base_rate for reference)

**+40%/200d** (base_rate 0.1189):

| model | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgb ffund (defaults) | 0.678 | 0.1013 | 0.330 | 0.327 | 0.338 | **0.364** | **0.368** |
| xgb ffundtune (`_275`) | 0.658 | 0.1015 | 0.580 | **0.460** | **0.414** | 0.359 | 0.335 |
| cb defaults | 0.692 | 0.1024 | **0.650** | 0.310 | 0.228 | 0.307 | 0.320 |
| cb depth-4 (loop pick) | **0.698** | 0.1026 | 0.500 | 0.290 | 0.252 | 0.307 | 0.354 |

**+20%/100d** (base_rate 0.2311):

| model | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgb ffund (defaults) | 0.575 | 0.1875 | 0.550 | 0.397 | 0.370 | 0.341 | 0.331 |
| cb defaults | 0.652 | 0.1859 | 0.270 | **0.533** | **0.456** | 0.403 | 0.384 |
| cb depth-4 (loop pick) | **0.657** | **0.1852** | **0.660** | 0.510 | **0.456** | **0.441** | **0.425** |

**+20%/50d** (base_rate 0.1277):

| model | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgb ffund (defaults) | 0.687 | 0.1134 | **0.440** | **0.393** | 0.340 | 0.317 | 0.292 |
| cb defaults | **0.734** | 0.1126 | 0.160 | 0.307 | 0.310 | **0.362** | **0.376** |
| cb depth-4 (loop pick) | 0.730 | **0.1117** | 0.170 | 0.343 | **0.350** | 0.357 | 0.373 |

## Findings

1. **CatBoost wins test AUC on all three cells** (cb-defaults vs xgb ffund: +0.014
   to +0.082) and test Brier
   on two — a consistent bulk-ranking advantage over xgboost on these long-horizon
   fundamentals cells, at defaults, before any tuning.
2. **The test book comparison is cell-shaped, not uniform.**
   - **+20%/100d: cb depth-4 beats the xgb single-fit at every K** (and the cell's
     prior best @1) — the strongest all-around book any arm has posted on this cell.
     The defaults arm shows @3–@20 + AUC are mostly the *backend*; depth-4 adds @1
     and @10/@20.
   - +20%/50d: CatBoost takes @5 (depth-4), the deep book (@10/@20) and AUC;
     xgboost keeps the top (@1–@3). Opposite tail concentration.
   - +40%/200d: cb-defaults takes @1 (0.650, the cell's best; 5.5× base); xgb
     ffundtune keeps @3/@5.
3. **The `_276` selection problem is backend-independent — and bidirectional.** On
   +20%/100d the loop pick's eval book was the *worst* in-loop (@1 0.18) yet its
   test book is the cell's best; on +40%/200d the harmonious in-loop winner
   (depth-4: val argmin + book held) tested no better than defaults. In-loop
   window metrics simply do not rank configs on these cells, in either direction.
   The 2-probe screen kept the damage bounded (loop pick ≈ defaults on 2 of 3
   cells) — small screens are the right discipline.
4. **Why CatBoost defaults transfer better: they're barely fit.** Train/val gaps of
   +0.004–0.030 (vs xgboost's +0.06–0.10) mean there's little window-specific
   sharpness to lose in transfer. The `_276` mechanism (regularization flattens the
   tail) shows up here too but milder — CatBoost's L2 degraded val *and* book
   together in-loop (an honest rejection signal), rather than xgboost's misleading
   split.

## Caveats

Single test window (Q=100) — @1 differences of ±0.1 are within noise; the @3–@20
pattern and AUC are the more robust reads. The 2-probe screen means CatBoost's
tuned ceiling is unexplored (deliberately — `_276` showed deeper screens
anti-select). Depth-4's cell-2 dominance is one window; same replication rule as
everything else.

## Verdict

- **CatBoost joins the trailing confirmation (`V1.7_TBD` §5) as a co-primary
  backend.** The lineup for each confirmed cell should now include the cb-defaults
  arm (and cb depth-4 on +20%/100d) alongside the xgb arms — the consistent AUC
  advantage + the cell-2 every-K dominance are exactly what a second window should
  test.
- **No adoption yet** — same independent-window rule (`_276`, CLAUDE.md). Best-known
  *confirmed* configs remain the xgb ffund single-fits; the cb arms are candidates.
- The deployed sp500 champions (different cells: +50%/50d, +20%/25d) are xgboost;
  nothing here touches them — but a cb-defaults pass over the champion cells is a
  cheap, well-motivated follow-up (parked in `V1.7_TBD`).

Registry: 6 rows (`*_ffundcbagent{,0}`), mode `agent_file_protocol`, backend
`catboost`. Specs: `configs/gbdt/experiments/*_ffundcbagent{,0}.yaml`. Decision
trail: `results/gbdt/experiments/<cell>_ffundcbagent/loop/` request+decision pairs.
Prior: `_276` (xgb agent loop), `_275` (default loop), `_274` (lattice). Plan:
`V1.7_fundamentals_features_plan.md`.
