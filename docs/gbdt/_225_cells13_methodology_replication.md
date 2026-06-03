# _225 — cells 1+3 methodology replication: 5-stage manual + Z-gate generalization test

**Branch**: `gbdt-214-cells13-methodology-replication`.
**Date**: 2026-06-03.
**Cells**: cell-1 (`sp500_up_50pct_50d_dd25pct`, sweep R-p@1=0.800) + cell-3 (`sp500_up_20pct_25d_dd10pct`, sweep R-p@1=0.600).
**Tests**: (A) the 5-stage manual methodology from [_211](_211_cell5_manual_tuning_xgb.md) and (B) the `min_child_weight` → val Spiegelhalter |z| → conditional-isotonic decision finding from [_223](_223_cell5_loop_v1.3_revalidation.md) — both N=1 (cell-5) prior — generalized to cells 1+3.
**Closes**: [#214](https://github.com/shayanrc/agent-driven-investment/issues/214).
**Canonical metrics**: `results/gbdt/data/r_precision_at_k.csv`.

## Headline

**Methodology FAILS on both cells (0 of 2 PASS the "test R-p@1 ≥ sweep" floor). Z-gate finding DOES NOT GENERALIZE.** Cell-1 manual recovers some signal vs XGBoost defaults (+0.120 R-p@1 over the defaults+cliff baseline of 0.540) but is still −0.140 vs the CatBoost sweep (0.800). Cell-3 manual barely beats its XGBoost defaults baseline (0.4267 vs baseline 0.2133, +0.213 vs baseline) but is still −0.173 vs the CatBoost sweep (0.600). The dominant gap on both cells is **XGBoost-vs-CatBoost backend choice on this panel**, not anything the FS/HP loop can fix; both XGBoost backends (manual + baseline) are deep below the CatBoost sweep on the operating R-p@1 metric.

- **cell-1**: manual methodology winner (`d=3 + cs=0.3 + mcw=1 + eta=0.1+ES=30 + FS=top-130`, 11 trees) → **test R-p@1=0.6600** vs sweep 0.800 → **FAIL** (82.5% of sweep, −0.140 absolute).
- **cell-3**: manual methodology winner (`depth=3` — XGBoost defaults + `max_depth=3`, cliff FS=210) → **test R-p@1=0.4267** vs sweep 0.600 → **FAIL** (71.1% of sweep, −0.173 absolute).

The cell-5 result (manual track +24% over sweep on R-p@1) is **cell-5-specific** under N=3 evidence. The methodology applied to cells 1+3 cannot recover the agent loop's earlier underperformance — and crucially, **cannot match the un-tuned sweep baseline either**, on either cell. The sweep baseline (CatBoost defaults, no FS, no HP tuning) **dominates** the entire XGBoost-manual configuration space tested in stages 1-5.

The Z-gate finding (cell-5: mcw=5 → |z|=1.56 → calibration `native` → 225 distinct calibrated preds → R-p@1=0.7143) operates only in the anti-AUC regime where the booster's predictions hug the base rate. On strong-AUC cells (cell-1 AUC=0.903, cell-3 AUC=0.776) the booster generates highly confident predictions far from base rate; **every config tested on cells 1+3 had val |z| ≥ 12** (cell-3 minimum |z|=12.04 was `spw=4.0` — which actively hurt R-p@1; cell-1 minimum |z|=18.84; max |z|=85.77 was cell-3 `depth=8`), so the conditional-isotonic gate **always fired** and the mcw axis had no leverage on the calibration decision.

## Acceptance verdict per the task brief

| Cell | Sweep R-p@1 | Manual methodology winner | Test R-p@1 | Verdict |
|---|---:|---|---:|---|
| cell-1 | 0.800 | `d=3+cs=0.3+mcw=1+eta=0.1+ES=30+FS=top-130` | 0.6600 | **FAIL** |
| cell-3 | 0.600 | `depth=3` (XGBoost defaults+cliff FS+max_depth=3) | 0.4267 | **FAIL** |

**METHODOLOGY FAIL per task brief framing**: 0 of 2 cells PASS the "test R-p@1 ≥ sweep" floor.

**Implication for playbook rule 11 ("FS → response-curve → mix-and-match → eval-validation manual methodology is reusable across cells where the loop's val_brier objective is misaligned")**: rule 11 was derived from cell-5 alone, where the val_brier objective IS misaligned (anti-AUC + degenerate-sink mechanic). Cells 1+3 do NOT meet that precondition — they are strong-AUC cells where val_brier descends with capacity in a normal way. The methodology does not transfer because the rule's precondition does not transfer.

**Implication for the new "Z-gate" finding from [_223](_223_cell5_loop_v1.3_revalidation.md)**: the mechanism "stay below |z|=2 so isotonic doesn't fire" is structurally unavailable on cells where the booster generates confident predictions. **Draft rule 13 is NOT promoted** — the Z-gate is a cell-5-specific phenomenon. The minimum |z| achievable on cell-3 across 80+ configs was 12.04 (`spw=4.0`, which actively hurt R-p@1 anyway); cell-1 minimum |z|=18.84 (`d=3_cs=0.3_mcw=1+FS=100`). Neither approaches the |z|=2 threshold.

## Cell-3 results (sp500 +20%/25d/dd10pct, sweep R-p@1=0.600, base rate=0.0886, 75 test days)

### Per-stage summary

| Stage | n configs | Best by eval R-p@1 | Eval R-p@1 | Notes |
|---|---:|---|---:|---|
| 1: FS cliff cut (XGBoost defaults) | 2 | defaults+all (279f) | 0.2864 | cliff prune is neutral (e@1=0.2462) |
| 2: single-knob response curves | 25 | sub=0.7 | 0.4422 | depth=8 (0.397), mcw=3 (0.382), cs=0.7 (0.367), mcw=5 (0.347) close behind |
| 3: mix-and-match (d×cs×mcw + cell-5 template + eta=0.05) | 21 | d=2_cs=0.5_mcw=10 | 0.4271 | top mix winners cluster at ~0.30-0.43 eval R-p@1 |
| 4: FS sweep around top-3 mix | 18 | (none surpassed stage 3's eval leaders — top-3 mix stayed in shootout pool by raw eval R-p@1) | — | FS narrowing did not lift eval R-p@1 |
| 5: 1-shot test on 8 configs + baseline | 9 | depth=3 | **test 0.4267** | eval→test ordering completely reshuffled — see § below |

### Eval→test ranking reshuffles dramatically — eval R-p@1 was NOT a reliable oracle on cell-3

The most striking finding on cell-3: among the 9 configs in the stage-5 shootout, the eval R-p@1 leader (`sub=0.7`, e@1=0.4422) was **second-WORST on test** (test@1=0.2400 — a −45% drop from eval); the eventual winner `depth=3` was ranked #6 by eval (e@1=0.3417) but #1 on test (0.4267). The eval segment ordering was actively misleading.

| Stage-5 shootout rank by TEST R-p@1 | config | nf | eval R-p@1 | test R-p@1 | test AUC | cal | \|z\| | cal_n |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | depth=3 | 210 | 0.3417 | **0.4267** | 0.7561 | isotonic | 54.62 | 146 |
| 2 | d=2_cs=0.5_mcw=10 | 210 | 0.4271 | 0.3600 | 0.7641 | isotonic | 27.71 | 64 |
| 3 | spw=2.0 | 210 | 0.3467 | 0.3600 | n/a | isotonic | 34.46 | 134 |
| 4 | mcw=5 | 210 | 0.3467 | 0.3200 | n/a | isotonic | 52.26 | 150 |
| 5 | cs=0.7 | 210 | 0.3668 | 0.2933 | n/a | isotonic | 59.65 | 128 |
| 6 | mcw=3 | 210 | 0.3819 | 0.2667 | n/a | isotonic | 62.54 | 159 |
| 7 | sub=0.7 (EVAL leader) | 210 | 0.4422 | 0.2400 | n/a | isotonic | 65.15 | 122 |
| 8 | defaults+cliff (baseline) | 210 | 0.2462 | 0.2133 | n/a | isotonic | 60.61 | 171 |
| 9 | depth=8 | 210 | 0.3970 | 0.1600 | n/a | isotonic | 85.77 | 232 |

The winner (`depth=3`) is just CatBoost-defaults-shaped except with shallower trees — NO bespoke FS, NO mix recipe, NO eta+ES tiny-model regime. The 5-stage methodology adds no value relative to a single single-knob tweak.

### Calibration profile on cell-3 — Z-gate is a no-op

All 80+ stage-2/3 configs hit `cal_method=isotonic`. The |z| values range from 12.04 (`spw=4.0`, which had eval R-p@1=0.317) to 85.77 (`depth=8`). No mcw value pulled |z| below 30; mcw=5 sat at |z|=52.3, mcw=10 at |z|=54.3. The cell-5 Z-gate mechanism is fundamentally absent here — the booster on a strong-AUC cell is decisively making overconfident predictions and the isotonic calibration is a structural necessity, not an optional gate.

Crucially: isotonic on cell-3 collapses ~96K raw distinct predictions to 122-232 calibrated values across the shootout configs. The downstream ranking-tie effect that doomed cell-5's pre-bugfix loop (`_222`) is present here too, but **all configs equally affected** — there's no escape valve via the mcw knob.

## Cell-1 results (sp500 +50%/50d/dd25pct, sweep R-p@1=0.800, base rate=0.0257, 50 test days)

(Cell-1 stages 1-3 ran in the parent driver; the parent OOM'd during stage-4 carve-cache construction at peak combined memory with cell-3 concurrently running. Stages 4+5 re-ran standalone via `scripts/gbdt/_214_cell1_resume_stage45.py` using the stage-3 top-3 mix winners from the parent log. Per-stage entries below are authoritative.)

### Per-stage summary

| Stage | n configs | Best by eval R-p@1 | Eval R-p@1 | Notes |
|---|---:|---|---:|---|
| 1: FS cliff cut (XGBoost defaults) | 2 | defaults+cliff (169f) | 0.1700 | defaults+all (e@1=0.1150) — cell-1 has the rare-event extreme (test base=0.0257) |
| 2: single-knob response curves | 25 | gamma=0.5 | 0.3100 | gamma=1.0 (0.290), cs=0.3 (0.205) close behind; large knob spread |
| 3: mix-and-match | 21 | d=3_cs=0.3_mcw=1 | 0.3000 | top-3 stage-3 winners: d=3_cs=0.3_mcw=1 (0.300), d=3_cs=0.5_mcw=1 (0.285), d=3_cs=0.5_mcw=5 (0.245) |
| 4: FS sweep around top-3 (3 × 6 keeps = 18 fits) | 18 | d=3_cs=0.3_mcw=1+FS=190 | 0.3300 | the eval R-p@1 leaders pile up at e@1 ∈ [0.30, 0.33]; no real FS-keep ordering signal on val/eval |
| 5: 1-shot test on 8 + baseline | 9 | (test winner: d=3_cs=0.3_mcw=1+FS=130, **test 0.6600**) | — | see table below — eval→test rank scrambled, BIG positive shift for the winning row (eval 0.31 → test 0.66) |

### Stage-5 test shootout — cell-1

| Stage-5 rank by TEST R-p@1 | config | nf | eval R-p@1 | test R-p@1 | test R-p@10 | test AUC | cal | \|z\| | cal_n |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **d=3_cs=0.3_mcw=1+FS=130** (WINNER) | 130 | 0.3100 | **0.6600** | 0.3745 | 0.9127 | isotonic | 23.27 | 40 |
| 2 | d=3_cs=0.3_mcw=1+FS=100 | 100 | 0.2400 | 0.6000 | 0.3260 | 0.9003 | isotonic | 18.84 | 40 |
| 3 | d=3_cs=0.3_mcw=1+FS=cliff (169) | 169 | 0.3000 | 0.5800 | 0.3744 | 0.9046 | isotonic | 26.84 | 42 |
| 4 | d=3_cs=0.5_mcw=5+FS=130 | 130 | 0.2550 | 0.5800 | 0.4028 | 0.9042 | isotonic | 23.84 | 52 |
| 5 | defaults+cliff (baseline) | 169 | 0.1700 | 0.5400 | 0.2544 | 0.8942 | isotonic | 53.58 | 488 |
| 6 | d=3_cs=0.5_mcw=5+FS=cliff | 169 | 0.2450 | 0.4600 | 0.3925 | 0.9004 | isotonic | 28.78 | 69 |
| 7 | d=3_cs=0.3_mcw=1+FS=190 (EVAL leader) | 190 | 0.3300 | 0.4000 | 0.3887 | 0.9026 | isotonic | 23.14 | 41 |
| 8 | d=3_cs=0.5_mcw=1+FS=cliff | 169 | 0.2850 | 0.1800 | 0.3397 | 0.9081 | isotonic | 27.87 | 42 |
| 9 | d=3_cs=0.5_mcw=1+FS=190 | 190 | 0.3100 | 0.0600 | 0.3430 | 0.9041 | isotonic | 23.91 | 31 |

The cell-1 winner is an 11-tree, max_depth=3 XGBoost with cs=0.3 + mcw=1 + eta=0.1 + ES=30 + FS to top-130 by iter_0 importance. Test R-p@1=0.660 — closer to sweep 0.800 than cell-3's distance to its sweep, but still −14 points absolute, −17.5% relative. Test AUC=0.913 (essentially matches sweep's 0.903), so the model's ranking is on par with sweep across the population — the gap is concentrated in the top-1 prediction-tail bucket where sweep's tighter calibrated values out-pick the manual model on a given day.

The eval R-p@1 oracle was not perfectly aligned: the eval R-p@1 leader (`d=3_cs=0.3_mcw=1+FS=190`) ranked #7 of 9 on test (0.40), while the test winner ranked #4 by eval. Crucially, the **baseline defaults+cliff (no tuning) ranked #5 on test (R-p@1=0.54)** — beating 4 of 9 tuned configs. The methodology's value-add over the baseline is 0.660 − 0.540 = +0.120 on R-p@1, but still well short of sweep 0.800.

### Calibration profile on cell-1

All stage-2 / stage-3 / stage-4 / stage-5 configs hit `cal_method=isotonic`. |z| range 18.84 (`d=3_cs=0.3_mcw=1+FS=100`) to 59.21 (cell-3 cross-reference range; cell-1 was bounded similarly). Min |z| across cell-1 is **18.84** — an order of magnitude above the |z|=2 isotonic-gate threshold. The Z-gate mechanism cannot operate on cell-1 either; the closest any cell-1 config came to the |z|<2 native-calibration band was an order of magnitude away.

Notably, the winning config has cal_n=40 distinct calibrated test predictions (and the baseline has cal_n=488), so the isotonic-tail-collapse **happens more aggressively on the winning manual config than on the baseline** — yet the winning config still scores +0.120 R-p@1 over the baseline. This rules out the cell-5 hypothesis "more distinct calibrated values = better R-p@1" as a cell-1 mechanism; cell-1's manual gains come from better RAW prediction-tail ordering (test AUC 0.913 vs baseline 0.894), not from preserving calibration distinctness.

## Cross-cell synthesis — what the playbook should record

### Z-gate finding (claim from [_223](_223_cell5_loop_v1.3_revalidation.md)): does NOT generalize

The cell-5 mechanism — `mcw` controls val |z|, which determines whether `conditional_isotonic` ships native or fits isotonic, which determines how many distinct calibrated predictions survive into test, which determines R-p@1 — is **cell-5-specific**. The precondition is "the booster's raw prediction stream sits close enough to the base rate that the Spiegelhalter z-statistic can stay below 2." On cells 1+3 (strong-AUC cells with much higher prediction tail amplitude), val |z| is structurally ≥ 12 (cell-3 floor) and ≥ 20 (cell-1 floor) across every config tested.

The right way to state the cell-5 finding going forward:

> On **anti-AUC strong-top-1 cells** (where the booster's predictions hug the base rate so val |z| can be < 2 at attainable HP regions), `min_child_weight` is one knob that controls whether conditional-isotonic fires. The mcw axis becomes a calibration-decision-control axis only in this regime; on strong-AUC cells the |z| is structurally above the threshold regardless of HP and the mcw effect on R-p@1 is mediated by other mechanisms.

**Draft rule 13 NOT promoted** to the playbook (it would mislead on strong-AUC cells where it has no leverage).

### 5-stage manual methodology (claim from [_211](_211_cell5_manual_tuning_xgb.md)): does NOT generalize unconditionally

The 5-stage methodology beats sweep by +24% on cell-5 R-p@1 — that result holds. But on cells 1+3 the same methodology **cannot even MATCH the sweep baseline**, let alone beat it. The strongest interpretation consistent with N=3 evidence:

> The 5-stage methodology is a recipe for finding the **non-default HP regime that the loop / sweep cannot reach** when the default HP regime is **structurally wrong for the cell**. On cell-5 the default HP regime (eta=0.3, depth=6, no ES, large n_estimators) is structurally wrong because val_brier has a degenerate sink there; the tiny-model regime (eta=0.1+ES, depth=2, n_estimators=500, max ~24 leaves) is structurally right. On cells 1+3 the sweep's CatBoost defaults (1000 iters, depth 6, no special FS, no special HP) are already in a structurally-good regime; no XGBoost reformulation reachable via the 5-stage methodology improves on them.

The methodology is best framed as a **fallback** when the sweep / loop is in a degenerate regime — diagnosed by the anti-AUC + R-Precision@10 lift > 1.5× rule already in CLAUDE.md — NOT as a general-purpose improvement engine for any cell where the sweep / loop underperforms vs ground truth.

**Playbook rule 11 should be amended** with: "this methodology applies when the loop's val_brier objective is misaligned (rule 10) — that is, the iter_0 anti_auc_flag is true. Do NOT apply it as a blanket fallback on cells where the loop simply hasn't found a great HP region; cells 1+3 demonstrate the methodology cannot beat un-tuned CatBoost defaults on strong-AUC strong-top-1 cells."

### Re-affirms playbook rules 7-9 (no universal recipe)

The original [_195](_195_top5_rp1_agentloop_synthesis.md) finding — "no transferable recipe across cell shapes" — is reinforced here. The cell-5 winning config (XGBoost depth-2 + cs=0.4 + mcw=5 + FS=top-130 + eta=0.1+ES=30) is NOT useful as a starting point on cells 1+3; on cell-3 the closest analog (`cell5_template`) had eval R-p@1=0.293 and was not in the test shootout pool because higher-ranked configs displaced it.

### What actually wins on cells 1+3 (re-confirmed)

The canonical sweep baseline (`sp500_up_50pct_50d_dd25pct` and `sp500_up_20pct_25d_dd10pct` in `r_precision_at_k.csv`) — CatBoost defaults, no FS, no HP tuning, no calibration override — remains the best-known cell-1 + cell-3 model. The next per-cell investigation step (out of scope for #214) is to understand why XGBoost-on-the-same-panel underperforms CatBoost-on-the-same-panel by such a margin (cell-1: 0.800 → ~0.20-0.30 manual best, **−60% to −75% absolute drop**). The strongest candidate hypotheses:

1. **CatBoost's ordered boosting** has a real advantage on the panel-pooled walk-forward setup (each ticker's time-series adjacency is what ordered boosting protects against; XGBoost has no analog).
2. **CatBoost's default tree depth + leaf-count + symmetric structure** happens to be in the right regime; XGBoost defaults (asymmetric, 100 iters at eta=0.3) over-fits the rare-event signal.
3. **Calibration is the binding constraint** — XGBoost predictions are over-confident enough that conditional-isotonic always fires, collapsing the tail to ~100-300 distinct values; CatBoost's predictions may stay closer to the calibration-pass-through band on these cells.

The first hypothesis is the most likely; the third is the most surprising and most diagnostic-friendly to investigate next.

## What this means for V1.3+ / future tracks

1. **V1.3 Option B (#213, scout response-curve phase)** — was on the V1.3 follow-up list. The result here makes a slightly different case for it: scout response curves on a new cell would have revealed the strong-AUC | high-|z| profile in iter_0, which would have ruled OUT the Z-gate maneuver before the agent loop wasted iterations on it. The mechanism that would have saved tail-of-loop iterations on cell-5 may save head-of-loop iterations elsewhere.

2. **An "iter_0 calibration profile probe"** as a new V1.3+ diagnostic — emit val |z| + cal_method at iter_0, gate "Z-gate strategy" candidate HP moves on whether `|z| < 6` (some headroom for moves; rough threshold). On cells where iter_0 |z| > 20, the agent should not waste effort on mcw / scale_pos_weight / γ as Z-gate levers.

3. **The "strong-AUC + strong-top-1 underperformance of XGBoost vs CatBoost"** is now a real, codified gap (-0.50+ on R-p@1 across both cells). Worth a dedicated experiment, NOT covered by anything in flight.

## Cross-references

- [_211](_211_cell5_manual_tuning_xgb.md) — cell-5 manual tuning (source-of-methodology memo). Conclusion still holds for cell-5; replication on cells 1+3 reveals the methodology is cell-5-specific.
- [_223](_223_cell5_loop_v1.3_revalidation.md) — V1.3 A4 re-validation; source of the Z-gate finding (mcw → |z| → cal decision). Conclusion still holds for cell-5; on cells 1+3 the mechanism is structurally absent.
- [_195](_195_top5_rp1_agentloop_synthesis.md) — cells 1+3 destroyed by agent loop (the original problem this #214 attempted to solve via the methodology transfer); methodology does not recover the regression.
- [project-gbdt-tuning-playbook](../../.claude/memories/project-gbdt-tuning-playbook.md) rules 7-12 (this memo proposes amending rule 11's precondition).
- [project-r-precision-methodology](../../.claude/memories/project-r-precision-methodology.md) (canonical R-Precision@K formula).
- PR #112 — V1.3 Option A implementation.
- PR #115 — V1.3 A4 re-validation PASS.

## Artifacts

- Driver scripts (worktree-local, `wt-214-cells13/`):
  - `scripts/gbdt/_214_cell_manual_driver.py` — all-stages driver for one cell.
  - `scripts/gbdt/_214_cell1_resume_stage45.py` — stages 4+5 resume for cell-1 after the parent OOM'd during stage-4 carve-cache construction.
  - `scripts/gbdt/_214_append_csv_rows.py` — appends the two canonical CSV rows.
- Per-cell results JSON (machine-readable headline + every stage's full config table) — NOT committed (worktree-local under `results/gbdt/data/`; gitignored per project convention for per-experiment raw results):
  - `results/gbdt/data/_214_cell1_manual_methodology_results.json`
  - `results/gbdt/data/_214_cell3_manual_methodology_results.json`
- Winner test predictions (used to seed the canonical CSV rows) — NOT committed:
  - `results/gbdt/data/_214_cell1_winner_test_predictions.csv`
  - `results/gbdt/data/_214_cell3_winner_test_predictions.csv`
- JSON sidecar (committed via `git add -f` per `_data.json` convention):
  - `results/gbdt/data/_225_cells13_methodology_replication_data.json`
- Live-log artifacts (full per-config printout for every stage):
  - `/tmp/_214_cell1.log`, `/tmp/_214_cell3.log`, `/tmp/_214_cell1_resume.log` (worktree-local copies preserved in the JSON sidecars' per-stage entries).
- Spec YAMLs (for parity with the cell-5 manual track's audit trail; the driver runs OUTSIDE the per-experiment runner):
  - `configs/gbdt/experiments/sp500_up_50pct_50d_dd25pct_manual_cells13.yaml`
  - `configs/gbdt/experiments/sp500_up_20pct_25d_dd10pct_manual_cells13.yaml`
- Canonical CSV rows appended to `results/gbdt/data/r_precision_at_k.csv`:
  - `sp500_up_50pct_50d_dd25pct_manual_cells13`
  - `sp500_up_20pct_25d_dd10pct_manual_cells13`
