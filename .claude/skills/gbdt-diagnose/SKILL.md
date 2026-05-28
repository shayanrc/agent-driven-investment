---
name: gbdt-diagnose
description: Emit a full diagnostic bundle for one fitted gbdt cell artifact — numeric feature importance, prevalence drift across segments, marginal monotonicity + model 1D-PDP audit per feature, pairwise interaction strength, correlation heatmap, pruned-feature redundancy investigation, and an auto-flagged tuning-guidance section keyed to the FS+HP tuning playbook. Read-only over the model; informs (does not perform) tuning decisions.
---

# /gbdt-diagnose

A single verb: **given a fitted gbdt cell artifact, produce the diagnostic bundle a data scientist would assemble before deciding how (or whether) to tune it.** It consolidates the five exploratory scripts from the nifty50 H=25 study (`docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md`) into one parametrized tool that works on any artifact dir.

This is **read-only over the model** — it never re-fits or mutates the artifact. It answers "what is this model doing, and what should I try next?", not "tune it for me." Tuning decisions stay with the agent/human (the judgment that `default_fs_hp_callback` can't supply — see `.claude/memories/project-gbdt-tuning-playbook.md`).

## When to use

- After a `/gbdt-experiment` run, before deciding on a tuning iteration (prune? change HP? constrain?).
- To decide whether a cell has FS/HP headroom at all, or is at its ceiling.
- To check, *before* applying monotone constraints, whether they're safe (they usually aren't — the skill flags it).
- To understand why feature selection helped / didn't (redundancy vs noise).

## Invocation

```
/gbdt-diagnose <artifact_dir>
```

`<artifact_dir>` is a fitted-cell artifact directory (contains `model.cbm`, `features.yaml`, `spec.yaml`, `predictions/`, and ideally `iterations.jsonl`). Canonical examples live under `results/gbdt/experiments/<name>/`.

CLI atom (non-agent / CI):

```
uv run python -m scripts.gbdt.diagnose <artifact_dir> [--top-n 30] \
    [--importance-threshold 0.01] [--out <dir>] [--no-pdp] [--no-figs]
```

- `--top-n` (default 30): how many top-importance features to deep-dive (monotonicity + 1D-PDP).
- `--importance-threshold` (default 0.01): the kept/pruned boundary for the redundancy investigation.
- `--no-pdp`: skip the per-feature 1D-PDP audit (faster; you lose the model-monotonicity column).
- `--out` (default `<artifact_dir>/diagnose`): output directory.

## What it computes

1. **Feature importance** — numeric top-N (the runner only emits a PNG).
2. **Prevalence drift** — positive-rate across train/val/eval/test from the saved predictions; flags non-stationarity (the calibration-ceiling signal — when the brier is capped by a regime shift the FS/HP loop can't touch).
3. **Marginal monotonicity** — per top feature: Spearman ρ(feature, target) + decile-consistency on in-sample rows. "Should this feature be monotone, and which way?"
4. **Model 1D-PDP audit** — per top feature: is the *fitted model* monotone in it? (The necessary-but-not-sufficient pre-check before constraining — a marginally-monotone feature can have a learned inverted-U the model uses.)
5. **Interaction strength** — top pairwise interactions + per-feature involvement, split into high/low. The features the model leans on for *conditional* structure.
6. **Correlation heatmap** — Spearman collinearity among the top features (`figs/corr_heatmap.png`). Explains redundancy.
7. **Pruned-feature investigation** — for sub-threshold-importance features: do they have a real monotone relationship, and are they redundant (collinear with a kept feature) vs genuine noise?

## Output

- `<out>/diagnose_report.md` — human-readable bundle, led by a **Tuning guidance** section that auto-applies the playbook rules:
  - *No overfit* (train/val gap ≥ 0, early-stop didn't fire) → **don't prune** (rule 1).
  - *Prevalence drift* across segments → calibration ceiling; the lever is recency / regime-conditional calibration, **out of the FS/HP loop** (rule 5).
  - *Pruned features mostly redundant, not noise* → FS will be neutral, not an accuracy win (rule 2).
  - *Per-feature monotone-constraint advice* — AVOID (high interaction) / AVOID (learned non-monotone) / NEUTRAL-at-best (rules 3–4).
- `<out>/diagnose.json` — all numerics, for programmatic use (e.g. as the per-iteration bundle a V1.1 agent-driven loop reads — see PR #48).
- `<out>/figs/corr_heatmap.png`.
- `<out>/_insample_matrix.parquet` — cached in-sample feature matrix (reused on re-run, so repeat diagnoses are fast).

## Pre-flight

0. **Infrastructure** (same contract as `/gbdt-experiment`): disk ≥10 G headroom; `readlink data` resolves to the shared cache; `sqlite3 data/processed.db 'PRAGMA quick_check'` prints `ok`. See `.claude/memories/{feedback-disk-wedge-pattern,feedback-worktree-symlink-contract,project-nse-data-quirks}.md`.
1. **Artifact validity** — `<artifact_dir>` must contain `model.cbm`, `features.yaml`, `spec.yaml`. `predictions/*.csv` enable the prevalence-drift check; `iterations.jsonl` enables the overfit read. The skill degrades gracefully if the optional pieces are missing (those sections are skipped).
2. **Cache coverage** — the cell's universe must be cached (the skill rebuilds the in-sample feature matrix via `data.load_panel` + `features.build_feature_matrix`). If the universe isn't cached, run `/gbdt-experiment`'s pre-flight first.

## Long-running pattern

The matrix build dominates and is **strongly universe-size-dependent**: ≈5–6 min for a ~50-ticker NSE panel, but **~28 min for a US panel** (nasdaq100, ~100 tickers — measured 2026-05-28), and longer still for sp500/russell1000. The per-feature 1D-PDP loop adds a few minutes. Run **foreground with `timeout`** (do NOT background+Monitor — see `.claude/memories/feedback-sub-agent-foreground.md`), but size the cap to the universe — a US cell nearly tripped a 1800s cap:

```bash
# NSE (~50 tickers): ~10 min
timeout 1800 uv run python -m scripts.gbdt.diagnose <nse_artifact> 2>&1 | tee logs/diagnose_<name>.log
# US / large universes (nasdaq100/sp500/russell1000): build ~28 min+, use a bigger cap
timeout 5400 uv run python -m scripts.gbdt.diagnose <us_artifact> 2>&1 | tee logs/diagnose_<name>.log
```

Re-runs reuse the cached `_insample_matrix.parquet`, so iterating on `--top-n` / `--no-pdp` is fast (the expensive build is skipped). `--no-pdp` also drops the per-feature 1D-PDP loop if you only need importance/interaction/pruned analysis.

## What it does NOT do

- It does **not** tune, prune, fit, or constrain anything — it's diagnostic-only. Feed its output to your tuning judgment (or to the `/gbdt-experiment` FS+HP loop).
- It does **not** apply monotone constraints; it *advises against* them where they'd bind on interaction-heavy or non-monotone features.

## References

- `.claude/memories/project-gbdt-tuning-playbook.md` — the 6 rules the tuning-guidance section applies.
- `docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md` — the study this generalizes; worked examples of every diagnostic.
- `docs/gbdt/CATBOOST_HP_REFERENCE.md` — per-parameter HP rubrics for acting on the diagnosis.
- `.claude/skills/gbdt-experiment/SKILL.md` — the run verb; `/gbdt-diagnose` consumes its artifacts.
