---
name: gbdt-experiment
description: Run a single gbdt experiment end-to-end from a YAML spec — builds the labeled panel for one (universe, direction, threshold, horizon) tuple, runs the synced FS+HP iteration loop (CatBoost), applies conditional isotonic calibration, and emits a self-contained artifact directory with predictions, metrics, iteration history, figures, and a human-readable report.
---

# /gbdt-experiment

End-to-end orchestrator for a single gbdt experiment. The skill consumes a YAML spec, drives the FS+HP iteration loop (the agent reads a diagnostic bundle each iteration and decides feature pruning + HP changes), and produces one artifact directory per invocation.

## When to use

- You have a YAML spec at `configs/gbdt/experiments/<name>.yaml` defining one `(universe, direction, threshold, horizon)` cell.
- You want a complete artifact: model, calibration, iteration history, predictions, figures, report.
- For multi-cell sweeps, invoke this skill once per spec file. v1 ships no sweep runner (parked in `docs/gbdt/V1.1_TBD.md`).

## Invocation

```
/gbdt-experiment <spec_path>
```

`<spec_path>` is a path to a YAML spec (canonical examples in `configs/gbdt/experiments/`). The skill resolves it, validates against the schema in `docs/gbdt/EXPERIMENT_SPEC.md`, and orchestrates the run.

Equivalent CLI atom (for non-agent invocation):

```
uv run python -m gbdt.experiment <spec_path>
```

The CLI atom runs without the per-iteration agent reasoning — it uses default HP starting points and a fixed prune heuristic. Use the skill (not the CLI atom) for production runs; use the CLI atom for smoke tests and CI.

## Pre-flight

Before launching the loop:

0. **Infrastructure checks** (added 2026-05-26 after v1 pilot wedge incident — see `.claude/memories/feedback-disk-wedge-pattern.md`):
   - **Disk space**: `df --output=avail $(pwd) | tail -1` must show ≥10 G of headroom on the volume hosting the repo. Else refuse to start; list candidate worktrees to prune.
   - **No competing experiment processes**: `ps -ef | grep "gbdt.experiment" | grep -v grep` must be empty. Don't run parallel experiments against the same SQLite cache (single-writer contract).
   - **Worktree symlink validation** (if running in a worktree): `readlink data` must return a non-empty absolute path to the shared cache (the main checkout's `data/` or a scratch dir like `~/exp_data`). Empty means `data/` is a real directory with a nested symlink — STOP and report; let the parent fix. See `.claude/memories/feedback-worktree-symlink-contract.md`.
   - **SQLite WAL integrity**: `sqlite3 data/processed.db 'PRAGMA quick_check' 2>&1` must print `ok`. If it errors with "unable to open database file" the WAL is filesystem-corrupted — STOP and use the scratch-dir workaround per `.claude/memories/project-nse-data-quirks.md`.
1. **Validate the spec** — `experiment.load_spec(spec_path)` runs the validation rules in `docs/gbdt/EXPERIMENT_SPEC.md` § "Validation rules". Bail with a clear error if any rule fails.
2. **Universe self-service** — if `spec.target.universe` is not one of the pre-registered presets under `configs/gbdt/default.yaml::universes` (v1 ships `nifty50` only), the agent registers it before going further. Three paths (try in this order):
   - **Inline tickers:** if the spec carries a top-level `tickers:` list, write a new universe YAML at `configs/data_pipelines/domains/nse_equities/universe_<name>.yaml` with those tickers + a `universes::<name>` block in `configs/gbdt/default.yaml` pointing at it.
   - **curl from archives.nseindia.com** (reliable; preferred for well-known NSE indices): `curl -L -A "Mozilla/5.0" https://archives.nseindia.com/content/indices/ind_<name>list.csv`. Known working: `ind_nifty50list`, `ind_niftynext50list`, `ind_nifty100list`, `ind_niftymidcap150list`, `ind_nifty500list`. **Filter `DUMMY*` placeholder tickers** (Vedanta demerger trick — non-tradeable; would otherwise burn ~80s/each on retry storms).
   - **`data_pipelines` adapter chain (jugaad/nselib)** — last resort, often blocked by anti-bot/SSL. Don't waste cycles here if curl works.

   After registration, fall through to step 3 to ensure each ticker is cached.
3. **Check the data cache** — verify `data_pipelines.fetch()` can serve every ticker in the universe over the requested `date_range`. For every ticker that is missing OR cached-but-shallow (rows below `spec.split.min_rows_per_ticker`, default 1,600 = sum of 800+400+200+100), call `data_pipelines.fetch(ticker, start='<spec.date_range.start or 2015-01-01>', end=today, back_extend=True)`. **`back_extend=True` is the default for universe self-service in this skill** — always pass it on every per-ticker fetch, do not gate it on a prior cache check. The alternative (relying on the library default `back_extend=False`) silently drops cached-but-shallow tickers under the row gate without ever attempting to top them up: Exp 2 lost 53/100 nifty100 tickers this way before the lesson was paid for (see `.claude/memories/project-nse-data-quirks.md` § 5). Note that `data_pipelines.fetch()` itself still defaults to `back_extend=False` — the default flip lives here in the skill, not in the library. Run sequentially (respect the SQLite single-writer contract) with a per-ticker 120 s hard timeout (mirror `scripts/seed_nifty50_deep.sh` from PR #6 — without it, one stuck ticker blocks the queue). After the back-extend pass, drop any tickers still below `spec.split.min_rows_per_ticker` with a logged note that lands in `metrics.json::data.tickers_excluded`. If many tickers need a cold pull, surface it and let the user decide whether to proceed; cold pulling 150–500 tickers can take 2–8 hours.
4. **Check the artifact path is free** — if `results/gbdt/experiments/<experiment_name>/` already exists, refuse to overwrite without explicit user confirmation. The artifact dir is the unit of currency; silent overwrites lose the prior result.
5. **Read the references** if unsure on any decision:
   - `docs/gbdt/V1_PLAN.md` § "Stage breakdown" for the layer-by-layer architecture.
   - `docs/gbdt/CATBOOST_HP_REFERENCE.md` for per-parameter "when to change" rubrics — this is the canonical guide for HP decisions inside the loop.
   - `docs/gbdt/EXPERIMENT_SPEC.md` for the YAML schema and artifact layout.

## Phases

### Phase 1 — Data build

- Load the universe panel via `data.load_universe(spec.target.universe, spec.date_range)`.
- Drop tickers below `spec.split.min_rows_per_ticker` (default 1,600). Log the exclusion list.
- Build the candidate feature matrix (`features.build_feature_matrix(panel, spec.features)`) — 279 columns by default.
- Build the binary target (`targets.compute_target(panel, spec.target)`). If `spec.target.max_drawdown` is set, the target builder applies the path-honesty filter described in `EXPERIMENT_SPEC.md` § "target".
- Carve segments per `spec.split` (train / val / eval / test, in time order).

### Phase 2 — Iteration 0

- Start with the full candidate feature pool.
- Use `spec.backend.hp_starting` if provided, else `default.yaml::backend.hp_starting`.
- Fit CatBoost on train, early-stop against val, score val + eval.
- Generate the diagnostic bundle (`fs_hp_loop.DiagnosticBundle`): train/val Brier, train-val gap, learning curve, early-stop iteration, feature importance, correlation matrix, calibration summary (Spiegelhalter Z on raw val preds), positive-class recall.
- Log iteration 0 to `iterations.jsonl` with rationale `"iteration 0 — full feature pool, default HPs"`.

### Phase 3 — FS+HP loop (iterations 1..N, N ≤ max_iterations)

Each iteration:

1. **Read the previous iteration's diagnostic bundle.** Look at: train Brier, val Brier, the gap between them, reliability deviation, val curve shape (descending? plateau? oscillating?), early-stop iteration (was the cap hit?), top-K feature importances, pairwise correlation among top features, positive-class recall.

2. **Decide the prune list.** Drop features per these heuristics (composed with judgment, not as hard rules):
   - **Low importance:** features with importance ≤ a small fraction (e.g. 1%) of the top feature and no domain reason to keep.
   - **Redundant:** of two features with |corr| > 0.95, drop the one with lower importance.
   - **Dominant noise:** if one feature has importance >50% of total and val Brier is plateaued, consider down-weighting via `feature_weights` or dropping it to see if the rest of the pool can compensate.
   - **Never drop:** the seed features (e.g. `stock_return_20`, `realized_vol_20`) unless explicitly justified — they anchor the cell's basic signal.

3. **Decide the HP changes.** Walk through `CATBOOST_HP_REFERENCE.md` § "Suggested per-iteration agent prompt" (Categories 1–11 + the 10-step decision rubric at the end). Common patterns:
   - Overfit signal (`train Brier << val Brier`): raise `l2_leaf_reg`, then `min_data_in_leaf`, then drop `depth`, then `rsm=0.7`.
   - Underfit signal (both Briers flat-high): raise `depth`, then raise `iterations` and lower `learning_rate` proportionally.
   - Rare cell + low recall (positives <5% AND recall ~0): set `auto_class_weights="SqrtBalanced"`; **always re-check calibration after weighting** (weights bias probabilities).
   - Cap hit (best iter within 10% of `iterations`): raise `iterations` 2× and raise `early_stopping_rounds` proportionally.
   - One-feature dominance: raise `random_strength` and/or set `rsm=0.7`.

   Stay within the documented ranges in `CATBOOST_HP_REFERENCE.md` per parameter; do not change pinned HPs (`has_time=True`, `loss_function`, `eval_metric`, `random_seed`).

4. **Record the rationale.** Every change must be logged with the signal that triggered it. The agent's value-add is the chain of reasoning, not the parameter values.

5. **Re-fit, re-score, generate the new diagnostic bundle, append to `iterations.jsonl`.**

6. **Check inner-stop:**
   - **Plateau:** val Brier improvement < `plateau_threshold` (default 0.005) over the last 2 iterations → stop.
   - **Degradation:** val Brier > `(1 + degradation_gate) × best_val_brier` (default `degradation_gate=0.01`) → stop.
   - **Cap:** iteration count ≥ `max_iterations` (default 8) → stop.

   Record the inner-stop signal in the iteration's row.

### Phase 4 — Calibration

- Apply `spec.backend.calibration_method` (default `conditional_isotonic`):
  - Run Spiegelhalter Z-test on the **best-checkpoint** iteration's raw val predictions.
  - If |z| < calibration_z_threshold (default 2.0) → ship native CatBoost outputs.
  - Else → fit `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` on val and persist.
- Record decision + Z + p-value in `metrics.json::calibration`.

### Phase 5 — Artifact emission

- Persist the best-checkpoint model (`model.cbm`) and calibrator (`calibration.pkl` or `None` marker).
- Write `features.yaml` (final pruned list) and `hp.yaml` (final HP dict).
- Score train/val/eval/test segments and emit `predictions/<segment>.csv`.
- Render figures into `figs/` (reliability diagram, calibration curve, per-iter learning curves, feature importance, train-val gap history).
- Compute headline metrics on eval (+ test sanity check) into `metrics.json`.
- Render `report.md` via `report.py` (sections per `V1_PLAN.md` Stage 8). The **per-experiment verdict** section is the agent's one-paragraph readout: "this cell looks calibrated and shippable" / "this cell is miscalibrated; recommend X" / "this cell is genuinely hard; the data doesn't support a model here." Explicitly not an automated pass/fail.

## Long-running pattern

Pick by **expected wall time of the compute being launched** (not iteration count):

**Sub-30-min runs** (cached-data nifty50 experiments, smoke tests): **foreground with `timeout` as a hard cap**. Do NOT background+Monitor — see `.claude/memories/feedback-sub-agent-foreground.md` for why (agent sessions exit prematurely under Monitor; orphans the run).
```bash
timeout 1800 uv run python -m gbdt.experiment <spec.yaml> 2>&1 | tee logs/exp_<name>.log
```
Stay in the session until `timeout` returns or the command exits naturally. Then read the artifact and report.

**Sub-2-hour runs** (single-universe self-service for ~100 tickers + experiment): foreground with `timeout 7200`. Still single-shell, agent stays alive.

**≥2-hour runs** (cold-pull universe self-service for 150+ tickers, multi-fold walk-forward tunes): background + Monitor + `ScheduleWakeup` chain. This is the original pattern:
> 1. Launch the long shell with `Bash run_in_background=true` and a stable log path (e.g. `logs/<experiment_name>_<UTC>.log`).
> 2. Use `Monitor` with a filtered `tail -f <log> | grep -E --line-buffered "iter=|elapsed|inner_stop|Error|Traceback|completed"` so progress events stream in without flooding context.
> 3. Use the `/schedule` skill (which wraps the underlying ScheduleWakeup capability) to self-pace — schedule a wakeup at 1200–1800 s with a self-contained prompt that re-checks progress, decides whether to schedule the next wake, **and explicitly chains to the next phase when the background work completes** (the agent that launched the background work may have exited its session; the wakeup is the chain).
> 4. Never block one tool call for hours — that wastes the session and prevents you from being interruptible.

See `.claude/memories/feedback-experiment-agent-loop.md` for the long-running guidance and `.claude/memories/feedback-sub-agent-foreground.md` for why the sub-30-min case differs.

The parent will check in periodically and the auto-completion notification reaches the parent directly — no need to do anything special to "signal done."

## References

- `docs/gbdt/V1_PLAN.md` — architecture, stage breakdown, decisions log.
- `docs/gbdt/EXPERIMENT_SPEC.md` — YAML schema, artifact layout, validation rules.
- `docs/gbdt/CATBOOST_HP_REFERENCE.md` — per-parameter "when to change" rubrics; the canonical guide for HP decisions inside the loop.
- `docs/gbdt/goal.md` — why the experiment-loop framing exists; per-experiment success criteria.
- `docs/gbdt/V0_INVESTIGATION_PLAN.md` — v0 base-rate findings that inform cell choice.
- `docs/gbdt/V1.1_TBD.md` — parked extensions (NDX, macro features, multi-target, Optuna HP search).
- `.claude/memories/feedback-experiment-agent-loop.md` — long-running compute pattern.
- `.claude/memories/project-overview.md` — module overview.
