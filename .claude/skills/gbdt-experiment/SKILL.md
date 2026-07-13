---
name: gbdt-experiment
description: Run a single gbdt experiment end-to-end from a YAML spec — builds the labeled panel for one (universe, direction, threshold, horizon) tuple, runs the synced FS+HP iteration loop (CatBoost), applies conditional isotonic calibration, and emits a self-contained artifact directory with predictions, metrics, iteration history, figures, and a human-readable report.
---

# /gbdt-experiment

End-to-end orchestrator for a single gbdt experiment. The skill consumes a YAML spec, drives the FS+HP iteration loop (the agent reads a diagnostic bundle each iteration and decides feature pruning + HP changes), and produces one artifact directory per invocation.

> **Use the canonical evaluation periods for any new training / fine-tuning** (set 2026-07-13; supersedes the `2019-01-01` D2 anchor): **train** 2015-01-01→2022-03-29 (fit) · **val** 2022-03-30→2023-06-30 (feature selection + early stopping) · **eval** 2023-07-01→2024-06-30 (hyperparameter tuning) · **test** 2024-07-01→2025-06-30 (final evaluation + comparison). Keep 2025-07-01→2026-06-30 reserved for **backtesting only** — never train/select on it. One role per window. Specify them with the `date_aligned` **explicit-boundary form** — `split.train_start` + `val_start`/`eval_start`/`test_start`/`test_end` (all four required together; literal dates, NOT row-count durations) — then verify `metrics.json::segment_dates` match the table. See `[[project-canonical-evaluation-periods]]` + `docs/gbdt/EXPERIMENT_SPEC.md` § `split`.

**Two loop modes** (`backend.fs_hp_loop.callback_mode`):

- **`default`** — the runner drives the loop end-to-end with a fixed algorithmic callback (importance-based prune + a small HP nudge). One process, no agent in the loop. This is the CLI-atom path and the default for unattended runs.
- **`agent_file_protocol`** (V1.1) — **the agent (this session) IS the data scientist in the loop.** The runner trains one iteration, writes a diagnostic bundle + a resume checkpoint, and **exits**; the agent reads the bundle, makes a feature-pruning + HP-tuning decision, writes it to a file, and relaunches with `--resume`. This is the production agent surface — it lets the agent reason about each iteration the way `docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md` did by hand, instead of applying one fixed heuristic. Documented in § "Agent-driven FS+HP loop" below.

## When to use

- You have a YAML spec at `configs/gbdt/experiments/<name>.yaml` defining one `(universe, direction, threshold, horizon)` cell.
- You want a complete artifact: model, calibration, iteration history, predictions, figures, report.
- For multi-cell sweeps, invoke this skill once per spec file. v1 ships no sweep runner (parked in `docs/gbdt/V1.1_TBD.md`).

## What to expect by cell type (cross-experiment priors)

Learnings from the completed US sweep (#107) snapshot + standalone cells, distilled in `docs/gbdt/_177_cross_experiment_analysis.md` (n ≈ 12 US sweep cells + 4 cross-market context cells; **provisional** — the full 57-cell sweep is unfinished). Use these as priors when reading a fresh run's report; they are NOT a substitute for the per-cell diagnostics.

- **Eval→test AUC decay is systemic — never trust eval AUC alone.** Every completed cell loses AUC eval→test (`dAUC` always negative, −0.02 to −0.14). The decay magnitude scales with horizon within a threshold family: nasdaq +10% goes 5d −0.04 → 10d −0.13 → 25d −0.14 (drops into the [0.45,0.55] null band on test) → 100d (no test window). **Always judge a cell on its TEST R-Precision@K, not eval AUC.** A strong eval AUC that decays into the null band on test is the expected failure mode for longer horizons, not a bug.
- **Short horizons (5d–10d) are the cleanest, and signal survives to test.** R-Precision@K lift is highest at 5d and declines monotonically with horizon. The 5d–10d up-move band on large US universes (sp500 ≳ russell1000 ≈ nasdaq) is the reliable, production-candidate region.
- **Higher thresholds (+20%, +40%) → much higher lift on a smaller base, modest absolute R-Precision@K.** E.g. sp500 +20%/5d: base ~0.6%, R-Precision@10 ~0.32 (lift ~53× test). These are "rare-but-clean top picks"; +10% cells are "more frequent, broader hit rate" (R-Precision@10 ~0.32-0.44 on 3-8% base, lift ~5–11×). They stay discriminating on test and decay less than the long-horizon +10% cells.
- **Bigger panel → stronger lift** (more positives/day → deeper F14 cross-section). sp500 > nasdaq at matched thresholds; nifty500 +30%/50d (lift ~3× test) beats nifty50 H=25 (~2×) cross-market. (n=1 per universe at most thresholds — treat as a prior, not a proven ablation.)
- **H ≥ 50 may produce zero test rows** under the standard 800/400/200/100 split — H=100 consumes the entire 100-row per-ticker test segment, leaving `n_rows_test=0` and no test metrics (nasdaq +10%/100d). If running a long-horizon cell, expect this and widen the per-ticker test allocation in the spec, or treat eval as the only held-out signal.
- **Ranking survives prevalence drift; calibrated-probability Brier often does not.** Multiple cells beat baseline on AUC/R-Precision@K while test Brier underperforms the base-rate constant (isotonic fit on a higher-prevalence val window over-predicts on a lower-prevalence test window). The trading rule consumes ranking — report R-Precision@K as the headline under drift, and flag Brier as a calibration-under-drift artifact, not a discrimination failure.
- **FS prunes hard with neutral-to-positive effect.** Final feature counts cluster at ~25–65 (out of 279) on most cells; the signal is low-complexity (a market-regime + volatility-level model). Aggressive pruning is the norm — don't be surprised by a 279→40 prune.
- **On US panels, P@k descends with k (top picks cleanest); on staggered NSE panels P@k ascends with k** — the latter is an artifact of R(d) < k days, NOT weak top picks (see the `_138` erratum). Read the prediction-extreme regime for the real top-tail signal.

## Invocation

```
/gbdt-experiment <spec_path>
```

`<spec_path>` is a path to a YAML spec (canonical examples in `configs/gbdt/experiments/`). The skill resolves it, validates against the schema in `docs/gbdt/EXPERIMENT_SPEC.md`, and orchestrates the run.

Equivalent CLI atom — `default` mode (no agent in the loop):

```
uv run python -m gbdt experiment <spec_path>
```

This runs without the per-iteration agent reasoning — it uses default HP starting points and a fixed prune heuristic, finalizing in one process. Use it for smoke tests and CI.

Agent-driven mode (`callback_mode: agent_file_protocol`) — the runner pauses each iteration and the agent drives it:

```
# launch (trains iter 0, writes the bundle + checkpoint, exits at the pause)
uv run python -m gbdt experiment <spec_path> --callback-mode agent_file_protocol
# ... agent reads loop/iter_0_request.json, writes loop/iter_0_decision.json ...
# resume (applies the iter-0 decision, trains iter 1, pauses again)
uv run python -m gbdt experiment <spec_path> --resume <run_id>
```

`<run_id>` is the value printed in the pause hint (it equals the spec file's stem; the artifact dir is `results/gbdt/experiments/<run_id>/`). Use the agent-driven mode for production runs where per-iteration judgment matters. Both modes load through the same `load_spec()` validator and emit the same final artifact set.

> CLI subcommand note: the entrypoint is `python -m gbdt experiment …` (space, the `experiment` subcommand), not `python -m gbdt.experiment …`. The flags are `--callback-mode {default|agent_file_protocol}`, `--resume <run_id>`, `--snapshot-end YYYY-MM-DD`, and `--overwrite`.

### Sweep orchestration — pin `--snapshot-end`

When running **≥2 cells back-to-back** (a sweep — same universe across multiple `(threshold, horizon, drawdown)` cells), every cell MUST be invoked with the **same** `--snapshot-end <today's UTC date>` value. This pins `date_range.end` for the lifetime of each run so the **universe-level feature cache key stays stable across cells** — every sibling hashes to the same `panel_signature` and shares the build (one cold rebuild + N warm hits, instead of N cold rebuilds at 5 min – 3 h apiece).

Without the pin, an auto-fetch between cells (e.g. the V1.4 Phase 0b cache-currency check appending a new trading bar to `processed.db`, or a scheduled refresh landing during the sweep) shifts `panel.date_max` / the `(date, ticker)` tuple set, drifts the universe-cache key, and forces every subsequent cell to rebuild the universe matrix from scratch. This is the bug-#226 pathology — three v2 russell1000 universe-cache keys with identical `(n_rows, n_cols)` shapes, written 2 days apart, that all should have collapsed to one shared entry. The in-process repro is bit-identical; the drift is strictly between-process.

Recipe:
```bash
END=$(date -I)   # today's UTC date — or any past date for reproducibility
for spec in configs/gbdt/experiments/<universe>_*.yaml; do
  uv run python -m gbdt experiment "$spec" --snapshot-end "$END"
done
```

`metrics.json::preflight.snapshot_end_override` records the pinned ISO date (or `null` when the flag is omitted) so post-hoc audits can confirm every cell of a sweep used the same snapshot. `metrics.json::cache.universe_hit == true` on every sibling after the first is the operational signal that the pin is working.

**Re-pass `--snapshot-end` on every `--resume`.** In `--callback-mode agent_file_protocol` the runner exits between iterations and argparse re-parses argv from scratch — the pin doesn't survive process exit. Each `--resume` invocation must carry the same `--snapshot-end <date>` value used at launch. Without it, the resumed cell loads the panel with `end=None`, rebuilds against a different snapshot, and the cell's iter-by-iter feature matrix silently drifts. (Single-process `default` mode is unaffected.)

**Phase 0b auto-fetch will mutate `processed.db` mid-sweep — that's by design.** When a sub-case-B currency check appends new bars to the cache between cells, sweep correctness is not compromised: with `--snapshot-end` pinned, `load_panel` sees a constrained view of the panel up to `end`, so post-`end` bars don't change the loaded data and `panel_signature` stays stable. The cross-cell sharing contract depends on the pinned end, not on the cache being frozen. See `src/gbdt/preflight.py::cache_currency_check` and PR #120 / bug #226 for the rationale.

**Single-cell runs don't need the flag** — the sharing dependency only matters when ≥2 cells run with overlapping universe-cache writes. The flag is also a no-op on cells whose spec already pins `date_range.end` to a past, fixed date (it just overrides one fixed value with another).

## Pre-flight

Before launching the loop:

0. **Infrastructure checks** (added 2026-05-26 after v1 pilot wedge incident — see `.claude/memories/feedback-disk-wedge-pattern.md`):
   - **Disk space**: `df --output=avail $(pwd) | tail -1` must show ≥10 G of headroom on the volume hosting the repo. Else refuse to start; list candidate worktrees to prune.
   - **No competing experiment processes**: `ps -ef | grep "gbdt.experiment" | grep -v grep` must be empty. Don't run parallel experiments against the same SQLite cache (single-writer contract).
   - **Worktree symlink validation** (if running in a worktree): `readlink data` must return a non-empty absolute path to the shared cache (the main checkout's `data/` or a scratch dir like `~/exp_data`). Empty means `data/` is a real directory with a nested symlink — STOP and report; let the parent fix. See `.claude/memories/feedback-worktree-symlink-contract.md`.
   - **SQLite WAL integrity**: `sqlite3 data/processed.db 'PRAGMA quick_check' 2>&1` must print `ok`. If it errors with "unable to open database file" the WAL is filesystem-corrupted — STOP and use the scratch-dir workaround per `.claude/memories/project-nse-data-quirks.md`.
   - **V1.4 cache-currency check (date-aligned mode only)**: when `spec.split.mode == "date_aligned"`, the runner's Phase 0 verifies the cache covers `test_end + horizon_days` for every ticker in the universe (plan §6 / `gbdt.preflight.cache_currency_check`). Sub-case A (universe-level shortfall — `test_end + horizon` past today, or the universe calendar doesn't extend that far) → run aborts with `CacheCurrencyError` and exit 2. Sub-case B (per-ticker right-edge staleness) → auto-fetch with 3-retry exponential backoff (1 s / 4 s / 16 s) on transient errors (`timeout`, `ConnectionError`, HTTP 429, HTTP 5xx, connection refused/reset). Definitive failures (HTTP 404, malformed/schema mismatch) skip retries. After retries, deficient tickers populate the worked-example REFUSE table (`ticker, latest_cached, needed_through, days_short, last_error_type`) and the run aborts with exit 2. Sub-case C (late-IPO left-edge) is auto-handled by the train-segment membership gate (`min_train_rows_per_ticker`, default 200 = `max(lookback_windows)`) — no REFUSE.
   - **V1.4 split-mode nudge**: when `spec.split.mode` is unspecified the runner emits a one-line info-log nudge at startup: `[split] info: spec.split.mode unspecified, defaulting to 'trailing'. New work should prefer 'date_aligned' (V1.4) — see docs/gbdt/V1.4_date_aligned_splits_plan.md.` Trailing mode remains the default for back-compat with the 38+ existing trailing-anchor specs.
1. **Validate the spec** — `experiment.load_spec(spec_path)` runs the validation rules in `docs/gbdt/EXPERIMENT_SPEC.md` § "Validation rules". Bail with a clear error if any rule fails.
2. **Universe self-service** — if `spec.target.universe` is not one of the pre-registered presets under `configs/gbdt/default.yaml::universes` (v1 ships `nifty50` only), the agent registers it before going further. Three paths (try in this order):
   - **Inline tickers:** if the spec carries a top-level `tickers:` list, write a new universe YAML at `configs/data_pipelines/domains/nse_equities/universe_<name>.yaml` with those tickers + a `universes::<name>` block in `configs/gbdt/default.yaml` pointing at it.
   - **curl from archives.nseindia.com** (reliable; preferred for well-known NSE indices): `curl -L -A "Mozilla/5.0" https://archives.nseindia.com/content/indices/ind_<name>list.csv`. Known working: `ind_nifty50list`, `ind_niftynext50list`, `ind_nifty100list`, `ind_niftymidcap150list`, `ind_nifty500list`. **Filter `DUMMY*` placeholder tickers** (Vedanta demerger trick — non-tradeable; would otherwise burn ~80s/each on retry storms).
   - **`data_pipelines` adapter chain (jugaad/nselib)** — last resort, often blocked by anti-bot/SSL. Don't waste cycles here if curl works.

   After registration, fall through to step 3 to ensure each ticker is cached.
3. **Check the data cache** — verify `data_pipelines.fetch()` can serve every ticker in the universe over the requested `date_range`. For every ticker that is missing OR cached-but-shallow (rows below `spec.split.min_rows_per_ticker`, default 1,600 = sum of 800+400+200+100), call `data_pipelines.fetch(ticker, start='<spec.date_range.start or 2015-01-01>', end=today, back_extend=True)`. **`back_extend=True` is the default for universe self-service in this skill** — always pass it on every per-ticker fetch, do not gate it on a prior cache check. The alternative (relying on the library default `back_extend=False`) silently drops cached-but-shallow tickers under the row gate without ever attempting to top them up: Exp 2 lost 53/100 nifty100 tickers this way before the lesson was paid for (see `.claude/memories/project-nse-data-quirks.md` § 5). Note that `data_pipelines.fetch()` itself still defaults to `back_extend=False` — the default flip lives here in the skill, not in the library. Run sequentially (respect the SQLite single-writer contract) with a per-ticker 120 s hard timeout (mirror `scripts/data_pipelines/seed_nifty50_deep.sh` from PR #6 — without it, one stuck ticker blocks the queue). After the back-extend pass, drop any tickers still below `spec.split.min_rows_per_ticker` with a logged note that lands in `metrics.json::data.tickers_excluded`. If many tickers need a cold pull, surface it and let the user decide whether to proceed; cold pulling 150–500 tickers can take 2–8 hours.
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

### Phase 1.4–1.6 — FS-prefit + scout + combine (V1.3 Option B, optional)

**When to enable** (default-OFF — opt in via `backend.scout.enabled: true` per `docs/gbdt/V1.3_OPTION_B_PLAN.md`):

- Anti-AUC + strong-top-1 cells (`anti_auc_flag == "true"` in the canonical CSV) where rule 12's "tiny-model bias" applies — the scout's depth=2/3 + eta=0.1+ES + colsample=0.3-0.5 grid lets the agent walk into a structurally-different regime than CatBoost defaults' depth=6/lr=0.05/iter=1000.
- Cells where rule 7's L2-grid winner does NOT apply (i.e. `sweep R-p@1` is not in the rare-event regime; the L2 plateau winner was calibrated on `sweep R-p@1 ≈ 0.03` cells per the r1k trio).

**Separate the two levers — FS-prefit vs scout.** `backend.fs_prefit.enabled` (Phase 1.4, the cliff-cut) and `backend.scout.enabled` (Phase 1.5, the HP-grid) are independent flags. Per **playbook rule 14**, turn the **FS-prefit cliff-cut ON whenever the finetune will apply any smoothing HP change** (lower `eta`, add `early_stopping_rounds`, lower `max_depth`, more reg) — on **ANY** cell, rare-event included. It is inert at default HP but it prevents the top-book anti-selection the moment you smooth (`docs/gbdt/_282`: R-p@3 0.716→0.055 without FS-prefit, →0.693 with, on the same schedule). The "leave OFF on rare-event" note below is about the **scout HP-grid only**, NOT the FS-prefit cliff-cut.

**When to leave the SCOUT HP-grid OFF** (`backend.scout.enabled` — the FS-prefit cliff-cut can still be on):

- Rare-event cells (rule 7) — the L2 grid `{1, 5, 10}` for `min_child_weight` is already calibrated; the scout grid adds machinery without value.
- Strong-AUC cells where defaults already dominate per `_225` (the cell-5 manual methodology FAILED on cells 1+3 because the gap was XGBoost-vs-CatBoost, not anything FS/HP could close).

The phases:

- **Phase 1.4 — FS-prefit** (D11): train one default-HP fit on the full feature matrix, cliff-cut features with importance < 1% of top (`backend.fs_prefit.cliff_pct` default 0.01). Matches the cell-5 manual workflow (~130 of 279 features kept).
- **Phase 1.5 — Scout** (D1): 8 single-knob response curves × ~4-5 values each + defaults zeroth = ~35-41 fits/cell on the cliff-cut pool. XGBoost canonical knob names; CatBoost translation applied internally (`gamma` is silently dropped — no CatBoost analog). Soft wall-clock cap (D3a.C: 5 min XGB / 30 min CB) + per-config timeout (30s XGB / 120s CB) bound the runtime.
- **Phase 1.6 — Combine** (D12):
  - In `default` mode: runner picks the lexicographic per-knob argmax (oracle: `eval_R_p@1 > 3 > 5 > 10 > 20`), composes one HP overlay, proceeds to iter_0. No exit-resume. D9.2.A degenerate-sink fallback → defaults if val_brier ≈ baseline AND train_val_gap ≈ 0.
  - In `agent_file_protocol` mode: **TWO new exit-resume cycles before iter_0** (see below).
  - In `sweep` mode: hard-OFF.

**The two new exit-resume cycles (agent_file_protocol mode)**:

```
Cycle 1 (scout-complete → combine_request):
  runner writes scout/scout_results.jsonl + scout/scout_bundle.json + scout/combine_request.json
  runner exits with the resume hint
  agent reads scout_results, writes scout/combine_decision.json (N ≤ 50 mix configs)

Cycle 2 (combine-fit → combine_results):
  --resume → runner fits each combine config in-process
  runner writes scout/combine_results.json, exits with the resume hint
  agent reads combine_results, writes scout/iter_0_decision.json with the winning HP

Cycle 3 (iter_0_decision → normal iter_0):
  --resume → runner reads iter_0_decision.json, uses its HP as iter_0's hp_starting
  + the cliff-cut feature pool from Phase 1.4
  Runner proceeds with Phase 2 (iter_0) → Phase 3 (the V1.1 FS+HP loop)
```

**`--snapshot-end` re-pass requirement** (V1.3 Option B P4 hard rule): every `--resume` MUST re-pass `--snapshot-end <DATE>` when the spec uses `callback_mode: agent_file_protocol` AND `backend.scout.enabled: true`. The runner refuses without it (clear error message). This keeps the universe-feature-cache key stable across cycles 1-3 (and the subsequent V1.1 loop iter resumes). Pre-V1.3-Option-B agent loops without scout do NOT trigger this requirement.

**The speed-biased combine prompt** (§ 3.5 of the plan) is prefilled into `combine_request.json::speed_biased_prompt` verbatim. It codifies the rule-12 tiny-model bias at the prompt level so the agent doesn't have to re-derive it per cell. The agent may override based on scout evidence (e.g. if depth=2 curves are flat at baseline and depth=6 has a clear R-p@1 lead, follow the data).

### Phase 2 — Iteration 0

- Start with the full candidate feature pool.
- Use `spec.backend.hp_starting` if provided, else `default.yaml::backend.hp_starting`.
- Fit CatBoost on train, early-stop against val, score val + eval.
- Generate the diagnostic bundle (`fs_hp_loop.DiagnosticBundle`): train/val Brier, train-val gap, learning curve, early-stop iteration, feature importance, correlation matrix, calibration summary (Spiegelhalter Z on raw val preds), positive-class recall.
- Log iteration 0 to `iterations.jsonl` with rationale `"iteration 0 — full feature pool, default HPs"`.

### Phase 3 — FS+HP loop (iterations 1..N, N ≤ max_iterations)

This phase is the **reasoning** the loop applies each iteration. In `default` mode the runner applies a fixed version of it (importance-prune + HP nudge) in one process. In `agent_file_protocol` mode **this session applies it by hand** via the exit-and-resume protocol — read § "Agent-driven FS+HP loop" below for the file mechanic; the per-iteration *decision-making* below is identical regardless of mode.

> **Diagnostic-first gate (read `.claude/memories/project-gbdt-tuning-playbook.md` first).** Before applying any heuristic below, check the train/val gap. `train_val_gap = val_brier − train_brier`; POSITIVE = val worse than train = overfit. **If `train_val_gap ≤ 0.02` (val not meaningfully worse than train) → the model is NOT overfitting; do NOT prune.** (Early-stopping *firing* is orthogonal — healthy tree-count selection, not an overfit signal; don't conjoin it.) Pruning a non-overfit model only removes capacity (nifty50 H=25: pruning raised val brier 0.1642→0.1663). Also: `importance≈0` usually means *redundant* (collinear with a kept feature), not unrelated — so FS is ~neutral on a non-overfit model, never an accuracy win. And watch for an **HP ceiling**: if val brier stays in a tiny band across depth {4,6,8} × lr scan, declare it and stop — a clean negative is a result. If val/eval brier is stuck but AUC/R-Precision@K are strong, suspect a **prevalence shift** (compare `positive_prevalence` across segments) — that lever is out of this loop's scope. **FS-first exception when smoothing (playbook rule 14 — do NOT skip this): the "don't prune a non-overfit model" rule is about FS-as-accuracy-lever at the CURRENT HP. The moment your HP decision ALSO smooths the model (lowers `eta`, adds `early_stopping_rounds`, lowers `max_depth`, or raises any regularizer), you MUST cliff-cut the sub-floor (importance < 0.01) + `|r|>0.9`-redundant features in the same or a prior iteration — even if the gap says no-overfit.** The smoothing spreads splits into the redundant / large-scale-noise features and anti-selects the top book (`docs/gbdt/_282`: R-p@3 0.716 → 0.055 without FS on a smoothing schedule, → 0.693 with the identical schedule + FS-first). FS is inert at default HP but protective under smoothing; enable `backend.fs_prefit.enabled: true` or hand-prune the cliff-cut in the iter_0 decision.

> **Sweep-top-1-aware iter_0 navigation (XGBoost backend; playbook rules 7-9).** Before applying the iter_0_decision, look up the cell's `sweep R-p@1` from `results/gbdt/data/r_precision_at_k.csv`. **There is no universal recipe** — the agent must navigate per-cell. (a) **If sweep R-p@1 ≤ 0.1** (rare-event cell, K=10 operating metric): the L2 grid `{1, 5, 10}` for `min_child_weight` is fine — iter_0 default → mcw=5 → mcw=10 (the r1k trio plateau winner). (b) **If sweep R-p@1 ≥ 0.5** (strong-top-1 cell): **DO NOT default `mcw=10`** — it regresses top-1 by −20% to −32% on cells 1+3+4 of _195. Also **DO NOT default the cell-4 "mix recipe"** (FS+mcw=5+cs=0.5+γ=0.5) — it WORKED on cell-4 (+21% R-p@1) but DESTROYED top-1 on cells 1+3 (−55% and −60% vs sweep). Mix is **cell-4-specific**, not transferable. Instead, explore per-cell: read iter_0 diagnostics + sweep baseline, pick ONE knob to test based on what the diagnostics say (gap > 0.02 → start with reg; gap < 0 → start with capacity), map ≥2 values, then pivot per rule 9. See `[[project-gbdt-tuning-playbook]]` rules 7-9 and `docs/gbdt/_195_*` for the full empirical evidence.

> **Anti-AUC + strong-top-1 cells — V1.3 Option A signals in the iter bundle (+ manual-tuning fallback for cells the flag misses).** Every iter bundle now carries five V1.3 fields the agent reads from `loop/iter_<N>_request.json::diagnostics`:
>
> - `anti_auc_flag` — set at iter_0 from the canonical CSV (`results/gbdt/data/r_precision_at_k.csv`); `"true"` when `AUC ∈ [0.46, 0.54] AND R-Precision@10 lift > 1.8×` (V1.3 D4 thresholds, tightened from the CLAUDE.md compound rule's [0.45, 0.55] + 1.5× to preempt false-positives for the auto-disables). `"unknown"` when no sweep row exists for the cell yet (new cell — auto-disables don't fire).
> - `eval_r_precision_at_k` — dict mapping K ∈ {1, 3, 5, 10, 20} to per-iter eval R-Precision@K (per-day macro, same formula + tie-break as `r_precision_at_k.csv`). Computed via an end-of-iter predict pass on the carved eval segment (D2; < 1s overhead).
> - `degenerate_sink_warning` — `true` when `val_brier <= 1.05 × weighted_base_rate_brier`, i.e. the model is approaching the trivial constant-predictor. The val_brier "winner" walking into the sink (per playbook rule 10) — pivot away from the current reg direction.
> - `weighted_base_rate_brier` — the trivial constant-predictor's Brier on val (`p_val * (1 - p_val)`). Constant for the run; included for context so the agent can see how close val_brier is to baseline.
> - `eval_segment_size` — number of eval rows used for scoring (R-p@K reliability indicator).
>
> When `anti_auc_flag == "true"`, the runner has **auto-disabled** two known-wrong mechanisms (visible in `loop/checkpoint.json::auto_disabled`): the L1 tie-break in `best_checkpoint` (which actively selects the degenerate sink — `docs/gbdt/_211_*` F5) and the val_brier auto-plateau in `inner_stop_check` (val_brier descending to a sink is NOT loop convergence — F1). The agent should:
>
> 1. **Use `eval_r_precision_at_k[1]` as the primary optimization signal** and `val_brier` as secondary on these cells.
> 2. **Do NOT call `should_stop` on a val_brier plateau alone** — pivot to a structurally-different knob family first (per playbook rule 9), and only call `should_stop` when **eval R-p@K has also plateaued across ≥2 knob families**.
> 3. **When `degenerate_sink_warning == true`**, pivot AWAY from the current reg direction (`gamma`, `alpha`) — the model is walking into rule-10's sink.
>
> At finalization the runner's `best_checkpoint` tie-breaks on `eval R-p@1` (higher wins) among tied configs when every tied iter carries an eval R-p@1 value; otherwise it falls back to strict val-Brier argmin among ties. L1 (gap + |z|) is bypassed entirely on these cells.
>
> **Anti-AUC cells the V1.3 flag can't reach** (`anti_auc_flag == "unknown"` — no sweep row; OR you want to push past the loop's HP envelope into the tiny-model regime cell-5's manual track explored): switch to the manual-tuning track per playbook rules 10-12. Pause iteration and execute: (1) cliff-cut FS, (2) single-knob response curves outside the loop, (3) mix-and-match around non-degenerate winners, (4) **eval-segment R-p@K validation** as the holdout oracle, (5) 1-shot test commit. The cell-5 manual track (`docs/gbdt/_211_*`) beat sweep R-p@1 by **+24%** (test 0.829 vs sweep 0.671) using a 6-tree depth-2 XGBoost (≤ 24 leaves, val_brier RIGHT at baseline) — a regime the loop's HP defaults never reach. See `[[project-gbdt-tuning-playbook]]` rules 10-12 for the full recipe + the per-stage scripts pattern. **Scope**: this in-loop track uses eval as the terminal holdout (3-window: train/val/eval) — it is NOT the **canonical-period finetune**. For the latter (5-window, with a separate held-out `test` + `backtest`), eval can INVERT vs test, so select on val and confirm on `test` (judge the test book), per `docs/gbdt/CANONICAL_FINETUNE_RECIPE.md`.

Each iteration:

1. **Read the previous iteration's diagnostic bundle.** Look at: train Brier, val Brier, the gap between them, reliability deviation, val curve shape (descending? plateau? oscillating?), early-stop iteration (was the cap hit?), top-K feature importances, pairwise correlation among top features, positive-class recall.

2. **Decide the prune list.** **FS-first when smoothing (rule 14): if this iteration's HP change lowers `eta` / adds early stopping / lowers depth / raises reg, you MUST cliff-cut the sub-floor + redundant features NOW — the "non-overfit → don't prune" gate does not apply once you smooth.** Drop features per these heuristics (composed with judgment, not as hard rules):
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

   - **Monotone constraints** (`monotone_constraints` passes through `hp_starting` as a named dict): use with caution and rarely. On an interaction-driven cell (good AUC/R-Precision@K; signal in feature *combinations*) they are **neutral-to-harmful** — CatBoost enforces monotonicity at the tree-structure level and degrades conditional interactions even when the net effect is monotone (nifty50 H=25: every constraint config lost vs baseline). Marginal feature-target correlation is NOT sufficient justification; check the unconstrained model's **1D partial dependence** first, and even then expect no gain. CatBoost has no interaction constraints. See `[[project-gbdt-tuning-playbook]]` rule 4.

4. **Record the rationale.** Every change must be logged with the signal that triggered it. The agent's value-add is the chain of reasoning, not the parameter values.

5. **Re-fit, re-score, generate the new diagnostic bundle, append to `iterations.jsonl`.**

6. **Check inner-stop:**
   - **Plateau:** val Brier improvement < `plateau_threshold` (default 0.005) over the last 2 iterations → stop. **⚠️ Bug #204 caveat (agent mode)**: this fires when ONE knob's value plateaued — NOT when the loop globally converged. In `agent_file_protocol` mode this is too aggressive and silently exits before the agent can pivot to a structurally-different knob. On cell-4: iter_2 (mcw=10) val Δ=0.0012 → runner plateau-stopped → couldn't reach iter_3 to test `colsample` (which gave +16% R-p@1) or `gamma=0.5` (which gave −0.0017 val). Workaround until #204 ships: set `plateau_threshold: 0.0001` in spec to disable, rely on agent's `should_stop` + the degradation_gate + max_iterations cap. Validation movement also doesn't track test ranking: cell-4 colsample variant val moved +0.00001 (noise) but test R-p@1 jumped 0.373→0.431 — don't trust val Δ as a single stop criterion (see `[[project-gbdt-tuning-playbook]]` rule 9).
   - **Degradation:** val Brier > `(1 + degradation_gate) × best_val_brier` (default `degradation_gate=0.01`) → stop.
   - **Cap:** iteration count ≥ `max_iterations` (default 8; max allowed by validator = 16) → stop.
   - **Agent stop** (`agent_file_protocol` only): the agent emits `should_stop: true` in a decision file → the next `--resume` finalizes without a new iteration (`inner_stop_signal: "agent_should_stop"`). Use this when you've exhausted your hypotheses across **multiple structurally-different knobs** (mcw, colsample, gamma, subsample, max_depth) — don't burn iterations on diminishing returns; equally don't `should_stop` after a single-knob plateau (rule 9).

   Record the inner-stop signal in the iteration's row. The plateau/degradation/cap gates fire automatically in BOTH modes (they bound the agent loop too); `agent_should_stop` is the agent's explicit early exit.

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

## Agent-driven FS+HP loop (`callback_mode: agent_file_protocol`)

This is the V1.1 surface where **this session is the data scientist in the loop**. The authoritative protocol spec is `docs/gbdt/V1.1_agent_driven_fs_hp_loop_plan.md` § 0; the decision-rules you apply each iteration are the ones rehearsed by hand in `docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md` (the answer key the automated loop must reproduce). The reasoning is Phase 3 above; this section is the **file mechanic** + how you (the agent) drive it.

### Architecture — exit-and-resume (NOT block-and-poll)

The runner does not stay alive waiting for you. Each iteration N it: trains iter N → writes the request bundle + a resume checkpoint → **exits cleanly (exit 0, a clean pause, not an error)**. Then you read the bundle, write a decision file, and relaunch with `--resume`. No long-lived blocked process; all state is on disk and survives session interrupts. The loop control files are co-located under the artifact dir:

```
results/gbdt/experiments/<run_id>/loop/
├── checkpoint.json            # resume state — full loop history, NO model blob (prior models re-fit at finalization)
├── iter_<N>_request.json      # the bundle you READ
└── iter_<N>_decision.json     # the decision you WRITE
```

`<run_id>` is the spec file's stem (printed in the pause hint). Use the absolute worktree path when reading/writing these files.

### The per-iteration cycle

1. **Launch (or relaunch).** Foreground, with a `timeout` cap (see § "Long-running pattern" — sub-30-min cached nifty50 iterations are foreground-with-`timeout`; do NOT background+Monitor for these). Each launch trains exactly ONE iteration and pauses:
   ```bash
   # first launch (iter 0):
   timeout 1800 uv run python -m gbdt experiment <spec.yaml> --callback-mode agent_file_protocol 2>&1 | tee -a logs/<run_id>.log
   # subsequent (iter N+1), after you've written iter_<N>_decision.json:
   timeout 1800 uv run python -m gbdt experiment <spec.yaml> --resume <run_id> 2>&1 | tee -a logs/<run_id>.log
   ```
   The process prints `[loop] paused at iter <N> — resume with: ...` and exits 0. Wait for it to exit (it's one iteration of compute), then proceed.

2. **Read `loop/iter_<N>_request.json`.** Use the `Read` tool (with `limit`/`offset` if the file is large — you don't need the whole 279-feature importance map in context). The envelope:
   - `schema_version`, `run_id`, `iter`, `max_iterations`.
   - `available_features` — the active feature set this iteration trained on. **A `prune_features` decision MUST be a subset of this list** or the resume rejects it.
   - `diagnostics` — the `diagnose.json`-shaped payload (built in-memory from the iteration's `DiagnosticBundle`; reuses the `/gbdt-diagnose` pure helpers, so it computes identical metrics to the on-disk diagnose without the matrix rebuild). Read in particular:
     - `metrics` (`train_brier`, `val_brier`, `train_val_gap`) + `overfit` (`no_overfit`, `train_val_gap`, `iteration_cap_hit`) — the diagnostic-first gate.
     - `prevalence_by_segment` + `prevalence_drift` (`drift_flag`, `monotone_decline`, `spread`) — the calibration-ceiling check (out-of-loop lever if it fires).
     - `calibration` (`spiegelhalter_z`, `spiegelhalter_p`, `reliability`).
     - `top_features` + `feature_importance` (full map) + `pruned_summary` (`pruned_count`, `pruned_features` below the importance floor).
     - `per_day_p_at_k` / `r_precision` / `per_ticker_hit_rate` / `prediction_range` — populated only when a per-segment prediction frame is threaded; in-loop these carry `available: false` (the runner only carves calibrated predictions over the *best* checkpoint at finalization). The `r_precision` JSON key records the legacy weighted form (per-day variable K = R(d)); R-Precision@K (the current 2026-06-01+ headline metric) is computed post-hoc via `scripts/gbdt/compute_r_precision.py` or sourced from `results/gbdt/data/r_precision_at_k.csv` — see `[[project-r-precision-methodology]]` for the relationship.
     - `tuning_guidance` — auto-flagged playbook lines (the same rules `/gbdt-diagnose` emits).
     - `full_diagnose_available: false` + `artifact_dir` — for the matrix-dependent analyses (1D-PDP model-monotonicity, interaction pairs, correlation heatmap, redundancy verdict) run the full `/gbdt-diagnose` against `artifact_dir` on demand.

3. **Decide** per Phase 3 (diagnostic-first gate → prune list → HP changes → rationale). Anchor on `_147`'s lessons: check train/val gap *before* pruning (negative/≤0.02 gap ⇒ do NOT prune for regularization — FS is neutral-to-harmful); scan depth+lr for an **HP ceiling** and declare it if val_brier stays in a tiny band; **monotone constraints are neutral-to-harmful on interaction-driven cells** (check the unconstrained 1D-PDP, not marginal correlation — and even then expect no gain); `importance≈0` usually means *redundant*, not unrelated. If a judgment call needs the user (e.g. "iter 5, val_brier flat, abandon this spec?"), just ask in chat before writing the decision — there is no `questions_for_user` field; you hand control back every iteration anyway.

4. **Write `loop/iter_<N>_decision.json`.** The schema validated by `loop_protocol.validate_decision`:
   ```jsonc
   {
     "iter": 0,                                   // informational; runner keys off the checkpoint
     "prune_features": ["realized_vol_20"],       // optional; each MUST be in available_features
     "hp_changes": { "l2_leaf_reg": 5.0, "depth": 3 },  // optional; tunable HPs only, within bounds, no pinned HPs
     "should_stop": false,                         // optional, default false; true ⇒ finalize on next --resume
     "rationale": "drop the slowest realized-vol window; deepen + raise L2."  // your lab-notebook entry
   }
   ```
   Validation rules (a violation raises a clear `DecisionError` on `--resume` and does NOT corrupt state — fix the file and relaunch):
   - `prune_features` ⊆ `available_features` (unknown feature name → reject).
   - `hp_changes` keys are real tunable HPs (`TUNABLE_HP_RANGES` / `ENUM_HP_VALUES` in `model.py` — `learning_rate ∈ [1e-4, 1.0]`, `depth ∈ [1, 16]`, `l2_leaf_reg ∈ [0, 100]`, `bootstrap_type ∈ {Bayesian, Bernoulli, MVS, Poisson, No}`, etc.); a spec-level `backend.fs_hp_loop.search_space` may narrow these further.
   - **Never** put a pinned HP in `hp_changes`: `has_time`, `loss_function`, `eval_metric`, `custom_metric`, and `calibration_method` (pinned-by-policy for the loop) are rejected.
   - `should_stop` must be a bool.

5. **Relaunch `--resume <run_id>`.** The resumed run loads the checkpoint, validates + applies the decision (`prune_features` removed, `hp_changes` merged), trains iter N+1 ONLY (0..N are threaded back, not re-trained), and pauses again — back to step 2. When you wrote `should_stop: true`, the resume finalizes instead: it selects the best checkpoint across the full prior history (re-fitting that config since model blobs aren't carried in the checkpoint), applies calibration, and emits the full artifact set.

### Termination

The loop ends when any of: you write `should_stop: true` (`agent_should_stop`); the runner's built-in **plateau** / **degradation** / **cap** gate fires (these bound the agent loop too — see Phase 3 step 6); or `iter == max_iterations`. On finalization the runner emits `model.cbm`, `calibration.pkl`, `features.yaml`, `hp.yaml`, `iterations.jsonl`, `predictions/*.csv`, `metrics.json`, `figs/`, and `report.md` — identical to `default` mode. `metrics.json::loop.inner_stop_signal` records which gate ended it; each iteration's `delta_attribution` (in `iterations.jsonl` + the `report.md` iteration table) carries your decision rationale.

### Worked example — one full cycle (matches `tests/gbdt/test_phase4_smoke.py`)

A tiny spec (`callback_mode: agent_file_protocol`, `max_iterations: 8`, plateau/degradation gates loosened so the agent's decisions drive the pauses) with 6 active features. The end-to-end cycle the Phase-4 smoke test exercises:

```bash
# 1. Fresh launch — trains iter 0, pauses.
uv run python -m gbdt experiment smoke_synth.yaml --callback-mode agent_file_protocol
#   -> writes results/.../smoke_synth/loop/iter_0_request.json + loop/checkpoint.json
#   -> prints "[loop] paused at iter 0 — resume with: ... --resume smoke_synth"; exits 0.
```
Read `loop/iter_0_request.json`: `iter=0`, `available_features` has the 6 features (incl. `realized_vol_20`), `diagnostics.metrics.val_brier` is set, `diagnostics.full_diagnose_available=false`. Suppose the gap is healthy and `realized_vol_20` is the lowest-importance window — write the iter-0 decision:
```jsonc
// loop/iter_0_decision.json
{ "iter": 0, "prune_features": ["realized_vol_20"],
  "hp_changes": { "l2_leaf_reg": 5.0, "depth": 3 },
  "should_stop": false,
  "rationale": "drop the slowest realized-vol window; deepen + raise L2." }
```
```bash
# 2. Resume — applies the iter-0 decision, trains iter 1, pauses.
uv run python -m gbdt experiment smoke_synth.yaml --resume smoke_synth
#   -> loop/iter_1_request.json: iter=1, realized_vol_20 GONE from available_features (now 5),
#      diagnostics.hp.depth==3, diagnostics.hp.l2_leaf_reg==5.0. checkpoint iter_idx=1, 2 val_briers.
```
Suppose two iterations is enough signal — write `should_stop`:
```jsonc
// loop/iter_1_decision.json
{ "iter": 1, "should_stop": true,
  "rationale": "two iters explored; HP ceiling — stopping per the agent's judgment." }
```
```bash
# 3. Resume(should_stop) — finalizes WITHOUT a new iteration.
uv run python -m gbdt experiment smoke_synth.yaml --resume smoke_synth
#   -> best checkpoint chosen across iters 0+1 (re-fit since no model blob in the checkpoint),
#      calibration applied, full artifact set emitted. metrics.json::loop.inner_stop_signal=="agent_should_stop",
#      n_iterations_run==0 (the finalizing process trained none — both explored iters were prior runs).
```

The negative paths are symmetric: a decision with an out-of-bounds HP (`depth: 99`), a pinned HP (`has_time: false`), an unknown feature, or malformed JSON raises a clear `DecisionError` on `--resume` and leaves the checkpoint + request intact — fix `loop/iter_<N>_decision.json` and relaunch. A *missing* decision file raises "decision file not found".

## Long-running pattern

Pick by **expected wall time of the compute being launched** (not iteration count):

**Sub-30-min runs** (cached-data nifty50 experiments, smoke tests, AND each single iteration of the `agent_file_protocol` loop): **foreground with `timeout` as a hard cap**. Do NOT background+Monitor — see `.claude/memories/feedback-sub-agent-foreground.md` for why (agent sessions exit prematurely under Monitor; orphans the run).
```bash
# default mode (one process, finalizes): the gbdt.experiment shim is fine
timeout 1800 uv run python -m gbdt.experiment <spec.yaml> 2>&1 | tee logs/exp_<name>.log
# agent_file_protocol — each launch/--resume is ONE iteration; MUST use the
# `gbdt experiment` subcommand (the gbdt.experiment shim does NOT accept
# --resume / --callback-mode), still foreground-with-timeout per iteration:
timeout 1800 uv run python -m gbdt experiment <spec.yaml> --resume <run_id> 2>&1 | tee -a logs/<run_id>.log
```
Stay in the session until `timeout` returns or the command exits naturally. Then read the artifact (or, for the agent loop, the iteration's `loop/iter_<N>_request.json`) and proceed. **Each agent-loop iteration is its own foreground launch** — do NOT try to keep one process alive across iterations (the runner exits at every pause by design); never background+Monitor a per-iteration run.

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
- `docs/gbdt/V1.1_agent_driven_fs_hp_loop_plan.md` — § 0 is the AUTHORITATIVE exit-and-resume protocol spec for `callback_mode: agent_file_protocol`.
- `docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md` — the hand-driven loop the automated loop must reproduce (the answer key); the source of the decision-rules surfaced in Phase 3 + § "Agent-driven FS+HP loop".
- `docs/gbdt/_177_cross_experiment_analysis.md` — cross-experiment landscape (US sweep #107 snapshot + standalone): the source of the § "What to expect by cell type" priors — eval→test decay, horizon/threshold/universe effects, the long-horizon test-window gap.
- `docs/gbdt/EXPERIMENT_SPEC.md` — YAML schema (incl. `callback_mode` + the `loop/` files), artifact layout, validation rules.
- `docs/gbdt/CATBOOST_HP_REFERENCE.md` — per-parameter "when to change" rubrics; the canonical guide for HP decisions inside the loop.
- `docs/gbdt/goal.md` — why the experiment-loop framing exists; per-experiment success criteria.
- `docs/gbdt/V0_INVESTIGATION_PLAN.md` — v0 base-rate findings that inform cell choice.
- `docs/gbdt/V1.1_TBD.md` — parked extensions (NDX, macro features, multi-target, Optuna HP search).
- `.claude/memories/project-gbdt-tuning-playbook.md` — the FS+HP diagnostic-first decision-rules keyed in the per-iteration reasoning.
- `.claude/memories/feedback-experiment-agent-loop.md` — long-running compute pattern (the agent driving the loop is subject to it).
- `.claude/memories/feedback-sub-agent-foreground.md` — foreground-vs-background split (each agent-loop iteration is a foreground-with-`timeout` launch).
- `.claude/memories/project-overview.md` — module overview.
