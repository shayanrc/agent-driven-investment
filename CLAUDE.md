# Agent-Driven Investment — Project Conventions

This repo hosts multiple forecasting/analytics modules — each independently versioned and namespaced:

- **analog_mc** — probabilistic price-path forecasting (analog Monte Carlo). [`docs/analog_mc/goal.md`](docs/analog_mc/goal.md)
- **data_pipelines** — generic time-series ingestion framework; `us_equities` + `nse_equities` domains shipped. [`docs/data_pipelines/goal.md`](docs/data_pipelines/goal.md)
- **forecasters** — agent-callable forecasting surface; presets as saved-model artifacts; analog_mc wired as backend #1. [`docs/forecasters/goal.md`](docs/forecasters/goal.md)
- **gbdt** — categorical-outcome GBDT classifiers; v1 ships **experiment-loop infrastructure** (per-experiment YAML spec → CatBoost on pooled panel → calibrated probabilities). [`docs/gbdt/goal.md`](docs/gbdt/goal.md)
- **calibration** — probability calibration (isotonic etc.); fit separately, handed to strategies already-fit. [`docs/calibration/goal.md`](docs/calibration/goal.md)
- **trading_strategies** — backend-agnostic decision policies that consume a `dict[date → (ticker, p_mean, p_low, p_high)]` and emit orders (e.g. `TopKDailyKellyLabelExit`); not forecasters, not execution engines. [`docs/trading_strategies/goal.md`](docs/trading_strategies/goal.md)
- **backtesting** — the execution engine (the `Strategy` Protocol, order queue, equity/drawdown/Sharpe results). [`docs/backtesting/goal.md`](docs/backtesting/goal.md)

**`backtests/` is not a packaged module** — it's the cross-module backtest harness + analysis area (`docs/backtests/` per-experiment memos `_001…_NNN` + `INDEX.md`, `scripts/backtests/` runners, `results/backtests/`) where the gbdt → calibration → trading_strategies → backtesting pipeline is evaluated. The daily **`/daily-predictions` forward-OOS cadence** (`_019`) lives here.

## Module goal docs (read first)

Before editing any file under a module's directory tree (`src/<module>/`, `tests/<module>/`, `configs/<module>/`, `dashboards/<module>/`, `docs/<module>/`, `results/<module>/`, `scripts/<module>/`), read `docs/<module>/goal.md` first. It defines what the module is optimizing for and which trade-offs are unacceptable. `IMPLEMENTATION_PLAN.md` / `V<N>_PLAN.md` describe *how* the module works; `goal.md` defines the success criteria your change must respect.

`docs/<module>/V<N>_TBD.md` is the parking lot for follow-ups discovered in a branch but out of scope for it. `docs/<module>/V0_INVESTIGATION_PLAN.md` (where present) frames pre-v1 data-exploration scans whose outputs inform v1 design (see `gbdt` for an example).

## Module namespacing

Every module-specific directory nests under the module name. The top-level directories are reserved for cross-module concerns (shared launchers, fixtures, etc.).

```
src/<module>/                  # package code
docs/<module>/                 # design docs (V1_PLAN.md, IMPLEMENTATION_PLAN.md) + per-experiment reports (_<id>_<name>.md)
tests/<module>/                # tests
configs/<module>/*.yaml        # YAML configs
scripts/<module>/              # ad-hoc orchestration / aggregation / report-rendering / v0 investigation scripts
runs/<module>/<timestamp>/     # raw per-fold run artifacts (gitignored)
results/<module>/data/         # aggregated experiment JSONs (_<id>_data.json) — checked in
dashboards/<module>/
  ├── app.py                   # module's runnable Streamlit entry point
  └── views/                   # view modules
dashboards/app.py              # thin global launcher
```

When adding new module-scoped files, always nest them under `<top>/<module>/`. Never put module-specific code in the top-level `dashboards/`, `tests/`, `configs/`, `scripts/`, or `runs/`.

**Docs vs results.** `docs/<module>/_<id>_<name>.md` holds the experiment narrative (setup, mechanistic reading, decision-rule verdict). `results/<module>/data/_<id>_data.json` holds the machine-readable headline metrics that back the narrative. New aggregate scripts write to `results/<module>/data/` by default.

## Claude Code skills

Agent-callable verbs live at `.claude/skills/<name>/SKILL.md` and are invoked as `/<name>` from any session.

- **One skill = one verb.** Inference, fitting, listing, fetching, health-checking are separate skills.
- **Skill is module-owned even when bundled in a cross-module PR.** SKILL.md files live in the shared `.claude/skills/` namespace (because that's where Claude Code reads them), but the runner script lives under the owning module (e.g. `/fetch-data` SKILL.md + `scripts/data_pipelines/skill_runner.py`). Ownership tracks the implementation, not the SKILL.md location.
- **Long-running skills (`/tune-preset`, `/gbdt-experiment`, and the like) MUST bake in the loop-pattern guidance** per `.claude/memories/feedback-experiment-agent-loop.md` AND the foreground-vs-background split per `.claude/memories/feedback-sub-agent-foreground.md` (sub-30-min runs = foreground with `timeout`; ≥2 h runs = background + Monitor + ScheduleWakeup chain).
- **No backend-internal hyperparameter flags** on the public skill surface — overrides via `--config-overrides path.yaml`, never `--n-eff 50`.

## Data and configs

- **analog_mc**: reads from a local CSV (default `data/NASDAQ100.csv`, FRED-style: `observation_date`, `NASDAQ100`). The loader takes `date_col` and `close_col` from config — asset-agnostic.
- **gbdt v1**: reads pooled NIFTY 50 panel via `data_pipelines.fetch()` from the SQLite cache; defaults to `cache_only=True` (no provider gap-fill during experiments). Universe self-service in Stage 1 of `/gbdt-experiment` registers new universes (curl from `archives.nseindia.com` is the reliable path; jugaad/nselib often blocked — see `[[project-nse-data-quirks]]`).
- **`data_pipelines`**: general-purpose loader. New modules that need historical price data should use `data_pipelines.fetch(identifier, start, end, back_extend=False)`. analog_mc stays on the CSV-first contract per `[[project-data-source]]`; wiring it to `data_pipelines` is a separate per-module plan.
- **Cache layout**: SQLite at `data/processed.db` (per-domain tables) + immutable per-provider raw downloads at `data/raw/<provider>/...`. Single-writer-per-`data_root` is the contract — concurrent seed processes risk SQLite `BUSY` contention; WAL serializes correctness-wise. **If `processed.db-wal` becomes filesystem-corrupted** (I/O error on stat/unlink), copy `processed.db` to a non-corrupted scratch dir and re-symlink — see `[[project-nse-data-quirks]]` for the workaround.
- **Cache `date` column carries a time component** (`'YYYY-MM-DD 00:00:00'`). When querying the SQLite `date` column in raw SQL, use a **half-open interval** `date >= start AND date < (end + 1 day)` (or `DATE(date) <= end`), NEVER a bare `date <= 'YYYY-MM-DD'` — a string compare sorts `'…16 00:00:00'` *after* `'…16'` and silently drops the end day (the #182 off-by-one that hid the latest bar from `load_panel` / fresh inference). `data_pipelines.fetch` + `gbdt.data.load_panel`/`_cache_read` already handle this; new queries must too. See `[[project-cache-date-time-component]]`.
- **gbdt v1.4 date-aligned splits**: new gbdt specs should set `split.mode: date_aligned` + `split.train_start: 2019-01-01` (the canonical anchor per V1.4 D2) — segments are anchored to universe-calendar dates (NYSE for US universes, NSE for NSE universes) so the cell is reproducible across cache growth. The trailing-anchor mode remains the default for back-compat. The canonical `results/gbdt/data/r_precision_at_k.csv` carries **8 calendar-date columns** (`train_start, train_end, val_start, val_end, eval_start, eval_end, test_start, test_end`) so every row self-describes the (train, val, eval, test) windows it was scored on — backtesters can use `test_end + 1` as a clean out-of-sample start without external lookup. Plan: `docs/gbdt/V1.4_date_aligned_splits_plan.md`.
- **gbdt F17 macro features (opt-in)**: daily FRED-style macro series (yields/curve/credit-OAS/breakevens/real-yield/VIX/USD) broadcast to the panel, 1-trading-day-lagged (C1 causal), `<id>_level`/`chg_{20,60}`/`z_{60,120}` (~45 cols). Enabled ONLY via the `features.candidates: all_macro` token (F1–F16 + F17); **NOT in the default `all` token, so existing models stay byte-identical**. The `fred_macro` cache currently holds 9 series under FRED ids but with **NON-FRED provenance** (Treasury / NY-Fed / Yahoo proxies; HY-OAS = `-log(HYG/IEF)` proxy) because FRED egress was unreachable on this host — re-running an `all_macro` spec reads whatever's cached. Status: opt-in, **NOT deployed / NOT promoted — failed second-window validation.** `_262`/`_263` had macro beating the sp500 champions + broadly helping top-of-book, but on a single (trailing 2026-Q1) window; `_264` re-ran the same matched lattice on an independent **date-aligned** window (2024-H2) and the edge did **NOT replicate** (8/12 cells flip sign at R-p@1; headline `+20%/50d` +0.30→−0.11). So the win was window-specific — reaffirms the `_258`–`_261` "contextually additive but not robust" read. **Do NOT wire macro into `/daily-predictions`.** See `[[project-gbdt-macro-features-f17]]`.
- **gbdt v1.3 Option B scout + FS-prefit (opt-in)**: anti-AUC + strong-top-1 cells (where rule 12's tiny-model bias applies) can opt in to Phase 1.4-1.6 (FS-prefit cliff-cut + 8 single-knob response curves + combine) BEFORE iter_0 via `backend.scout.enabled: true` + `backend.fs_prefit.enabled: true`. Default-mode picks the lex auto-compose (oracle: `R-p@1 > 3 > 5 > 10 > 20`); agent_file_protocol mode adds TWO new exit-resume cycles (`scout/combine_request.json` → agent writes `combine_decision.json` ≤ 50 mix configs → runner fits → agent writes `iter_0_decision.json` → iter_0 starts from the agent's HP + cliff-cut feature pool). `--resume` MUST re-pass `--snapshot-end` when scout is enabled in agent mode. Schema details: `docs/gbdt/EXPERIMENT_SPEC.md` § `backend.scout` + `backend.fs_prefit`. Plan: `docs/gbdt/V1.3_OPTION_B_PLAN.md`.
- **Daily forward-prediction cadence (`/daily-predictions`)**: `scripts/backtests/daily_forward_predictions.py` refreshes US data → incrementally re-scores the validated sp500 champions (`infer_fresh_predictions --since`, a trailing ~7y slice; ~2 min/cell since the V1.5 feature-build vectorize) → SMA200 regime gate → appends **complete-day** top-K to the committed `results/backtests/data/forward_predictions_log.csv` (the durable signal record; full per-(date,ticker) CSVs gitignored, regenerable). Idempotent + self-gating + backfilling; only logs days **strictly before today** (partial-intraday-bar guard, #182). Unattended via a systemd **user** timer (`scripts/backtests/systemd/`). The strategy it tracks (`TopKDailyKellyLabelExit`) is **hold-through-horizon with exit-driven turnover, NOT daily top-K chasing** (daily churn is the high-turnover variant costs destroyed, `_015`). Memo: `docs/backtests/_019_forward_oos_pipeline.md`.

## Plans and branches

Each new plan doc (`V<N>_PLAN.md`, `V<N>_EXPERIMENTS_PLAN.md`, `ABLATION_STUDIES_PLAN.md`, etc.) gets its own git branch — don't stream commits to main. Plan + its implementation + reports land together as a PR. Pure refactors of already-merged work may go on main; the rule is per-*plan*, not per-*change*.

Merged plan/feature branches stay around for reference — don't offer to delete them after merge (local or remote). The history is wanted. See `[[feedback-branch-retention]]`.

Follow-ups discovered during a branch that are out of scope for that branch go in `docs/<module>/V<N+1>_TBD.md` (parking lot), not in scratch files or new branches. Promote to a real `V<N>_PLAN.md` when a coherent slice is big enough to be "a project, not a chore."

## Commit and PR attribution

Never attribute commits or PRs to Claude / an AI / a code-generation tool. No `Co-Authored-By: Claude …` footers, no `🤖 Generated with [Claude Code]` lines, no "generated by"/"written by AI" markers anywhere in commit messages or PR titles/bodies. Commit messages and PR text should read as the user's own work.

**Make granular commits** — one focused, self-contained logical change per commit, not one lump; stage selectively (`git add <paths>`, or `git add -p` for hunks). Granularity is about commits *within* a branch (the per-plan branch rule and `[[feedback-branch-retention]]` are unchanged), and merge stays rebase-and-merge — never `--squash`, which would collapse the granular history. See `[[feedback-granular-commits]]`.

## Reporting conventions

**Do not put lift (metric / base_rate) in tables unless the table is explicitly framed as a lift-comparison.** Lift compresses two pieces of information into one number; future readers lose the base rate, the units, and the actual hit-rate scale. Tables should show raw metric values + a base rate column for reference; readers compute lift on demand. This rule applies to all per-experiment memos (`docs/<module>/_<id>_*.md`), aggregated reports, and the gbdt runner's standard `report.md`. Lift may appear in narrative prose ("nasdaq P@1 was 1.78× base rate"), but not as a column in a data table.

For top-K classifier metrics, the per-day denominator MUST be `min(R(d), K)` where R(d) is the number of positives that day, NOT the count of picks actually made. The picks-made denominator silently mis-normalizes on staggered panels where R(d) < K for many days, producing artifact values that have nothing to do with model skill (this is the 2026-05-28 lesson — see `[[project-r-precision-methodology]]`). The runner's `src/gbdt/topk_diagnostics.py` is fixed; post-hoc analyses must use the same denominator via `scripts/gbdt/compute_r_precision.py`.

Standard cross-cell comparison metric for gbdt experiments: **R-Precision@K** at K ∈ {1, 3, 5, 10, 20}, defined as `(1/Q) · Σ_q r_q / min(K, R_q)` — per-day fixed K, macro-averaged across days where R_q > 0 (see `[[project-r-precision-methodology]]`). Renamed from the prior "weighted R-precision" (per-day variable K = R(d), micro-aggregated) on 2026-06-01; the legacy metric and the current one are **different metrics**, not just different aggregations of the same thing — pre-2026-06-01 memos quote the legacy form. Canonical cell-by-cell registry: `results/gbdt/data/r_precision_at_k.csv`.

**Show times to the user in IST** (UTC+5:30) unless they specify another timezone — the project owner is in India. Internal artifacts (logs, commit metadata) may stay UTC; convert cron next-fire times from machine-local before display. See `[[feedback-report-time-in-ist]]`.

**When the user asks to be shown something** (a table, ranking, metric, diff, file contents), re-render the result as properly-formatted markdown **in the reply itself** — compute it in the shell if needed, then paste the formatted output. Don't reply with "see the table above" pointing at raw tool output, which isn't always rendered in chat as intended. Same table rules apply (raw metric + base_rate, no lift columns). See `[[feedback-replicate-output-in-dialog]]`.

## Source of truth for analog_mc

`docs/analog_mc/IMPLEMENTATION_PLAN.md` is the spec for the analog_mc pipeline and was the output of a long design conversation. Every decision in it was made for a reason. **Do not silently change architectural decisions.** If implementation reveals a problem with a decision, surface it explicitly and ask before deviating.

Follow the 11-stage build order in that doc strictly — the diagnostic infrastructure (Stages 6, 9) is what makes the pipeline trustworthy, not the optimizer (Stage 7). Don't skip ahead.

The 6 critical correctness constraints (C1–C6 in the plan) are non-negotiable:
- C1: causal rolling features (zero look-ahead)
- C2: n_eff parameterization for distance → probability
- C3: per-analog vol scaling = demean → clip ratio → rescale → add drift
- C4: running EWMA σ for blocks 2+
- C5: strictly forward sampling
- C6: walk-forward boundary discipline

## Environment

- Python ≥3.12 via uv. Venv at `.venv/`, lockfile at `uv.lock`.
- `analog_mc`, `data_pipelines`, `forecasters`, and `gbdt` are all installed as editable packages via hatchling (`pyproject.toml` `[tool.hatch.build.targets.wheel] packages` list) — `import <module>.foo` works from anywhere.
- Run things with `uv run <cmd>` (e.g., `uv run pytest`, `uv run streamlit run dashboards/analog_mc/app.py`, `uv run python -m data_pipelines fetch ...`, `uv run python -m scripts.gbdt.v0_opportunity_scan`).
- **gh CLI is installed and authenticated** as `shayanrc` (per per-user memory `gh-cli-installed`). Use `gh pr create / view / diff / comment / review` directly for GitHub ops; no need to surface compare URLs.
- **Autonomous PR review/merge pipeline is the default for this project.** When the assistant opens a PR via `gh pr create`, immediately fire the autonomous review/merge background sub-agent for it; do not pre-ask per PR. The sub-agent uses `gh pr comment` (not `gh pr review --approve` — author-self-approval is blocked) for the approval comment, then `gh pr merge <num> --rebase` (**rebase-and-merge is the standing method per the user's direction — NOT `--squash`**; NEVER `--delete-branch` — branch retention policy). Sub-agents executing the review should treat this CLAUDE.md line as authorization for the merge; the per-user `gh-cli-installed` "don't auto-merge without explicit OK" rule is overridden at the project level here. See `[[feedback-auto-fire-review-merge]]` (per-user memory) for the why. Skip the auto-fire ONLY when the user explicitly asks to hold a specific PR (e.g., "don't auto-merge #X — I want to read it first") or when the PR touches CLAUDE.md / spec / policy docs that warrant human judgment.
- **Worktree workflow for parallel agents**: parent agent creates the sibling worktree (`wt-<scope>/` next to the main checkout), symlinks `data/` → `${SCRATCH_CACHE}` and `.env` from the main checkout, runs `git update-index --skip-worktree data/.gitkeep` once in the worktree, then `uv sync --frozen`. `${SCRATCH_CACHE}` is a **per-machine scratch directory** — the literal lives in your per-user memory `scratch-cache-path` (substitute inline or `export SCRATCH_CACHE=<path>` once per shell). See `[[feedback-worktree-symlink-contract]]` for the rationale + verification steps + the `ln -snf` foot-gun + the `.gitkeep` invariant.
- **Disk pre-flight before long-running runs**: the FS hosting the project can wedge when near-full (≥95%) — kernel writes enter D-state, processes become unkillable, cascading. Long-running skills should pre-flight `df --output=avail $(pwd) | tail -1 ≥ 10 G`. See `[[feedback-disk-wedge-pattern]]`.
- **Checking a background sub-agent's progress**: to report a background sub-agent's progress, parse its `*.output` JSONL with a bounded query (event count + recent tool names + last assistant text + last tool result, capped). See `[[feedback-subagent-transcript-parsing]]` for the recipe.

## Memories

Project-shared facts (architecture decisions, layout conventions, workflow rules) live in `.claude/memories/` (indexed in `.claude/memories/INDEX.md`) or as bullets in this file. The detail in `.claude/memories/` includes the *why* and *how-to-apply* per topic; CLAUDE.md is the summary.

Per-user/per-machine items (personal preferences, role context, machine paths) stay at `~/.claude/projects/<hash>/memory/`. Don't duplicate project facts there — refer to the project memories instead.

## Presenting plans and priorities

- **Always render a dependency graph to PNG when optimizing/prioritizing tasks.** Whenever the work involves task sequencing — a dependency graph, "what's unblocked?", critical-path/parallelization, "optimize for time" — produce the graph AND render it to a PNG (plus SVG) and display it via `SendUserFile`; don't stop at inline mermaid text. Trigger it automatically, without being asked. Render **locally with graphviz `dot`** (`dot -Tpng -Gdpi=150 g.dot -o g.png` + `-Tsvg`); `mmdc`/mermaid-cli is NOT usable in this environment (its install skips the headless-browser download and fails). **House style** (full spec + canonical DOT skeleton in `.claude/memories/feedback-task-priority-graph.md`): `rankdir=TB`; title carries **date + time**; group nodes into labeled **module/phase cluster boxes** with recurring loops/monitors in their own **`scheduled tasks / loops`** cluster; the **legend is a single HTML-table node in an isolated top rank, centered above all clusters via a spacer-row + invisible weight=100 anchor edges to every cluster's top node** (not title text; the older `cluster_legend` swatch-row approach is deprecated — couldn't be reliably centered or rank-isolated). Node fills: green=actionable, blue=in-flight, gold=goal, **white+dashed-outline=pending/gated**, **filled-gray=done/merged** (kept visually distinct), blue+dotted=recurring (paused), red=blocked-on-data; solid edge=hard dep, dashed=soft, colored edge per workstream/critical path. Don't upload graphs to external mermaid/Kroki renderers — render locally. See `.claude/memories/feedback-task-priority-graph.md`.

## What not to do — analog_mc

(Module-specific anti-patterns. Future modules append their own `## What not to do — <module>` sections rather than dumping into a shared list.)

- Don't use scikit-learn's `StandardScaler` — it batch-fits across the array. Implement causal z-scoring directly.
- Don't swap grid search for BayesOpt — the grid was chosen for diagnostic interpretability.
- Don't implement v2 features (trailing-momentum drift, conditional block sampling, tail inflation) in v1. They are gated on specific diagnostic findings; premature implementation contaminates the diagnostics that decide whether v2 is needed.
- Don't report aggregate CRPS as the headline result without PIT and weight-trajectory diagnostics.
- Don't add transaction costs, position sizing, or PnL to this pipeline. Those belong downstream.

## What not to do — gbdt

- Don't ship the 18-cell lattice as a single deliverable — v1 is the experiment-loop infrastructure; each experiment is one `(universe, direction, threshold, horizon, max_drawdown?)` tuple. The lattice was v0 EDA scope.
- Don't add per-asset constants in features — v1 features must be **asset-agnostic** (the panel-pooled model has no per-stock metadata). Cross-sectional features (rank/zscore across the panel at each `t`) are explicitly OK.
- Don't disable `has_time=True` in the CatBoost wrapper — it's mandatory for walk-forward correctness (C6). Pinned in `configs/gbdt/default.yaml::backend.hp_pinned`; never overridable.
- Don't treat the per-experiment verdict as automated — `report.md` is the agent's one-paragraph readout; the PASS/FAIL judgment is for the user. AUC ∈ [0.46, 0.54] **alone** is no longer sufficient to flag null — PR #28 + memo #138 (H=25 cross-market) found AUC=0.51 cells with R-Precision@10 lift 1.5–2.0× (real top-tail signal masked by null AUC). Use the compound rule: AUC ∈ [0.46, 0.54] **AND** R-Precision@10 lift < 1.2× → null signal; AUC ∈ [0.46, 0.54] **AND** R-Precision@10 lift > 1.8× → **anti-AUC strong-top-1 cell** (top-tail signal hidden by AUC; the loop's val_brier objective is structurally misaligned) — V1.3+ auto-disables the L1 tie-break + val_brier plateau gate on these cells, see the V1.3 Option A bullet below. Thresholds tightened from [0.45, 0.55] + 1.5× per V1.3 plan D4 (`docs/gbdt/V1.3_option_a_loop_anti_auc_integration_plan.md`) so the auto-disables don't false-positive; single source of truth with the iter_0 flag. R-Precision@K (K ∈ {1, 3, 5, 10, 20}) is per-day fixed K, macro-averaged via `(1/Q) · Σ r_q / min(K, R_q)` over days where R_q > 0; reports the precision-vs-K curve a portfolio manager actually uses for position sizing. The pre-2026-06-01 "weighted R-precision" (per-day variable K, micro-aggregated) is a different metric, retained in old memos for cross-walk. See `[[project-r-precision-methodology]]` for the full definition + relationship.
- Don't introduce lagged-target features without strict masking — they're the classic foot-gun (target's event window overlaps prediction time). v1 covers the "recent event" intuition via the F16 `signed_days_outside_<X>sig` family, which has no leakage trap.
- Don't default `min_child_weight=10` on cells where sweep R-p@1 ≥ 0.5. The L2 grid `{1, 5, 10}` plateau winner from r1k trio + _185 was calibrated on rare-event cells (sweep R-p@1 ≈ 0.03). On strong-top-1 cells, mcw=10 over-regularizes the prediction tail (cells 1+3+4 of _195 regressed loop R-p@1 by −20% to −32% vs sweep). And do NOT default to the cell-4 "mix recipe" (FS+mcw=5+cs=0.5+γ=0.5) either — it won on cell-4 but DESTROYED top-1 on cells 1+3 (−55% and −60%). **There is no universal recipe for strong-top-1 cells** — the agent must explore per-cell; see `[[project-gbdt-tuning-playbook]]` rules 7-9 and `docs/gbdt/_195_*`.
- Don't ship the val_brier winner as the final model on anti-AUC cells. On cells flagged by the AUC ∈ [0.46, 0.54] + R-Precision@10 lift > 1.8× rule (V1.3 thresholds, tightened from [0.45, 0.55] + 1.5× per D4), val_brier has a degenerate global minimum at the **constant predictor** — any monotonically-shrinking reg knob (`gamma`, `alpha`) walks toward it; the L1 tie-break (gap + Z) actively selects it; the resulting "winner" has eval R-p@1 below baseline. V1.3 Option A auto-disables the L1 tie-break + val_brier auto-plateau when the iter_0 `anti_auc_flag == "true"` (visible in `loop/checkpoint.json::auto_disabled`); the agent should read `eval_r_precision_at_k[1]` as the primary signal and treat **eval R-Precision@K as the holdout oracle** when val signals are misaligned. Val R-p@K is ALSO unreliable on these cells (Q_days × 1 pick/day is too small a sample). The manual-tuning track (`docs/gbdt/_211_*`) remains available for cells the flag misses (`unknown` — no sweep row) or to push past the loop's HP envelope. See `[[project-gbdt-tuning-playbook]]` rules 10-11.
- Don't dismiss tiny models on top-1 metrics. Cell-5 manual winner is a **6-tree, max_depth=2 XGBoost** (≤ 24 leaves total, val_brier RIGHT at the weighted-base-rate baseline) — it beats CatBoost-default sweep (1000 iters, depth 6) by **+24%** on test R-p@1 (0.829 vs 0.671). "val_brier close to baseline = model didn't learn" intuition is wrong on anti-AUC cells: the prediction-tail shape is what matters for R-p@1, not bulk Brier. Explicitly try `max_depth ∈ {2, 3}` + `eta ∈ {0.05, 0.1}` + early stopping + `colsample_bytree ∈ {0.3, 0.4, 0.5}` when rule 10 triggers. See `[[project-gbdt-tuning-playbook]]` rule 12.
- Don't trust the runner's auto-plateau in `agent_file_protocol` mode — it stops on single-knob plateau, NOT global convergence. Set `plateau_threshold: 0.0001` in spec to disable; rely on agent's `should_stop` + `degradation_gate` + `max_iterations` (cap 16 per validator). Bug #204 fix lifts this in a future runner patch; until then, the workaround is mandatory or the agent silently exits before it can pivot knobs.
- Don't compare feature variants (e.g. base vs `+macro`) by running each arm under the auto-loop (`callback_mode: default`). The loop tunes EACH ARM INDEPENDENTLY, so the per-arm HP divergence confounds the feature effect with the search's choices. This is exactly what produced the `_260` sp500 +20% "macro sign-flip" (base loop → R-p@1 0.347, macro loop → 0.120 — purely different HP, not a macro property). For a **clean feature A/B**, PIN identical HP for both arms via `backend.hp_starting` + `max_iterations: 1` (single fit) so ONLY `features.candidates` differs; the matched HP need not be each arm's optimum since the per-cell DELTA is what's measured. The `_262` matched re-baseline did this (both arms `mcw=10`) and reversed the verdict — macro beat the champion's config on both sp500 cells. See `[[project-gbdt-macro-features-f17]]` + `docs/gbdt/_262`/`_263`; tooling: `scripts/gbdt/{gen_macro_sweep_specs.py,run_macro_sweep.sh}`.
- Don't add transaction costs, position sizing, or PnL — same as analog_mc; downstream concerns.
