# Project Memories — Index

Shared, checked-in project memory. Auto-loading is not guaranteed for this folder (Claude Code's auto-memory reads from `~/.claude/projects/<hash>/memory/`), but `CLAUDE.md` at the repo root references this folder so future sessions know to consult it for project-scoped context.

When adding a new memory, append a one-line pointer below. Before adding, check that the fact isn't already in `CLAUDE.md`, in `docs/<module>/goal.md`, or derivable from the code — those are not memory material.

## Project facts

- [project-overview.md](project-overview.md) — Four modules: analog_mc + data_pipelines + forecasters + gbdt (all v1 shipped to main as of 2026-05-26). Per-module `docs/<module>/goal.md` for specifics
- [project-data-source.md](project-data-source.md) — analog_mc reads from `data/NASDAQ100.csv`; gbdt v1 reads NIFTY 50 panel via `data_pipelines.fetch()`; per-module switchover for analog_mc is a separate plan
- [project-streamlit.md](project-streamlit.md) — Local dashboard designed for concurrent runs (per-run lockfile, no global mutex)
- [project-results-layout.md](project-results-layout.md) — Aggregated experiment JSONs live in `results/<module>/data/`, separated from `docs/<module>/` narratives
- [project-fat-tail-eval.md](project-fat-tail-eval.md) — Operational how-to for the mandatory 15-anchor fat-tail eval panel (scripts, fig paths, regeneration rules)
- [project-nse-data-quirks.md](project-nse-data-quirks.md) — NSE fetch quirks: jugaad/nselib blocked → use curl from archives.nseindia.com; filter DUMMY* placeholders; processed.db-wal can corrupt → /tmp/exp_data workaround; per-ticker fetch is 30–80s; cached-but-short tickers need `back_extend=True`
- [project-gbdt-uniqueness-weights.md](project-gbdt-uniqueness-weights.md) — gbdt applies LdP §4.4 sample-uniqueness weights by default; corrects the 2.15× overlap-prevalence inflation surfaced by sweep exp #1; opt-out via `target.uniqueness_weighting: false`
- [project-r-precision-methodology.md](project-r-precision-methodology.md) — R-precision (per-day variable K=R(d)) is the headline cross-cell metric for gbdt; AUC alone misclassifies top-tail-signal cells as null; compound rule + compute_r_precision.py script
- [project-gbdt-tuning-playbook.md](project-gbdt-tuning-playbook.md) — diagnostic-first FS+HP tuning rules (nifty50 H=25 study): check train/val gap before pruning; HP-ceiling detection; monotone constraints neutral-to-harmful on interaction-driven cells (check 1D PDP not marginal corr); importance≈0 = redundant not unrelated
- [project-xgboost-interaction-analysis.md](project-xgboost-interaction-analysis.md) — XGBoost native TreeSHAP (`pred_contribs`/`pred_interactions`) for feature-interaction analysis: method, O(rows×feat²) cost mitigations, drop-only-if-low-main-AND-low-interaction pruning rule, GPU-OK-for-diagnostics-not-training split; grounds V1.2 (#165)
- [project-xgboost-training-essentials.md](project-xgboost-training-essentials.md) — XGBoost backend training contract: load-bearing DETERMINISM pins (seed + `tree_method=exact` + `n_jobs=1` + cpu, hard-fail) for the finalization-retrain; HP-name mapping vs CatBoost (no `has_time`); built-in L1/L2 + missing-value handling; calibration unchanged; portability

## Workflow feedback

- [feedback-branch-retention.md](feedback-branch-retention.md) — Don't propose deleting merged plan/feature branches; history is wanted
- [feedback-experiment-agent-loop.md](feedback-experiment-agent-loop.md) — Bake loop-pattern guidance (background shell + Monitor filter + ScheduleWakeup + parallel-worker w/ per-item timeout) into sub-agent launch prompts up front; SendMessage isn't exposed
- [feedback-track-agent-tasks.md](feedback-track-agent-tasks.md) — Every sub-agent launch → immediate TaskCreate + status=in_progress; mark completed only when the work is actually closed out
- [feedback-disk-wedge-pattern.md](feedback-disk-wedge-pattern.md) — NTFS-on-Linux volumes can wedge when near-full → kernel D-state procs, unkillable, cascading; pre-flight disk check + cleanup via paths off the wedged FS
- [feedback-sub-agent-foreground.md](feedback-sub-agent-foreground.md) — Sub-30-min sub-agent tasks: run in FOREGROUND with `timeout`, NOT background+Monitor (the latter exits prematurely, orphans procs); ≥2h tasks still use background+Monitor+ScheduleWakeup chain
- [feedback-worktree-symlink-contract.md](feedback-worktree-symlink-contract.md) — Sub-agents can't `git worktree add` (sandbox blocks); parent pre-creates. Symlink: `rm -rf data && ln -s` (not `ln -snf` — creates nested `data/data` when data/ exists as dir). Then `git update-index --skip-worktree data/.gitkeep` so the shadowed tracked file doesn't break `git rebase --autostash` / `git checkout`
- [feedback-agent-pkill-antipattern.md](feedback-agent-pkill-antipattern.md) — Sub-agents MUST NOT use `pkill -f <pattern>` in bash wrappers; pattern-based kills catch other concurrent processes (cost us 2 sp500 H=25 retries 2026-05-27). Use direct PID targeting or plain `timeout(1)` instead
- [feedback-task-priority-graph.md](feedback-task-priority-graph.md) — Project convention: when optimizing/prioritizing tasks, ALWAYS render the dependency graph to PNG (+SVG) via local graphviz `dot` (mermaid-cli unavailable) and display it via SendUserFile — automatically, not just inline text. Styling + recipe inside
