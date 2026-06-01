# V1.1 Phase 6 — nifty50 H=25 agent-loop acceptance runbook

**Goal** (plan `docs/gbdt/V1.1_agent_driven_fs_hp_loop_plan.md` § 0.4 + the Phase 6 row in § 10): drive the **automated** `agent_file_protocol` FS+HP loop on the nifty50 H=25 cell and verify it reaches the same end-state conclusions the **hand-driven** loop documented in `docs/gbdt/_147_nifty50_h25_manual_fs_hp_loop.md` (the answer key). This is the acceptance test for the V1.1 capability: *an agent acting as the data scientist each iteration, driven through the real exit-and-resume CLI, reproduces a human data scientist's findings.*

This file is the **runbook** — what you (or a supervised agent) execute to run the full acceptance. The scaffolding for it ships in this PR:

| Piece | Path |
|---|---|
| Acceptance spec (the cell) | `configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_acceptance.yaml` |
| Comparison harness (the verdict) | `scripts/gbdt/acceptance_check_147.py` |
| Comparison-logic unit tests | `tests/gbdt/test_acceptance_check_147.py` |
| This runbook | `docs/gbdt/PHASE6_ACCEPTANCE_RUNBOOK.md` |

> **Why this is a runbook, not a script-that-runs-itself.** The heart of the acceptance is the *agent acting as the data scientist each iteration* (read the diagnostic bundle → decide FS pruning + HP changes → resume), which is exactly the capability under test. That is a multi-hour, human-in-the-loop-shaped process and is **not** a single autonomous command. The scaffolding makes the run launchable + checkable; a person (or a supervised agent session) drives the iterations.

---

## What the acceptance must reproduce (the `_147` findings → the checks)

`scripts/gbdt/acceptance_check_147.py::evaluate_acceptance` encodes nine findings from `_147` as pass/fail checks with explicit tolerances (each justified from the memo, cited in the script's `ANSWER_KEY` and per-check `finding_147`):

| Check | `_147` finding | Tolerance (and why) |
|---|---|---|
| `hp_ceiling_band` | All explored configs land in val_brier ~[0.1641, 0.1664]. | Band = the memo's full explored range (HP-only 0.1641–0.1661 + monotone up to 0.1664). |
| `hp_ceiling_spread` | The HP-only (non-monotone) configs span a *tiny* band ⇒ declare the HP ceiling. | spread ≤ 0.0020 = the documented HP-only width (0.1661 − 0.1641). |
| `depth_optimal` | Clean inverted-U: depth 4/6/8 → 0.1661/0.1642/0.1652; **depth 6 optimal**. | Best depth == 6; if 4/6/8 all explored, require 6 ≤ both neighbours. |
| `no_meaningful_improvement` | No HP/FS config meaningfully beats baseline (best "win" +0.0001 = noise). | best improvement over iter-0 baseline ≤ 0.0005 (the memo calls 0.0001 noise). |
| `monotone_contraindicated` | **Every** monotone config (iters 5–9) is worse than baseline; best monotone +0.0012. | all monotone > baseline AND best-monotone harm ≥ 0.0010. |
| `no_overfit_baseline` | Iter-0 train/val gap −0.0048 (val below train) ⇒ no overfit ⇒ FS hurts. | iter-0 gap ≤ +0.02 (the memo's no-overfit threshold / lesson 1). |
| `prevalence_drift_ceiling` | train 0.280 → val 0.204 → eval 0.138, declining — the calibration ceiling. | train−eval prevalence decline ≥ 0.05 (observed 0.142). |
| `ranking_robust` | Ranking strong + robust: legacy weighted R-precision ~2.1× throughout (~1.8× under R-Precision@10 — the post-2026-06-01 headline metric; see `[[project-r-precision-methodology]]`). | lift ≥ 1.5× base rate. |
| `final_features_not_collapsed` | FS neutral-to-harmful — best model keeps a substantial set (all-279 / 88-feat). | final feature count ≥ 80 (no aggressive prune). |

**SKIP vs FAIL.** A check **SKIPs** (does not fail) when the data it needs isn't in the artifacts — e.g. the loop never explored a monotone constraint (`monotone_contraindicated`), or `metrics.json` doesn't carry R-precision (`ranking_robust`, which then points you at `scripts/gbdt/compute_r_precision.py` — that script emits both legacy weighted R-precision and the current R-Precision@K at K ∈ {1, 3, 5, 10, 20}). The overall verdict is **PASS iff no check FAILED**; SKIPs are surfaced loudly so a *complete* acceptance resolves them. The acceptance is only fully demonstrated when the agent has explored depth, lr, FS, **and** at least one monotone-constraint iteration (so the contraindication check is evaluable, not skipped).

---

## Step 1 — launch + drive the agent loop

The cell is `nifty50 UP +10% / 25d / dd5%`, `callback_mode: agent_file_protocol`, `max_iterations: 10`. `<run_id>` = the spec stem = `nifty50_up_10pct_25d_dd5pct_acceptance`. Artifacts land at `results/gbdt/experiments/<run_id>/`; loop control files under `results/gbdt/experiments/<run_id>/loop/`.

**Pre-flight:** confirm the nifty50 panel is in the cache (`/data-health nifty50`) and `df` ≥ 10 G free on the project FS (long-run wedge guard, CLAUDE.md § Environment). Each iteration is **foreground with a `timeout` cap** — do NOT background+Monitor a per-iteration run (`.claude/memories/feedback-sub-agent-foreground.md`).

Each iteration is one launch that trains, writes the bundle, and exits 0:

```bash
cd /mnt/122CEE982CEE765F/Workspace/wt-gbdt-v11-phase6   # or the main checkout

# iter 0 — first launch:
timeout 2400 uv run python -m gbdt experiment \
  configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_acceptance.yaml \
  --callback-mode agent_file_protocol \
  2>&1 | tee -a logs/nifty50_up_10pct_25d_dd5pct_acceptance.log
#   -> writes results/gbdt/experiments/<run_id>/loop/iter_0_request.json + loop/checkpoint.json
#   -> prints "[loop] paused at iter 0 — resume with: ... --resume <run_id>"; exits 0.

# iter N+1 — after writing loop/iter_<N>_decision.json:
timeout 2400 uv run python -m gbdt experiment \
  configs/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_acceptance.yaml \
  --resume nifty50_up_10pct_25d_dd5pct_acceptance \
  2>&1 | tee -a logs/nifty50_up_10pct_25d_dd5pct_acceptance.log
```

> The first iteration pays the one-time feature-matrix build for the nifty50 panel (the bundle is cheap thereafter). `--callback-mode` is only needed on the first launch (it's snapshotted into the spec); resumes just need `--resume <run_id>`.

### What the driving agent must do EACH iteration

This is the `_147` decision chain, applied through the file protocol (full detail: `.claude/skills/gbdt-experiment/SKILL.md` § "Agent-driven FS+HP loop"; reasoning rules: SKILL.md Phase 3 + `_147` "Reusable lessons"):

1. **Read `loop/iter_<N>_request.json`** (use `Read` with `limit`/`offset` — the 279-feature importance map is large). Focus on: `diagnostics.metrics` (`val_brier`, `train_val_gap`), `diagnostics.overfit.no_overfit`, `diagnostics.prevalence_drift`, `diagnostics.top_features` / `pruned_summary`, `diagnostics.tuning_guidance`.
2. **Form a hypothesis** about what the last decision changed and what to try next — don't apply a fixed template. Track prior iterations (don't repeat a failed experiment).
3. **Decide**, anchored on the `_147` lessons:
   - **Check the train/val gap *before* pruning.** A negative / ≤0.02 gap ⇒ no overfit ⇒ FS is neutral-to-harmful. Do **not** prune for regularization here.
   - **Scan depth + lr** for the HP ceiling. depth {4, 6, 8} and lr {0.02, 0.05} are the diagnostic sweep; if val_brier stays in a tiny band, **declare the ceiling and stop** — don't burn iterations.
   - **Monotone constraints are contraindicated** on this interaction-driven cell. To make the `monotone_contraindicated` check *evaluable* (not skipped), the agent must try at least one monotone-constraint iteration (e.g. `+1` on the clean vol estimators) and observe the harm — exactly as `_147` iters 5–9 did. Check the unconstrained model's 1D-PDP (`/gbdt-diagnose` against `artifact_dir`), not the marginal correlation.
   - `importance ≈ 0` usually means **redundant**, not unrelated — pruning it is ~neutral on a non-overfit model.
4. **Write `loop/iter_<N>_decision.json`** (`prune_features` ⊆ `available_features`; `hp_changes` = tunable HPs within bounds, no pinned HPs; optional `should_stop`; a `rationale` lab-notebook entry). A malformed/out-of-bounds decision raises a clear `DecisionError` on resume and leaves state intact — fix + relaunch.
5. **Relaunch `--resume <run_id>`** → iter N+1. When you've mapped the ceiling and exhausted hypotheses, write `should_stop: true` (or let the plateau/degradation/cap gate fire) to finalize.

If a judgment call needs the human (e.g. "iter 6, val_brier flat at 0.164, abandon the spec?"), **ask in chat** before writing the decision — there is no `questions_for_user` field; control returns to the agent every iteration anyway.

### Expected shape of the run

- **Iteration count:** ~7–10. `_147` explored 10 distinct configs (iter 0 baseline + iters 1–9: depth 8, depth 4, lr 0.02, 88-feat FS, then five monotone-ablation configs). A faithful automated run covers the same ground; the plateau gate (or `should_stop`) ends it once the ~0.164 ceiling is mapped. `max_iterations: 10` is the hard cap.
- **Wall-clock:** iter 0 ≈ a few minutes (the nifty50 feature-matrix build dominates; the fit on 46 tickers × ~1600 rows is fast). Subsequent iterations ≈ 1–3 min of compute each. **Total *compute* ≈ 20–40 min.** The dominant cost is the **agent's attention** — reading each bundle + reasoning + writing each decision is ~15–30 min of focused work per iteration (plan § 12 R1), so the **wall-clock for a full acceptance is 2–4 hours of human-in-the-loop work**, not compute-bound.
- **Termination signal:** `metrics.json::loop.inner_stop_signal` ∈ {`agent_should_stop`, `plateau`, `degradation`, `cap`}. Either an explicit agent stop after mapping the ceiling or the plateau gate firing is the expected, `_147`-consistent end.

---

## Step 2 — run the acceptance check

Once the loop finalizes (the run dir has `iterations.jsonl`, `metrics.json`, `features.yaml`, `model.cbm`, …):

```bash
uv run python -m scripts.gbdt.acceptance_check_147 \
  results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_acceptance
# (the run dir is also the default arg, so a bare invocation works from repo root)

# machine-readable:
uv run python -m scripts.gbdt.acceptance_check_147 \
  results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_acceptance --json
```

It prints a PASS/FAIL/SKIP table (one row per finding, with the observed value, the answer-key target + tolerance, and a note) and a one-line overall verdict. Exit code: **0** if the overall verdict is PASS (no check FAILED), **1** if any check FAILED, **2** if the run dir is missing.

**If `ranking_robust` SKIPs** (R-precision not in `metrics.json`): compute it from the run's predictions and re-read —
```bash
uv run python -m scripts.gbdt.compute_r_precision \
  results/gbdt/experiments/nifty50_up_10pct_25d_dd5pct_acceptance/predictions/test.csv
```
then confirm the R-Precision@10 lift is ≥ 1.5× (the post-2026-06-01 form — `_147` was originally reported at ~2.1× under the legacy weighted form, which lands at ~1.8× under R-Precision@10; either way clears the threshold). The script emits both the current R-Precision@K and the legacy weighted form. The check reads R-precision from `metrics.json` if present; otherwise verify the lift manually against the `ranking_robust` tolerance.

**If `monotone_contraindicated` SKIPs:** the loop never tried a monotone constraint. Drive one more iteration that applies `monotone_constraints` on the clean vol estimators (as `_147` iter 5 did), then re-run the check. The acceptance is only *fully* demonstrated when this check is evaluable, not skipped.

---

## Step 3 — write the acceptance memo

When the check passes (or surfaces a documented divergence), write the per-experiment memo `docs/gbdt/_<id>_phase6_acceptance.md` (narrative: did the automated agent loop reach the `_147` conclusions? where did its path differ from the hand-driven one, and why?) + the machine-readable headline at `results/gbdt/data/_<id>_data.json`, per the repo's docs-vs-results convention (CLAUDE.md § Reporting conventions). The `_147` memo is the comparison baseline; the acceptance memo documents the *automated* loop's trajectory against it.

---

## Recovery / failure modes (plan § 9)

- **Malformed / out-of-bounds / pinned-HP / unknown-feature decision** → `--resume` raises a clear `DecisionError` and leaves the checkpoint + request intact. Fix `loop/iter_<N>_decision.json` and relaunch.
- **Missing decision file** → `DecisionError: decision file not found`. Write it and relaunch.
- **Runner crash mid-iteration** → the request + checkpoint persist; relaunch `--resume <run_id>` from the last good iteration.
- **Disk fills** → the write raises and the run aborts with a clear error; the pre-flight `df ≥ 10 G` guard should prevent it.
