---
name: feedback-logs-evidence-policy
description: "logs/ is gitignored (PR #150 / task #253) but five pre-existing tracked logs remain in git as memo evidence. New evidence-worthy logs must be force-staged via `git add -f`. Captured stdout shouldn't be auto-committed because Python tracebacks + preflight lines historically leaked `/mnt/<UUID>/...` host paths."
metadata:
  type: feedback
---

The `logs/` directory is **gitignored as of PR #150** (task #253). Five historical tracked logs remain in git for memo-evidence traceability — these were *kept* through the cleanup, just sanitized in-place to remove the partition UUID prefix:

- `logs/nasdaq100_sweep.log` — backs `_192` (nasdaq100 sweep results)
- `logs/russell1000_sweep.log` + `logs/russell1000_up_10pct_100d_dd5pct.rerun.log` — back `_188` (russell1000 sweep results)
- `logs/nifty50_up_10pct_25d_dd5pct_xgb_acceptance.log` — backs investigation-B XGBoost acceptance run (nifty50 H=25)
- `logs/nifty50_deep_pre.json` — backs the NIFTY 50 deep-history seed (per-ticker row-count snapshot, before back-extend)

**Why captured stdout shouldn't be auto-committed.** Run logs include Python tracebacks (which contain absolute `/mnt/<UUID>/.../site-packages/...` file paths the agent runner can't sanitize) and historically included the runner's `[preflight] cache_db=/mnt/<UUID>/... data_root=/mnt/<UUID>/...` line. The preflight emission is now sanitized at the source (PR #150) — `[preflight] cache_db=data/processed.db data_root=data` — but tracebacks still leak via the stack-trace path component. So default-tracking `logs/` would silently re-introduce the path-leak we just cleaned up.

**How to apply.**

1. **For routine run logs**: don't commit. They're for local debugging. If they leak host paths into git, future contributors clone in confusion.
2. **For evidence-worthy logs** (sweep outcomes paired with a memo, per-ticker fingerprint snapshots backing a back-extend memo, acceptance-run traces backing an investigation memo): `git add -f <path>` to override the ignore. Before staging:
   - Scrub host-specific paths with the sed recipe used for the five historical logs:
     ```bash
     sed -i \
       -e 's|/mnt/<UUID>/Workspace/wt-[a-zA-Z0-9_-]*|<repo>|g' \
       -e 's|/mnt/<UUID>/Workspace/agent-driven-investment|<repo>|g' \
       -e 's|/mnt/<UUID>/cache_data|data|g' \
       -e 's|/mnt/<UUID>|<external>|g' \
       <path>
     ```
     Substitute `<UUID>` with the partition UUID per per-user memory `scratch-cache-path`.
   - Verify with `grep -c '/mnt/' <path>` → 0 before staging.
3. **For sweep memo PRs**: the sweep drivers (`scripts/gbdt/run_*_sweep.sh`) write to `logs/<universe>_sweep.log` AND `logs/<universe>_<cell>.cell.log`. Only the top-level sweep summary log is memo-worthy (the per-cell logs are noisy + reproducible from the artifact dirs). Force-add the summary log; leave the per-cell logs untracked.

**Why this convention rather than untracking everything.** The five historical logs are referenced from their memos as evidence (cell run order, timing, exit codes). Removing them from git would break those memos' traceability for marginal benefit. Going forward, the `git add -f` path keeps the option open without the auto-commit foot-gun.

**Don't.**

- Don't commit `logs/*.log` without scrubbing host paths — the `.gitignore` is a guardrail, not a license to skip the cleanup step.
- Don't paste captured stdout into a memo or PR body without scrubbing — same path-leak.
- Don't remove the five tracked historical logs to "tidy up" — they're load-bearing for `_188` / `_192` / inv-B memos.

See `[[feedback-worktree-symlink-contract]]` for the `${SCRATCH_CACHE}` / `${WORKSPACE_ROOT}` placeholder convention the scrub recipe targets; the runner-side preflight sanitization that prevents future automatic leakage is at `src/gbdt/__main__.py::_sanitize_preflight_for_emission` (PR #150).
