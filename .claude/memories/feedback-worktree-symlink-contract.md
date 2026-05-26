---
name: feedback-worktree-symlink-contract
description: Sub-agents can't `git worktree add` to sibling paths (sandbox blocks). Parent must pre-create worktrees. Symlink for data/ must `rm -rf data && ln -s` because `ln -snf` creates a nested `data/data` when `data/` exists as a real dir (with `.gitkeep`). After symlinking, run `git update-index --skip-worktree data/.gitkeep` so the shadowed tracked file doesn't show as a phantom modification and break `git rebase --autostash` / `git checkout`.
metadata:
  type: feedback
---

Three related foot-guns surfaced during the GBDT v1 pilot experiment launches (2026-05-26):

**1. Sub-agent worktree creation is sandboxed.**

`git worktree add <sibling-path>` from inside a sub-agent gets auto-classifier-denied — the harness's `EnterWorktree` tool only creates worktrees under `.claude/worktrees/` (or a configured base path), NOT at an arbitrary sibling path next to the main checkout. Sub-agents will repeatedly fail with permission errors, burning tool budget, before giving up.

**Fix:** parent agent pre-creates the worktree before launching the sub-agent. Use a path relative to the repo's parent directory (matches the `wt-<scope>/` convention in `CLAUDE.md` § Environment):
```bash
git worktree add "$(dirname "$(pwd)")/wt-<scope>" -b <branch-name> origin/main
```
Then sub-agent prompt says "Worktree pre-created at `<sibling>/wt-<scope>`; cd there and proceed."

**2. data/ symlink contract.**

The repo has `data/.gitkeep` tracked. When `git worktree add` checks out, `data/` materializes as a real directory containing `.gitkeep`. Subsequent `ln -snf <main>/data data` does NOT replace the directory — `ln -snf` only replaces an existing symlink or empty path, never a non-empty directory. Instead it creates `data/data` (a symlink INSIDE the dir, named `data`). `ls data/` then shows the inner-symlink AND the original `.gitkeep` — looks vaguely correct to a casual `ls`, but the cache reads fail because the runner expects `data/processed.db` and gets `data/data/processed.db`.

**Fix:** `rm -rf data && ln -s <absolute-path-to-main>/data data`. The `.gitkeep` loss is irrelevant (it's a tracking-only file; the symlink target has its own `.gitkeep`).

**Validation:** after symlink, `readlink data` should print the ABSOLUTE PATH to the target, not empty. If empty → wrong (data/ is still a directory). Sub-agents should pre-flight validate this before any `uv run` step.

**3. `data/.gitkeep` phantom modification after symlinking.**

Once `data/` is replaced with a symlink (per foot-gun #2), the tracked `data/.gitkeep` path resolves through the symlink to the main checkout's `.gitkeep` — which still exists, but git compares index-vs-worktree per the worktree's own index, and the symlink-traversal makes it look modified/missing depending on git version. The visible failure is `git rebase --autostash` or `git checkout <branch>` aborting with "Your local changes to the following files would be overwritten" on `data/.gitkeep`. Stashing doesn't help (re-applies the phantom diff); reverting the file does nothing (symlink shadows the write).

**Fix:** in the new worktree, run once:
```bash
git update-index --skip-worktree data/.gitkeep
```
This tells git's index to stop comparing the worktree copy of `data/.gitkeep` — exactly what we want, since the symlink target's `.gitkeep` is the same tracked content. It's a per-worktree index flag (not a tree-wide config), so it must be re-run for each new worktree. The main checkout is unaffected.

**Why all three:** observed 2026-05-26 across multiple experiment relaunches. Foot-gun #1 caused immediate abort at step 1. Foot-gun #2 was hit by an agent that debugged inline (+5 min) and by another that correctly stopped + reported. Foot-gun #3 surfaced later when sub-agents tried to rebase their worktree branches onto updated main and got blocked by the phantom `data/.gitkeep` diff. Pattern: parent pre-creates + symlinks correctly + runs skip-worktree + verifies, then launches sub-agent.

**How to apply:**

1. Parent agent does worktree creation + symlink setup for every long-running experiment (or any task that needs filesystem isolation).
2. Symlink command MUST be `rm -rf data && ln -s <abs-path>/data data`, not `ln -snf`.
3. Immediately after the symlink, run `git update-index --skip-worktree data/.gitkeep` in the worktree (once per new worktree).
4. Parent verifies with `readlink data` returning the absolute target path before launching sub-agent.
5. Sub-agent prompt explicitly tells the agent: "Worktree pre-created; symlinks verified; skip-worktree applied; do NOT touch the symlinks; if `readlink data` returns empty, STOP and report — don't try to fix."

**Don't apply when:** the sub-agent doesn't need a separate worktree (single-file edits, doc updates, etc.). Use the main checkout directly.

See `[[feedback-disk-wedge-pattern]]` (when disk wedges, symlink + worktree ops cascade-fail) and `CLAUDE.md` § Environment for the `wt-<scope>/` convention.
