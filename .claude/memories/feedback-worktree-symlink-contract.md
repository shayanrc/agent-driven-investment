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

**Fix:** `rm -rf data && ln -s /mnt/122CEE982CEE765F/cache_data data`. The `.gitkeep` loss is irrelevant (it's a tracking-only file; the symlink target has its own `.gitkeep`).

*`data/` MUST point at the scratch cache `/mnt/122CEE982CEE765F/cache_data`, NOT the main checkout's `data/`. The main checkout's `data/processed.db-wal` is filesystem-corrupted (2026-05-26 disk wedge — see `[[project-nse-data-quirks]]` § 3 for the wedge story and the move to the scratch cache).*

*The scratch cache's `raw/` symlinks back to the main checkout, so immutable per-provider downloads still carry over automatically — sub-agents debugging "where are my raw downloads?" will find them at `data/raw/<provider>/...` resolving through that secondary symlink.*

*Also symlink `.env`: `ln -sf <main-checkout-abs>/.env .env` (NOT `ln -snf` — same nested-dir foot-gun as `data/`; if `.env` were ever a directory the snf flag would create `.env/.env`).*

**Validation:** after symlink, `readlink data` should print `/mnt/122CEE982CEE765F/cache_data`, not empty. If empty → wrong (data/ is still a directory). Then run the integrity gate: `sqlite3 "$(readlink -f data)/processed.db" 'PRAGMA quick_check'` MUST return `ok`. If the DB is corrupt, fix it BEFORE running any gbdt experiment against this worktree — a corrupt cache will silently feed wrong/missing rows into the feature build and contaminate the entire run. Sub-agents should pre-flight both checks before any `uv run` step.

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
2. Symlink command MUST be `rm -rf data && ln -s /mnt/122CEE982CEE765F/cache_data data`, not `ln -snf`. (Scratch cache, NOT main checkout's `data/` — see `[[project-nse-data-quirks]]` § 3.)
3. Symlink `.env`: `ln -sf <main-checkout-abs>/.env .env` (not `ln -snf`).
4. Immediately after the symlinks, run `git update-index --skip-worktree data/.gitkeep` in the worktree (once per new worktree).
5. Parent verifies with `readlink data` returning `/mnt/122CEE982CEE765F/cache_data` AND `sqlite3 "$(readlink -f data)/processed.db" 'PRAGMA quick_check'` returning `ok` before launching sub-agent.
6. Sub-agent prompt explicitly tells the agent: "Worktree pre-created; symlinks verified; skip-worktree applied; do NOT touch the symlinks; if `readlink data` returns empty or `PRAGMA quick_check` returns anything other than `ok`, STOP and report — don't try to fix."

**Don't apply when:** the sub-agent doesn't need a separate worktree (single-file edits, doc updates, etc.). Use the main checkout directly.

See `[[feedback-disk-wedge-pattern]]` (when disk wedges, symlink + worktree ops cascade-fail), `[[project-nse-data-quirks]]` § 3 (scratch cache rationale + the WAL corruption that motivated the move), and `CLAUDE.md` § Environment for the `wt-<scope>/` convention.

---

**Addendum 2026-06-04 — `.git/info/exclude` `data` rule masks ALL `data` subdirectories.**

The main checkout's `.git/info/exclude` carries an unanchored `data` line — added to keep the top-level `data` directory (the cache symlink) out of `git status` while it's a per-machine pointer. The trap: gitignore patterns without a leading slash match **anywhere in the tree**, so `data` masks any file under any subdir named `data/` — `results/gbdt/data/`, `dashboards/<module>/data/`, future `<module>/data/` paths, etc. New files added under these subdirs are silently absent from `git status` and skipped by `git add -A`. We hit this twice this session:

- V1.4 P3 backfill: `results/gbdt/data/v1.4_backfill_log.json` invisible to `git add -A`; required `git add -f` to stage.
- (#218 candidate path) any `_<id>_data.json` written under `results/<module>/data/` per the CLAUDE.md "Docs vs results" convention has the same trap.

**Two fixes**:
1. **Per-developer**: anchor the pattern in `.git/info/exclude` — replace the bare `data` with `/data` so it only matches the top-level cache dir. Self-applied; survives across worktrees of this checkout because `.git/info/exclude` is shared via `$GIT_DIR/info`.
2. **Per-file**: `git add -f <path>` when staging a new file under any nested `data/` subdir. Cheaper to remember once you know the trap exists; harder to remember out-of-context.

`.git/info/exclude` is NOT tracked (it's per-checkout, like `.gitignore_global`). A fresh `git clone` won't carry the unanchored `data` line. If you're collaborating with a contributor whose `git status` looks different from yours, this is a common cause — surface the line + suggest the `/data` anchor.

**How to apply**: when writing a new artifact under `results/<module>/data/`, `dashboards/<module>/data/`, or any subpath ending in `/data/`, default to `git add -f <path>` until the contributor base has migrated to the anchored pattern. The runner already does this for canonical files like `r_precision_at_k.csv`; new aggregator scripts should follow.
