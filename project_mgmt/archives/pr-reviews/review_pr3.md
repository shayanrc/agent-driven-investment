# PR #3 (docs-meta-refresh → main) — Review

## Verdict
COMMENT (non-blocking; small accuracy + brevity nits)

## Severity summary
- Blocking: 0
- High: 0
- Medium: 2
- Low: 3

## Factual accuracy

- **analog_mc — shipped to main; V5.A.2 on `v5-experiments` not merged.** Verified. `git log main..v5-experiments --oneline` shows `92d1752 refactor(analog_mc): drop dead mix_paths/mix_ratios + rework V5.A.2 ensemble tests` and `65558ae feat(analog_mc): v5.a.2 path-level ensemble — canonical + fat-tail panel`. Main's most recent analog_mc work is `5b405e1 Merge branch 'v4.5-investigation' into main` + `0db4db9 docs(analog_mc): goal.md`. Claim correct.
- **data_pipelines — shipped via PR #1; v1.7 NSE added; `data-seed-nifty-total` active.** Verified. PR #1 is MERGED (`gh pr list --state all` confirms `"number":1,"state":"MERGED"`). `data-seed-nifty-total` exists locally (`git branch -a`). The `v1.7 NSE` work is on `main` (`b155614 feat(data_pipelines): v1.7 NSE domain ...`). Claim correct.
- **forecasters — PR #2 open on `v1-skills`.** Verified. `gh pr view 2` returns OPEN, `headRefName: v1-skills`. Claim correct.
- **gbdt — PR open on `gbdt-v1` with scaffolding + v0 scans.** Verified. `gh pr view 4` shows OPEN, title `"gbdt v1: module scaffolding + v0 NIFTY 50 opportunity scans (3)"`. `git log main..origin/gbdt-v1` shows `c0a52c3 scaffold(gbdt)` + v0.1/v0.2/v0.3 scans. Claim correct.

All four module-status claims are accurate.

## Internal consistency

Memory wiki-refs (resolved against `.claude/memories/` on the PR head `d5a3b24`):

- `[[project-overview]]` → `project-overview.md` — exists.
- `[[feedback-branch-retention]]` → `feedback-branch-retention.md` — exists (already on `main`, not new in this PR; PR description's claim that the new file is `feedback-experiment-agent-loop.md` is correct).
- `[[feedback-experiment-agent-loop]]` → `feedback-experiment-agent-loop.md` — exists (the new file).
- `[[project-data-source]]` → `project-data-source.md` — exists.
- `[[project-csv-schema]]` (referenced from `project-data-source.md` line 17 with "(TBD)" tag) — does NOT exist. Pre-existing (not introduced by this PR), kept because the refresh preserved the original `See [[project-csv-schema]]` footer. Low priority but worth noting.

CLAUDE.md `.claude/memories/feedback-experiment-agent-loop.md` reference (CLAUDE.md:44) resolves on this branch.

`docs/<module>/goal.md` links in CLAUDE.md:
- `docs/analog_mc/goal.md` — exists on `main` and on this branch.
- `docs/data_pipelines/goal.md` — exists on `main` and on this branch.
- `docs/forecasters/goal.md` — exists ONLY on `v1-skills` (verified via `git ls-tree origin/v1-skills docs/forecasters/`); not on `main` and not on this branch.
- `docs/gbdt/goal.md` — exists ONLY on `gbdt-v1`; not on `main` and not on this branch.

When PR #3 merges to `main`, two of the four `goal.md` links will be 404s until PR #2 and PR #4 also merge. The PR description acknowledges this ("once the corresponding module branches merge"), but in practice it creates a window of dead links on `main`. See Finding 1.

## CLAUDE.md authority

- **"Claude Code skills" section (lines 38-45)** is crisp and unambiguous. The "one skill = one verb" / runner-script-owns-implementation / no-backend-flags rules are testable when reviewing future skill additions. Good.
- **"Worktree workflow" bullet (line 85)** is reasonable but hardcodes the user's machine path (`/mnt/<UUID>/Workspace/wt-<scope>`). Fine for now (single-user repo) but if the repo ever gains a second contributor that path becomes wrong. Low priority.
- **"What not to do (analog_mc)" section (lines 93-99)** was kept at the bottom of CLAUDE.md. With 4 modules now, this analog_mc-specific block is a bit out of place at root level. `docs/analog_mc/goal.md` would be the natural home. Not blocking but worth a tidy. See Finding 2.
- **"Data and configs" generalization (lines 49-51)** correctly reflects the cache: `data/processed.db` + `data/raw/<provider>/...` is confirmed by `ls data/raw` → `jugaad nselib tiingo yfinance`. The single-writer / WAL contract claim matches the data_pipelines module's known behavior. Accurate.

## Style / brevity

CLAUDE.md is supposed to be the summary; the detail belongs in `.claude/memories/`. The new sections (especially "Claude Code skills" and the "Environment" worktree bullet) are dense but each line earns its place — there's no single bullet that obviously wants relocation. The 36-line growth is acceptable. The one place I'd watch is the "Environment" section getting long (5 bullets, each multi-line); if a 5th module lands, this section should probably be split out into a memory and reduced to a pointer.

## Findings

### [Medium] Editable-packages claim is aspirational on `main` until PR #2 / PR #4 merge
**File:** `CLAUDE.md:82`
**Observation:** Claim is `"analog_mc, data_pipelines, forecasters, and gbdt are all installed as editable packages via hatchling"`. On this branch's `pyproject.toml`, `packages = ["src/analog_mc", "src/data_pipelines"]` only. `forecasters` is added in `origin/v1-skills` (`packages = [..., "src/forecasters"]`), `gbdt` is added in `origin/gbdt-v1` (`packages = [..., "src/gbdt"]`). If PR #3 merges first, CLAUDE.md will assert four editable packages while `pyproject.toml` only declares two — and `import forecasters` / `import gbdt` will fail on a fresh `uv sync`.
**Suggested action:** Either (a) merge PRs #2 and #4 before #3, or (b) hedge the wording, e.g., "as the module PRs land, each becomes an editable package via hatchling — see `pyproject.toml` `[tool.hatch.build.targets.wheel] packages` for the current set." The same risk applies to the `docs/<module>/goal.md` links for forecasters and gbdt (already noted in the PR description, but the same merge-order hedge applies).

### [Medium] Two `goal.md` links in CLAUDE.md will be broken on `main` until PRs #2 and #4 merge
**File:** `CLAUDE.md:7-8`
**Observation:** `docs/forecasters/goal.md` and `docs/gbdt/goal.md` don't exist on this branch (verified via `git ls-tree`). The PR description acknowledges this but it's worth surfacing as a concrete merge-order constraint, not just a footnote.
**Suggested action:** Merge in order PR #2 → PR #4 → PR #3, OR add a short caveat in CLAUDE.md (e.g., "links are populated as the corresponding module PRs land"). Same root cause as Finding 1; pick a single resolution that covers both.

### [Low] `[[project-csv-schema]]` wiki-link still points to a non-existent memory
**File:** `.claude/memories/project-data-source.md:17`
**Observation:** Footer reads `See [[project-csv-schema]] (TBD) for the analog_mc CSV file format.` The `(TBD)` marker is honest, but the memory has been TBD for a while now; either write the schema memory or drop the reference.
**Suggested action:** Out of scope for this PR (pre-existing), but consider parking in a future memory-hygiene pass.

### [Low] `"What not to do (analog_mc)"` section in CLAUDE.md is now module-specific noise at the project-conventions root
**File:** `CLAUDE.md:93-99`
**Observation:** When CLAUDE.md was an analog_mc-mostly doc, this fit at root. With 4 modules and the new intro that puts all 4 on equal footing, having one module's "don't do" list at the root feels imbalanced. The content itself is correct and worth keeping somewhere.
**Suggested action:** Move the section to `docs/analog_mc/goal.md` (or a sibling `docs/analog_mc/anti_patterns.md`) and replace it in CLAUDE.md with a one-line pointer. Not blocking; could be a follow-up cleanup.

### [Low] Worktree workflow bullet hardcodes user-specific path
**File:** `CLAUDE.md:85`
**Observation:** `"create a worktree under /mnt/<UUID>/Workspace/wt-<scope>"` — that path only exists on this user's machine. If the repo ever onboards a second contributor (or runs in CI), the convention should be path-shape, not a literal path.
**Suggested action:** Change to something like `"create a sibling worktree (we use `wt-<scope>/` next to the main checkout)"`. Minor.

## Recommendation

Merge after either (a) reordering merges so PRs #2 and #4 land before #3 (cleanest), or (b) hedging the editable-packages and `goal.md`-link wording in CLAUDE.md to be merge-order-tolerant. Everything else is style / cleanup that can land in a follow-up. Anti-attribution rule is correctly observed throughout.
