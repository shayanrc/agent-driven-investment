---
name: feedback-read-before-edit
description: Always call Read on the target path before Edit/Write, even if you "know" the content. Saves a tool round-trip
metadata:
  type: feedback
---

**Rule:** before the first `Edit` or `Write` against any file path in a conversation, call `Read` on that exact path. The tool will error with "File has not been read yet. Read it first before writing to it." otherwise, costing a round-trip.

**Why:** the harness tracks per-path read state per conversation, **not** globally. The state does NOT survive:
- Sub-agent boundaries — when a sub-agent edits files, my own conversation has not Read them. I have to Read before I can Edit.
- Worktree creation — files in a freshly-created worktree haven't been Read through this session yet, even if I Read the same path in the main checkout.
- Switching between worktrees — `/mnt/.../wt-A/file.md` and `/mnt/.../wt-B/file.md` are tracked independently.
- Files I just authored via `Write` — those count as written, but the read state may not chain to a subsequent `Edit` cleanly.

The user pointed out this anti-pattern explicitly (2026-06-01 during the R-Precision@K rename PR #101 — I tripped into Read-then-Edit at least 5 times in one session).

**How to apply:**
- **Default to Read first.** Even if I think I know the content (e.g. from `grep` output, a sub-agent's report, or a CSV I just wrote), Read costs one tool call and saves the failed-Edit round-trip plus the re-Read.
- **Batch Read + Edit in adjacent tool calls.** Don't space them apart — the Read result is the working snapshot for the Edit.
- **Beware "I read part of it earlier."** Reading a slice of a file (`offset` + `limit`) is enough to satisfy the harness's read requirement, but if my Edit's `old_string` falls outside that slice I won't catch a stale match. Re-read the surrounding lines if I edited the file then need to edit it again much later.
- **After a sub-agent edits files I will subsequently Edit myself, Read the affected files before my first Edit.** The sub-agent's Read state isn't mine.
- **`Write` is the special case.** It also requires a prior `Read` to OVERWRITE an existing file (but not to create a new file). When I'm overwriting, Read first.

**Exception (don't waste a Read for this):** Trivial renames or single-character fixes via `Bash` with `sed -i` are fine when the path/content is unambiguous. But for anything multi-line or context-sensitive, use `Read` → `Edit`.

**Trigger to remember this:** if I see the error string "File has not been read yet" more than once in a session, slow down and adopt Read-first by default for the rest of the session.
