---
name: feedback-disk-wedge-pattern
description: NTFS volumes can wedge when full — kernel writes enter D-state (uninterruptible sleep, unkillable), cascading to every IO on the same volume. Recovery requires reboot or freeing space via paths NOT on the wedged FS.
metadata:
  type: feedback
---

When the project volume (an NTFS-mounted disk on the dev machine where this repo was first developed) gets near-full (≥95%), kernel-level writes can wedge in `ntfs_mark_rec_free` and similar states. Processes hold the SQLite write lock and become unkillable (`kill -9` ineffective on D-state). Cascading effect: every new IO on the volume queues behind the wedge, including `ls`, `rm`, `git worktree`, `uv sync`. Cleanup attempts from within the wedged FS also block.

**Why:** First observed 2026-05-26 during 4 parallel GBDT experiment launches; each experiment competed for disk + SQLite, one wedged inside `data_pipelines.fetch()` retry storms, and once one was stuck the rest cascaded. The user had to reboot to recover. Disk went from 14 G free → 283 G free after intervention.

**Generality:** the specific failure mode (`ntfs_mark_rec_free` D-state) is NTFS-on-Linux specific, but the broader pattern (full filesystem → write-wedged process → cascading IO block) can occur on any FUSE-mounted FS, slow network mount, or near-full ext4 with concurrent fsync pressure. The mitigations below are filesystem-agnostic.

**How to apply:**

1. **Pre-flight check in long-running skills:** before any sub-agent that fetches or builds large datasets, assert the volume hosting the project has ≥10 G free (rule of thumb). If not, refuse to start and list candidate worktrees to prune. Use `df --output=avail $(pwd) | tail -1` to query the relevant mount portably.
2. **Pre-flight check on competing processes:** `ps -ef | grep "<expected runner>" | grep -v grep` should be empty before launching another instance. Don't run experiments in parallel that touch the same SQLite cache.
3. **Don't run cleanup commands that touch the wedged FS** when a wedge is suspected — they will themselves wedge. Free space via paths NOT on the wedged volume first (the user's HOME on a different mount, `/tmp`, etc.).
4. **Symptom triage:** `ps -o state,wchan -p <pid>` showing `D` plus a kernel WCHAN like `ntfs_mark_rec_free` means the process is wedged at kernel level — `kill -9` won't help. Recovery requires either disk pressure relief (sometimes thaws the wait), `umount -f` (rare success), or reboot.
5. **Secondary damage:** SQLite WAL files can be left filesystem-corrupted after a wedge (I/O errors on stat/read/unlink), independent of the underlying DB integrity. See `[[project-nse-data-quirks]]` for the `/tmp/exp_data/` workaround that keeps experiments running until the FS is repaired.

**Don't apply when:** running on a known-healthy mount or on a tmpfs/RAM-backed path (these wedges don't happen there). The pre-flight check is cheap regardless.

See `[[feedback-sub-agent-foreground]]` (parallel agents + Monitor-armed-then-exit makes wedges harder to detect) and `[[feedback-worktree-symlink-contract]]` (the wedge cascade made worktree creation also fail).
