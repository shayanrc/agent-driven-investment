---
name: feedback-sweep-best-cell-reporting
description: Sweep status reports must surface the best-cell-so-far by weighted R-precision test lift, not just progress counts
metadata:
  type: feedback
---

For any sweep status report — hourly check, completion notification, ad-hoc progress update — **always surface the best-cell-so-far**, not just cell-count progress.

**Why:** "9/20 done" tells the user nothing about *whether the work is producing signal*. "9/20 done, best so far: `russell1000_up_40pct_10d_dd20pct` at weighted R-prec 42.1× test (base rate 0.18%)" tells them whether to keep going, where to focus follow-up, and whether the sweep is worth its compute cost. The user explicitly asked for this on 2026-05-31 during the parallel r1k+nasdaq sweep work.

**How to apply:**
- Definition of "best": **weighted R-precision lift on test** (the standard cross-cell comparison metric per `[[project-r-precision-methodology]]`). Use eval-only-discriminating if no cell has test data yet.
- Cite the raw numbers + base rate (e.g. `R-prec 0.077 / base rate 0.0018, lift 42.1×`), not lift alone — lift hides the underlying hit-rate scale (this is the project's broader anti-lift-in-tables rule from CLAUDE.md).
- Cite the specific cell name (e.g. `russell1000_up_40pct_10d_dd20pct`) so the user can act on it.
- One line per running sweep is enough — don't expand into the full table on every status update.
- During the cold phase (no cells done yet), state that explicitly: "0 done — no best yet."
- For multi-sweep status (e.g. concurrent r1k + nasdaq), report one line per sweep.

Applies to all sweep reporting surfaces: the hourly cron (`[[feedback-hourly-wakeup-mandate]]`), sweep-agent reports, ad-hoc progress checks, dependency graphs (`[[feedback-task-priority-graph]]`).
