---
name: feedback-replicate-output-in-dialog
description: When asked to show something, re-render the result as properly-formatted markdown in the reply — don't point at shell/tool output
metadata:
  type: feedback
---

**Rule:** when the user asks to be shown something (a table, a ranking, a metric, a
diff, a file's contents), **replicate the result as properly-formatted markdown in the
response text.** Compute it in the shell if needed, but then reproduce the output
cleanly in the dialog. Do NOT reply with "see the table above" / "the output shows…"
pointing at a tool-result block.

**Why:** shell/tool output is not always rendered in the chat in the format intended,
so the user may not see it the way it was meant to look. The user asked for this
explicitly on 2026-06-22 after a top-10 ranking was left as raw shell output.

**How to apply:**
- After running the command, paste the formatted result into the reply.
- Render tabular data as markdown tables; format numbers consistently (ints where
  appropriate, fixed decimals for metrics).
- Follow the repo reporting conventions: raw metric + base_rate columns, **no lift
  columns** in data tables (see CLAUDE.md § Reporting conventions and
  [[feedback-predictions-table-format]]).
- Keep [[feedback-report-time-in-ist]] polish (IST timestamps).
