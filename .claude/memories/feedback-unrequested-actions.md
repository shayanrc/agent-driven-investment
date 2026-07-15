---
name: feedback-unrequested-actions
description: A partial echo of a bundled offer authorizes only that part — don't stop/restart the user's servers or take state-changing actions unasked.
metadata:
  type: feedback
---

The assistant offered "shut the server down and return to main, or leave the dashboard
up?" — the user replied "return to main", and the assistant killed the running Streamlit
dashboard server too. The user corrected: "I didn't tell you to kill the server." Only
"return to main" was authorized. (Migrated from per-user memory 2026-07-15.)

**Why:** The user controls their own long-running processes. A partial echo of a bundled
option authorizes ONLY the echoed part, not the whole bundle. Stopping/restarting a
server is a state-changing action that needs an explicit ask.

**How to apply:**
- Don't bundle a state-changing action with a benign one in a single offered option.
- When the user's reply matches only part of what was proposed, do exactly that part and
  leave the rest untouched (ask separately if needed).
- Treat stopping/restarting the dashboard server — or any process the user started — as
  the user's call, per the standing "confirm hard-to-reverse actions" rule.

Pairs with [[feedback-process-death-verification]] (don't kill long jobs on a hunch) and
the sub-agent kill rules in [[feedback-agent-pkill-antipattern]].
