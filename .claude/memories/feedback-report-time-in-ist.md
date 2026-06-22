---
name: feedback-report-time-in-ist
description: Report all times shown to the user in IST (UTC+5:30) unless they ask for another timezone
metadata:
  type: feedback
---

**Rule:** when reporting any time to the user (ETAs, check-in times, timestamps,
schedules, "as of" stamps), express it in **IST (Indian Standard Time, UTC+5:30)**
unless the user explicitly asks for another timezone.

**Why:** the project owner is in India; UTC/local stamps force a mental conversion.
The user asked for this explicitly on 2026-06-22 during the macro-lattice sweep
heartbeats.

**How to apply:**
- Convert UTC/machine-local to IST before displaying; label it "IST".
- Internal artifacts (logs, process timestamps, commit metadata) may stay UTC — this
  rule is about what is *shown to the user*.
- Cron expressions are interpreted in the machine's local tz, so when computing a
  next-fire time, find it in local time then convert to IST for the display string.

Pairs with [[feedback-replicate-output-in-dialog]] (clean, properly-formatted output).
