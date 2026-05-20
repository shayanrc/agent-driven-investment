---
name: project-results-layout
description: Aggregated experiment metrics (JSON) live in `results/<module>/data/`, not `docs/<module>/`
metadata:
  type: project
---

Aggregated per-experiment metrics (mean CRPS, per-vol-regime breakdown, ACF curves, decision-rule values) live in `results/<module>/data/_<id>_data.json` — module-namespaced like everything else (see [[project-layout]]).

For analog_mc: `results/analog_mc/data/_e<N>_data.json`. Originally these lived alongside the markdown reports in `docs/analog_mc/`; separated 2026-05-20 (commit `9e660c4`) — docs hold narratives, results hold data.

**Why:** The raw per-fold artifacts under `runs/<module>/<timestamp>/` are gitignored. The aggregated JSON in `results/` is therefore the only **durable, checked-in** record of an experiment's headline metrics — future experiments compare against these.

**How to apply:**
- New aggregate scripts write to `results/<module>/data/_<id>_data.json` by default — don't restore the old `docs/<module>/` path.
- Markdown reports under `docs/<module>/_<id>_<name>.md` reference the json via the new path in their Deliverables section.
