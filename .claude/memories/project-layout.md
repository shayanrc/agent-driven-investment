---
name: project-layout
description: Per-module namespacing convention — analog_mc is one of several future modules
metadata:
  type: project
---

Every module-specific directory is namespaced under the module name. Top-level directories are reserved for cross-module concerns.

```
src/<module>/                  # package code
docs/<module>/                 # design docs
tests/<module>/                # tests
configs/<module>/*.yaml        # YAML configs
runs/<module>/<timestamp>/     # output artifacts
dashboards/<module>/
  ├── app.py                   # module's runnable Streamlit entry point
  └── views/                   # view modules
dashboards/app.py              # thin global launcher
```

**Why:** Deviates from the original IMPLEMENTATION_PLAN.md layout (which assumed `dashboards/views/` without a module namespace). User explicitly requested per-module namespacing because analog_mc is just one of many planned modules — see [[project-overview]].

**How to apply:** When adding new module-scoped files (tests, configs, run artifacts, dashboard views), always nest them under `<top>/<module>/`. Never put module-specific code in the top-level `dashboards/`, `tests/`, `configs/`, or `runs/` directories — those are reserved for cross-module concerns (the global launcher, shared fixtures, etc.). This convention is also documented in [[CLAUDE.md]] at the repo root.
