---
name: project-streamlit
description: Streamlit dashboard runs locally but must support concurrent walk-forward runs
metadata:
  type: project
---

The Streamlit dashboard is local-only (no hosting), but the run-experiment view must tolerate multiple concurrent walk-forward runs without corrupting shared state. Use lockfiles inside each `runs/<module>/<timestamp>/` directory rather than a global lock.

**Why:** User picked "Local but plan for multi-run" — wants the ability to kick off several runs in parallel (e.g., one per ticker or hyperparameter sweep) without one clobbering another's artifacts.

**How to apply:** When implementing `dashboards/analog_mc/views/run_experiment.py`:
- Use `subprocess.Popen` to launch `walk_forward.py` — do NOT run the loop inside the Streamlit process (it blocks the event loop).
- Each run gets its own timestamped directory; write a `lock` file at start, remove at exit.
- Progress display polls per-fold parquet/JSON artifacts in the run dir — no in-process shared state.
- The dashboard does not need a global concurrency cap; the OS handles process scheduling.
