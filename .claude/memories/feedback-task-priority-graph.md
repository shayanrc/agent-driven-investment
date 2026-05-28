---
name: feedback-task-priority-graph
description: Project convention — when optimizing/prioritizing tasks (dependency graph, what's unblocked, sequencing, "optimize for time"), ALWAYS render the graph to a PNG and display it via SendUserFile, not just inline text. Render locally with graphviz `dot` (mermaid-cli is unavailable in this environment).
metadata:
  type: feedback
---

**Project convention.** When the work involves **optimizing or prioritizing tasks** — a dependency graph, "what's unblocked?", sequencing, "optimize for time", critical-path / parallelization questions — ALWAYS produce a task-dependency graph **and render it to a PNG (plus SVG) and display it via `SendUserFile`**. Inline mermaid text alone is not enough; the rendered picture is the deliverable.

**Why:** The user repeatedly asked for the dependency graph throughout the 2026-05 gbdt sessions ("show me a dependency graph… optimize it for time", asked many times) and then asked to make it a standing **project-level** convention to always create the graph and always display it as a PNG. Prioritization is consumed visually here.

**How to apply:**
- **Trigger automatically** (don't wait to be asked) whenever about to present multi-task sequencing, answer "is anything unblocked?", lay out a plan's phases, or discuss critical path / parallelization.
- **Render locally with graphviz `dot`** — installed at `/usr/bin/dot`. `mmdc` / `@mermaid-js/mermaid-cli` is NOT usable here (its npm install skips the headless-browser download and fails), so do NOT depend on mermaid-cli. Also present: `rsvg-convert`, `npx`, `node`; absent: `mmdc`, `inkscape`. Python: `graphviz` + `matplotlib` present; `pydot`/`networkx`/`cairosvg` absent.
- **Recipe** (proven 2026-05-28):
  1. Write a DOT file (e.g. `/tmp/depgraph.dot`).
  2. `dot -Tpng -Gdpi=150 depgraph.dot -o depgraph.png` and `dot -Tsvg depgraph.dot -o depgraph.svg` — give both (SVG vector + PNG raster).
  3. `SendUserFile` the PNG (and SVG) — `status:"proactive"` if surfacing unprompted, `"normal"` if replying.
  4. Also drop the mermaid source inline in the reply (the chat's CommonMark UI renders it) so there's a text version too.
- **Styling convention** (keep consistent so graphs read at a glance):
  - green `#d4edda` = actionable now / ready · blue `#d1ecf1` = in-flight / running · gold `#ffe69c` (penwidth 2) = goal · dashed gray `#f3f4f6` = pending/blocked · light gray `#e2e3e5` = backlog or context.
  - solid edge = hard dependency; dashed edge = soft/informational; color a key edge (e.g. `#0066cc`) to highlight the critical path or a key synergy.
  - cluster independent/parallel prerequisites in a labeled subgraph; do NOT combine a cluster with `rank=same` (graphviz drops the cluster box).
- **Do not** upload graphs to an external mermaid/Kroki web renderer — render locally (privacy + reliability).

CLAUDE.md § "Presenting plans and priorities" is the one-line summary of this convention.
