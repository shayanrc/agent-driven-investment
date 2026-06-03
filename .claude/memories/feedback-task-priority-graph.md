---
name: feedback-task-priority-graph
description: Project convention — when optimizing/prioritizing tasks (dependency graph, what's unblocked, sequencing, "optimize for time"), ALWAYS render the graph to a PNG (+SVG) and display it via SendUserFile, not just inline text. Render locally with graphviz `dot` (mermaid-cli is unavailable). House style is top-to-bottom, module/phase cluster boxes, an HTML-table legend node in an isolated top rank (centered above all clusters via spacer-row anchors), and a date+time-stamped title.
metadata:
  type: feedback
---

**Project convention.** When the work involves **optimizing or prioritizing tasks** — a dependency graph, "what's unblocked?", sequencing, "optimize for time", critical-path / parallelization questions — ALWAYS produce a task-dependency graph **and render it to a PNG (plus SVG) and display it via `SendUserFile`**. Inline mermaid text alone is not enough; the rendered picture is the deliverable.

**Why:** The user repeatedly asked for the dependency graph throughout the 2026-05 gbdt sessions and asked to make it a standing **project-level** convention to always create the graph and always display it as a PNG. Prioritization is consumed visually here. The house style below was dialed in interactively on 2026-05-29 (the user supplied a reference image and refined it).

**How to apply:**
- **Trigger automatically** (don't wait to be asked) whenever about to present multi-task sequencing, answer "is anything unblocked?", lay out a plan's phases, or discuss critical path / parallelization.
- **Render locally with graphviz `dot`** — installed at `/usr/bin/dot`. `mmdc` / `@mermaid-js/mermaid-cli` is NOT usable here (its npm install skips the headless-browser download and fails), so do NOT depend on mermaid-cli. Also present: `rsvg-convert`, `npx`, `node`; absent: `mmdc`, `inkscape`. Python: `graphviz` + `matplotlib` present; `pydot`/`networkx`/`cairosvg` absent.
- **Recipe** (proven 2026-05-29):
  1. Write a DOT file (e.g. `/tmp/optimize.dot`) following the house style below.
  2. `dot -Tpng -Gdpi=150 g.dot -o g.png` and `dot -Tsvg g.dot -o g.svg` — give both (SVG vector + PNG raster).
  3. `SendUserFile` the PNG (and SVG) — `status:"proactive"` if surfacing unprompted, `"normal"` if replying.
- **Do not** upload graphs to an external mermaid/Kroki web renderer — render locally (privacy + reliability).

## House style (keep consistent so graphs read at a glance)

- **Layout: `rankdir=TB`** (top-to-bottom flow), `splines=true`.
- **Title:** `Agent-Driven Investment — <subject> (<YYYY-MM-DD HH:MM TZ>)`. The title MUST carry **both date AND time** (get it from `date '+%Y-%m-%d %H:%M %Z'`). Do NOT put the legend text in the title — the legend is its own subgraph (below).
- **Group nodes into labeled cluster boxes** by module / phase / workstream (e.g. `gbdt · V1.2 — XGBoost backend`, `gbdt · V1.1 — agent-driven loop`, `data_pipelines / sweep`). Rounded, `color="#b0b0b0"`, `labelloc="t"`, bold cluster label. Cross-cluster edges are fine.
- **Recurring loops / monitors live in their own `scheduled tasks / loops` cluster** — the hourly wakeup (#94) and any other always-on schedulers/monitors go here, NOT mixed into the work clusters.
- **Node fills/styles** (each visually distinct — pending vs done was a confusable pair, now resolved):
  - actionable now: `fillcolor="#d4edda"`, `color="#155724"` (green)
  - in flight / running: `fillcolor="#d1ecf1"`, `color="#0c5460"` (blue)
  - goal: `fillcolor="#ffe69c"`, `color="#856404"`, `penwidth=2` (gold)
  - **pending / gated: `fillcolor="#ffffff"`, `color="#adb5bd"`, `style="rounded,filled,dashed"`** (WHITE box, dashed gray outline — reads as "not started")
  - **done / merged: `fillcolor="#ced4da"`, `color="#495057"`** (filled medium-gray — reads as "complete"; clearly darker than the white pending box)
  - recurring (paused between ticks): `fillcolor="#d1ecf1"`, `color="#0c5460"`, `style="rounded,filled,dotted"` (blue, DOTTED border — always-on but idle right now)
  - blocked on data: `fillcolor="#f8d7da"`, `color="#842029"` (red)
- **Edges:** solid = hard dependency; `style=dashed` = soft/informational; color a workstream's critical path (e.g. orange `#d9534f` = catboost/V1.1 path, purple `#5a3e9e` = xgboost/V1.2 path). Put the meaning in the legend, not the title.
- **Legend = a single HTML-table `[shape=plaintext, label=<<TABLE>...</TABLE>>]` node in an isolated top rank, centered above all clusters via spacer-row anchors.** (NOT a text blob; NOT a bottom table; NOT the older `cluster_legend` swatch-row approach which couldn't be reliably centered or rank-isolated.) See the canonical skeleton below for the spacer-row + `weight=100` anchor pattern. **Optional "done this session" swatch** (`fillcolor="#a8d8c0"`, mid-green between done-gray and actionable-bright-green) can be added to visually distinguish work completed in the current session from older completed work.
- **SVG/PNG render parity**: set explicit `FACE="Helvetica" POINT-SIZE="N"` on every `FONT` tag inside the HTML table. Without this, SVG and PNG can use different default fonts → different cell widths → different layouts.

## Canonical DOT skeleton (current — HTML-table legend + spacer-row pattern, proven 2026-06-03)

```dot
digraph deps {
  rankdir=TB;
  newrank=true;
  graph [fontname="Helvetica", bgcolor="white", splines=true, nodesep=0.35, ranksep=0.7,
         labelloc="t", fontsize=15,
         label="Agent-Driven Investment — <subject>\n2026-06-03 11:33 IST"];
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.16,0.09", color="#444444"];
  edge [fontname="Helvetica", fontsize=8, color="#888888"];

  /* ---------- legend (single wide HTML-table node, isolated top rank) ---------- */
  legend [shape=plaintext, label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="6" CELLPADDING="6" BGCOLOR="#f6f6f6">
      <TR>
        <TD COLSPAN="8" BORDER="0"><FONT FACE="Helvetica" POINT-SIZE="11"><B>legend</B></FONT></TD>
      </TR>
      <TR>
        <TD BGCOLOR="#ced4da"><FONT FACE="Helvetica" POINT-SIZE="10">done / merged</FONT></TD>
        <TD BGCOLOR="#a8d8c0"><FONT FACE="Helvetica" POINT-SIZE="10">done this session</FONT></TD>
        <TD BGCOLOR="#d4edda"><FONT FACE="Helvetica" POINT-SIZE="10">actionable</FONT></TD>
        <TD BGCOLOR="#d1ecf1"><FONT FACE="Helvetica" POINT-SIZE="10">in flight</FONT></TD>
        <TD BGCOLOR="#ffffff" STYLE="dashed"><FONT FACE="Helvetica" POINT-SIZE="10">pending / gated</FONT></TD>
        <TD BGCOLOR="#f8d7da"><FONT FACE="Helvetica" POINT-SIZE="10">blocked on data</FONT></TD>
        <TD BGCOLOR="#d1ecf1"><FONT FACE="Helvetica" POINT-SIZE="10">recurring (paused)</FONT></TD>
        <TD BGCOLOR="#ffe69c"><FONT FACE="Helvetica" POINT-SIZE="10">goal</FONT></TD>
      </TR>
    </TABLE>
  >];
  /* Dedicated empty spacer row enforces visual gap between legend (rank=min) and ALL clusters. */
  spacer_L [shape=point, width=0.01, style=invis];
  spacer_C [shape=point, width=0.01, style=invis];
  spacer_R [shape=point, width=0.01, style=invis];
  {rank=min;  legend;}
  {rank=same; spacer_L; spacer_C; spacer_R;}
  /* Strong vertical pull keeps legend ABOVE the spacer row */
  legend -> spacer_L [style=invis, weight=1];
  legend -> spacer_C [style=invis, weight=100];
  legend -> spacer_R [style=invis, weight=1];
  /* EVERY cluster's top-most node must have a weight=100 invisible edge from a spacer,
     so the cluster cannot drift up to the legend's rank. Distribute across L/C/R to control x-position. */
  spacer_L -> <leftmost_cluster_top_node>  [style=invis, weight=100];
  spacer_C -> <middle_cluster_top_node>    [style=invis, weight=100];
  spacer_R -> <rightmost_cluster_top_node> [style=invis, weight=100];
  /* repeat spacer_? → <top_node> for every cluster you add */

  /* ---------- work clusters ---------- */
  subgraph cluster_work { label="<module · phase>"; labelloc="t"; fontsize=12; fontname="Helvetica-Bold"; style="rounded"; color="#b0b0b0"; margin=12;
    /* work nodes here */ }
  subgraph cluster_scheduled { label="scheduled tasks / loops"; labelloc="t"; fontsize=12; fontname="Helvetica-Bold"; style="rounded"; color="#b0b0b0"; margin=12;
    WAKE [label="#94 · hourly wakeup", fillcolor="#d1ecf1", color="#0c5460", style="rounded,filled,dotted"]; }

  /* dependency edges ... */
}
```

**Notes**:
- Set `newrank=true` at graph level — required so the `{rank=min; legend;}` pin works across clusters.
- Do NOT combine a cluster with a global `rank=same` of its members (graphviz drops the cluster box).
- The spacer-row pattern is what makes the legend BOTH centered (via L/C/R distribution of anchor edges) AND in its own rank (via `weight=100` strong vertical pull on every cluster top node).
- The old `cluster_legend` swatch-row approach is deprecated — it couldn't be reliably centered or rank-isolated against multiple cluster anchors.

CLAUDE.md § "Presenting plans and priorities" is the one-line summary of this convention.
