# /task-graph

Maintain the project's pending-task dependency graph at `docs/task_dependencies.{dot,png,svg}`.

The DOT file is the canonical source of truth for in-flight work. The agent updates it on every task transition (create / status change / close), re-renders PNG + SVG, and commits all three together.

## When to use

- **Automatic (no invocation needed):** every TaskCreate / TaskUpdate transition that changes the *shape* of the dependency graph — new task added, status moves between actionable / in-flight / done / blocked, a dependency edge is added or removed, or a goal node fires. Routine status checks that don't change the shape (e.g. a `/loop` hourly wakeup that just reports "all green") do NOT need a re-render.
- **User-invoked (`/task-graph`):** render current state and surface the PNG to the user. Useful when the user asks "what's pending?" / "what's next?" / "show me the graph."

## Where the files live

| File | Purpose |
|---|---|
| `docs/task_dependencies.dot` | Canonical source of truth — the DOT graph |
| `docs/task_dependencies.png` | Rendered PNG at 150 DPI for inline display |
| `docs/task_dependencies.svg` | Rendered SVG for crisp zooming |

All three are checked into git. Update them as a unit, never the PNG/SVG alone.

## Invocation

```
/task-graph                # render current DOT + send PNG to user
/task-graph add NNN "..."  # placeholder syntax; agent infers structure
```

When invoked, the agent reads the current `docs/task_dependencies.dot`, applies any pending changes, re-renders, and surfaces.

The skill is mostly a *convention* the agent follows automatically. Manual invocation is for "show me the graph" moments.

## House style

Inherit from `.claude/memories/feedback-task-priority-graph.md`. Key rules:

- `rankdir=TB`
- Title carries **date + time** of the last edit (the live snapshot moment).
- Group nodes into labeled **module/phase cluster boxes**.
- Recurring loops/monitors (e.g. `#94 hourly wakeup`) go in a **`scheduled tasks / loops`** cluster.
- Legend is a single **HTML table node in an isolated top rank**, with invisible `weight=100` anchor edges to every cluster's top node so it's centered above all clusters (not embedded in any one cluster).
- Node fills:
  - **green** (`#a4e9a4`) — actionable, ready to start
  - **blue** (`#a3c8e8`) — in-flight
  - **gold** (`#ffd966`) — goal node (octagon shape)
  - **white + dashed outline** — pending / gated (not yet actionable; waiting on a hard dep)
  - **filled gray** (`#cccccc`) — done / merged (kept for context; visually distinct from actionable)
  - **blue + dotted outline** — recurring (paused between firings)
  - **red** (`#dcb3b3`) — blocked on external state (data wait, sudo wait, etc.)
- Edge style:
  - **solid** — hard dep (downstream cannot start until upstream finishes)
  - **dashed** — soft dep / "surfaced by" / "may obviate"
  - **dotted** — recurring supervision (hourly wakeup → goals)
  - Color edges per workstream — pick a consistent color per logical track (e.g. `#cc3333` for the disk-cleanup critical path, `#338866` for the clean-memo path).
- **Don't upload graphs to external mermaid/Kroki renderers** — render locally with graphviz `dot`. `mmdc`/mermaid-cli is NOT usable in this environment.

## Rendering

```bash
cd docs/
dot -Tpng -Gdpi=150 task_dependencies.dot -o task_dependencies.png
dot -Tsvg              task_dependencies.dot -o task_dependencies.svg
```

The PNG goes inline with `SendUserFile`; the SVG is for crisp zoom.

## Bookkeeping rules

1. **Single source of truth.** The DOT file is canonical. Always re-render PNG + SVG together and commit all three.
2. **Update on every transition that changes the shape.** New task → add a node. Status change → recolor. Dep added → add an edge. Goal fires → recolor the goal node + propagate any unblocks.
3. **Keep done/merged nodes around** for the current plan cycle so the graph shows *why* the pending work has its current shape. Don't prune done nodes until they're truly stale (the surrounding pending work has all closed).
4. **Title timestamp.** Update the `label=` in the `graph [...]` line on every edit. Format: `"<repo name> — pending task dependencies (YYYY-MM-DD HH:MM TZ)\n<one-line subtitle of what changed last>"`.
5. **Commit message** when updating: short, names the task ID that triggered the edit. Examples:
   - `tasks: mark #244 done, add #253 follow-up`
   - `tasks: #251 + #252 → in-flight (V1.4 follow-up workstream)`
   - `tasks: re-render after V1.4 plan close-out`

## How to apply this skill

- After any TaskCreate / TaskUpdate that meaningfully changes the dependency picture: edit `docs/task_dependencies.dot`, re-render, stage all three files, commit.
- Do NOT edit the DOT file when the only thing that changed is a routine `[in_progress]` flip that's part of an already-modeled task (the node was already colored correctly).
- When the user asks "what's pending" / "what's next" / "audit the todos" / similar: re-render and surface PNG via `SendUserFile`.
- When opening PRs that close out plan phases: include the corresponding DOT edit in the SAME PR (the graph state and the merged work should land together).

## Cross-references

- `.claude/memories/feedback-task-priority-graph.md` — the canonical house-style spec (DOT skeleton, legend rules, color palette).
- CLAUDE.md § "Presenting plans and priorities" — the original mandate for rendering graphs to PNG on every prioritization turn.
