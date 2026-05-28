---
name: feedback-subagent-transcript-parsing
description: When you need a background sub-agent's progress, DON'T blindly Read/cat/tail its JSONL transcript — parse it with a bounded python/jq/grep query that extracts only progress signals and caps output. The harness "do not read" reminder guards against dumping the whole file, not against inspecting it.
metadata:
  type: feedback
---

When a background sub-agent is running and you want its progress, **parse its `*.output` JSONL transcript with a bounded query — do not blindly `Read`/`cat`/`tail` the whole file.**

**Why:** the harness emits a reminder ("Do NOT Read or tail this file … it will overflow your context") on the sub-agent output path. The user pushed back (2026-05-28): that warning guards against dumping the *entire* transcript (individual JSONL lines can be enormous tool-results) into context — it is NOT a blanket ban on inspecting it. A field-extracting parser that truncates every field is safe and gives real progress visibility on demand. Treating the file as untouchable wastes the signal; dumping it whole overflows context. The bounded parse is the middle path the user wants.

**How to apply:** on "what's the progress?" (or when you proactively want to know if a long agent is stuck), run a `python`/`jq`/`grep` one-liner against the agent's `output_file` that prints ONLY: total event count, tool-use count, the last ~10 tool-use names, the last assistant text tail (~700 chars), and the last tool-result snippet (~160 chars). Truncate every field; never print whole lines; never dump the array. This stays well within context. Do this for liveness/progress; you'll still get the formal completion notification when the agent finishes (don't poll in a tight loop).

**Reusable recipe** (handles the Claude Code session JSONL shape — `message.content` list of `text`/`tool_use`/`tool_result` blocks; defensive against schema drift):

```bash
OUT=<agent output_file path>
python3 - "$OUT" <<'PY'
import json, sys, os
path = sys.argv[1]
if not os.path.exists(path): print("no output file yet"); sys.exit(0)
lines = open(path, errors="replace").read().splitlines()
tools, last_text, last_tr = [], "", ""
for ln in lines:
    try: o = json.loads(ln)
    except Exception: continue
    msg = o.get("message") or o
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str) and msg.get("role") == "assistant": last_text = content
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict): continue
            t = b.get("type")
            if t == "tool_use": tools.append(b.get("name"))
            elif t == "text" and msg.get("role") == "assistant": last_text = b.get("text", "")
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    for x in c:
                        if isinstance(x, dict) and x.get("type") == "text": last_tr = x.get("text", "")[:160]
print(f"events={len(lines)} tool_uses={len(tools)}")
print("recent_tools:", tools[-10:])
print("last_tool_result_snip:", last_tr.replace(chr(10), " "))
print("last_assistant_text (tail):", last_text[-700:].replace(chr(10), " "))
PY
```

Adapt field caps as needed; the invariant is **bounded output, extracted fields, never the raw transcript**. See `[[feedback-sub-agent-foreground]]` and `[[feedback-experiment-agent-loop]]` for the launch-side patterns.
