# ComfyUI-H3-Blended-Inject

ComfyUI custom node that injects keyframed latents into a MiniMax H3 A/V generation and blends
the rest of the video into them via intuitive img2img-style per-row denoise (no ghosting).

## Prototype branches

Branches prefixed `proto-` exist to test and prove a METHOD, not to implement a durable feature.
On a prototype branch:

- Code as efficiently and quickly as possible. Move fast; minimal ceremony.
- Do NOT write tests (no unit tests, no regression tests, no AST/signature tests) unless the
  user explicitly asks. The "bugs get regression tests" rule applies to durable code, not
  prototypes.
- Dead toggles, GPU-only unverified paths, and rough edges are acceptable; the user runs all
  GPU verification.
- Coverage consequence: untested prototype code fails the pre-push 90% diff-coverage hook.
  Either mark prototype-only code `# pragma: no cover`, or leave it uncovered and the user
  pushes with `git push --no-verify`. State which applies when handing off.

## Dev knowledge base (the wiki)

Start at [`.claude/docs/PER_ROW_IMG2IMG_NOTES.md`](docs/PER_ROW_IMG2IMG_NOTES.md) — the top-level
summary + direction doc. It links into `.claude/docs/per-row-img2img/` for drill-down detail.
Read the top doc first, then only the child docs the current task needs.

### Wiki edits and audit passes run in a subagent

All wiki updates and audit passes MUST be performed inside a subagent — UNLESS the wiki update
and audit IS the sole purpose of the current thread. In a mixed thread (analysis, coding, testing,
etc. plus a wiki write), delegate the wiki edit + its budget/link audit to a subagent; do not edit
the wiki inline. Give the subagent the exact content to record and the target doc(s); it performs
the edit, the char/line/link-budget audit, and the `Last updated:` pointer update, then reports
back. When the thread exists only to write/audit the wiki, edit it directly.

### Always record new work, developments, or results in the wiki

Whenever a new **theory, finding, hypothesis, plan, design decision, or correction** emerges in a
session (like the stochastic-recovery theory), write it into the appropriate wiki page BEFORE
acting on it — do not leave it only in chat. Add a new detail doc under `per-row-img2img/` if it
doesn't fit an existing one, link it from the index and any sibling docs whose conclusions it
revises, and keep the index's "core finding / current direction" current. Stale or contradicted
docs get fixed immediately, not deferred.

**Index freshness:** Any edit to a wiki page must also update the `Last updated: <date> (branch
\`<name>\`)` line in `PER_ROW_IMG2IMG_NOTES.md`. It went stale by two branches once — update it
every time, no exceptions.

### Memory vs wiki boundary

Technical findings about H3, ComfyUI internals, or this codebase belong in the wiki — one
canonical home per fact. The auto-memory system holds only cross-session pointers to wiki docs,
user/workflow preferences, and non-project facts. Never record the same technical fact in both:
duplicated facts drift (a memory kept an outdated claim the wiki had already corrected). If a
memory is found duplicating wiki content, reduce it to a pointer into the relevant wiki doc.

### Every wiki page needs a provenance/purpose tag

The FIRST line of every doc under `.claude/docs/` must be an HTML-comment provenance tag naming
what kind of content it is, so a reader knows its epistemic status at a glance:

```
<!-- provenance: theory (UNVERIFIED — analytical, no experimental confirmation) -->
```

Tag vocabulary (extend as needed):
- `confirmed` — verified by test/GPU/source; safe to rely on.
- `theory` — analytical hypothesis, NOT yet verified; mark `UNVERIFIED`.
- `reference` — source/file-line map or external-code notes.
- `status` — current state / open paths / planning.
- `bug` — a defect record (add `fixed`/`open`).

Existing docs may be missing tags — add them opportunistically as you touch each page; no need to
backfill all at once.

### Doc length & when to child-split

The 50–100 line guideline exists to **cap tokens per file in a way a CLI can check**. Measure BOTH
dimensions — neither alone is sufficient, and packing one to dodge the other is cheating:

- **Lines:** aim **50–100**; a doc may run shorter/longer when content genuinely calls for it.
- **Characters:** aim **≤ 8,000 chars/file**; **hard ceiling 12,000**. Past the ceiling, split.
- **Per-line chars:** **≤ ~250 chars/line** (roughly one wrapped paragraph). NEVER stuff a
  changelog, a findings-list, or multiple paragraphs onto one physical line to keep the line count
  down — that defeats the whole point. One idea per line/paragraph; let the line count reflect reality.

Audit any doc from the CLI before editing it:

```
awk '{ if (length > m) m = length } END { print FILENAME, NR" lines", m" max-line-chars" }' <file>
wc -c <file>            # total chars
```

A file is **over budget** if it exceeds the char ceiling, OR has any line past ~250 chars, OR runs
well past ~120 lines. When over budget, **split into a nested subfolder** (see
`per-row-img2img/native-h3-mechanism/` for the pattern):

1. Create `docs/<parent-name>/` (drop the `.md`), add an `index.md` (or keep the parent file as the
   index) that states the through-line and links each child.
2. Carve the content into **topic-coherent children, each 50–100 lines / ≤8k chars**, each with its
   own provenance tag on line 1 and verified stamp on line 2.
3. **Preserve every cross-link** — update inbound links (grep the docs tree for the old path) and
   keep outbound links working.

**The `Last updated:` line in `PER_ROW_IMG2IMG_NOTES.md` is a POINTER, not a changelog** — one
short sentence naming the latest thread + the child docs that hold it. Never grow it into a running
history; that history lives in the child docs and in git.
