---
name: clio-kit-archify
description: 'Use when this workflow is requested: Validated interactive system maps
  and diagrams as standalone HTML from a typed JSON spec, for one of five diagram
  types (architecture, workflow, sequence, dataflow, lifecycle). Use when the user
  asks to visualize architecture, a workflow, a call sequence, a data pipeline, or
  a state machine, or to map this repository. Not for ad hoc drawings or slide art,
  and not for Mermaid output; use the artifact tool for plain Markdown. Triggers on
  "architecture diagram", "system map".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/planning/archify
  upstream-name: archify
  upstream-eval-status: scenarios-recorded
license: MIT
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Archify

Create a self-contained interactive HTML diagram from a small typed JSON
specification. This library package contains Clio instructions; the renderer is
an independently installed upstream Archify v2.16.0 package. Before authoring,
locate its directory containing `bin/archify.mjs`, `schemas/` and `examples/`.
If unavailable, ask the operator to prepare that upstream renderer; do not claim
that installing this instruction package installed it. Below, `<renderer>` is
that verified directory, and renderer-relative schema/example paths refer to it.
The renderer needs Node 18 or later; ordinary rendering then needs no network.

## Fast authoring path

1. Choose `architecture`, `workflow`, `sequence`, `dataflow`, or `lifecycle`
   from the question. When ambiguous, run
   `node <renderer>/bin/archify.mjs guide "<scenario>" --json`.
2. Read one matching schema in `schemas/`, `schemas/common.schema.json`, and
   one matching example in `examples/`. Read only those files. The example
   supplies field shape, never facts: author new stable ids, domain wording,
   and layout. New workflows use `schema_version: 2`.
3. Write the candidate JSON as the very next action. Do not plan coordinates
   in prose. Start with one clear main path, short side branches, sparse
   labels, and at most 12 primary nodes. Set `meta.quality_profile` to
   `"showcase"` unless the user asks for a dense `standard` map. Start with
   automatic routes and labels; add `via`, `channelX`, `channelY`, or
   `labelAt` only when a diagnostic asks for one, and at most one per repair.
4. Validate after every edit and immediately before delivery:

   ```bash
   node <renderer>/bin/archify.mjs validate <type> <candidate.json> --quality showcase --json
   ```

   A showcase pass reports all artifact checks with 0 composition errors
   and 0 warnings. If validation fails, change only the diagnosed `subject`,
   verify `evidence`, choose from `supportedFixes`, and rerun. Stop and report
   the unresolved diagnostics truthfully when two consecutive rounds do not
   lower the error count.
5. Deliver once, as the final acceptance command:

   ```bash
   node <renderer>/bin/archify.mjs deliver <type> <candidate.json> <output.html> --quality showcase --json
   ```

   A non-zero exit is never success. A failed delivery preserves any previous
   output at that path.

Do not read renderer, validator, or geometry source before the first
candidate exists. Inspect implementation only for an unsupported internal
diagnostic or after two focused repairs fail.

## Type router

| Type | Use for |
|---|---|
| `architecture` | Components, services, boundaries, infrastructure, repository maps |
| `workflow` | Processes, approval gates, tool calls, runbooks, CI/CD |
| `sequence` | API call chains, request lifecycles, async traces, returns |
| `dataflow` | Pipelines, ETL/ELT, lineage, consumers |
| `lifecycle` | State transitions, retries, waiting and terminal states |

Pasted Mermaid is read for topology and meaning, then re-authored as fresh
archify JSON: `flowchart` becomes `workflow` (or `architecture` for a
component map), `sequenceDiagram` becomes `sequence`, `stateDiagram` becomes
`lifecycle`.

## Authoring invariants

- One obvious main path; side branches leave the nearest main-path node.
  Remove low-value edges before adding routing controls.
- Omit `meta.visual_preset`, `meta.subtitle`, `meta.legend`, and
  `meta.engineering_profile` by default. Set a visual preset only when the
  user names that style.
- Component types are `frontend`, `backend`, `database`, `cloud`,
  `security`, `messagebus`, and `external`. Variants are `default`,
  `emphasis`, `security`, and `dashed`.
- Relationship labels are semantic data. When one collides, move the label
  or adjust spacing first, then shorten wording while preserving meaning.
  Deleting a label is not a geometry repair.
- Preserve exact product names, identifiers, commands, protocols, and paths.
  `meta.locale` is `"en"` or `"zh-CN"`; omit it for any other language and
  say that the viewer chrome falls back to English.
- Brand identity is explicit. Put a canonical built-in id in `brand` only
  when the node names that real product; query
  `node <renderer>/bin/archify.mjs brands "<name>" --json`. Never infer a brand
  from a role such as "database".
- Never accept an edge crossing an unrelated opaque node, an ambiguous shared
  corridor, or a label masking another route.

Read `<renderer>/references/authoring-contract.md` only when you need field enums,
spacing math, geometry repair rules, or mode-specific placement.

## Placement

Write `<name>.<type>.json` and `<name>.html` to `.clio-coder/artifacts/maps/`
unless the user names an output path. Create the directory first. An
operator-named path is a human deliverable and lands in the working tree
where they said. Archify creates `.archify-delivery-*` staging directories
beside the output during `deliver`, so the output directory must be a
writable directory inside the workspace, never a read-only or external
location.

## Repository evidence

When asked to map a repository, do not author from scratch. First run:

```bash
clio-coder context map --json
```

It reads the codewiki index and writes
`.clio-coder/artifacts/maps/<repo>.architecture.json`: an architecture spec
whose components are the repository's largest directory areas and whose
connections are collapsed import edges. When the origin remote is a GitHub
URL and `HEAD` is a full revision, the seed also carries `meta.repository`
and per-component `sources` pointing at real files and lines; otherwise it
carries neither, because archify accepts source citations only against a
pinned revision. If it reports that no index exists, run
`clio-coder context index` and retry. Then refine the seed: rename labels to
what the areas mean, drop noise edges, and keep `meta.repository` and every
`sources` entry that came from the seed. Cite additional line ranges only
from `code_nav mode=outline` results, never from memory, and only when
`meta.repository` is present. Keep at most 12 primary components; fold
smaller areas into their parent rather than adding nodes. When the seed
carries `meta.repository`, pass `--repo-root .` to both `validate` and
`deliver` so the source paths and line numbers are verified against the
working tree; without it, run both commands without `--repo-root`.

## Delivery and verification

`validate` during repair, `deliver` once for acceptance. Delivery freezes the
exact spec bytes into a same-directory snapshot, renders and checks it,
atomically commits the HTML, and reports SHA-256 plus byte counts for both
spec and artifact. That is deterministic artifact evidence; it does not
exercise the viewer in a browser.

After `deliver` exits 0, run the Clio-side check on the committed file:

```
verify(check="frontend", path="<output.html>")
```

That validates structure, local references, and syntax, and loads the page
in a headless browser when one is on PATH. Archify's own
`node <renderer>/bin/archify.mjs visual-check <output.html> --json` is optional:
it needs a system Chrome and fails with `viewer/chrome-unavailable` on
headless nodes. Report that outcome as what it is, never as a pass.

Keep the three claims separate: `deliver` proves deterministic artifact
checks, a browser load proves bounded behavior in a real browser, and
perceptual visual review requires a human or an image-capable reviewer.

## Output

Return the HTML path, the diagram type, the validation summary, the
spec and artifact receipt, the `verify` result, and the visual-review
status. Do not claim success for a non-zero command or claim inspection you
did not perform.
