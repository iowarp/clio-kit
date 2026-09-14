---
name: materio-research-explorer
description: 'Use when this workflow is requested: Synthesize candidate materials-science
  questions and write researcher-confirmed identity. Triggers on "materio research
  explorer".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/plugins/materio/skills/materio-research-explorer
  upstream-eval-status: unspecified
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

For a new materials research direction, interview the domain, phenomenon, processing conditions, hypothesis, scope exclusions, resources, and timeline. Present three calibrated candidate questions, then write the selected identity and decisions under .research/ using the linked template. Preserve existing research state. Use the research state helper to initialize missing config.

Read the [shared research policy](assets/references/research-policy.md) for
state, provenance, execution boundaries, and advisory validation.

Read the linked references when needed. Resolve links from this skill directory.
If acting as a worker, return questions to the orchestrator; when acting as the
interactive assistant, collect the researcher answers directly using the host's
conversation facility.

- [research-domains.md](assets/references/research-domains.md)
- [RESEARCH.md](assets/templates/RESEARCH.md)
- [config.json](assets/templates/config.json)

## What Makes a Good Research Prompt

A good research prompt is:
- **Specific**: Names specific material, property, condition, or phenomenon
- **Answerable**: Can be confirmed or refuted by an achievable experiment or calculation
- **Novel**: Not already fully answered in the literature (this comes from literature-review, but initial framing matters)
- **Scoped**: Has clear in/out boundaries so the researcher knows when they're done

Bad: "Study high-entropy alloys"
Better: "Characterize mechanical properties of HEAs"
Good: "How does Cr content affect yield strength and ductility in CoCrFeMnNi alloys processed by arc melting?"

## Socratic Narrowing Is the Orchestrator's Job

The interview narrows in five moves: domain, then property or phenomenon, then conditions and processing route, then hypothesis, then explicit out-of-scope. The orchestrator runs those rounds. You read the transcript and judge whether the narrowing went far enough. If it did not, say exactly which move is missing and what to ask.

Do not fill gaps with assumptions. Vague prompts lead to unfocused research.



Generate three candidate research prompts at different levels of ambition. For each, propose a scope and keywords so the researcher can decide in one turn:

**Option A (Focused):** Narrowest, most achievable for the stated timeline and resources
**Option B (Standard):** Moderate scope, typical for a journal article
**Option C (Ambitious):** Broader, suitable if timeline and resources allow

For every option include:
- The prompt as one specific, answerable question
- In scope: 3 to 5 concrete items (materials, phenomena, conditions, methods)
- Out of scope: 2 to 4 explicit exclusions
- Keywords: 5 to 8 terms, marked primary or secondary
- Feasibility note: one line on why it fits (or strains) the stated resources and timeline

Return the three candidates for researcher selection; write identity files only after the selection is settled.



## Complete action guides

Read only the guide for the requested operation; it contains its interview and
state transitions. Resolve these links from this skill directory.

- [identify-research](assets/actions/identify-research.md)
- [wtfp](assets/actions/wtfp.md)
- [help](assets/actions/help.md)
- [status](assets/actions/status.md)
- [progress](assets/actions/progress.md)
- [settings](assets/actions/settings.md)
- [checkpoint](assets/actions/checkpoint.md)
- [pause-research](assets/actions/pause-research.md)
- [resume-research](assets/actions/resume-research.md)
- [upload-data](assets/actions/upload-data.md)
- [add-task](assets/actions/add-task.md)
- [remove-task](assets/actions/remove-task.md)
- [archive-task](assets/actions/archive-task.md)
