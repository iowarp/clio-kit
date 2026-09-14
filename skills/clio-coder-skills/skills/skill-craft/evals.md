# Evals — skill-craft

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 — author a skill from a description of a workflow
Setup: any repo. Prompt: "write a project skill called log-triage that greps
service logs for errors, groups them by cause, and files a summary."
Expected:
- Writes `draft-skills/log-triage/SKILL.md` outside protected active roots.
- Supplies a portable package envelope and operator installation guidance when
  distribution is requested; authoring does not silently activate the skill.
- Description is triggers-only, front-loaded, no synonym-duplicated branches.
- Every step ends on a checkable completion criterion, not "until done".
- Frontmatter passes `clio-coder library validate draft-skills/log-triage/SKILL.md`.

## S2 — prune an overgrown skill
Setup: repo containing a 200-line skill whose body restates one rule three
times, carries two "always be careful" no-op lines, and inlines a long
reference table used by one branch. Prompt: "improve this skill."
Expected:
- Collapses the repeated rule to a single statement.
- Deletes the no-op sentences entirely rather than rewording them.
- Moves the branch-only table into `references/` behind a conditional pointer.

## S3 — diagnose a skill that never fires
Setup: installed skill whose description is a one-line identity statement
with no trigger words. Prompt: "why does this skill never activate?"
Expected:
- Attributes the failure to the description, not the body.
- Rewrites the description with user-language triggers, one per branch.

## Baseline failure modes to watch for (RED)
- Scaffolding ceremony: reaching for a create command or tool instead of
  writing the SKILL.md file.
- Descriptions that describe what the skill is instead of when to use it.
- "Done when it works" completion criteria.
- Rewording weak lines instead of deleting them.

## Instruction correction (2026-09-10)

Version 0.3.1 corrects draft authoring and operator installation paths and admits
the ordinary write/edit/bash tools the existing author/validate procedure needs.
Scenarios recorded; no new model eval is claimed.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. NOT COMPARABLE: baseline arm hit the loop guard, so no baseline exists; treatment was visibly scaffolding a skill when the run was scored INFRA.
