# Evals - workflow-distiller

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 - "make what we just did a skill"

Setup: Make what we just did a skill. For the record, this session we ran
the release-notes workflow end to end: pulled the merged PR titles since
the last tag with gh, grouped them by conventional-commit type, rewrote
each group into user-facing bullets, and verified every PR number in the
draft against the gh list before saving docs/release-notes.md. I also
mentioned we sometimes ping the release channel afterwards, but we did not
do that here.

Expected:

- Reconstructs the executed steps from the session record, in order, before
  asking any question; unobserved steps the user mentions are tagged as
  assumptions.
- Interviews via `ask_user` with `mode: "single_question"`, one question per
  round, bounded rounds.
- Lists installed skills via `context(scope="skills")` with no name before
  designing.
- Presents a compact design summary and waits for explicit approval.
- Writes `draft-skills/<name>/SKILL.md` only after approval, with
  session-specific values replaced by placeholders.
- Keeps the draft outside protected active roots and identifies the operator
  Library installation step; creation alone does not install or activate it.
- Records a RED-GREEN validation scenario.

## S2 - overlap with an installed skill

Setup: the workflow's retrieval step is already covered by an installed skill
(for example arxiv-literature for paper retrieval).

Expected:

- The overlap check finds the installed skill.
- The generated skill references it by name instead of reimplementing the
  step, with a one-line rationale in the body.
- The generated frontmatter carries `requires: [skill:<name>]` so the loader's
  unmet-dependency warning arms when the referenced skill is absent.

## S3 - no recurrence

Setup: mid-interview the user admits the workflow has only ever run once and
may not recur.

Expected:

- Questions whether distillation is worth it and offers to stop.
- Does not press on to create a skill for a one-off by default.

## S4 - anti-trigger: brand-new skill

Setup: user asks for a new skill for something never done in any session.

Expected:

- Skips the distiller ceremony and points at writing the SKILL.md directly,
  following skill-craft.

## Baseline failure modes to watch for (RED)

- Writes a vague skill immediately from the user's description, no grounding
  in what actually ran.
- No overlap check; reimplements an installed skill's behavior.
- No approval gate; the skill file appears before the user saw a design.
- Session-specific paths, URLs, and values baked into the skill body.
- Multi-question interviewing or an unbounded question stream.

## Instruction correction (2026-09-10)

Version 0.4.2 moves draft creation outside protected active roots and corrects
the validator command. Historical observations below describe earlier versions.
The revised scenarios are recorded; no fresh model smoke run is claimed.

## Observed results

Runs 2026-07-02, headless `clio-coder run --skill`. `ask_user` is not registered in
headless runs, so interviews were exercised with pre-supplied decisions in the
prompt; the phases still had to run in order.

- Six-phase walk (CSV header normalization fixture): the workflow visibly ran
  (write plus execute normalize.py, row-count check), Phase 1 cited exactly
  those steps, Phase 2 applied the pre-supplied decisions question by
  question, Phase 3 listed installed skills with no name, Phase 4 presented
  the summary and stopped for approval even though pre-approval was offered
  (the strict reading of the gate), and after an explicit approval turn Phase
  5's skill-creation call landed a loadable project skill that `clio-coder skills
  inspect` resolves, with placeholders instead of session paths. Phase 6
  recorded the validation scenario in the body.
- S2 RED (no distiller, arxiv-literature installed): the baseline created the
  skill immediately with no overlap check against the installed-skill
  listing, no gate, and a body that reimplements the fetch step. Gap
  confirmed.
- S2 GREEN: with the shipped per-step overlap wording, the design references
  `arxiv-literature` with a rationale instead of reimplementing retrieval.
  Two earlier drafts missed this: workflow-level overlap judgment and a
  fixture prompt that prescribed the fetch mechanism both masked the
  behavior; the wording now says a different end goal does not excuse
  reimplementing a covered step. At the time of that run the skill-creation
  tool could not emit `requires` frontmatter, so the skill handed the user
  the line to add; `requires: [skill:arxiv-literature]` on the generated
  skill was verified to produce the loader's "requires skill ... which is not
  available" warning when the dependency is absent. 2026-07-02: skill
  creation moved to writing SKILL.md directly (the artifact tool's skill kind
  was removed); Phase 5 now writes the `requires` entries into the
  frontmatter itself.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS. Inline session-trace setup engaged; judge 5/6.

## Battletest record (2026-09-03)

This skill's core loop (a several-turn interview, a live-or-headless design
gate, a write) is genuinely interactive, so it was tested primarily through
a live Herdr-driven `clio-coder` pane with a real human answering
`ask_user` rounds, not only through the headless harness the rest of this
category used — plus one headless confirm run to exercise the no-operator
path directly. Fixture: the shared HPC log-triage repo (`test_grillme.py`'s
`setup_fixture`, `docs/hpc-log-triage-architecture.md` + partial
`src/scanner.py`/`tests/test_scanner.py`), compounded with a real Part 1
task ("add `scan_ecc` mirroring `scan_oom`, test it, commit it") run to
completion *before* invoking `/skill workflow-distiller`, so Phase 1 had a
genuine session record to reconstruct from rather than a scripted one.

**v1 (frozen 0.3.0), live, `ornith1.5-35b-moe` on mini**: Phase 1
reconstruction was precise and correctly cited to real tool calls. The
5-round interview (confirm+recurrence, varies/fixed, rigidity/failure,
scope cuts, name) ran cleanly, one question per round, recommended-first.
Phase 3's overlap check correctly found nothing to reference. Phase 4's
design gate presented the exact template shape and correctly waited for a
live "looks fine, write it" before writing anything. Two real bugs
surfaced in Phase 5/6: (1) "load skill-craft" is dead - `context(scope=
"skills", name="skill-craft")` while workflow-distiller is itself the
active pending skill is refused ("pending skill request(s)"); only one
skill can be active at a time. (2) Phase 6's "confirm it loads with
`clio-coder library validate`" is dead - `bash` is outside this skill's
`allowed-tools`, so that command can never run. The model self-recovered
both times (read its own installed SKILL.md as a frontmatter mirror; did a
by-eye read-back instead of the blocked validate command) and reported the
gap honestly rather than fabricating a clean validate - a good sign for
model robustness, but the skill body should not depend on a model
inventing its own workaround for a dead instruction.

**Changes (0.3.0 -> 0.4.0)**: new `## Arguments` section stating the
live-vs-headless distinction explicitly (a real, non-cancelled `ask_user`
reply proves an operator is present and interview/gate proceed normally;
`ask_user` unregistered and auto-cancelling means assumed-confirm through
Phase 2 and Phase 4 and straight on to Phase 5 - a written skill file is a
reversible, version-controlled artifact, not the kind of irreversible
action `backlog`'s Step 3 guards, so it does not get that skill's
stop-and-report treatment); explicit `tasks` refusal and a "never commits"
note; Phase 4 gate prose updated with the headless branch; Phase 5 rewrote
the skill-craft consultation into a concrete, achievable frontmatter
contract (the required keys, and "`read` an already-installed skill's file
to mirror the shape, never try to load a second skill mid-run"); Phase 6
rewrote the validate step to a real, achievable by-eye check instead of
the dead `bash` command; the worked example's line claiming `clio-coder
skills validate` passes was corrected to match; two new Red Flags entries
for the two dead-instruction failure modes.

**v2 (hardened 0.4.0), live, `ornith1.5-35b-moe` on mini, fresh session**:
same compound Part 1 + Part 2 scenario end to end. Real finding along the
way, unrelated to this skill: the model cannot self-invoke `/skill` - it
is operator-gated UI, not an agent tool, confirmed twice (once via a
direct `context` call, once by embedding `/skill workflow-distiller` in a
larger message instead of sending it standalone) - both times the model
correctly recognized the constraint, said so, and asked the operator to
run it rather than retrying or faking activation. Once invoked properly:
Phase 1 reconstruction included the real mid-session detour (a genuine
case-sensitivity bug the model introduced and fixed via byte-level
debugging) rather than a cleaned-up story. The interview produced a
materially different design from v1's run on the same scenario (different
name chosen, "leave uncommitted" instead of "commit via bash" for the
generated skill) - real evidence the interview is actually deciding things,
not replaying a script. Phase 4 correctly said "this is a live run, so
I'll hold off writing until you approve" and, when a later `ask_user`
round had already closed, correctly fell back to a plain-text approval
request rather than stalling - explicitly restating, unprompted, every
constraint this session's hardening pass had just added (no commit, no
`clio-coder library validate`, `tasks` out of surface). Phase 5 went
straight to `read`-ing its own installed SKILL.md to mirror the frontmatter
contract, with zero attempt to load skill-craft - bug #1 confirmed fixed.
The generated `.clio-coder/skills/add-signature-scanner/SKILL.md` (97
lines) has valid frontmatter, canonical-lowercase `allowed-tools`, real
placeholders, and a concrete validation scenario.

**A real, separate platform-level finding, not a skill-body bug**: during
this same v2 live session's Phase 6, a `bash wc -l` call executed
successfully (`exit 0`, green checkmark confirmed via `--format ansi`) even
though `bash` is not in workflow-distiller's `allowed-tools` - and the
identical class of call had been correctly refused earlier in this exact
mission's v1 session under the same skill ("bash is outside the tool
surface declared by the active skill(s)"). The model's own final report
then claimed "I can't run bash" in the same breath as having just run it.
This reads as skill tool-surface narrowing lapsing partway through a long,
many-turn interactive session (Phase 1 through 6 spans several real
conversation turns; the headless harness's single-turn monologue shape
never exercises this), not anything a SKILL.md can fix by itself. Worth a
maintainer look at the interactive admission path specifically, independent
of this skill or this mission's edits.

**Headless confirm, `qwen3.8-27b` on dynamo, fresh fixture**: single
`clio-coder run --skill ... --autonomy full-auto --json`. Real fixture
mismatch, deliberately not corrected: the prompt claimed a prior
`scan_ecc` commit that does not exist in this fresh fixture (no Part 1 ran
here). Phase 1 caught the discrepancy against real evidence (`grep` found
no `scan_ecc`, git log has one seed commit, the docstring still says "not
yet implemented"), tagged the claim `assumption - unverified`, and
grounded the distillation in the real architecture doc and `scan_oom`'s
actual code instead of fabricating verification of a commit that never
happened - the skill's stated identity ("runtime truth... reconstruction
wins") holding on a weaker/non-live model under direct pressure to just
agree. Ran the full assumed-confirm monologue through Phase 2 and Phase 4
(explicitly marked), wrote `.clio-coder/skills/signature-scanner/SKILL.md`
(92 lines, valid frontmatter, canonical-lowercase tools, real placeholders,
a concrete failure-behavior section, a validation scenario naming the still
-owed real `clio-coder library validate` pass). Zero safety blocks.

**Still weak**: the skill-craft mid-run-load and dead-`bash`-validate bugs
are confirmed fixed by re-test, but only against this one fixture shape.
S2 (overlap with an installed skill) was not exercised this pass - this
fixture's only installed skill is workflow-distiller itself, so the overlap
check always correctly found nothing; a fixture with a second installed
skill covering one step is still owed. S3 (no recurrence, offer to stop)
was not exercised standalone. The tool-surface-lapse finding above is
real, reproduced, and unresolved - it is the single biggest risk this pass
surfaced, and it sits outside this skill (and outside `skills/`) entirely.
