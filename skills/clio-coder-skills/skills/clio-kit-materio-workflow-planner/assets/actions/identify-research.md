# Action: identify-research

This guide is linked from an installed portable skill. Action requests use the
actual installed skill names; this standard package registers no slash commands.
The interactive assistant applies a role inline or assigns its skill to a generic
worker if the host supports delegation. No named worker registration is presumed.
If a native peer package supplies command or agent bindings, follow its binding
section below while preserving the same researcher gates and return checkpoints.

This guide is linked from an installed portable skill. Determine the package root
from that skill's directory. Before executing a shell example, bind PACKAGE_ROOT
to that actual root in the same command. Treat $ARGUMENTS as validated user input;
these are examples to fill with actual paths, not automatic host substitutions.



Use the host's file reading/editing and local command execution capabilities. Read
assets/references/research-policy.md, resolved from this package. If this task
needs researcher answers, the interactive assistant asks them through the host's
conversation facility before gated writes. Workers return questions with their
checkpoint kind. Use the named role's portable skill for the scientific work;
if the host does not support delegation, the interactive assistant performs that
role with the same input/output and decision boundaries. Do not invent worker,
search, fleet, or interactive capabilities. Inspect actual files before completion.
Use package scripts by resolving their links from the installed skill directory.
For an optional commit, set CHANGED_FILES to only the named files actually changed
and read back during this action. The research-state helper rejects directory staging.


Use the assigned role skill with complete research context and exact permitted outputs. Return structured checkpoints and inspect actual files after writing.

<execution_context>
@${PACKAGE_ROOT}/assets/templates/RESEARCH.md
@${PACKAGE_ROOT}/assets/templates/config.json
@${PACKAGE_ROOT}/assets/references/research-domains.md
</execution_context>

<objective>
Initialize a new material science research project through a deep guided interview. Creates `.research/` and populates RESEARCH.md with domain, sub-field, scope, researcher profile, and a confirmed research prompt.

**Orchestrator role:** Check preconditions, detect existing research context, run the whole researcher interview (foundation, profile, and Socratic drilling), apply the materio-research-explorer skill for synthesis, present its candidate prompts, collect the choice, apply the role again to write the files, verify, record.

**Why the interview lives here:** These worker roles return questions to the orchestrator (worker roles cannot interview the researcher). The orchestrator owns every question. The agent gets a fresh context window for the reasoning that benefits from it: turning the transcript into sharp, calibrated candidate prompts.
</objective>

<context>
No arguments. Runs in current directory.
</context>

<process>

Worker write scope: .research/RESEARCH.md and .research/STATE.md, only after selection.


## 1. Validate Environment

```bash
[ -f .research/RESEARCH.md ] && echo "WARN: Research already initialized. Use Run progress using the materio-research-explorer skill to see current state, or continue to refine."
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" config
# Read-only before researcher selection; initialize at the approved write step.
```

Materio never runs `git init`. Version control of `.research/` is the researcher's choice, controlled by `commit_research` in `.research/config.json` (default false).

## 2. Detect Existing Context

```bash
find . -name "*.pdf" -o -name "*.csv" -o -name "*.bib" -o -name "*.txt" 2>/dev/null | grep -v ".research" | grep -v ".claude" | grep -v ".git" | head -10
```

If data files found: ask via the host conversation facility whether to base the research framing on existing files, and suggest `Run upload-data using the materio-research-explorer skill` afterwards so the literature review can read them.

## 3. Gather Research Foundation (Batched)

Use the host conversation facility; collect core framing in one turn:
- header: "Research Exploration; Material Science"
- question: "Let's explore your research direction.\n\n1. **Domain**: Which area of material science? (e.g., Structural, Energy, Biomaterials, Computational, Nanomaterials, Functional, Polymers, Characterization)\n2. **Driving question**: What phenomenon, material, or problem interests you? (free-form, even vague is fine)\n3. **Motivation**: Fundamental understanding, application/device, optimization, or review/synthesis?\n4. **Resources available**: Experimental lab, computational cluster, literature only, or combination?"
- options: "Provided details" | "I'm not sure yet; help me explore" | "I have a specific topic already"

**If "I'm not sure yet":** Ask about domain first, then walk through sub-fields in that domain using research-domains.md as reference, one the host conversation facility per level, before continuing.

**If "I have a specific topic already":** Capture it verbatim, then go to Step 4 and run only Round 3 of Step 5.

## 4. Gather Researcher Profile

Use the host conversation facility:
- header: "Researcher Context"
- question: "Help me understand your context:\n1. **Career stage**: Undergrad / Grad student / Postdoc / PI / Industry researcher\n2. **Timeline**: How long for this research? (months)\n3. **Prior work**: Have you already read papers in this area? Any specific authors or groups you follow?\n4. **Constraints**: Any materials, methods, or approaches you CANNOT use?"
- options: "Provided details" | "Skip; use reasonable defaults"

## 5. Socratic Drilling (as needed)

Narrow in up to three rounds. Skip any round whose answers the transcript already contains. Use the domain's sub-fields, key phenomena, and common methods from research-domains.md to make the examples concrete for this researcher.

**Round 1; Narrow the sub-field:**
- header: "Narrowing Your Research Focus"
- question: "You mentioned [topic]. Let's narrow it down:\n\n1. What specific aspect? (e.g., for batteries: anode material? electrolyte? capacity fade mechanism? cycling stability?)\n2. What material system specifically? (e.g., Li-S, Na-ion, solid-state with sulfide electrolyte)\n3. What scale? (atomic/nano/micro/device level)"
- options: "Provided details" | "Help me choose; show sub-fields"

**Round 2; Identify the gap:**
- header: "Finding the Angle"
- question: "For [narrowed topic]:\n1. What do you think is currently UNKNOWN or UNSOLVED?\n2. What would change if you answered your question? (better battery? new understanding? design rule?)\n3. What's your hypothesis or intuition about the answer?"
- options: "Provided details" | "I don't know yet; the literature review should tell me"

**Round 3; Scope confirmation:**
- header: "Scoping the Research"
- question: "Final scoping:\n1. What methods will you use? (synthesis, characterization, simulation; be specific)\n2. What will you NOT study? (even if related and interesting)\n3. What's the minimum result that would make this a successful research project?"
- options: "Provided details" | "Let the candidates propose scope"

Assemble every answer from Steps 2 through 5, verbatim, into the `<researcher_input>` transcript.

## 6. Apply the materio-research-explorer skill

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► EXPLORING RESEARCH SPACE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Apply the materio-research-explorer role skill using the assignment and full context; delegate if supported. The prompt carries the full inlined context:
- `<researcher_input>`; the complete interview transcript from Steps 2–5
- `<domains_ref>`; the relevant section of research-domains.md
- `<research_path>`; `.research/RESEARCH.md`

Before selection, request candidates only and authorize no writes; omit
any write authorization and state explicitly that this is the no-write candidate phase.
On selection, authorize exactly `.research/RESEARCH.md` and `.research/STATE.md`.
Include all three original candidates, the verbatim selection, scope corrections,
and the complete RESEARCH template in the writing dispatch.

## 7. Present Candidates and Collect the Choice

**`candidates_ready:`:**

Use the host conversation facility, one turn:
- header: "Your Research Prompts"
- question: "Based on our conversation, here are 3 research prompts:\n\n**A (Focused):** [prompt A]\n  In: [scope] · Out: [scope] · Keywords: [list]\n\n**B (Standard):** [prompt B]\n  In / Out / Keywords\n\n**C (Ambitious):** [prompt C]\n  In / Out / Keywords\n\nRecommended: [X] because [reason].\n\nWhich fits your goals? If you pick one, also say whether its scope and keywords need any correction."
- options: "Prompt A" | "Prompt B" | "Prompt C" | "Combine or edit; I'll describe"

Apply the research-explorer skill again with a `<selection>` block containing the choice and any corrections. Apply the materio-research-explorer skill with the full `<researcher_input>` transcript plus `<selection>`.

**`needs_input: checkpoint:decision`, `needs_input: checkpoint:human-action`, or `needs_input: checkpoint:human-verify`:** Ask the batched questions it lists via the host conversation facility, append the answers to the transcript, and apply the role again.

**`exploration_inconclusive:`:** Show what's missing and what it suggested. Offer to re-run with more guidance.

After the researcher selects a candidate and confirms scope, run
`python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" init` before the writing dispatch.

## 8. Verify and Record

**`research_identified:`:**

```bash
test -s .research/RESEARCH.md && test -s .research/STATE.md && echo "OK: files on disk" || echo "ERROR: agent reported files it did not write"
grep -n "Selected Prompt" -A 2 .research/RESEARCH.md
```

If either file is missing, do not report success; apply the role again and tell it exactly which file is absent.

Optional git record, only if the researcher enabled it:
```bash
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" record --message "research: initialize; [research prompt one-liner]" --files "${CHANGED_FILES[@]}"
```

</process>

<offer_next>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► RESEARCH IDENTIFIED ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prompt:  [selected research prompt]
Domain:  [field + sub-field]
Profile: .research/RESEARCH.md

───────────────────────────────────────────

## ▶ Next Up

**Register the papers you already have, then review the literature**

`Run upload-data using the materio-research-explorer skill` (optional, but the review is only as good as its corpus)
`Run literature-review using the materio-literature-reviewer skill`

<sub>Each dispatch starts with fresh worker context.</sub>

───────────────────────────────────────────

**Also available:**
- `Run define-research-tasks using the materio-workflow-planner skill`; skip literature review if already done

</offer_next>

<success_criteria>
- [ ] Every question to the researcher was asked by this command, not by the agent
- [ ] Domain and sub-field clearly identified
- [ ] Research prompt is a single answerable question
- [ ] Scope (in/out) explicitly defined
- [ ] Researcher profile captured (career stage, resources, timeline)
- [ ] Three candidate prompts presented before final selection
- [ ] RESEARCH.md and STATE.md verified on disk with test -s
- [ ] config.json created from template; no git init; commit only if commit_research is true
</success_criteria>
