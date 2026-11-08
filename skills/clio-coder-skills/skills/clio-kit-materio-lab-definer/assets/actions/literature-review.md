# Action: literature-review

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
@.research/RESEARCH.md
@.research/config.json
</execution_context>

<objective>
Conduct a structured literature review based on the research identity in RESEARCH.md. Identifies current state of knowledge, open gaps, key authors/groups, and proposes refined search keywords. Creates LITERATURE.md.

**Orchestrator role:** Validate RESEARCH.md exists, gather the supplied corpus and scope guidance from the user, apply the materio-literature-reviewer skill, run the advisory citation check, collect the researcher's decisions on flagged entries and proposed keywords, handle loop-back to identify-research if a major problem with the prompt is found.

**Why subagent:** Literature synthesis requires sustained reading, cross-referencing, and gap reasoning across many papers. Fresh context = more rigorous gap analysis. The agent never asks the researcher anything; every decision comes back here.
</objective>

<context>
No arguments. Reads .research/RESEARCH.md and .research/config.json.
</context>

<process>

Worker write scope: .research/LITERATURE.md.


## 1. Gather Supplied Materials First

Before any other research read, use ls on `.research/data/` and read
`.research/DATA-INDEX.md`; inspect readable registered materials first. Record
unreadable formats as access limits and ask for usable text when needed. Then
validate the research identity and config below.

### Validate Environment

```bash
test -f .research/RESEARCH.md || echo "ERROR: No RESEARCH.md. Run identify-research using the materio-research-explorer skill first."
[ -f .research/LITERATURE.md ] && echo "WARN: LITERATURE.md exists. Running again will update it."
cat .research/RESEARCH.md
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" config
```

## 2. Gather the Supplied Corpus

```bash
ls -la .research/data/ 2>/dev/null
cat .research/DATA-INDEX.md 2>/dev/null
```

Supplied papers, BibTeX files, and notes are the primary corpus for the review; researcher-provided URLs may extend it. If nothing is registered, say so in the scope question below and offer `Run upload-data using the materio-research-explorer skill` first. Use only retrieval capabilities actually available. State that the review is built from supplied
materials plus any specific URLs the researcher provides. Ask for those URLs in
the scope question; use the host URL-fetch capability only when web_search is true and the tool is
available. With web_search false, use supplied materials only. Record this exact
coverage and access limits in LITERATURE.md under Sources Reviewed.

## 3. Confirm Scope with User

Use the host conversation facility:
- header: "Literature Review Scope"
- question: "I'll review literature based on your research prompt:\n\n**[research prompt from RESEARCH.md]**\n\nCorpus: [N] supplied files [list] · Provided-URL fetching: [allowed|disabled]\n\nAnything to adjust before I start?\n1. Any specific papers, authors, or groups to prioritize?\n2. Any journals or conferences to focus on?\n3. Year range? (default: all years, emphasis on last 10)\n4. Any sub-topics to explicitly include or exclude?\n5. Any specific URLs to add to the supplied corpus?"
- options: "Start with current scope" | "I'll add more context" | "Upload papers first; Run upload-data using the materio-research-explorer skill"

## 4. apply the materio-literature-reviewer skill

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► REVIEWING LITERATURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Apply the materio-literature-reviewer role skill using the assignment and full context; delegate if supported. The prompt carries:
- `<research>`; full RESEARCH.md content
- `<uploaded_data>`; every registered file path with its DATA-INDEX.md description
- `<user_guidance>`; scope adjustments from Step 3
- `<web_allowed>`; true only when config permits URL fetching and the host URL-fetch capability is available
- `<provided_urls>`; the specific URLs supplied by the researcher, or none
- `<literature_path>`; `.research/LITERATURE.md`

## 5. Handle Agent Return

**`literature_review_complete:`:**

1. Use read on `.research/LITERATURE.md` and verify the file is on disk:
   ```bash
   test -s .research/LITERATURE.md && echo OK || echo "ERROR: LITERATURE.md missing"
   ```
   If missing, apply the role again and say so; do not continue.

2. Advisory citation check. Verify the Key Papers table against Crossref to catch fabricated papers or dead DOIs. This is advisory; it never deletes anything on its own:
   ```bash
   if command -v python3 >/dev/null 2>&1; then
     citation_flags=()
     [ "$(python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" config --key web_search)" = true ] || citation_flags=(--offline)
     if python3 "${PACKAGE_ROOT}/assets/scripts/verify_citations.py" "${citation_flags[@]}" .research/LITERATURE.md; then echo "CHECK EXIT: 0"; else echo "CHECK EXIT: $? (findings or incomplete coverage)"; fi
   fi
   ```
   - If python3 is absent, record citation verification as skipped. If every entry is UNVERIFIABLE with network errors, record the access limitation in Review Notes and continue. With web_search false, run only the offline structural check.
   - If any entry is **NOT_FOUND** or **MISMATCH**: show each flagged entry next to the closest Crossref match and ask via the host conversation facility, one batched question:
     - header: "Citation Check"
     - question: "Crossref could not confirm [N] entries:\n\n1. [entry] → closest match: [title, year, DOI] (score [x])\n2. ...\n\nFor each: keep as-is, mark unverified, correct to the match, or remove?"
     - options: "Keep all, mark unverified" | "Correct to matches where shown" | "I'll decide per entry" | "Remove all flagged"
     Apply the decision to LITERATURE.md. A flagged entry is never removed without the researcher saying so.

3. Keyword gate. The agent proposed changes in the Refined Keywords section and in its return. Ask via the host conversation facility:
   - header: "Keyword Refinement"
   - question: "Based on the literature, I suggest refining keywords:\n\nAdd: [terms]\nRemove: [terms]\nReplace: [old] → [new]\n\nAny adjustments?"
   - options: "Accept refined keywords" | "Keep original keywords" | "I'll adjust"
   Apply the accepted set to the Keywords section of RESEARCH.md.

4. Record (optional git):
   ```bash
   python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" record --message "research: literature review; [N] sources, [N] gaps identified" --files "${CHANGED_FILES[@]}"
   ```

**`loop_back:`:**
- Read LITERATURE.md back and run the same advisory citation check and researcher decision gate above before applying the loop-back decision.
- The reviewer found the prompt is too broad, too narrow, already answered, or rests on a false premise. LITERATURE.md was still written.
- Present the finding and its three suggested prompts.
- Ask via the host conversation facility:
  - header: "Prompt Needs Revision"
  - question: "The literature shows: [finding].\n\nSuggested revisions:\n1. [narrowed]\n2. [adjacent gap]\n3. [replication/extension]\n\nHow do you want to proceed?"
  - options: "Narrow to suggestion 1" | "Narrow to suggestion 2" | "Narrow to suggestion 3" | "Proceed with the current prompt anyway"
- If a suggestion is chosen: update the Selected Prompt and Key Decisions in RESEARCH.md and STATE.md, note the revision reason, then run the keyword gate above. If the change is larger than a rewording, recommend re-running `Run identify-research using the materio-research-explorer skill` with the new framing.

**`needs_input: checkpoint:decision`, `needs_input: checkpoint:human-action`, or `needs_input: checkpoint:human-verify`:**
- Present the question to the user, collect the answer via the host conversation facility, apply the role again.

</process>

<offer_next>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► LITERATURE REVIEW COMPLETE ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Review:   .research/LITERATURE.md
Sources:  [N] supplied, [N] researcher-provided URLs fetched
Gaps:     [N] identified
Keywords: [primary keywords]
Citations: [N verified, N marked unverified]

───────────────────────────────────────────

## ▶ Next Up

**Map your available lab resources**
(equipment, HPC, software, collaborations)

`Run define-virtual-lab using the materio-lab-definer skill`

<sub>Each dispatch starts with fresh worker context.</sub>

───────────────────────────────────────────

**Also available:**
- `Run identify-research using the materio-research-explorer skill`; refine prompt based on gaps found
- `Run upload-data using the materio-research-explorer skill`; add more papers/datasets to review
- `Run define-research-tasks using the materio-workflow-planner skill`; skip virtual lab setup and go straight to workflow

</offer_next>

<success_criteria>
- [ ] Every question to the researcher was asked by this command, not by the agent
- [ ] Supplied corpus listed and passed to the agent; web availability passed explicitly
- [ ] Current state of knowledge summarized by sub-topic
- [ ] Evidence-supported gaps identified with limitations; zero gaps is a valid result
- [ ] Key papers listed with quoted titles and provenance
- [ ] Citation check run; flagged entries decided by the researcher, never auto-removed
- [ ] Keywords refined only after the researcher accepted
- [ ] Loop-back handled if the prompt needs revision
- [ ] LITERATURE.md verified on disk; commit only if commit_research is true
</success_criteria>
