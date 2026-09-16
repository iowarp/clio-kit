# Action: define-research-tasks

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
@.research/LITERATURE.md
@.research/VIRTUAL-LAB.md
@${PACKAGE_ROOT}/assets/references/traditional-workflows.md
@${PACKAGE_ROOT}/assets/templates/WORKFLOW.md
</execution_context>

<objective>
Create the research workflow for the project. Either select a traditional workflow template and customize it, or build from scratch. Each task gets an assumption interview. Outputs WORKFLOW.md.

**Orchestrator role:** Load research context, present workflow options, gather customizations and scope, run the per-task assumption interview, apply the materio-workflow-planner skill to generate WORKFLOW.md, present resource flags and planner defaults for decision, apply the role again with revisions, create task directories, record.

**Why the interview lives here:** These worker roles return questions to the orchestrator. Assumptions are research decisions and only the researcher can make them, so this command collects them before the planner runs. The planner gets a fresh context for the reasoning that benefits from it: feasibility against VIRTUAL-LAB.md, dependency ordering, and writing a WORKFLOW.md the executor can run without interpretation.
</objective>

<context>
No arguments. Requires RESEARCH.md to exist.
</context>

<process>

Worker write scope: .research/WORKFLOW.md.


## 1. Validate Environment

```bash
test -f .research/RESEARCH.md || echo "ERROR: No RESEARCH.md. Run identify-research using the materio-research-explorer skill first."
[ -f .research/WORKFLOW.md ] && echo "WARN: WORKFLOW.md exists; running again will replace it."
cat .research/RESEARCH.md
[ -f .research/LITERATURE.md ] && cat .research/LITERATURE.md | head -60
```

Check for VIRTUAL-LAB.md and warn if missing:
```bash
if [ ! -f .research/VIRTUAL-LAB.md ]; then
  echo "NOTE: No VIRTUAL-LAB.md found. Run define-virtual-lab using the materio-lab-definer skill to map your resources first."
  echo "Proceeding without resource constraints; tasks may be planned that require unavailable equipment."
fi
[ -f .research/VIRTUAL-LAB.md ] && cat .research/VIRTUAL-LAB.md
```

## 2. Present Workflow Template Options

Based on the research type from RESEARCH.md (experimental / computational / mixed / literature-only), present the relevant traditional workflow templates from traditional-workflows.md.

Use the host conversation facility:
- header: "Research Workflow Design"
- question: "Based on your research (**[research prompt]**), I suggest one of these workflow templates:\n\n**A. [Template name]**; [1-line description, N tasks]\n**B. [Template name]**; [1-line description, N tasks]\n**C. Build from scratch**; I'll define each step\n\nWhich fits best?"
- options: "Template A" | "Template B" | "Template C; Build from scratch" | "Show me all templates"

## 3. Gather Customizations

After template selection, ask about modifications:
Use the host conversation facility:
- header: "Workflow Customization"
- question: "Template **[selected]** has these tasks:\n\n[numbered list of tasks from template]\n\nWhat changes?\n1. **Add tasks**: Any steps missing for your specific research?\n2. **Remove tasks**: Any steps you'll skip (and why)?\n3. **Reorder**: Any dependencies that differ from the template?"
- options: "Use as-is" | "I have changes" | "Add one task" | "Remove one task"

## 4. Confirm Scope and Exclusions

Use the host conversation facility:
- header: "Scope & Exclusions"
- question: "Before planning assumptions, confirm:\n1. **Hard constraints**: Which tasks are definitely IN scope?\n2. **Out of scope**: Which standard steps will you skip?\n3. **First milestone**: Which task marks the first major checkpoint?\n4. **Data availability**: Do you have any existing data that eliminates early tasks?"
- options: "Confirmed; proceed to assumptions" | "Let me adjust"

## 5. Assumption Interview, Per Task

You now hold the final task list. For each task, ask the question block for its type. Batch tasks of the same type into one the host conversation facility when there are three or fewer; otherwise one question per task. Use the "Key assumptions to interview about" hints in traditional-workflows.md for the selected template to sharpen the prompts. Every question offers "Use planner defaults for this task" so the researcher can skip and confirm later.

**For experimental tasks:**
- header: "Assumptions: Task [NN]; [name]"
- question: "For '[task description]', I need to understand your assumptions:\n1. **Sample preparation**: What processing route? What contamination risks?\n2. **Characterization**: Which tools are available? What resolution/sensitivity?\n3. **Test conditions**: Temperature, strain rate, environment, standards (ASTM/ISO)?\n4. **Sample size/replicates**: How many samples? What's statistically sufficient?\n5. **Success criteria**: What result would confirm or disprove your hypothesis?"
- options: "Provided details" | "Use planner defaults for this task"

**For computational tasks:**
- header: "Assumptions: Task [NN]; [name]"
- question: "For '[task description]':\n1. **Method**: DFT, MD, phase-field, CALPHAD; which and why?\n2. **Software**: VASP, LAMMPS, Thermo-Calc, etc.?\n3. **Functional/force field**: Exchange-correlation functional or interatomic potential?\n4. **System size and timescale**: Feasible with available resources?\n5. **Validation benchmark**: What experimental data exists to validate against?"
- options: "Provided details" | "Use planner defaults for this task"

**For data-analysis tasks:**
- header: "Assumptions: Task [NN]; [name]"
- question: "For '[task description]':\n1. **Data source**: Which uploaded files? Which task outputs?\n2. **Statistical approach**: What statistical tests are appropriate?\n3. **Outlier handling**: Expected outliers? How to treat them?\n4. **Visualization**: What plots are needed for the paper?\n5. **Software**: Python/MATLAB/R? Which libraries?"
- options: "Provided details" | "Use planner defaults for this task"

**For literature tasks:**
- header: "Assumptions: Task [NN]; [name]"
- question: "For '[task description]':\n1. **Scope**: Which sub-topics to cover?\n2. **Inclusion criteria**: Year range, journal type, minimum citation count?\n3. **Key authors/groups**: Anyone to prioritize?\n4. **Output format**: Structured table? Narrative summary? BibTeX file?"
- options: "Provided details" | "Use planner defaults for this task"

**For analytical tasks:**
- header: "Assumptions: Task [NN]; [name]"
- question: "For '[task description]':\n1. **Governing model**: Which physical model or theory?\n2. **Boundary conditions and simplifications**: What is held fixed or neglected?\n3. **Validation**: Which limiting cases or data will the model be checked against?\n4. **Implementation**: Analytical only, or numerical (Python/MATLAB)?"
- options: "Provided details" | "Use planner defaults for this task"

Writing tasks need no interview; they bridge to `Run wtfp using the materio-research-explorer skill`.

Collect every answer verbatim into `<assumptions>`, keyed by task name, with "planner defaults requested" where the researcher skipped.

## 6. apply the materio-workflow-planner skill

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► PLANNING RESEARCH WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Apply the materio-workflow-planner role skill using the assignment and full context; delegate if supported. The prompt carries:
- `<research>`; full RESEARCH.md
- `<literature>`; LITERATURE.md gaps and methods landscape
- `<virtual_lab>`; full VIRTUAL-LAB.md (or "not defined" if missing)
- `<selected_template>`; the chosen template tasks
- `<customizations>`; additions, removals, reorderings
- `<scope_decisions>`; inclusions, exclusions, first milestone, existing data
- `<assumptions>`; the per-task answers from Step 5
- `<workflow_path>`; `.research/WORKFLOW.md`

## 7. Decide on Flags and Defaults

**`workflow_complete:`:**

```bash
test -s .research/WORKFLOW.md && echo OK || echo "ERROR: WORKFLOW.md missing"
grep -c "^### Task" .research/WORKFLOW.md
```
If missing, apply the role again and say so.

Present the planner's return and ask via the host conversation facility, one turn:
- header: "Workflow Review"
- question: "[N] tasks planned. Critical path: [A → B → C]. First checkpoint after Task [NN].\n\n**Resource flags** ([N]):\n- ⚠ Task [NN]: [gap] → applied: [alternative]\n- BLOCKED Task [NN]: [gap] → proposed: [remove/replace]\n\n**Planner defaults to confirm** ([N]):\n- Task [NN]: [assumption]\n\nDoes this order make sense? Decide on each flag and default, or accept as written."
- options: "Accept as written" | "I have decisions on the flags" | "Confirm defaults with edits" | "Change the task order"

If anything changes: apply the workflow-planner skill again with a `<revisions>` block (or continue with the full context plus `<revisions>`), then re-verify the file.

**`needs_input: checkpoint:decision`, `needs_input: checkpoint:human-action`, or `needs_input: checkpoint:human-verify`:** Present the question via the host conversation facility, collect the domain-expert answer, apply the role again.

## 8. Create Task Directories and State

```bash
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" validate-workflow
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" task-dirs
ls .research/tasks/
```

Update STATE.md: current phase = executing, current task = first eligible active task in dependency order, status = planning. Keep the Decisions Made list and append the workflow decisions.

Optional git record:
```bash
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" record --message "research: workflow defined; [N] tasks, [research type]" --files "${CHANGED_FILES[@]}"
```

</process>

<offer_next>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► WORKFLOW DEFINED ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Workflow: .research/WORKFLOW.md
Tasks:    [N] tasks ([type])

| # | Task | Type | Depends On |
|---|------|------|------------|
[table from WORKFLOW.md]

───────────────────────────────────────────

## ▶ Next Up

**Execute the first task**

`Run execute-task using the materio-task-executor skill 1`

<sub>Each dispatch starts with fresh worker context.</sub>

───────────────────────────────────────────

**Task management:**
- `Run add-task using the materio-research-explorer skill`; add a task to the workflow
- `Run remove-task using the materio-research-explorer skill [N]`; remove a task
- `Run archive-task using the materio-research-explorer skill [N]`; archive without deleting

</offer_next>

<success_criteria>
- [ ] Every question to the researcher was asked by this command, not by the agent
- [ ] Traditional workflow template presented as starting point
- [ ] User customizations captured (add/remove/reorder)
- [ ] Scope and exclusions explicitly confirmed
- [ ] Assumption interview run per task before planning; skips recorded as planner defaults
- [ ] Resource flags and planner defaults decided by the researcher
- [ ] Each task has: type, description, assumptions, inputs, outputs, dependencies
- [ ] Task directories created with at least two-digit numbering for every task count
- [ ] WORKFLOW.md verified on disk; commit only if commit_research is true
</success_criteria>
