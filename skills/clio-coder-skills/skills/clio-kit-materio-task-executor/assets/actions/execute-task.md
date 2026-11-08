# Action: execute-task

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
@.research/WORKFLOW.md
@.research/RESEARCH.md
@.research/LITERATURE.md
@.research/STATE.md
@.research/config.json
</execution_context>

<objective>
Execute a specific research task (or the full workflow sequentially) by applying the task-executor skill with the task's type and context. Verifies outputs on disk, runs advisory guardrails, updates task status, and creates outputs in .research/tasks/task-NN/.

**Orchestrator role:** Resolve target task, check dependencies, gate-confirm assumptions, apply the materio-task-executor skill with full task context, verify outputs exist, run guardrails and surface findings, update WORKFLOW.md and STATE.md, checkpoint, record.

**Why subagent:** Each task type (literature, computational, data-analysis, experimental-protocol) requires different reasoning depth. Fresh context per task = focused execution. The executor never asks the researcher anything; checkpoints come back here.
</objective>

<context>
Target: $ARGUMENTS (task number, or "all")
</context>

<process>

Worker write scope: the selected .research/tasks/task-NN/; include .research/LITERATURE.md only for an approved literature update.


An optional native fleet may require an existing Git worktree to enforce its
write boundary. Clio's v4 fleet does. If absent, keep the boundary and use direct
role execution with the selected task output grant followed by a separate read-only
review. Never initialize git to enable the fleet. If the host lacks worker
delegation, perform labeled artifact readback in the main conversation and disclose
that no independent worker review occurred.

## 1. Inspect Supplied Materials First

Use ls on `.research/data/` and read `.research/DATA-INDEX.md` before other
research inputs. For literature tasks, inspect these registered materials before
fetching anything. When general search is unavailable: build literature work from
supplied materials plus specific researcher-provided URLs. Request URLs with the
task confirmation if more coverage is needed. Honor web_search false by staying
with supplied materials and record coverage in Sources Reviewed in each summary
and any LITERATURE.md update.

### Validate Environment

```bash
test -f .research/WORKFLOW.md || echo "ERROR: No WORKFLOW.md. Run define-research-tasks using the materio-workflow-planner skill first."
cat .research/WORKFLOW.md
cat .research/STATE.md 2>/dev/null
cat .research/DATA-INDEX.md 2>/dev/null
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" config
```

## 2. Resolve Target Task

**If $ARGUMENTS is empty:** Run `python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" next-task`.
Use its actual dependency-ready pending/in-progress task ID in workflow order.
If id is null, show the blocked dependencies or that no pending task is ready;
do not invent a task number. Preserve the task confirmation and assumption gate.

**If $ARGUMENTS = "all":** Resolve the next eligible task with the same next-task helper after each completed task; preserve all interview/checkpoint gates.

**If $ARGUMENTS = number:** Find that specific task. Verify status and dependencies.

Always derive the two-digit form and the output directory:
```bash
# Bind N to the validated decimal task number in this call.
NN=$(printf '%02d' "$((10#$N))"); OUT=".research/tasks/task-$NN"; echo "$OUT"
# Validate the actual active workflow IDs before creating their directories.
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" task-dirs
```

**Dependency check:** If the target task has `Dependencies: Task XX`, verify Task XX is `☑ complete`. If not:
- Warn the user: "Task [N] depends on Task [XX] which is not complete."
- Ask via the host conversation facility: proceed anyway, or execute Task XX first?

## 3. Show Task Summary

Display the target task entry from WORKFLOW.md:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Task [NN]: [name]
 Type:     [literature | computational | experimental | data-analysis | analytical | writing]
 Inputs:   [what's needed]
 Outputs:  [what this produces]
 Assumptions: [list]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 4. Gate Check; Confirm Before Executing

Use the host conversation facility:
- header: "Execute Task [NN]: [name]"
- question: "Ready to execute this task?\n\n**Assumptions being made:**\n[list assumptions, marking any planner defaults not yet confirmed]\n\n**Expected output:**\n[expected outputs]\n\nAny assumptions wrong or missing data?"
- options: "Proceed" | "Update assumptions first" | "I need to upload data first; Run upload-data using the materio-research-explorer skill" | "Skip this task"

If assumptions are updated, write them back to the task entry in WORKFLOW.md before applying the role.

## 5. Handle Task Type Routing

**If type = `writing`:**
- Tell user: "This task involves writing. Use `Run wtfp using the materio-research-explorer skill` to bridge into wtf-p for paper writing."
- Exit gracefully.

**If type = `experimental`:** protocol-generation mode → protocol document + data recording template + checklist

**If type = `literature`:** supplied-corpus and provided-URL review mode → summary + BibTeX + LITERATURE.md updates; supplied materials first, provided-URL fetching only if allowed

**If type = `computational`:** script-generation mode → input files + submission script + analysis script + README

**If type = `data-analysis`:** check for registered data in DATA-INDEX.md and `.research/data/`; if none, stop and offer `Run upload-data using the materio-research-explorer skill` → analysis code + results summary + figure descriptions

**If type = `analytical`:** theoretical mode → derivation, validation, optional implementation

## 6. apply the materio-task-executor skill

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► EXECUTING TASK [NN]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Apply the materio-task-executor role skill using the assignment and full context; delegate if supported. The prompt carries:
- `<task>`; full task entry from WORKFLOW.md (with any updated assumptions)
- `<research>`; full RESEARCH.md
- `<literature>`; relevant LITERATURE.md sections
- `<virtual_lab>`; the relevant equipment and software rows from VIRTUAL-LAB.md, if it exists
- `<prior_outputs>`; outputs from dependency tasks (paths and SUMMARY.md contents)
- `<data_files>`; DATA-INDEX.md entries and paths under `.research/data/`
- `<web_allowed>`; true only when config permits URL fetching and the host URL-fetch capability is available
- `<provided_urls>`; the researcher's specific URLs, or none
- `<output_dir>`; `.research/tasks/task-[NN]/`

## 7. Handle Executor Return

**`task_complete:`:**

1. Use ls and read on every reported output and task-NN-SUMMARY.md. Verify outputs on disk. Self-report is not completion:
   ```bash
   ls -la .research/tasks/task-[NN]/
   test -s .research/tasks/task-[NN]/task-[NN]-SUMMARY.md && echo "SUMMARY OK" || echo "ERROR: no summary"
   ```
   Every file the executor listed under Outputs must exist and be non-empty. If any is missing, apply the role again naming the missing file; do not mark the task complete.

1b. Apply the materio-task-verifier skill to the approved task,
   assumptions, dependency context, actual output inventory and summary. Use a
   separate read-only worker with no write grant when available; otherwise perform
   labeled main-conversation readback and disclose that independent worker review
   was unavailable. Resolve every returned researcher checkpoint. Read/ls findings
   are artifact inspection, not command-backed scientific validation. Preserve
   host grounding diagnostics; a rejected receipt remains incomplete and cannot
   be promoted to completion. Advisory commands have separate evidence.

2. Advisory guardrails. Run what applies, read the findings and exit status, and surface them; nothing here deletes or rewrites on its own. Record a skipped check if a script or python3 is missing.
   Use bash to collect only existing files and keep stderr visible. The scripts
   accept files, not directory arguments; never mistake a skipped file for a
   successful scan. All findings remain advisory. Run these shell blocks only
   after deriving OUT as the selected task directory:
   ```bash
   if command -v python3 >/dev/null 2>&1; then
     shopt -s nullglob globstar
     physics_files=("$OUT"/**/*.md)
     citation_files=("$OUT"/**/*.bib "$OUT"/**/*summary*.md "$OUT"/**/*SUMMARY*.md)
     script_files=("$OUT"/**/*.py "$OUT"/**/*.sh)
     citation_flags=()
     [ "$(python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" config --key web_search)" = true ] || citation_flags=(--offline)
     if ((${#physics_files[@]})); then
       if python3 "${PACKAGE_ROOT}/assets/scripts/check_physics.py" "${physics_files[@]}"; then echo "CHECK EXIT: 0"; else echo "CHECK EXIT: $? (findings or incomplete coverage)"; fi
     else echo "CHECK SKIPPED: no applicable files"; fi
     if ((${#citation_files[@]})); then
       if python3 "${PACKAGE_ROOT}/assets/scripts/verify_citations.py" "${citation_flags[@]}" "${citation_files[@]}"; then echo "CHECK EXIT: 0"; else echo "CHECK EXIT: $? (findings or incomplete coverage)"; fi
     else echo "CHECK SKIPPED: no applicable files"; fi
     if ((${#script_files[@]})); then
       if python3 "${PACKAGE_ROOT}/assets/scripts/check_scripts.py" "${script_files[@]}"; then echo "CHECK EXIT: 0"; else echo "CHECK EXIT: $? (findings or incomplete coverage)"; fi
     else echo "CHECK SKIPPED: no applicable files"; fi
   fi
   ```
   Include any other actual reference-list files in citation_files after reading
   them. If a literature task updated `.research/LITERATURE.md`, read it back and
   include it in citation_files too. A nonzero checker exit records findings;
   it never authorizes deletion, edits or an automatic task rejection.
   If any finding is **IMPOSSIBLE** (physics), **NOT_FOUND** / **MISMATCH** (citations), or **WILL NOT RUN** (scripts): show the findings to the researcher and ask via the host conversation facility:
   - header: "Guardrail Findings; Task [NN]"
   - question: "[N] findings:\n\n1. [file:line] [finding] → [suggested fix or closest Crossref match]\n2. ...\n\nHow should I handle them?"
   - options: "Fix them; resume the executor" | "They're fine; note and continue" | "I'll decide per finding"
   Apply the task-executor skill again with the accepted fixes when asked, then read all changed files back and rerun the applicable advisory checks before completing. Warnings (IMPLAUSIBLE, UNVERIFIABLE, SUSPECT) are listed in the completion output and noted in the task SUMMARY, not gated.

3. After the researcher has resolved every gated finding and the outputs have been read back, mark task as `☑ complete` in WORKFLOW.md. Update STATE.md: current task = next pending, last completed = [NN].

4. Checkpoint if `auto_checkpoint` is true (default):
   ```bash
   if [ "$(python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" config --key auto_checkpoint)" = true ]; then
  python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" checkpoint save after-task-NN
fi
   ```

5. Optional git record:
   ```bash
   python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" record --message "research(task-[NN]): [task name]; complete" --files "${CHANGED_FILES[@]}"
   ```

**`needs_input: checkpoint:decision`, `needs_input: checkpoint:human-action`, or `needs_input: checkpoint:human-verify`:**
- Present what's needed (data, decision, domain knowledge) via the host conversation facility
- Apply the task-executor skill again with a `<resume>` block containing the answer (or continue with the full context plus `<resume>`)

**`task_blocked:`:**
- Show blocker clearly
- Offer via the host conversation facility: upload missing data, revise assumptions, execute the blocking task first, or skip this task

**If $ARGUMENTS = "all":** After each task completes, loop to the next pending task. Stop the loop at any needs_input: checkpoint or task_blocked: return and at the first `writing` task.

</process>

<offer_next>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► TASK [NN] COMPLETE ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Task:    [NN]; [name]
Status:  [N]/[total] tasks complete

Evidence ladder (report each rung with its actual value; never promote a rung):
  Artifacts inspected:     [N] files under .research/tasks/task-[NN]/ read back; verifier
                           receipt [task_complete | task_blocked | checkpoint] (read/ls only)
  Prepared, not executed:  [list scripts, input decks, submission files, proposed experiments]
  Executed computation:    [none | the exact commands the researcher or orchestrator actually
                           ran in this session, with exit status and output paths]
  Advisory checks:         physics [exit N | skipped], citations [exit N | skipped, offline?],
                           scripts [exit N | skipped]; [N] findings resolved, [N] warnings noted
  Scientific validation:   not performed by this workflow; the researcher owns physical,
                           statistical and experimental validation of these outputs

───────────────────────────────────────────

## ▶ Next Up

**Execute the next dependency-ready task**
`Run execute-task using the materio-task-executor skill `

**Or check progress**
`Run progress using the materio-research-explorer skill`

<sub>Each dispatch starts with fresh worker context.</sub>

───────────────────────────────────────────

</offer_next>

<success_criteria>
- [ ] Target task resolved (from argument or next pending); at least two-digit directory used
- [ ] Dependency check completed
- [ ] Gate confirmation always shown with assumptions listed
- [ ] Task type correctly routed; web availability passed explicitly
- [ ] Outputs read back on disk with ls and test -s before marking complete
- [ ] Completion output states the evidence ladder honestly: inspected, prepared, executed, advisory, validation not performed
- [ ] Guardrail findings surfaced to the researcher; nothing auto-removed
- [ ] WORKFLOW.md and STATE.md updated
- [ ] Checkpoint saved if auto_checkpoint; commit only if commit_research is true
</success_criteria>
