# Action: progress

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


<execution_context>
@.research/STATE.md
@.research/WORKFLOW.md
@.research/RESEARCH.md
</execution_context>

<objective>
Display a rich progress report with statusline, recent task completions, current position, and routing to the next action. Equivalent to wtf-p action progress but for research workflow.
</objective>

<process>

## 1. Validate

```bash
[ ! -f .research/RESEARCH.md ] && echo "No research project. Run identify-research using the materio-research-explorer skill." && exit 0
cat .research/STATE.md 2>/dev/null
cat .research/WORKFLOW.md 2>/dev/null
cat .research/RESEARCH.md 2>/dev/null | head -20
```

## 2. Build Statusline

```
Materio ► [N]/[total] tasks ◆ [phase: exploring|reviewing|executing|writing] ► [domain]
```

Display this first, before all other output.

## 3. Calculate Progress Metrics

From WORKFLOW.md, count:
- Total active tasks
- Complete tasks (☑ complete)
- In-progress tasks
- Pending tasks
- Archived tasks

Determine current phase:
- No LITERATURE.md → "exploring"
- LITERATURE.md exists, no WORKFLOW.md → "reviewing"
- WORKFLOW.md exists, tasks pending → "executing"
- All tasks complete → "writing"

## 4. Show Recent Work

Find task output files modified in the last 7 days:
```bash
find .research/tasks/ -newer .research/RESEARCH.md -name "*.md" 2>/dev/null | head -5
```

Show 1-line summary of recent task completions.

## 5. Display Progress Report

```
Materio ► [statusline]

# [Research Prompt]

**Phase:**    [exploring | reviewing | executing | writing]
**Progress:** [████████░░] [N]/[total] tasks
**Domain:**   [domain + sub-field]

## Recent Work
- Task [N] ([name]): [1-line output summary]

## Current Position
[Next pending task or current state]

## Key Decisions Made
[from STATE.md]

## Open Questions
[unresolved assumptions or decisions from WORKFLOW.md]
```

## 6. Route to Next Action

**Phase = exploring:** → `Run literature-review using the materio-literature-reviewer skill`
**Phase = reviewing:** → `Run define-virtual-lab using the materio-lab-definer skill` if VIRTUAL-LAB.md is missing, otherwise `Run define-research-tasks using the materio-workflow-planner skill`
**Phase = executing, task in-progress:** → `Run execute-task using the materio-task-executor skill [N]`
**Phase = executing, next pending task:** → `Run execute-task using the materio-task-executor skill `
**Phase = writing:** → `Run wtfp using the materio-research-explorer skill` or `wtf-p action progress`

Show the routed command clearly. If the researcher chooses it, follow its prompt flow with the existing context.

</process>

<success_criteria>
- [ ] Statusline displayed first
- [ ] Phase correctly determined
- [ ] Task progress shown with counts and visual bar
- [ ] Recent work summarized
- [ ] Smart routing to next command
</success_criteria>
