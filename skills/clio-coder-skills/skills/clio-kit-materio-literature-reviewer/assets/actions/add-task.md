# Action: add-task

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
@.research/WORKFLOW.md
</execution_context>

<objective>
Add a new task to WORKFLOW.md through a short interview. Inserts at the end or at a specified position.
</objective>

<context>
No arguments.
</context>

<process>

## 1. Read Current Workflow

```bash
test -f .research/WORKFLOW.md || echo "ERROR: No WORKFLOW.md. Run define-research-tasks using the materio-workflow-planner skill first."
cat .research/WORKFLOW.md
```

Count current tasks and show the list.

## 2. Gather Task Details

Use the host conversation facility:
- header: "Add Task"
- question: "Define the new task:\n1. **Name**: Short descriptive name\n2. **Type**: literature | experimental | computational | data-analysis | analytical | writing\n3. **Description**: What needs to be done?\n4. **Inputs**: What data or prior task output does this need?\n5. **Expected output**: What will this task produce?\n6. **Position**: After which task? (default: end of list)\n7. **Dependencies**: Which tasks must complete first?"
- options: "Provided all details" | "Guide me through each field"

## 3. Generate New Task Entry

Run `python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" next-task-id` to allocate a stable ID above active, archived, and on-disk IDs. Insert the entry at the chosen presentation position without renumbering any existing task, directory, output, or dependency. IDs use at least two digits (Task 07, Task 12). Interview the new task assumptions using the task-type guidance before finalizing its entry.

Write the new task block into WORKFLOW.md at the correct position.

## 4. Create Task Directory

```bash
mkdir -p ".research/tasks/task-$(printf '%02d' "$((10#$N))")"
```

## 5. Record

Only if `commit_research` is true in `.research/config.json`:
```bash
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" record --message "research: add task [NN]; [task name]" --files "${CHANGED_FILES[@]}"
```

</process>

<success_criteria>
- [ ] New task has all required fields
- [ ] Task identity is unique and stable; numbering gaps are allowed
- [ ] Task directory created
- [ ] WORKFLOW.md updated; committed only if commit_research is true
</success_criteria>
