# Action: remove-task

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
Permanently remove a task from WORKFLOW.md after user confirmation. Use Run archive-task using the materio-research-explorer skill instead if you want to preserve it for reference.
</objective>

<context>
Task number: $ARGUMENTS
</context>

<process>

## 1. Validate

```bash
test -f .research/WORKFLOW.md || echo "ERROR: No WORKFLOW.md."
[ -z "$ARGUMENTS" ] && echo "ERROR: Provide task number. Usage: Run remove-task using the materio-research-explorer skill 3" && exit 1
cat .research/WORKFLOW.md
```

## 2. Show Task and Confirm

Display the full task entry to be removed.

**Check for dependents:** Find any tasks that list this task as a dependency. Warn the user if other tasks depend on the one being removed.

Use the host conversation facility:
- header: "Remove Task [N]: [name]?"
- question: "This will permanently delete Task [N] from WORKFLOW.md.\n\n[full task block]\n\n[If dependents exist: WARNING; Tasks [X, Y] depend on this task. Removing it will break their dependency chain.]\n\nAlternative: `Run archive-task using the materio-research-explorer skill [N]` preserves it for reference.\n\nProceed with removal?"
- options: "Yes, remove it" | "Archive it instead" | "Cancel"

## 3. Remove Task from WORKFLOW.md

Edit WORKFLOW.md to remove the task block. Update dependency references in other tasks if needed.

## 4. Record

Only if `commit_research` is true in `.research/config.json`:
```bash
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" record --message "research: remove task [N]; [task name]" --files "${CHANGED_FILES[@]}"
```

</process>

<success_criteria>
- [ ] Task displayed before removal
- [ ] Dependent tasks warned if applicable
- [ ] User explicitly confirmed removal
- [ ] Task block cleanly removed from WORKFLOW.md
- [ ] Dependency references updated in remaining tasks
</success_criteria>
