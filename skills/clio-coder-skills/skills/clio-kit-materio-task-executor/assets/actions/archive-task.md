# Action: archive-task

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
Move a task to the Archived Tasks section of WORKFLOW.md. Archived tasks are preserved for reference but excluded from Run execute-task using the materio-task-executor skill and progress tracking. Preferred over remove-task when the task might be relevant later.
</objective>

<context>
Task number: $ARGUMENTS
</context>

<process>

## 1. Validate

```bash
test -f .research/WORKFLOW.md || echo "ERROR: No WORKFLOW.md."
[ -z "$ARGUMENTS" ] && echo "ERROR: Provide task number. Usage: Run archive-task using the materio-research-explorer skill 3" && exit 1
```

## 2. Show Task and Gather Reason

Display the task entry.

Use the host conversation facility:
- header: "Archive Task [N]: [name]"
- question: "Why are you archiving this task? (This note is saved for future reference)\n\n[full task block]"
- options: "No longer needed for this research" | "Deferred to future work" | "Replaced by another task" | "Resource/time constraint" | "Other reason"

## 3. Move Task to Archived Section

- Change task status to `☐ archived`
- Move the task block from the active Tasks section to the `## Archived Tasks` section in WORKFLOW.md
- Add archive note and reason

## 4. Update Dependencies

If any active task depends on this task, warn user and ask how to handle the dependency.

## 5. Record

Only if `commit_research` is true in `.research/config.json`:
```bash
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" record --message "research: archive task [N]; [reason]" --files "${CHANGED_FILES[@]}"
```

</process>

<success_criteria>
- [ ] Task moved to Archived section (not deleted)
- [ ] Archive reason recorded
- [ ] Active task list clean
- [ ] Dependencies in other tasks flagged if affected
</success_criteria>
